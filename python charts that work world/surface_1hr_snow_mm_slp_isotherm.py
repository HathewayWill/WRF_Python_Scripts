#!/usr/bin/env python3
"""
SFC_1hrSnow_SLP_mm.py

Plot WRF mean sea-level pressure (SLP, hPa) and 1-hour accumulated
snowfall (mm, from SNOWH) on a Cartopy map.

Playbook v3 features:
    * Supports multiple wrfout_<domain>* files with one or more timesteps.
    * Supports a single wrfout file containing many timesteps.
    * One frame = one (file, time_index) pair (PNG).
    * Handles static and moving nests by recomputing geometry per frame.
    * NetCDF4 + wrf-python only for field access (no xarray).

STANDALONE REPAIR NOTE: Snowfall water equivalent uses ACSNOW (or labeled
SNOWNC grid-scale accumulation) when available. Otherwise the script plots
a clearly marked snowpack net change or snapshot, NOT an invented snowfall
accumulation. A red map footer and colorbar identify the actual product.
No wrf_accumulation_fix.py module is required.

MOVING-NEST UPDATE: When prior and current WRF grids overlap, accumulated
snowfall is geographically interpolated to the current grid; pixels without
earlier coverage remain missing and are hatched. Interpolation is not
conservative budget remapping. No companion helper module required.
"""

###############################################################################
# Imports (clean, ordered)
###############################################################################
import glob
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import cartopy.crs as crs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import wrf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
from netCDF4 import Dataset
from PIL import Image
from scipy.ndimage import gaussian_filter
from wrf import ALL_TIMES, to_np  # ALL_TIMES kept for physics invariants

###############################################################################
# Warning suppression
###############################################################################

###############################################################################
# Standalone WRF snow calculation and time pairing (no separate Python module)
#
# Source distinction:
#   ACSNOW: model accumulated snowfall water equivalent (kg m-2 = mm H2O)
#   SNOWNC: accumulated grid-scale snow (not necessarily total snowfall)
#   SNOW:   instantaneous snowpack water equivalent (kg m-2)
#   SNOWH:  instantaneous snowpack depth (m)
# In the absence of a valid complete accumulation, a snapshot is plotted
# with an explicit description. Never label a snapshot as an accumulation.
###############################################################################

def plot_time_pairs(dated_frames, hours):
    """Every frame is drawable. Use exact UTC endpoints when available.

    A missing earlier endpoint causes a labeled *snapshot* rather than an
    invented hour/day accumulation or a silently missing PNG. A self-pair
    signals that snapshot mode to snow_diagnostic().
    """
    from datetime import timedelta, timezone
    if hours <= 0:
        raise ValueError("hours must be positive")
    times = {}
    for path, index, valid in dated_frames:
        if isinstance(valid, np.datetime64):
            valid = datetime.fromisoformat(np.datetime_as_string(valid, unit="us"))
        if valid.tzinfo is not None:
            valid = valid.astimezone(timezone.utc).replace(tzinfo=None)
        if valid in times:
            print(f"WARNING: duplicate WRF time {valid}; using most recently listed frame", flush=True)
        times[valid] = (path, index)
    result = []
    for valid in sorted(times):
        current = times[valid]
        old = times.get(valid - timedelta(hours=hours))
        if old is None:
            print(f"INFO: no exact {hours}-hour predecessor for {valid}; plotting snow snapshot", flush=True)
            old = current
        result.append((*current, *old))
    return result


def _snow_array(nc, variable_name, time_index):
    if variable_name not in nc.variables:
        return None
    var = nc.variables[variable_name]
    dimensions = getattr(var, "dimensions", ())
    arr = var[time_index, ...] if dimensions and dimensions[0] == "Time" else var[...]
    arr = np.asarray(np.ma.filled(np.ma.asarray(arr, dtype=float), np.nan)).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"WRF {variable_name} expected a 2D field; got {arr.shape}")
    return arr


def _snow_geo_coords(nc, time_index):
    """Read actual WRF cell coordinates at this model time, including moving nests."""
    lat_name = next((n for n in ('XLAT', 'XLAT_M') if n in nc.variables), None)
    lon_name = next((n for n in ('XLONG', 'XLONG_M') if n in nc.variables), None)
    if not lat_name or not lon_name:
        raise ValueError('WRF XLAT/XLONG geographic coordinates are missing')
    return (_snow_array(nc, lat_name, time_index),
            _snow_array(nc, lon_name, time_index))


def _snow_geo_remap(previous_values, previous_lat, previous_lon,
                    current_lat, current_lon):
    """Interpolate previous field to current WRF geographic locations.

    Fast path for stationary grids. For shifted meshes, piecewise-linear
    triangulation (no extrapolation) yields NaN outside previous geographic
    coverage. Longitude wrapping handles the date line.

    This is an APPROXIMATE geographic interpolation, not mass-conservative
    remapping or a reconstruction of snow budget history on moving WRF cells.
    """
    old = np.ma.filled(np.ma.asarray(previous_values, dtype=float), np.nan)
    olat = np.ma.filled(np.ma.asarray(previous_lat, dtype=float), np.nan)
    olon = np.ma.filled(np.ma.asarray(previous_lon, dtype=float), np.nan)
    nlat = np.ma.filled(np.ma.asarray(current_lat, dtype=float), np.nan)
    nlon = np.ma.filled(np.ma.asarray(current_lon, dtype=float), np.nan)
    if olat.shape != old.shape or olon.shape != old.shape or nlat.shape != nlon.shape:
        raise ValueError('Snow field and WRF latitude/longitude shapes do not match')
    if (olat.shape == nlat.shape and
        np.allclose(olat, nlat, atol=1.e-5, rtol=0, equal_nan=False) and
        np.allclose((olon - nlon + 180) % 360 - 180, 0,
                    atol=1.e-5, rtol=0, equal_nan=False)):
        return old.copy(), False

    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError
    finite = np.isfinite(nlat) & np.isfinite(nlon)
    if not finite.any():
        raise ValueError('Current nest lacks valid geographic coordinates')
    clat = float(np.nanmedian(nlat[finite]))
    lon_rad = np.deg2rad(nlon[finite])
    ref_lon = float(np.rad2deg(np.arctan2(np.mean(np.sin(lon_rad)),
                                          np.mean(np.cos(lon_rad)))))
    xscale = max(float(np.cos(np.deg2rad(clat))), 0.05)

    def locations(lat, lon):
        x = ((lon - ref_lon + 180) % 360 - 180) * xscale
        return np.column_stack((x.ravel(), lat.ravel()))

    src = locations(olat, olon)
    dst = locations(nlat, nlon)
    valid_src = np.all(np.isfinite(src), axis=1) & np.isfinite(old.ravel())
    valid_dst = np.all(np.isfinite(dst), axis=1)
    if np.count_nonzero(valid_src) < 3:
        raise ValueError('Too few valid snow locations in previous nest')

    # Restrict Delaunay triangulation near the target domain when possible.
    low, high = np.min(dst[valid_dst], axis=0), np.max(dst[valid_dst], axis=0)
    span = np.maximum(high - low, 1.e-4)
    margin = np.maximum(0.08 * span, 0.002)
    near = valid_src & np.all((src >= low-margin) & (src <= high+margin), axis=1)
    if np.count_nonzero(near) >= 4:
        valid_src = near
    try:
        interpolator = LinearNDInterpolator(
            src[valid_src], old.ravel()[valid_src], fill_value=np.nan)
        remapped = np.full(len(dst), np.nan)
        remapped[valid_dst] = interpolator(dst[valid_dst])
        remapped = remapped.reshape(nlat.shape)
    except (QhullError, ValueError) as exc:
        raise ValueError(f'Snow moving-grid interpolation failed: {exc}') from exc
    return remapped, True


def _snow_missing_hatching(ax, lons, lats, plotted):
    """Hatch cells without prior geographic overlap or with invalid differences."""
    field = np.ma.filled(np.ma.asarray(plotted, dtype=float), np.nan)
    missing = ~np.isfinite(field)
    if np.any(missing) and not np.all(missing):
        ax.contourf(lons, lats, missing.astype(float), levels=(0.5, 1.5),
                    colors='none', hatches=['////'],
                    transform=crs.PlateCarree(), zorder=8)


def snow_diagnostic(current, current_idx, previous, previous_idx,
                    kind, scale=1.0, label='snow'):
    """Return (geographic snow field, scientifically explicit description).

    The previous accumulated field is compared at the CURRENT geographic
    locations. Outside previous nest coverage, values remain missing (NaN).
    Accumulator resets are masked, while negative snowpack net changes are
    retained as valid net changes. Missing inputs yield labeled snapshots.
    """
    depth = kind == 'snow_depth'
    if kind not in ('snow_depth', 'snow_swe'):
        raise ValueError(f'Unknown snow product kind: {kind}')
    unit = ('in' if (abs(scale-0.3937007874015748) < 1.e-6 or
                     abs(scale-0.03937007874015748) < 1.e-6) else 'mm')
    descriptor = ('estimated snowfall depth (assumed 10:1)' if depth else
                  'snowfall water equivalent')
    same_frame = (current_idx == previous_idx and
                  getattr(current, 'filepath', lambda: None)() ==
                  getattr(previous, 'filepath', lambda: None)())
    aligned = False
    moved = False
    clat = clon = plat = plon = None
    if not same_frame:
        try:
            clat, clon = _snow_geo_coords(current, current_idx)
            plat, plon = _snow_geo_coords(previous, previous_idx)
            aligned = True
        except (ValueError, KeyError, IndexError) as exc:
            print(f'WARNING {label}: cannot compare nest geography: {exc}', flush=True)

    if aligned:
        for name, qualifier in (('ACSNOW', 'model accumulated snowfall'),
                                ('SNOWNC', 'grid-scale accumulated snowfall')):
            new = _snow_array(current, name, current_idx)
            old = _snow_array(previous, name, previous_idx)
            if new is None or old is None:
                continue
            if new.shape != clat.shape or old.shape != plat.shape:
                print(f'WARNING {label}: {name} grid shape mismatch', flush=True)
                continue
            try:
                prior, moved = _snow_geo_remap(old, plat, plon, clat, clon)
            except (ValueError, IndexError) as exc:
                print(f'WARNING {label}: cannot align {name}: {exc}', flush=True)
                continue
            change = new - prior
            invalid = np.isfinite(change) & (change < -0.02)
            change[invalid] = np.nan
            tiny = np.isfinite(change) & (change < 0.0)
            change[tiny] = 0.0
            coverage = 100.0 * np.count_nonzero(np.isfinite(change)) / change.size
            if coverage < 1.0:
                print(f'WARNING {label}: {name} lacks reliable previous coverage', flush=True)
                continue
            description = f'{qualifier}: {descriptor} ({unit})'
            if moved:
                description += f'; moving-grid INTERPOLATED, {coverage:.1f}% overlap (approx.)'
            if np.any(invalid):
                description += f'; {np.count_nonzero(invalid)} reset cells masked'
            print(f'INFO {label}: {description}', flush=True)
            return change * scale, description

        fallback = 'SNOWH' if depth else 'SNOW'
        new = _snow_array(current, fallback, current_idx)
        old = _snow_array(previous, fallback, previous_idx)
        if new is not None and old is not None and new.shape == clat.shape and old.shape == plat.shape:
            try:
                prior, moved = _snow_geo_remap(old, plat, plon, clat, clon)
                change = new - prior
                coverage = 100.0 * np.count_nonzero(np.isfinite(change)) / change.size
                if coverage >= 1.0:
                    factor = ((1000.0 if unit == 'mm' else 39.37007874015748)
                              if depth else scale)
                    description = ('NET snowpack-depth change (NOT snowfall)' if depth else
                                   'NET snowpack-SWE change (NOT snowfall)') + f' ({unit})'
                    if moved:
                        description += f'; moving-grid INTERPOLATED, {coverage:.1f}% overlap (approx.)'
                    print(f'WARNING {label}: {description}', flush=True)
                    return change * factor, description
            except (ValueError, IndexError) as exc:
                print(f'WARNING {label}: no net snowpack change: {exc}', flush=True)

    # No usable baseline: plot a truthful instantaneous/since-init diagnostic.
    if depth:
        candidates = (
            ('SNOWH', (1000.0 if unit == 'mm' else 39.37007874015748),
             f'SNAPSHOT snowpack depth ({unit}); NOT interval snowfall'),
            ('SNOW', scale, f'SNAPSHOT 10:1 estimated snowpack depth ({unit}); NOT interval snowfall'))
    else:
        candidates = (
            ('SNOW', scale, f'SNAPSHOT snowpack SWE ({unit}); NOT interval snowfall'),
            ('ACSNOW', scale, f'SINCE-INITIALIZATION ACSNOW ({unit}); NOT requested interval'),
            ('SNOWNC', scale, f'SINCE-INITIALIZATION SNOWNC ({unit}); NOT requested interval'),
            ('SNOWH', scale * 100,
             f'SNAPSHOT snowpack SWE ({unit}) estimated 10:1 from SNOWH; NOT interval snowfall'))
    for name, factor, description in candidates:
        values = _snow_array(current, name, current_idx)
        if values is not None:
            print(f'WARNING {label}: {description}', flush=True)
            return values * factor, description
    raise RuntimeError(f'{label}: no snow fields ACSNOW, SNOWNC, SNOW or SNOWH found')

warnings.filterwarnings("default")

###############################################################################
# Canonical helper function block (v9 – contiguous, order-locked)
###############################################################################


def add_feature(
    ax, category, scale, facecolor, edgecolor, linewidth, name, zorder=None, alpha=None
):
    feature = cfeature.NaturalEarthFeature(
        category=category,
        scale=scale,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        name=name,
        zorder=zorder,
        alpha=alpha,
    )
    ax.add_feature(feature)


def parse_valid_time_from_wrf_name(path: str) -> datetime:
    base = os.path.basename(path)

    match = re.search(
        r"wrfout_.*?_(\d{4}-\d{2}-\d{2})_(\d{2}[:_]\d{2}[:_]\d{2})",
        base,
    )
    if match:
        date_str = match.group(1)
        time_str = match.group(2).replace("_", ":")
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H:%M:%S")
        except Exception:
            pass

    try:
        year = base[11:15]
        month = base[16:18]
        day = base[19:21]
        hour = base[22:24]
        minute = base[25:27]
        second = base[28:30]
        return datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(second)
        )
    except Exception:
        return datetime.utcfromtimestamp(os.path.getmtime(path))


def get_valid_time(ncfile: Dataset, ncfile_path: str, time_index: int) -> datetime:
    try:
        valid = wrf.extract_times(ncfile, timeidx=time_index)

        if isinstance(valid, np.ndarray):
            valid = valid.item()

        if isinstance(valid, np.datetime64):
            valid = valid.astype("datetime64[ms]").tolist()

        if isinstance(valid, datetime):
            return valid
    except Exception:
        pass

    return parse_valid_time_from_wrf_name(ncfile_path)


def compute_grid_and_spacing(lats, lons):
    lats_np = to_np(lats)
    lons_np = to_np(lons)

    dx, dy = mpcalc.lat_lon_grid_deltas(lons_np, lats_np)

    dx_km = dx.to(units.kilometer)
    dy_km = dy.to(units.kilometer)

    dx_km_rounded = np.round(dx_km.magnitude, 2)
    dy_km_rounded = np.round(dy_km.magnitude, 2)

    avg_dx_km = round(np.mean(dx_km_rounded), 2)
    avg_dy_km = round(np.mean(dy_km_rounded), 2)

    if avg_dx_km >= 9 or avg_dy_km >= 9:
        extent_adjustment = 0.50
        label_adjustment = 0.35
    elif 3 < avg_dx_km < 9 or 3 < avg_dy_km < 9:
        extent_adjustment = 0.25
        label_adjustment = 0.20
    else:
        extent_adjustment = 0.15
        label_adjustment = 0.15

    return lats_np, lons_np, avg_dx_km, avg_dy_km, extent_adjustment, label_adjustment


def add_latlon_gridlines(ax):
    gl = ax.gridlines(
        crs=crs.PlateCarree(),
        draw_labels=True,
        linestyle="--",
        color="black",
        alpha=0.5,
    )

    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.x_inline = False
    gl.top_labels = False
    gl.right_labels = False

    return gl


def plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km):
    plot_extent = [
        lons_np.min(),
        lons_np.max(),
        lats_np.min(),
        lats_np.max(),
    ]

    cities_within_extent = cities.cx[
        plot_extent[0] : plot_extent[1],
        plot_extent[2] : plot_extent[3],
    ]

    sorted_cities = cities_within_extent.sort_values(
        by="POP_MAX", ascending=False
    ).head(150)

    if sorted_cities.empty:
        return

    if avg_dx_km >= 9 or avg_dy_km >= 9:
        min_distance = 1.0
    elif 3 < avg_dx_km < 9 or 3 < avg_dy_km < 9:
        min_distance = 0.75
    else:
        min_distance = 0.40

    gdf_sorted = gpd.GeoDataFrame(
        sorted_cities,
        geometry=gpd.points_from_xy(
            sorted_cities.LONGITUDE,
            sorted_cities.LATITUDE,
        ),
    )

    selected_rows = []
    selected_geoms = []

    for row in gdf_sorted.itertuples():
        geom = row.geometry
        if not selected_geoms:
            selected_geoms.append(geom)
            selected_rows.append(row)
        else:
            distances = [g.distance(geom) for g in selected_geoms]
            if min(distances) >= min_distance:
                selected_geoms.append(geom)
                selected_rows.append(row)

    if not selected_rows:
        return

    filtered_cities = gpd.GeoDataFrame(selected_rows).set_geometry("geometry")

    for city_name, loc in zip(filtered_cities.NAME, filtered_cities.geometry):
        ax.plot(
            loc.x,
            loc.y,
            marker="o",
            markersize=6,
            color="r",
            transform=crs.PlateCarree(),
            clip_on=True,
        )
        ax.text(
            loc.x,
            loc.y,
            city_name,
            transform=crs.PlateCarree(),
            ha="center",
            va="bottom",
            fontsize=11,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.08",
                facecolor="white",
                alpha=0.4,
            ),
            clip_on=True,
        )


def handle_domain_continuity_and_polar_mask(lats_np, lons_np, *fields):
    """
    Detect and correct dateline continuity and polar masking for WRF domains.

    Ensures proper handling of longitude wrapping across the 180° meridian
    and masking for domains including polar caps.

    This function is field-agnostic: pass any number of fields (or none).
    All provided fields are reordered/masked consistently with lats/lons.
    """
    lats_min = np.nanmin(lats_np)
    lats_max = np.nanmax(lats_np)
    lons_min = np.nanmin(lons_np)
    lons_max = np.nanmax(lons_np)

    lon_span = lons_max - lons_min
    dateline_crossing = lon_span > 180.0
    polar_domain = (abs(lats_min) > 70.0) or (abs(lats_max) > 70.0)

    fields_out = list(fields)

    if dateline_crossing:
        lons_wrapped = np.where(lons_np < 0.0, lons_np + 360.0, lons_np)
        sort_idx = np.argsort(lons_wrapped[0, :])

        lons_np = lons_wrapped[..., sort_idx]
        lats_np = lats_np[..., sort_idx]
        fields_out = [(f[..., sort_idx] if f is not None else None) for f in fields_out]

    if polar_domain and dateline_crossing:
        polar_cap_lat = 88.0
        polar_mask = (lats_np >= polar_cap_lat) | (lats_np <= -polar_cap_lat)

        fields_out = [
            (np.ma.masked_where(polar_mask, f) if f is not None else None)
            for f in fields_out
        ]

    return (lats_np, lons_np, *fields_out)


###############################################################################
# Natural Earth features (features list can differ from golden script)
###############################################################################
# List of Natural Earth features to add (keep commented-out options intact)
features = [
    ("physical", "10m", cfeature.COLORS["land"], "black", 0.50, "minor_islands"),
    ("physical", "10m", "none", "black", 0.50, "coastline"),
    ("physical", "10m", cfeature.COLORS["water"], None, None, "ocean_scale_rank", -1),
    ("physical", "10m", cfeature.COLORS["water"], "lightgrey", 0.75, "lakes", 0),
    ("cultural", "10m", "none", "grey", 1.00, "admin_1_states_provinces", 2),
    ("cultural", "10m", "none", "black", 1.50, "admin_0_countries", 2),
    ("cultural", "10m", "none", "black", 0.60, "admin_2_counties", 2, 0.6),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_lake_centerlines"),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_north_america", None), 0.75),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_australia", None), 0.75),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_europe", None), 0.75),
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None,
    #  "lakes_north_america", None), 0.75),
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None,
    #  "lakes_australia", None), 0.75),
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None,
    #  "lakes_europe", None), 0.75),
]

###############################################################################
# Cities (module scope)
###############################################################################
# Load populated places (cities) once per worker process
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


###############################################################################
# SLP + 1-hour snow (mm) plotting for one (file, time_index) frame
###############################################################################
def process_frame(args):
    """
    Process a single frame: one (file, time_index) pair.

    Steps:
        * Read SLP and cumulative snow depth at this time.
        * Read cumulative snow at previous frame (if any).
        * Compute 1-hour accumulation as difference of cumulative totals (mm).
        * Build map (features, cities, gridlines).
        * Plot:
            - Smoothed SLP contours (hPa) with H/L markers.
            - 1-hour accumulated snow (mm) as filled contours.
        * Save a PNG with a valid-time-based timestamp.
    """
    (
        ncfile_path,
        time_index,
        prev_ncfile_path,
        prev_time_index,
        domain,
        path_figures,
    ) = args

    # Open current WRF file
    if prev_ncfile_path is None or prev_time_index is None:
        return  # No complete accumulation window; do not fabricate zero.

    ncfile = Dataset(ncfile_path)

    # Valid time for current frame
    valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
    print(f"Plotting data: {valid_dt:%H:%M:%SZ %Y-%m-%d}")

    # -------------------------------------------------------------------------
    # Physics: SLP and cumulative snow depth (SNOWH → mm)
    # -------------------------------------------------------------------------
    slp = wrf.getvar(ncfile, "slp", timeidx=time_index)
    # Exact 1-hour UTC accumulation, source: ACSNOW.
    # Snow depth is an explicit 10:1 estimate, not a change in SNOWH.
    with Dataset(prev_ncfile_path) as _previous_accum_nc:
        one_hour_snow, _snow_description = snow_diagnostic(
            ncfile, time_index, _previous_accum_nc, prev_time_index,
            'snow_depth', scale=10.0, label='surface_1hr_snow_mm_slp_isotherm.py')
    earliest_dt = valid_dt - timedelta(hours=1)
    latest_dt = valid_dt



    # -------------------------------------------------------------------------
    # Additional variables for physics invariants (not plotted, kept as-is)
    # -------------------------------------------------------------------------
    temp = wrf.getvar(ncfile, "T2", timeidx=time_index)


    # -------------------------------------------------------------------------
    # Smoothing (sigma preserved from original)
    # -------------------------------------------------------------------------
    slp_np = to_np(slp)
    smooth_slp = gaussian_filter(slp_np, sigma=5.0)
    smooth_temp = gaussian_filter(to_np(temp), sigma=1.0)

    # -------------------------------------------------------------------------
    # Geometry: lat/lon, grid spacing, extent (moving-nest safe)
    # -------------------------------------------------------------------------
    lats, lons = wrf.latlon_coords(slp)
    (
        lats_np,
        lons_np,
        avg_dx_km,
        avg_dy_km,
        extent_adjustment,
        label_adjustment,
    ) = compute_grid_and_spacing(lats, lons)

    cart_proj = wrf.get_cartopy(slp)

    # -------------------------------------------------------------------------
    # Dateline continuity and polar masking (v9 canonical helper)
    # -------------------------------------------------------------------------
    one_hour_snow_np = to_np(one_hour_snow)
    (
        lats_np,
        lons_np,
        smooth_slp,
        one_hour_snow_np,
    ) = handle_domain_continuity_and_polar_mask(
        lats_np,
        lons_np,
        smooth_slp,
        one_hour_snow_np,
    )

    # -------------------------------------------------------------------------
    # Figure and map setup
    # -------------------------------------------------------------------------
    dpi = plt.rcParams.get("figure.dpi", 400)
    fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

    # Map extent: slightly larger than model domain
    ax.set_extent(
        [
            lons_np.min() - extent_adjustment,
            lons_np.max() + extent_adjustment,
            lats_np.min() - extent_adjustment,
            lats_np.max() + extent_adjustment,
        ],
        crs=crs.PlateCarree(),
    )

    # Base land feature
    ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])

    # Additional Natural Earth features
    for feat in features:
        add_feature(ax, *feat)

    # Plot cities (filtered and thinned)
    plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)

    # Add lat/lon gridlines
    add_latlon_gridlines(ax)

    ax.tick_params(labelsize=12, width=2)

    # -------------------------------------------------------------------------
    # SLP contours (hPa) with H/L markers
    # -------------------------------------------------------------------------
    # Contour interval adjustment (physics invariant)
    if avg_dx_km >= 9 or avg_dy_km >= 9:
        contour_interval = 4
    else:
        contour_interval = 2

    SLP_start = 870
    SLP_end = 1090
    SLP_levels = np.arange(SLP_start, SLP_end, contour_interval)

    slp_contours = ax.contour(
        lons_np,
        lats_np,
        smooth_slp,
        levels=SLP_levels,
        colors="k",
        linewidths=1.0,
        transform=crs.PlateCarree(),
    )
    ax.clabel(slp_contours, inline=1, fontsize=10, fmt="%1.0f")

    # High and low centers
    slp_min_loc = np.unravel_index(np.argmin(smooth_slp), smooth_slp.shape)
    slp_max_loc = np.unravel_index(np.argmax(smooth_slp), smooth_slp.shape)

    min_pressure = smooth_slp[slp_min_loc]
    max_pressure = smooth_slp[slp_max_loc]

    min_lat, min_lon = lats_np[slp_min_loc], lons_np[slp_min_loc]
    max_lat, max_lon = lats_np[slp_max_loc], lons_np[slp_max_loc]

    ax.text(
        min_lon,
        min_lat,
        "L",
        color="red",
        fontsize=18,
        ha="center",
        va="center",
        transform=crs.PlateCarree(),
    )
    ax.text(
        max_lon,
        max_lat,
        "H",
        color="blue",
        fontsize=18,
        ha="center",
        va="center",
        transform=crs.PlateCarree(),
    )

    ax.text(
        min_lon,
        min_lat - label_adjustment,
        f"{min_pressure:.0f}",
        color="black",
        fontsize=12,
        ha="center",
        va="center",
        transform=crs.PlateCarree(),
    )
    ax.text(
        max_lon,
        max_lat - label_adjustment,
        f"{max_pressure:.0f}",
        color="black",
        fontsize=12,
        ha="center",
        va="center",
        transform=crs.PlateCarree(),
    )

    # Keep this unused SLP_levels line for invariance, though it's not replotted
    _ = np.arange(800, 1150, 2)

    # -------------------------------------------------------------------------
    # 1-hour accumulated snow (mm) filled contours
    # -------------------------------------------------------------------------
    # 0 to 175 mm per 1 hour interval (original levels preserved)
    Snow_levels = np.array([5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200])

    color_map_rgb = (
        np.array(
            [
                (189, 215, 231),
                (107, 174, 214),
                (49, 130, 189),
                (8, 81, 156),
                (8, 38, 148),
                (255, 255, 150),
                (255, 196, 0),
                (255, 135, 0),
                (219, 20, 0),
                (158, 0, 0),
                (105, 0, 0),
                (54, 0, 0),
            ],
            np.float32,
        )
        / 255.0
    )
    snow_map = plt.matplotlib.colors.ListedColormap(color_map_rgb[:-1])
    snow_map.set_over(color_map_rgb[-1])
    snow_norm = plt.matplotlib.colors.BoundaryNorm(Snow_levels, snow_map.N)

    snow_contour = ax.contourf(
        lons_np,
        lats_np,
        one_hour_snow_np,
        levels=Snow_levels,
        cmap=snow_map,
        norm=snow_norm,
        extend="max",
        transform=crs.PlateCarree(),
    )

    # Colorbar for snow
    cbar = fig.colorbar(
        snow_contour,
        ax=ax,
        orientation="vertical",
        shrink=0.8,
        pad=0.05,
        ticks=Snow_levels,
    )
    cbar.set_label("Estimated snowfall depth (mm, 10:1)", fontsize=14)
    cbar.ax.set_yticklabels([f"{level:.0f}" for level in Snow_levels])

    # -------------------------------------------------------------------------
    # Titles and saving the figure
    # -------------------------------------------------------------------------
    plt.title(
        "Weather Research and Forecasting Model\n"
        f"Average Grid Spacing:{avg_dx_km}x{avg_dy_km}km\n"
        "SLP (hPa)\n"
        "1-hour Total Snow Precipitation (mm)\n"
        "Estimated snowfall depth (10:1 from ACSNOW)",
        loc="left",
        fontsize=13,
    )
    plt.title(
        f"Valid: {earliest_dt:%H:%M:%SZ %Y-%m-%d} – {latest_dt:%H:%M:%SZ %Y-%m-%d}",
        loc="right",
        fontsize=13,
    )

    # Filename timestamp from valid time → alphabetical sort == chronological
    fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
    file_out = f"wrf_{domain}_SLP_Snow_1hrmm_{fname_time}.png"

    image_folder = os.path.join(path_figures, "Images")
    # The colorbar must also reflect snapshot/net-change fallback semantics.
    cbar.set_label(_snow_description, fontsize=12)
    # Hatch missing prior-grid coverage or invalid counters; keep original map.
    _snow_missing_hatching(ax, lons_np, lats_np, one_hour_snow_np)
    # Label the measured/estimated quantity, especially snapshot fallbacks.
    fig.text(0.5, 0.012, _snow_description, ha="center", va="bottom",
             fontsize=11, fontweight="bold", color="darkred",
             bbox=dict(facecolor="white", alpha=0.94, edgecolor="none", pad=3))
    plt.savefig(
        os.path.join(image_folder, file_out),
        bbox_inches="tight",
        dpi=100,
    )

    plt.close(fig)
    ncfile.close()


###############################################################################
# Frame Discovery (v9 canonical)
###############################################################################
def discover_frames(ncfile_paths):
    frames = []

    for path in ncfile_paths:
        with Dataset(path) as nc:
            if "Time" in nc.dimensions:
                n_times = len(nc.dimensions["Time"])
            elif "Times" in nc.variables:
                n_times = nc.variables["Times"].shape[0]
            else:
                n_times = 1

        for t in range(n_times):
            frames.append((path, t))

    return frames


###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Parse command-line arguments
    # -------------------------------------------------------------------------
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example: SFC_1hrSnow_SLP_mm.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # -------------------------------------------------------------------------
    # Prepare output directories
    # -------------------------------------------------------------------------
    path_figures = "wrf_SFC_1hrSNOW_mm_figures"
    image_folder = os.path.join(path_figures, "Images")
    animation_folder = os.path.join(path_figures, "Animation")

    for folder in (path_figures, image_folder, animation_folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    # -------------------------------------------------------------------------
    # Find all WRF output files for this domain
    # -------------------------------------------------------------------------
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print(f"No wrfout files found in {path_wrf} matching wrfout_{domain}*")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Build list of frames (file, time_index) to be processed
    # -------------------------------------------------------------------------
    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    # Sort frames by valid time to ensure chronological order
    frames_with_time = []
    for path, t_idx in frames:
        with Dataset(path) as nc:
            vt = get_valid_time(nc, path, t_idx)
        frames_with_time.append(((path, t_idx), vt))

    frames_with_time_sorted = sorted(frames_with_time, key=lambda ft: ft[1])
    sorted_frames = [ft[0] for ft in frames_with_time_sorted]

    # Build args_list including previous frame info (for 1-hr accumulation)
    # Match elapsed UTC time, never frame count.
    dated_frames = []
    for _path, _idx in frames:
        with Dataset(_path) as _nc:
            dated_frames.append((_path, _idx, get_valid_time(_nc, _path, _idx)))
    args_list = [
        (p, k, pp, pk, domain, path_figures)
        for p, k, pp, pk in plot_time_pairs(dated_frames, hours=1)
    ]
    if not args_list:
        print("No complete 1-hour accumulation windows; no graphics produced.")
        sys.exit(0)

    max_workers = min(4, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("SFC SLP + 1-hour snow (mm) plot generation complete.")

    # -------------------------------------------------------------------------
    # Build animated GIF from the sorted PNG files
    # -------------------------------------------------------------------------
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        sys.exit(0)

    # Filenames contain YYYYMMDDHHMMSS, so simple sort is chronological
    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files")

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]
    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        sys.exit(0)

    gif_file_out = f"wrf_{domain}_1-hour_Total_Snow_mm_SLP_Isotherm.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
