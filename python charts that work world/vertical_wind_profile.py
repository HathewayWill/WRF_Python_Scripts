#!/usr/bin/env python3
"""
Time–height wind barbs at a point.

- X axis: time
- Y axis 1 (left / PRIMARY): height AGL (m)
- Y axis 2 (right / SECONDARY): pressure (hPa)

v3 structure:
- Supports many wrfout_<domain>* files with one or more timesteps.
- Uses wrf.extract_times + filename fallback for valid times.
- Uses ProcessPoolExecutor over (file, time_index) frames to compute
  interpolated profiles, then builds a single time–height figure.
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
from datetime import datetime

import cartopy.crs as crs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import wrf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
from netCDF4 import Dataset
from PIL import Image
from scipy.ndimage import gaussian_filter
from wrf import ALL_TIMES, getvar, ll_to_xy, to_np

###############################################################################
# Warning suppression
###############################################################################
# Quiet noisy warnings (optional, but standard in v3)
warnings.filterwarnings("ignore")

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
# Natural Earth features (v9 canonical – verbatim, order-locked)
###############################################################################
# List of Natural Earth features to add (keep commented-out options intact)
features = [
    ("physical", "10m", cfeature.COLORS["land"], "black", 0.50, "minor_islands"),
    ("physical", "10m", "none", "black", 0.50, "coastline"),
    ("physical", "10m", cfeature.COLORS["water"], None, None, "ocean_scale_rank", -1),
    ("physical", "10m", cfeature.COLORS["water"], "lightgrey", 0.75, "lakes", 0),
    ("cultural", "10m", "none", "grey", 1.00, "admin_1_states_provinces", 2),
    ("cultural", "10m", "none", "black", 1.50, "admin_0_countries", 2),
    # ("cultural", "10m", "none", "black", 0.60, "admin_2_counties", 2, 0.6),
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
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)




class PointOutsideMovingNestError(RuntimeError):
    """Raised when the requested lat/lon is outside the moving nest for a frame."""


def ll_to_xy_moving_safe(nc, lat, lon, time_index, sample_da, ncfile_path=""):
    """
    Convert lat/lon to x/y for the current frame of a moving nest.

    This intentionally uses timeidx=time_index. It also checks the returned
    indices against the current frame's grid shape so a point outside the
    moving nest can be skipped cleanly instead of crashing or sampling the
    wrong static column.
    """
    xy = ll_to_xy(nc, lat, lon, timeidx=time_index, meta=False)
    x_idx = int(xy[0])
    y_idx = int(xy[1])

    arr = to_np(sample_da)
    ny, nx = arr.shape[-2], arr.shape[-1]

    if x_idx < 0 or x_idx >= nx or y_idx < 0 or y_idx >= ny:
        raise PointOutsideMovingNestError(
            f"lat={lat:.4f}, lon={lon:.4f} is outside grid "
            f"for {os.path.basename(ncfile_path)} time_index={time_index} "
            f"(x={x_idx}, y={y_idx}, nx={nx}, ny={ny})"
        )

    return x_idx, y_idx


def derive_pressure_height_mapping(frames, lat, lon, p_levels_all):
    """
    Build the pressure-to-height reference mapping from the first frame that
    actually contains the requested point.

    Moving nests may not cover the point at the first file/time. This routine
    searches forward until the point is inside the nest, then uses that frame as
    the reference for the vertical axis mapping.
    """
    skipped = 0

    for ncfile_path, time_index in frames:
        with Dataset(ncfile_path) as nc:
            valid_dt = get_valid_time(nc, ncfile_path, time_index)
            p_da = getvar(nc, "pres", timeidx=time_index, units="hPa")
            z_da = getvar(nc, "height_agl", timeidx=time_index, units="m")

            try:
                x_idx, y_idx = ll_to_xy_moving_safe(
                    nc, lat, lon, time_index, p_da, ncfile_path
                )
            except PointOutsideMovingNestError as exc:
                skipped += 1
                print(f"SKIP reference frame: {exc}")
                continue

            p_prof0 = to_np(p_da[:, y_idx, x_idx]).astype(float)
            z_prof0 = to_np(z_da[:, y_idx, x_idx]).astype(float)

            pmin = min(p_prof0.min(), p_prof0.max())
            pmax = max(p_prof0.min(), p_prof0.max())
            mask = (p_levels_all >= pmin) & (p_levels_all <= pmax)
            p_levels = p_levels_all[mask]

            if p_levels.size == 0:
                skipped += 1
                print(
                    "SKIP reference frame: no requested pressure levels within "
                    f"model range for {os.path.basename(ncfile_path)} "
                    f"time_index={time_index}"
                )
                continue

            sort_p = np.argsort(p_prof0)
            p_sorted = p_prof0[sort_p]
            z_sorted = z_prof0[sort_p]
            z_levels = np.interp(p_levels, p_sorted, z_sorted)

            sort_z = np.argsort(z_levels)
            z_levels = z_levels[sort_z]
            p_levels = p_levels[sort_z]

            print(
                "Using "
                f"{ncfile_path} time_index={time_index} "
                f"({valid_dt:%H:%M:%SZ %Y-%m-%d}) "
                "to derive standard pressure→height mapping."
            )
            if skipped:
                print(f"Skipped {skipped} earlier reference frame(s).")

            return p_levels, z_levels

    raise RuntimeError(
        "The requested lat/lon was not inside the moving nest for any frame, "
        "or no usable pressure levels were available."
    )

###############################################################################
# Per-frame processing (runs in worker processes)
###############################################################################
def process_frame(args):
    """
    Worker function that interpolates U and V winds to standard pressure levels
    for a single (file, time_index) frame.

    Moving-nest safety:
        * Converts the requested lat/lon to x/y with timeidx=time_index.
        * Skips frames where the point is outside the moving nest.
    """
    ncfile_path, time_index, domain, lat, lon, p_levels = args

    print(f"Processing {ncfile_path} (time_index={time_index})")

    with Dataset(ncfile_path) as nc:
        valid_dt = get_valid_time(nc, ncfile_path, time_index)

        # Physics: identical getvar calls and units
        p_da = getvar(nc, "pres", timeidx=time_index, units="hPa")
        u_da = getvar(nc, "ua", timeidx=time_index, units="kt")
        v_da = getvar(nc, "va", timeidx=time_index, units="kt")

        try:
            x_idx, y_idx = ll_to_xy_moving_safe(
                nc, lat, lon, time_index, p_da, ncfile_path
            )
        except PointOutsideMovingNestError as exc:
            print(f"SKIP frame: {exc}")
            return None

        p_prof = to_np(p_da[:, y_idx, x_idx]).astype(float)
        u_prof = to_np(u_da[:, y_idx, x_idx]).astype(float)
        v_prof = to_np(v_da[:, y_idx, x_idx]).astype(float)

        # Sort by pressure ascending for interpolation
        srt = np.argsort(p_prof)
        p_s = p_prof[srt]
        u_s = u_prof[srt]
        v_s = v_prof[srt]

        # Interpolate U/V to the full standard p_levels
        u_std = np.interp(p_levels, p_s, u_s, left=np.nan, right=np.nan)
        v_std = np.interp(p_levels, p_s, v_s, left=np.nan, right=np.nan)

    return valid_dt, u_std, v_std


###############################################################################
# Frame Discovery (v9 canonical)
###############################################################################
# NOTE: Original script included an expanded docstring here. That text is preserved
# below as a commented legacy block so the active function remains v9-canonical.
#
# """
# Discover all (file, time_index) combinations.
#
# Supports:
#     * Many wrfout_<domain>* files with one or more Time steps.
#     * A single wrfout file with multiple Time steps.
#
# Returns
# -------
# frames : list[tuple[str, int]]
#     Each tuple is (ncfile_path, time_index).
# """
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
# Script entry point
###############################################################################
if __name__ == "__main__":
    ###############################################################################
    # Deviation Justification (required by v9)
    #
    # Deviation type: outputs
    # What changed (exactly):
    #   This script aggregates many (file, time_index) frames into ONE time–height PNG.
    # Why it is technically required:
    #   The diagnostic is a time–height cross-section; it is not meaningful as per-frame PNGs.
    # Why v9 canonical cannot satisfy this case:
    #   v9 frame semantics define one PNG per (file, time_index); this product is inherently multi-time.
    # Evidence:
    #   N/A (product definition)
    # Scope:
    #   This script only (time–height point wind barbs)
    # Rollback plan:
    #   Revert to per-frame PNGs only if product definition changes.
    ###############################################################################

    # -------------------------------------------------------------------
    # Command line args
    # -------------------------------------------------------------------
    if len(sys.argv) != 5:
        print("Usage: script.py <path_to_wrf_files> <domain> <latitude> <longitude>")
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]
    lat = float(sys.argv[3])
    lon = float(sys.argv[4])

    output_dir = "Vertical_Wind_Profiles"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Find WRF output files
    # -------------------------------------------------------------------
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}_*")))
    if not ncfile_paths:
        print(f"No WRF files found in {path_wrf} matching wrfout_{domain}_*")
        sys.exit(1)

    print(f"Found {len(ncfile_paths)} WRF files.")

    # -------------------------------------------------------------------
    # STEP 1: Discover frames, then derive standard pressure→height mapping
    #         from the first frame that contains the requested lat/lon.
    # -------------------------------------------------------------------
    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    p_levels_all = np.array(
        [
            1013,
            1000,
            950,
            900,
            850,
            800,
            750,
            700,
            650,
            600,
            550,
            500,
            450,
            400,
            350,
            300,
            250,
            200,
            150,
            100,
        ],
        dtype=float,
    )

    p_levels, z_levels = derive_pressure_height_mapping(
        frames, lat, lon, p_levels_all
    )

    print("Standard pressure → height mapping (from first valid moving-nest frame):")
    for p, z in zip(p_levels, z_levels):
        print(f"  {p:6.1f} hPa  ~  {z:7.1f} m")

    # -------------------------------------------------------------------
    # STEP 2: Discover frames and interpolate U/V in parallel
    # -------------------------------------------------------------------
    tasks = [
        (ncfile_path, time_index, domain, lat, lon, p_levels)
        for (ncfile_path, time_index) in frames
    ]

    times = []
    u_list = []
    v_list = []

    max_workers = min(4, len(tasks)) if tasks else 1

    # Interpolating U and V onto standard pressure levels is CPU-bound per frame.
    # We parallelize this per (file, time_index); plotting remains single-process.
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(process_frame, tasks):
            results.append(res)

    for res in results:
        if res is None:
            continue
        valid_dt, u_std, v_std = res
        times.append(valid_dt)
        u_list.append(u_std)
        v_list.append(v_std)

    if not times:
        print("No valid frames contained the requested point. No plot created.")
        sys.exit(0)

    skipped_frames = len(results) - len(times)
    if skipped_frames:
        print(f"Skipped {skipped_frames} frame(s) where the point was outside the moving nest.")

    # Sort by time to ensure monotonic time axis
    times = np.array(times)
    order = np.argsort(times)
    times = times[order]
    u_levels = np.array(u_list)[order]
    v_levels = np.array(v_list)[order]

    nt, nlev = u_levels.shape
    print(f"Interpolated winds: ntimes={nt}, nlevels={nlev}")

    # -------------------------------------------------------------------
    # STEP 3: Build grids for barbs (time × height)
    # -------------------------------------------------------------------
    time_nums = mdates.date2num(times)

    MAX_TIME_COLUMNS = 40
    time_stride = max(1, nt // MAX_TIME_COLUMNS)
    t_idx = np.arange(0, nt, time_stride)
    time_sel = time_nums[t_idx]

    # 2D grids: y = z_levels, x = selected times
    T, Z = np.meshgrid(time_sel, z_levels)
    U_plot = u_levels[t_idx, :].T
    V_plot = v_levels[t_idx, :].T

    # -------------------------------------------------------------------
    # STEP 4: Plot (primary Y = height, secondary Y = pressure)
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=150)

    ax.barbs(T, Z, U_plot, V_plot, length=6, clip_on=False)

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Height (m)")
    ax.set_title(f"Time–Height Wind Barbs at {lat:.3f}, {lon:.3f}")

    pad_z = 200.0
    z_min = max(0.0, z_levels.min() - pad_z)
    z_max = z_levels.max() + pad_z
    ax.set_ylim(z_min, z_max)

    ax.grid(True, linestyle="--", alpha=0.3)

    tick_times = mdates.num2date(time_sel)

    tick_labels = []
    for dt in tick_times:
        if dt.hour == 0 and dt.minute == 0:
            tick_labels.append(
                dt.strftime("%H:%M:%SZ %Y-%m-%d")
            )  # date (no year) + time at 00Z
        else:
            tick_labels.append(dt.strftime("%H:%M:%SZ %Y-%m-%d"))  # time only otherwise

    ax.set_xticks(time_sel)
    ax.set_xticklabels(tick_labels, rotation=90, ha="center")

    # Primary y-axis ticks at uniform height intervals
    y_ticks = np.arange(0, z_max + 1, 2000.0)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(y)} m" for y in y_ticks])

    # -------------------------------------------------------------------
    # Secondary y-axis (right): pressure (hPa)
    # -------------------------------------------------------------------
    def z_to_p(z):
        """Map height (m) -> pressure (hPa) for secondary axis."""
        z = np.asarray(z)
        return np.interp(z, z_levels, p_levels)

    def p_to_z(p):
        """Map pressure (hPa) -> height (m) for secondary axis."""
        p = np.asarray(p)
        return np.interp(p, p_levels[::-1], z_levels[::-1])

    secax = ax.secondary_yaxis("right", functions=(z_to_p, p_to_z))
    secax.set_ylabel("Pressure (hPa)")

    p_ticks = np.linspace(100, 1000, 10)
    secax.set_yticks(p_ticks)
    secax.set_yticklabels([f"{int(pt)} hPa" for pt in p_ticks])

    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.25, top=0.90)

    # Original output name preserved here as a comment:
    # out_file = os.path.join(output_dir, "time_height_wind_barbs_height_primary.png")

    # v9-convergent naming uses a valid time token (first time in series)
    fname_time = times[0].strftime("%Y%m%d%H%M%S") if len(times) else "00000000000000"
    out_file = os.path.join(
        output_dir,
        f"wrf_{domain}_time_height_wind_barbs_{fname_time}.png",
    )

    fig.savefig(out_file, dpi=150)
    plt.close(fig)

    print(f"Saved plot to {out_file}")
