#!/usr/bin/env python3
"""
850hPa_QVector_Divergence_WND_PRESS_multicorevgolden.py

Plot WRF 500-hPa geopotential height, traditional QG height tendency forcing
(terms A + B), and 700-hPa Q-vectors / Q-vector divergence on a WRF map
projection, using multiprocessing and building an animated GIF.

Usage:
    python 850hPa_QVector_Divergence_WND_PRESS_multicorevgolden.py <path_wrf> <domain>

Example:
    python 850hPa_QVector_Divergence_WND_PRESS_multicorevgolden.py \
        /home/workhorse/WRF_Intel/WRF-4.7.1/run/ d01

This script follows the 250 hPa "golden" playbook:
    * Supports multiple wrfout_<domain>* files, each with one or more timesteps.
    * Supports single wrfout files with many timesteps.
    * One frame = one (file, time_index) pair.
    * Geometry (lat/lon, dx/dy, extent, cities) recomputed each frame,
      safe for static and moving nests.
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
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import metpy.constants as mpconstants
import numpy as np
import wrf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
from netCDF4 import Dataset
from PIL import Image
from scipy.ndimage import gaussian_filter
from wrf import ALL_TIMES, to_np  # ALL_TIMES unused but kept for consistency

###############################################################################
# Warning suppression
###############################################################################
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


###############################################################################
# Physics helper
###############################################################################
def smooth_gaussian(field, sigma=1.5):
    """
    Fast Gaussian smoothing for a Pint-quantity array or plain ndarray.
    Preserves units if present.
    """
    if hasattr(field, "magnitude") and hasattr(field, "units"):
        data = np.asarray(field.magnitude, dtype=np.float32)
        units_ = field.units
        smoothed = gaussian_filter(data, sigma=sigma, mode="nearest")
        return smoothed * units_
    else:
        data = np.asarray(field, dtype=np.float32)
        smoothed = gaussian_filter(data, sigma=sigma, mode="nearest")
        return smoothed


###############################################################################
# Q-vector / QG forcing plotting for one (file, time_index) frame
###############################################################################
def process_frame(args):
    """
    Process a single frame: one file and one time index.

    Steps:
        * Read WRF variables (z, temp, pressure, ua, va, slp) at this time.
        * Interpolate to 500, 700, and 300 hPa.
        * Smooth fields with Gaussian filtering.
        * Compute traditional QG height tendency forcing (A + B).
        * Compute 700-hPa Q-vectors and Q-vector divergence.
        * Build map (features, cities, gridlines).
        * Plot:
            - Filled contours: total QGHT forcing (-(A+B) × 10^13 s^-3)
            - Line contours: 500-hPa geopotential height (m)
            - Contours: -2 × Q-vector divergence (×10^18)
            - Arrows: 700-hPa Q-vectors
            - Surface SLP H/L markers + values
        * Save a PNG file named with the valid time.
    """
    ncfile_path, time_index, domain, path_figures = args

    with Dataset(ncfile_path) as ncfile:
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        ###########################################################################
        # Get WRF base fields at this time index
        ###########################################################################
        p = wrf.getvar(ncfile, "pressure", timeidx=time_index)  # hPa
        z = wrf.getvar(ncfile, "z", timeidx=time_index, units="m")  # m
        t = wrf.getvar(ncfile, "temp", timeidx=time_index)  # K
        u = wrf.getvar(ncfile, "ua", timeidx=time_index)  # m/s
        v = wrf.getvar(ncfile, "va", timeidx=time_index)  # m/s

        slp = wrf.getvar(ncfile, "slp", timeidx=time_index)  # hPa

        ###########################################################################
        # Vertical interpolation using wrf.interplevel
        ###########################################################################
        level_500 = 500
        level_700 = 700
        level_300 = 300

        z_500 = wrf.vinterp(
            ncfile,
            z,
            "pressure",
            [level_500],
            field_type="z",
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()
        t_500 = wrf.vinterp(
            ncfile,
            t,
            "pressure",
            [level_500],
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()
        u_500 = wrf.vinterp(
            ncfile,
            u,
            "pressure",
            [level_500],
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()
        v_500 = wrf.vinterp(
            ncfile,
            v,
            "pressure",
            [level_500],
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()

        t_700 = wrf.vinterp(
            ncfile,
            t,
            "pressure",
            [level_700],
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()
        u_700 = wrf.vinterp(
            ncfile,
            u,
            "pressure",
            [level_700],
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()
        v_700 = wrf.vinterp(
            ncfile,
            v,
            "pressure",
            [level_700],
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()

        t_300 = wrf.vinterp(
            ncfile,
            t,
            "pressure",
            [level_300],
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()
        u_300 = wrf.vinterp(
            ncfile,
            u,
            "pressure",
            [level_300],
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()
        v_300 = wrf.vinterp(
            ncfile,
            v,
            "pressure",
            [level_300],
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()

        ###########################################################################
        # Convert to numeric arrays + attach units, then smooth (Gaussian)
        ###########################################################################
        hght_500 = np.asarray(to_np(z_500), dtype=np.float32) * units.m
        tmpk_500 = np.asarray(to_np(t_500), dtype=np.float32) * units.kelvin
        uwnd_500 = np.asarray(to_np(u_500), dtype=np.float32) * (units.m / units.s)
        vwnd_500 = np.asarray(to_np(v_500), dtype=np.float32) * (units.m / units.s)

        tmpk_700 = np.asarray(to_np(t_700), dtype=np.float32) * units.kelvin
        uwnd_700 = np.asarray(to_np(u_700), dtype=np.float32) * (units.m / units.s)
        vwnd_700 = np.asarray(to_np(v_700), dtype=np.float32) * (units.m / units.s)

        tmpk_300 = np.asarray(to_np(t_300), dtype=np.float32) * units.kelvin
        uwnd_300 = np.asarray(to_np(u_300), dtype=np.float32) * (units.m / units.s)
        vwnd_300 = np.asarray(to_np(v_300), dtype=np.float32) * (units.m / units.s)

        SIGMA_UV = 2.0
        SIGMA_T = 1.5
        SIGMA_Z = 1.0

        hght_500s = smooth_gaussian(hght_500, sigma=SIGMA_Z)

        uwnd_500s = smooth_gaussian(uwnd_500, sigma=SIGMA_UV)
        vwnd_500s = smooth_gaussian(vwnd_500, sigma=SIGMA_UV)
        tmpk_500s = smooth_gaussian(tmpk_500, sigma=SIGMA_T)

        tmpk_700s = smooth_gaussian(tmpk_700, sigma=SIGMA_T)
        uwnd_700s = smooth_gaussian(uwnd_700, sigma=SIGMA_UV)
        vwnd_700s = smooth_gaussian(vwnd_700, sigma=SIGMA_UV)

        tmpk_300s = smooth_gaussian(tmpk_300, sigma=SIGMA_T)
        uwnd_300s = smooth_gaussian(uwnd_300, sigma=SIGMA_UV)
        vwnd_300s = smooth_gaussian(vwnd_300, sigma=SIGMA_UV)

        ###########################################################################
        # Lat/lon & grid spacing (2D, moving-nest safe)
        ###########################################################################
        lats, lons = wrf.latlon_coords(z_500)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,
        ) = compute_grid_and_spacing(lats, lons)

        ###########################################################################
        # Dateline continuity and polar masking (v9 canonical helper)
        ###########################################################################
        (
            lats_np,
            lons_np,
            slp_np,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            to_np(slp),
        )

        ###########################################################################
        # dx, dy in meters for MetPy derivatives
        ###########################################################################
        dx, dy = mpcalc.lat_lon_grid_deltas(lons_np, lats_np)

        ###########################################################################
        # Traditional QG height tendency forcing: terms A + B (MetPy logic)
        ###########################################################################
        sigma = 2.0e-6 * units("m^2 Pa^-2 s^-2")
        f0 = 1.0e-4 * units("s^-1")
        Rd = mpconstants.Rd

        latitude = lats_np * units.degrees

        avor_500 = mpcalc.absolute_vorticity(
            uwnd_500s, vwnd_500s, dx=dx, dy=dy, latitude=latitude
        )
        vortadv_500 = mpcalc.advection(avor_500, uwnd_500s, vwnd_500s, dx=dx, dy=dy)
        term_A = (f0 * vortadv_500).to_base_units()

        tadv_700 = mpcalc.advection(tmpk_700s, uwnd_700s, vwnd_700s, dx=dx, dy=dy)
        tadv_300 = mpcalc.advection(tmpk_300s, uwnd_300s, vwnd_300s, dx=dx, dy=dy)

        diff_tadv = (
            Rd / (700 * units.hPa) * tadv_700 - Rd / (300 * units.hPa) * tadv_300
        ) / (400 * units.hPa)
        diff_tadv = diff_tadv.to_base_units()

        term_B = (-(f0**2) / sigma * diff_tadv).to_base_units()

        ###########################################################################
        # 700-hPa Q-vectors and Q-vector divergence
        ###########################################################################
        u_qvect, v_qvect = mpcalc.q_vector(
            uwnd_700s, vwnd_700s, tmpk_700s, 700 * units.hPa, dx=dx, dy=dy
        )
        Qdiv = mpcalc.divergence(u_qvect, v_qvect, dx=dx, dy=dy)

        ###########################################################################
        # Plot setup (Cartopy + dynamic extent + features + cities)
        ###########################################################################
        cart_proj = wrf.get_cartopy(z_500)

        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        ax.set_extent(
            [
                lons_np.min() - extent_adjustment,
                lons_np.max() + extent_adjustment,
                lats_np.min() - extent_adjustment,
                lats_np.max() + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])

        for feature in features:
            add_feature(ax, *feature)

        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)

        add_latlon_gridlines(ax)

        ###########################################################################
        # Surface SLP H/L markers
        ###########################################################################
        smooth_slp = gaussian_filter(slp_np, sigma=5.0)

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

        ###########################################################################
        # Contour levels and slices
        ###########################################################################
        clevs_500_hght = np.arange(0, 8000, 60)
        clevs_QGHT = np.arange(-12, 12, 0.5)

        wind_slice = (slice(2, None, 5), slice(2, None, 5))

        ###########################################################################
        # Filled contours: total QG height tendency forcing (-(A + B) × 10^13)
        ###########################################################################
        qght_field = (-(term_A + term_B) * 1e13).magnitude
        qght_field = gaussian_filter(qght_field, sigma=2.0)

        cf = ax.contourf(
            lons_np,
            lats_np,
            qght_field,
            clevs_QGHT,
            cmap=plt.cm.bwr,
            extend="both",
            transform=crs.PlateCarree(),
        )
        cb = plt.colorbar(
            cf,
            ax=ax,
            orientation="vertical",
            pad=0.01,
            aspect=50,
            extendrect=True,
        )
        cb.set_label("Total QG Height Tendency Forcing (×10$^{13}$ s$^{-3}$)")

        ###########################################################################
        # 500-hPa geopotential height (m) contours
        ###########################################################################
        hgt_field = hght_500s.magnitude
        cs = ax.contour(
            lons_np,
            lats_np,
            hgt_field,
            clevs_500_hght,
            colors="black",
            linewidths=0.80,
            transform=crs.PlateCarree(),
        )
        plt.clabel(cs, fmt="%d")

        ###########################################################################
        # Q-vector divergence contours (−2 × Qdiv × 10^18)
        ###########################################################################
        qdiv_field = (-2 * Qdiv * 1e18).magnitude
        qdiv_field = gaussian_filter(qdiv_field, sigma=3.0)

        cs2 = ax.contour(
            lons_np,
            lats_np,
            qdiv_field,
            np.arange(-100, 100, 25),
            colors="grey",
            linewidths=0.75,
            transform=crs.PlateCarree(),
        )
        plt.clabel(cs2, fmt="%d")

        ###########################################################################
        # 700-hPa Q-vectors
        ###########################################################################
        uq_field = u_qvect.magnitude
        vq_field = v_qvect.magnitude

        ax.quiver(
            lons_np[wind_slice],
            lats_np[wind_slice],
            uq_field[wind_slice],
            vq_field[wind_slice],
            pivot="mid",
            scale=1e-10,
            scale_units="height",
            transform=crs.PlateCarree(),
        )

        ###########################################################################
        # Titles and figure saving
        ###########################################################################
        plt.title(
            f"Weather Research and Forecasting Model\n"
            f"Average Grid Spacing: {avg_dx_km} x {avg_dy_km} km\n"
            f"500 hPa Geopotential Height (m)\n"
            f"Traditional QG Height Tendency Forcing (A + B) "
            f"(×10$^{{13}}$ s$^{{-3}}$)\n"
            f"700 hPa Q-Vectors and Q-Vector Divergence\n"
            f"Surface Sea-Level Pressure Highs and Lows (hPa)",
            loc="left",
            fontsize=13,
        )

        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_QVector_Div_{fname_time}.png"
        plt.savefig(
            os.path.join(path_figures, "Images", file_out),
            bbox_inches="tight",
            dpi=150,
        )

        plt.close(fig)


###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example:\n"
            "    850hPa_QVector_Divergence_WND_PRESS_multicorevgolden.py "
            "/home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    path_figures = "wrf_QVector_Div"
    image_folder = os.path.join(path_figures, "Images")
    animation_folder = os.path.join(path_figures, "Animation")

    for folder in (path_figures, image_folder, animation_folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print(f"No wrfout files found in {path_wrf} matching wrfout_{domain}*")
        sys.exit(0)

    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    args_list = [
        (ncfile_path, time_index, domain, path_figures)
        for (ncfile_path, time_index) in frames
    ]

    max_workers = min(4, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("Q-vector / QG forcing plots complete.")

    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        sys.exit(0)

    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files")

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]
    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        sys.exit(0)

    gif_file_out = f"wrf_{domain}_QVector_Div.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
