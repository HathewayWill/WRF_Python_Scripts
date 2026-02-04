#!/usr/bin/env python3
"""
SFC_SLP_WIND_HI.py

Plot WRF surface fields:
    * Sea level pressure (hPa)
    * 10-m wind barbs (knots)
    * Heat Index (°F)

This script can handle:
    * Multiple wrfout_<domain>* files, each with one or more timesteps.
    * A single wrfout file containing many timesteps.

It does NOT assume the domain is static:
    * For each timestep, lat/lon, grid spacing, extent, and cities
      are recomputed from the WRF fields. This automatically works
      for both static nests and moving/vortex-following nests.
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
import numpy as np
import wrf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
from netCDF4 import Dataset
from PIL import Image
from scipy.ndimage import gaussian_filter
from wrf import ALL_TIMES, to_np  # ALL_TIMES kept for consistency with v3 pattern

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


def calculate_heat_index(T, RH):
    """
    Calculate Heat Index (°F) given temperature (°F) and relative humidity (%).
    Handles NumPy arrays correctly.

    This is the exact same formula as the original script.
    """
    # Simple calculation first
    simple_HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    average_HI = 0.5 * (simple_HI + T)

    # Initialize HI array with average_HI
    HI = np.where(
        average_HI < 80,
        average_HI,
        (
            -42.379
            + 2.04901523 * T
            + 10.14333127 * RH
            - 0.22475541 * T * RH
            - 0.00683783 * T**2
            - 0.05481717 * RH**2
            + 0.00122874 * T**2 * RH
            + 0.00085282 * T * RH**2
            - 0.00000199 * T**2 * RH**2
        ),
    )

    # Apply special condition adjustments using masks
    adjustment_1 = ((13 - RH) / 4) * np.sqrt((17 - np.abs(T - 95)) / 17)
    condition_1 = (RH < 13) & (T >= 80) & (T <= 112)

    adjustment_2 = ((RH - 85) / 10) * ((87 - T) / 5)
    condition_2 = (RH > 85) & (T >= 80) & (T <= 87)

    # Adjust HI where conditions are met
    HI = np.where(condition_1, HI - adjustment_1, HI)
    HI = np.where(condition_2, HI + adjustment_2, HI)

    return HI


###############################################################################
# Map features, Natural Earth configuration, and cities
###############################################################################
## List of Natural Earth features to add (keep commented-out options intact)
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
## Load populated places (cities) once per worker process.
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


###############################################################################
# Frame processing (one file + one time_index)
###############################################################################
def process_frame(args):
    """
    Process a single frame: one (file, time_index) pair.

    Physics is kept identical to the original SFC Heat Index script, but
    wrapped in the v3 structural pattern (multi-file + multi-time).
    """
    ncfile_path, time_index, domain, path_figures = args

    # Open the WRF file for this frame
    with Dataset(ncfile_path) as ncfile:
        # Valid time from metadata (preferred) or filename
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # -------------------------------------------------------------------------
        # Physics / diagnostics (unchanged in spirit)
        # -------------------------------------------------------------------------
        slp = wrf.getvar(ncfile, "slp", timeidx=time_index)
        u10 = wrf.getvar(ncfile, "U10", timeidx=time_index)
        v10 = wrf.getvar(ncfile, "V10", timeidx=time_index)
        temp = wrf.getvar(ncfile, "T2", timeidx=time_index)
        rh2 = wrf.getvar(ncfile, "rh2", timeidx=time_index)

        u10_knots = u10 * 1.94384449
        v10_knots = v10 * 1.94384449

        # Convert temp to degrees Fahrenheit (K → °F)
        temp_f_raw = (temp - 273.15) * 9.0 / 5.0 + 32.0

        # Smooth temperature and RH for HI calculation (unchanged sigma)
        temp_f_smooth = gaussian_filter(temp_f_raw, sigma=1.0)
        rh2_smooth = gaussian_filter(rh2, sigma=1.0)

        # Calculate Heat Index (°F)
        heat_index = calculate_heat_index(temp_f_smooth, rh2_smooth)

        # Wind speed in knots (diagnostic, but kept as original)
        wind_speed_knots = np.sqrt(u10_knots**2 + v10_knots**2)

        # Lat/lon grid
        lats, lons = wrf.latlon_coords(slp)

        # Compute grid spacing and extent padding (per frame, moving-nest safe)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,
        ) = compute_grid_and_spacing(lats, lons)

        # -------------------------------------------------------------------------
        # SLP smoothing and continuity handling (v9 canonical helper)
        # -------------------------------------------------------------------------
        smooth_slp = gaussian_filter(to_np(slp), sigma=5.0)

        u10_knots_np = to_np(u10_knots)
        v10_knots_np = to_np(v10_knots)
        temp_np = to_np(temp)

        (
            lats_np,
            lons_np,
            smooth_slp,
            heat_index,
            u10_knots_np,
            v10_knots_np,
            temp_np,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            smooth_slp,
            heat_index,
            u10_knots_np,
            v10_knots_np,
            temp_np,
        )

        # Map projection (native WRF)
        cart_proj = wrf.get_cartopy(slp)

        # -------------------------------------------------------------------------
        # Figure setup
        # -------------------------------------------------------------------------
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Map extent padding
        ax.set_extent(
            [
                lons_np.min() - extent_adjustment,
                lons_np.max() + extent_adjustment,
                lats_np.min() - extent_adjustment,
                lats_np.max() + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        # Base land and other Natural Earth features
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])
        for feature in features:
            add_feature(ax, *feature)

        # Plot cities (thinned by resolution)
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)

        # Gridlines
        add_latlon_gridlines(ax)

        # -------------------------------------------------------------------------
        # SLP smoothing and H/L markers
        # -------------------------------------------------------------------------
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

        # -------------------------------------------------------------------------
        # Heat Index and SLP contour settings (unchanged ranges)
        # -------------------------------------------------------------------------
        HI_Levels = np.arange(75, 125, 5)

        color_map_rgb = (
            np.array(
                [
                    [255, 237, 160],
                    [254, 209, 118],
                    [254, 174, 42],
                    [253, 141, 60],
                    [252, 78, 42],
                    [227, 26, 28],
                    [177, 0, 38],
                    [128, 0, 38],
                    [89, 0, 66],
                    [40, 0, 40],
                ]
            )
            / 255
        )
        HI_map = plt.matplotlib.colors.ListedColormap(color_map_rgb[:-1])
        HI_map.set_over(color_map_rgb[-1])
        HI_norm = plt.matplotlib.colors.BoundaryNorm(HI_Levels, HI_map.N)

        contour_interval = 2
        SLP_start = 870
        SLP_end = 1090
        SLP_levels = np.arange(SLP_start, SLP_end, contour_interval)

        # SLP contours
        SLP_contours = ax.contour(
            lons_np,
            lats_np,
            smooth_slp,
            levels=SLP_levels,
            colors="k",
            linewidths=1.0,
            transform=crs.PlateCarree(),
        )
        ax.clabel(SLP_contours, inline=1, fontsize=10, fmt="%1.0f")

        # Heat Index (°F) filled contours
        heat_index_plot = ax.contourf(
            lons_np,
            lats_np,
            to_np(heat_index),
            levels=HI_Levels,
            cmap=HI_map,
            norm=HI_norm,
            extend="max",
            transform=crs.PlateCarree(),
        )

        # 105°F isotherm
        ax.contour(
            lons_np,
            lats_np,
            to_np(heat_index),
            levels=[105],
            colors="red",
            linewidths=2.0,
            linestyles="-",
            transform=crs.PlateCarree(),
        )

        # Label SLP contours
        plt.clabel(SLP_contours, inline=True, fontsize=12, fmt="%d", colors="k")

        # -------------------------------------------------------------------------
        # Wind barbs (same brightness / masking logic as original)
        # -------------------------------------------------------------------------
        ny, nx = np.shape(u10_knots_np)
        desired_barbs = 15

        barb_density_x = max(nx // desired_barbs, 1)
        barb_density_y = max(ny // desired_barbs, 1)
        barb_density = max(barb_density_x, barb_density_y)

        color_brightness = np.dot(color_map_rgb, [0.299, 0.587, 0.114])
        brightness_threshold = 0.4

        norm = plt.Normalize(vmin=HI_Levels[0], vmax=HI_Levels[-1])
        HI_normalized = norm(to_np(heat_index))

        brightness_map = np.interp(
            HI_normalized, np.linspace(0, 1, len(color_brightness)), color_brightness
        )

        dark_region = brightness_map <= brightness_threshold
        light_region = brightness_map > brightness_threshold

        # Note: The original script used `temp` here; we preserve that behavior
        outside_contour_mask = (
            (temp_np < HI_Levels[0]) | (temp_np > HI_Levels[-1]) | np.isnan(temp_np)
        )
        inside_contour_mask = ~outside_contour_mask

        inside_contour_light_region = (inside_contour_mask & light_region)[
            ::barb_density, ::barb_density
        ]
        inside_contour_dark_region = (inside_contour_mask & dark_region)[
            ::barb_density, ::barb_density
        ]
        outside_contour = outside_contour_mask[::barb_density, ::barb_density]

        lons_ds = lons_np[::barb_density, ::barb_density]
        lats_ds = lats_np[::barb_density, ::barb_density]
        u_ds = u10_knots_np[::barb_density, ::barb_density]
        v_ds = v10_knots_np[::barb_density, ::barb_density]

        # Barbs in light regions inside contour range (black)
        ax.barbs(
            lons_ds[inside_contour_light_region],
            lats_ds[inside_contour_light_region],
            u_ds[inside_contour_light_region],
            v_ds[inside_contour_light_region],
            length=6,
            sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
            linewidth=0.8,
            color="black",
            transform=crs.PlateCarree(),
        )

        # Barbs in dark regions inside contour range (light gray)
        ax.barbs(
            lons_ds[inside_contour_dark_region],
            lats_ds[inside_contour_dark_region],
            u_ds[inside_contour_dark_region],
            v_ds[inside_contour_dark_region],
            length=6,
            sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
            linewidth=0.8,
            color="lightgray",
            transform=crs.PlateCarree(),
        )

        # Barbs outside contour range (black)
        ax.barbs(
            lons_ds[outside_contour],
            lats_ds[outside_contour],
            u_ds[outside_contour],
            v_ds[outside_contour],
            length=6,
            sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
            linewidth=0.8,
            color="black",
            transform=crs.PlateCarree(),
        )

        # -------------------------------------------------------------------------
        # Colorbar and titles
        # -------------------------------------------------------------------------
        cbar2 = fig.colorbar(
            heat_index_plot, ax=ax, orientation="vertical", shrink=0.8, pad=0.05
        )
        cbar2.set_label("Heat Index (°F)", fontsize=14)

        plt.title(
            f"Weather Research and Forecasting Model\n"
            f"Average Grid Spacing: {avg_dx_km} x {avg_dy_km} km\n"
            f"Sea Level Pressure (hPa)\n"
            f"10-m Wind Barbs (knots)\n"
            f"Heat Index (°F)",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        # Timestamp-based filename for correct GIF sorting
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_SLP_WIND_HI_{fname_time}.png"

        plt.savefig(
            os.path.join(path_figures, "Images", file_out),
            bbox_inches="tight",
            dpi=150,
        )

        plt.close(fig)


###############################################################################
# Frame discovery: multi-file + multi-time (v9 canonical)
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
            "For example:\n"
            "    SFC_SLP_WIND_HI.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # -------------------------------------------------------------------------
    # Prepare output directories
    # -------------------------------------------------------------------------
    path_figures = "wrf_SFC_HeatIndex_degF_figures"
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
    # Discover frames (file, time_index) and build arg list
    # -------------------------------------------------------------------------
    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    args_list = [
        (ncfile_path, time_index, domain, path_figures)
        for (ncfile_path, time_index) in frames
    ]

    max_workers = min(4, len(args_list)) if args_list else 1

    # -------------------------------------------------------------------------
    # Process frames in parallel
    # -------------------------------------------------------------------------
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("Surface SLP / wind / Heat Index plots complete.")

    # -------------------------------------------------------------------------
    # Build animated GIF from sorted PNGs
    # -------------------------------------------------------------------------
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

    gif_file_out = f"wrf_{domain}_SLP_WIND_HI.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
