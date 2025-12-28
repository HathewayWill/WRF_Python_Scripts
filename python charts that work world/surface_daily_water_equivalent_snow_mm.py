#!/usr/bin/env python3
"""
SFC_DailySnow_Inch.py

Daily (00–00 UTC) accumulated snowfall (inches) based on SNOWH (m)
difference between consecutive 00Z times.

Key behaviors (physics preserved):
    * Daily accumulation between consecutive 00Z valid times:
        total_snow      = SNOWH * 39.3700787402  (m -> inches)
        prev_total_snow = prev_SN0WH * 39.3700787402
        daily_snow_00z  = total_snow - prev_total_snow

    * Uses SNOWH only; SLP is loaded solely for projection/lat-lon.

v3 structure:
    * Supports multiple wrfout_<domain>* files and multi-time files.
    * Discovers all (file, time_index) combinations via metadata.
    * Selects only 00Z times, pairs consecutive 00Z frames for daily totals.
    * One PNG per daily accumulation:
        wrf_{domain}_Snow_{YYYYMMDDHHMMSS}.png
    * NetCDF4 + wrf-python only; no xarray/metpy accessors for fields.
    * Geometry (lat/lon, grid spacing, extent, cities) recomputed per frame,
      safe for static and moving/vortex-following nests.
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
from wrf import ALL_TIMES, to_np  # ALL_TIMES kept for consistency, even if unused

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


###############################################################################
# Natural Earth features (keep commented-out options intact)
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
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


###############################################################################
# Frame processing: one daily 00Z–00Z snowfall accumulation
###############################################################################
def process_frame(args):
    """
    Process a single daily frame (00Z–00Z snowfall accumulation).

    args:
        ncfile_path      : path to current 00Z WRF file
        time_index       : time index for current 00Z in ncfile_path
        prev_ncfile_path : path to previous 00Z WRF file
        prev_time_index  : time index for previous 00Z in prev_ncfile_path
        domain           : WRF domain string (e.g., 'd01')
        path_figures     : base path for output
    """
    (
        ncfile_path,
        time_index,
        prev_ncfile_path,
        prev_time_index,
        domain,
        path_figures,
    ) = args

    # Need a previous 00Z frame to compute a daily accumulation
    if prev_ncfile_path is None:
        return

    # Open current and previous WRF files (worker-local; no shared handles)
    with Dataset(ncfile_path) as ncfile, Dataset(prev_ncfile_path) as prev_ncfile:

        # Valid times for title / bookkeeping
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        prev_valid_dt = get_valid_time(prev_ncfile, prev_ncfile_path, prev_time_index)

        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        earliest_datetime = prev_valid_dt
        latest_datetime = valid_dt

        # -------------------------------------------------------------------------
        # Extract variables (current 00Z) — physics preserved
        # -------------------------------------------------------------------------
        slp = wrf.getvar(ncfile, "slp", timeidx=time_index)
        snow = wrf.getvar(ncfile, "SNOW", timeidx=time_index)  # kg m-2 = mm liquid H2O
        total_snow = snow * 10  # H2O mm to snow (10:1 ratio) → mm of snow depth

        # -------------------------------------------------------------------------
        # Extract variables (previous 00Z)
        # -------------------------------------------------------------------------
        prev_snow = wrf.getvar(prev_ncfile, "SNOWH", timeidx=prev_time_index)  # m
        prev_total_snow = prev_snow * 39.3700787402  # m -> inches

        # -------------------------------------------------------------------------
        # Compute daily accumulated snowfall between two 00Z times (inches)
        # -------------------------------------------------------------------------
        daily_snow_00z = total_snow - prev_total_snow
        daily_snow_np = to_np(daily_snow_00z)

        # -------------------------------------------------------------------------
        # Geometry: lat/lon, projection, grid spacing (moving-nest safe)
        # -------------------------------------------------------------------------
        lats, lons = wrf.latlon_coords(slp)
        cart_proj = wrf.get_cartopy(slp)

        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,
        ) = compute_grid_and_spacing(lats, lons)

        # -------------------------------------------------------------------------
        # Dateline continuity and polar masking (v9 canonical helper)
        # -------------------------------------------------------------------------
        (
            lats_np,
            lons_np,
            daily_snow_np,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            daily_snow_np,
        )

        # ------------------------------------------------------------------ #
        # Create figure and map
        # ------------------------------------------------------------------ #
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Set extent with resolution-based padding (per frame, moving-nest safe)
        ax.set_extent(
            [
                lons_np.min() - extent_adjustment,
                lons_np.max() + extent_adjustment,
                lats_np.min() - extent_adjustment,
                lats_np.max() + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        # Land + features
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])
        for feature in features:
            add_feature(ax, *feature)

        # Cities & gridlines
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)
        add_latlon_gridlines(ax)

        ax.tick_params(labelsize=12, width=2)

        # -------------------------------------------------------------------------
        # Daily snow (mm) filled contours: levels & colormap preserved
        # -------------------------------------------------------------------------
        Snow_levels = np.array(
            [0.50, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200]
        )

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
                    (204, 204, 255),
                    (159, 140, 216),
                    (124, 82, 165),
                    (86, 28, 114),
                    (46, 0, 51),
                ],
                np.float32,
            )
            / 255.0
        )
        snow_map = plt.matplotlib.colors.ListedColormap(color_map_rgb[:-1])
        snow_map.set_over(color_map_rgb[-1])
        snow_norm = plt.matplotlib.colors.BoundaryNorm(Snow_levels, snow_map.N)

        Snow_contour = ax.contourf(
            lons_np,
            lats_np,
            daily_snow_np,
            levels=Snow_levels,
            cmap=snow_map,
            norm=snow_norm,
            extend="max",
            transform=crs.PlateCarree(),
        )

        # Colorbar
        cbar = fig.colorbar(
            Snow_contour,
            ax=ax,
            orientation="vertical",
            shrink=0.8,
            pad=0.05,
            ticks=Snow_levels,
        )
        cbar.set_label("Daily Total Snow Precipitation (Inch)", fontsize=14)
        cbar.ax.set_yticklabels([f"{level:.2f}" for level in Snow_levels])

        # -------------------------------------------------------------------------
        # Titles
        # -------------------------------------------------------------------------
        plt.title(
            "Weather Research and Forecasting Model\n"
            f"Average Grid Spacing:{avg_dx_km}x{avg_dy_km}km\n"
            "Daily Total Snow Precipitation (Inch)\n"
            "Model Snow Height Difference",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {earliest_datetime:%Y-%m-%d %H:%M} UTC\n"
            f"{latest_datetime:%Y-%m-%d %H:%M} UTC",
            loc="right",
            fontsize=13,
        )

        # -------------------------------------------------------------------------
        # Save PNG with timestamp from valid_dt for GIF sorting
        # -------------------------------------------------------------------------
        fname_time = latest_datetime.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_Snow_{fname_time}.png"

        image_folder = os.path.join(path_figures, "Images")
        plt.savefig(os.path.join(image_folder, file_out), bbox_inches="tight", dpi=250)

        plt.close(fig)


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
    # CLI arguments
    # -------------------------------------------------------------------------
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example: SFC_DailySnow_Inch_multicore_v3.py "
            "/home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # -------------------------------------------------------------------------
    # Output directories (Images + Animation)
    # -------------------------------------------------------------------------
    path_figures = "wrf_SFC_DailySnow_Inch_figures"
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
    # Discover all 00Z frames and pair consecutive days
    # -------------------------------------------------------------------------
    frames_00z = []
    for path in ncfile_paths:
        with Dataset(path) as nc:
            if "Time" in nc.dimensions:
                n_times = len(nc.dimensions["Time"])
            elif "Times" in nc.variables:
                n_times = nc.variables["Times"].shape[0]
            else:
                n_times = 1

            for t in range(n_times):
                valid_dt = get_valid_time(nc, path, t)
                if valid_dt.hour == 0:
                    frames_00z.append((path, t, valid_dt))

    frames_00z.sort(key=lambda x: x[2])

    if not frames_00z:
        print("No 00Z timesteps found in provided WRF files.")
        sys.exit(0)

    args_list = []
    for idx, (path, t_idx, valid_dt) in enumerate(frames_00z):
        if idx == 0:
            prev_path = None
            prev_t_idx = None
        else:
            prev_path, prev_t_idx, prev_valid_dt = frames_00z[idx - 1]
        args_list.append((path, t_idx, prev_path, prev_t_idx, domain, path_figures))

    # -------------------------------------------------------------------------
    # Process frames in parallel (only frames with a previous 00Z produce PNGs)
    # -------------------------------------------------------------------------
    max_workers = min(4, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("SFC daily snow (inches) plot generation complete.")

    # -------------------------------------------------------------------------
    # Build animated GIF from sorted PNG files (timestamped filenames)
    # -------------------------------------------------------------------------
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        sys.exit(0)

    # Filenames contain YYYYMMDDHHMMSS → simple sort is chronological
    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files")

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]
    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        sys.exit(0)

    gif_file_out = f"wrf_{domain}_Daily_Total_Snow_Inch_SLP_Isotherm.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
