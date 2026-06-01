#!/usr/bin/env python3
"""
SFC_3hr_Water_Equivalent_Snow_mm_SLP.py

Plot WRF 3-hour accumulated snow (mm, 10:1 ratio from SNOW water equivalent),
and mean sea level pressure (SLP, hPa) on a Cartopy map.

v3 pattern:
    * Supports multiple wrfout_<domain>* files with one or more timesteps.
    * Treats each (file, time_index) pair as a frame.
    * Recomputes lat/lon, grid spacing, extent, and cities per frame.
    * Uses netCDF4 + wrf-python only (no xarray).

Important fixes:
    * Sort frames by valid time before linking “idx-2”.
    * Avoid unused 850-hPa temperature interpolation, which can fail when wrf-python receives a squeezed/non-3D field.
    * Includes a dateline/polar continuity helper (safe for worldwide/polar domains).
"""

import glob

###############################################################################
# Standard library imports
###############################################################################
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

###############################################################################
# Third-party imports
###############################################################################
import numpy as np

###############################################################################
# WRF imports
###############################################################################
import wrf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
from netCDF4 import Dataset
from PIL import Image
from scipy.ndimage import gaussian_filter
from wrf import to_np

warnings.filterwarnings("ignore")


###############################################################################
# Map features, Natural Earth configuration, and cities
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


features = [
    ("physical", "10m", cfeature.COLORS["land"], "black", 0.50, "minor_islands"),
    ("physical", "10m", "none", "black", 0.50, "coastline"),
    ("physical", "10m", cfeature.COLORS["water"], None, None, "ocean_scale_rank", -1),
    ("physical", "10m", cfeature.COLORS["water"], "lightgrey", 0.75, "lakes", 0),
    ("cultural", "10m", "none", "grey", 1.00, "admin_1_states_provinces", 2),
    ("cultural", "10m", "none", "black", 1.50, "admin_0_countries", 2),
    ("cultural", "10m", "none", "black", 0.60, "admin_2_counties", 2, 0.6),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_lake_centerlines"),
]

# Load populated places (cities) once per worker process
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


###############################################################################
# Time parsing helpers
###############################################################################
def parse_valid_time_from_wrf_name(path: str) -> datetime:
    base = os.path.basename(path)
    match = re.search(r"wrfout_.*?_(\d{4}-\d{2}-\d{2})_(\d{2}[:_]\d{2}[:_]\d{2})", base)
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


###############################################################################
# Grid / projection helpers
###############################################################################
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
    plot_extent = [lons_np.min(), lons_np.max(), lats_np.min(), lats_np.max()]

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
        geometry=gpd.points_from_xy(sorted_cities.LONGITUDE, sorted_cities.LATITUDE),
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
            clip_on=True,
            transform=crs.PlateCarree(),
        )
        ax.text(
            loc.x,
            loc.y,
            city_name,
            transform=crs.PlateCarree(),
            clip_on=True,
            ha="center",
            va="bottom",
            fontsize=11,
            color="black",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", alpha=0.4),
        )


def handle_domain_continuity_and_polar_mask(lats_np, lons_np, *fields):
    """
    If domain crosses the dateline, wrap longitudes to 0..360 and reorder columns.
    Optionally mask a tiny polar cap when combined with dateline crossing.
    """
    lon_span = np.nanmax(lons_np) - np.nanmin(lons_np)
    dateline_crossing = lon_span > 180.0

    lats_min = np.nanmin(lats_np)
    lats_max = np.nanmax(lats_np)
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
# Frame processing: SLP + 3-hour WE snow (mm, 10:1) for one frame
###############################################################################
def process_frame(args):
    (
        ncfile_path,
        time_index,
        prev2_ncfile_path,
        prev2_time_index,
        domain,
        path_figures,
    ) = args

    with Dataset(ncfile_path) as ncfile:
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        earliest_dt = valid_dt - timedelta(hours=3)

        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # Core fields
        slp = wrf.getvar(ncfile, "slp", timeidx=time_index)

        # SNOW: kg m-2 == mm water -> mm snow (10:1)
        snow_we = wrf.getvar(ncfile, "SNOW", timeidx=time_index)
        total_snow_mm = snow_we * 10.0

        # 3-hour accumulation via frame 2 steps back
        if prev2_ncfile_path is not None and prev2_time_index is not None:
            with Dataset(prev2_ncfile_path) as prev2_ncfile:
                prev2_snow_we = wrf.getvar(
                    prev2_ncfile, "SNOW", timeidx=prev2_time_index
                )
                prev2_total_snow_mm = prev2_snow_we * 10.0
            three_hour_snow_mm = total_snow_mm - prev2_total_snow_mm
        else:
            three_hour_snow_mm = np.zeros_like(to_np(total_snow_mm))

        # Lat/lon per frame
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

        # numpy + continuity
        slp_np = to_np(slp)
        snow_np = to_np(three_hour_snow_mm)

        lats_np, lons_np, slp_np, snow_np = handle_domain_continuity_and_polar_mask(
            lats_np, lons_np, slp_np, snow_np
        )

        # Figure
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
        for feat in features:
            add_feature(ax, *feat)

        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)
        add_latlon_gridlines(ax)
        ax.tick_params(labelsize=12, width=2)

        # Smooth fields (same sigmas)
        smooth_slp = gaussian_filter(slp_np, sigma=5.0)

        # SLP contours
        contour_interval = 4 if (avg_dx_km >= 9 or avg_dy_km >= 9) else 2
        SLP_levels = np.arange(870, 1090, contour_interval)

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

        # H/L
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

        # 3-hour snow (mm, 10:1 ratio) filled contours
        Snow_levels = np.array([5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400])

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

        snow_cf = ax.contourf(
            lons_np,
            lats_np,
            snow_np,
            levels=Snow_levels,
            cmap=snow_map,
            norm=snow_norm,
            extend="max",
            transform=crs.PlateCarree(),
        )

        cbar = fig.colorbar(
            snow_cf,
            ax=ax,
            orientation="vertical",
            shrink=0.8,
            pad=0.05,
            ticks=Snow_levels,
        )
        cbar.set_label("3-hour Total Snow (mm, 10:1)", fontsize=14)
        cbar.ax.set_yticklabels([f"{lvl:.0f}" for lvl in Snow_levels])

        # Titles
        plt.title(
            "Weather Research and Forecasting Model\n"
            f"Average Grid Spacing:{avg_dx_km}x{avg_dy_km}km\n"
            "SLP (hPa)\n"
            "3-hour Total Snow (mm)\n"
            "10:1 Ratio (from SNOW water equivalent)",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {earliest_dt:%Y-%m-%d %H:%M} UTC\n{valid_dt:%Y-%m-%d %H:%M} UTC",
            loc="right",
            fontsize=13,
        )

        # Save
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_SLP_3hrWESnow_mm_{fname_time}.png"

        image_folder = os.path.join(path_figures, "Images")
        plt.savefig(os.path.join(image_folder, file_out), bbox_inches="tight", dpi=150)
        plt.close(fig)


###############################################################################
# Frame discovery
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
# Main
###############################################################################
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example:\n"
            "    SFC_3hr_Water_Equivalent_Snow_mm_SLP_multicore_v3.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    path_figures = "wrf_SFC_3hr_Water_equivalent_Snow_mm_figures"
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

    # Sort frames by valid time (critical for multi-file / mixed layouts)
    frames_with_time = []
    for path, t_idx in frames:
        with Dataset(path) as nc:
            vt = get_valid_time(nc, path, t_idx)
        frames_with_time.append(((path, t_idx), vt))
    frames_sorted = [ft[0] for ft in sorted(frames_with_time, key=lambda x: x[1])]

    # Build args list using idx-2 linkage
    args_list = []
    for idx, (ncfile_path, time_index) in enumerate(frames_sorted):
        if idx >= 2:
            prev2_ncfile_path, prev2_time_index = frames_sorted[idx - 2]
        else:
            prev2_ncfile_path, prev2_time_index = (None, None)

        args_list.append(
            (
                ncfile_path,
                time_index,
                prev2_ncfile_path,
                prev2_time_index,
                domain,
                path_figures,
            )
        )

    max_workers = min(4, len(args_list)) if args_list else 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print(
        "SFC SLP and 3-hour water-equivalent snow (mm, 10:1) plot generation complete."
    )

    # Build GIF
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        sys.exit(0)

    png_files_sorted = sorted(png_files)

    # Optional: skip first two frames (no real 3-hour window when using idx-2)
    if len(png_files_sorted) > 2:
        png_files_for_gif = png_files_sorted[2:]
    else:
        png_files_for_gif = png_files_sorted

    print("Creating .gif file from sorted .png files")
    images = [Image.open(os.path.join(image_folder, fn)) for fn in png_files_for_gif]
    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        sys.exit(0)

    gif_file_out = f"wrf_{domain}_3-hour_WaterEquivalentSnow_mm_SLP.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
