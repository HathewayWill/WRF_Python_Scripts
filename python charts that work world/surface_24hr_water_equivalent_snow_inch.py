#!/usr/bin/env python3
"""
SFC_24hr_Water_equivalent_Snow_Inch.py

Plot WRF 24-hour accumulated water-equivalent snow (10:1 ratio; inches) on a Cartopy map.

This script can handle:
    * Multiple wrfout_<domain>* files, each with one or more timesteps.
    * A single wrfout file containing many timesteps.
    * Mixed situations.

It does NOT assume the domain is static:
    * For each (file, time_index) frame, lat/lon, grid spacing, and extent
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
from wrf import to_np

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
# Natural Earth features (verbatim feature intent preserved; list may differ)
###############################################################################
# List of features to add
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
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None, "lakes_north_america", None), 0.75),
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None, "lakes_australia", None), 0.75),
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None, "lakes_europe", None), 0.75)]
]

###############################################################################
# Cities (module scope)
###############################################################################
# Add cities to plot
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


###############################################################################
# 24-hr water-equivalent snow (10:1 ratio; inches) plotting for one frame
###############################################################################
def process_frame(args):
    """
    Process a single frame: one file and one time index.

    Physics / diagnostics preserved from the original script intent:

        * slp = wrf.getvar(ncfile, "slp")
        * snow = wrf.getvar(ncfile, "SNOW")  # units is kg m-2 which is equal to 1mm of liquid h20
        * total_snow = snow * 10 * 0.0393701  # mm to inch snow using 10:1 ratio
        * 24-hr accumulation = total_snow - prev24_total_snow (24 frames back)
        * Snow_levels array and RGB colormap unchanged.
        * City thinning / labeling and gridlines preserved via canonical helpers.
    """
    (
        ncfile_path,
        time_index,
        prev24_ncfile_path,
        prev24_time_index,
        domain,
        path_figures,
    ) = args

    with Dataset(ncfile_path) as ncfile:

        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        earliest_dt = valid_dt - timedelta(hours=24)

        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # Extract the required variables from the NetCDF file
        slp = wrf.getvar(ncfile, "slp", timeidx=time_index)
        snow = wrf.getvar(
            ncfile, "SNOW", timeidx=time_index
        )  # units is kg m-2 which is equal to 1mm of liquid h20
        total_snow = snow * 10 * 0.0393701  # mm to inch snow using 10:1 ratio

        # Calculate the 24-hour accumulated snow
        if prev24_ncfile_path is not None and prev24_time_index is not None:
            with Dataset(prev24_ncfile_path) as prev24_ncfile:
                prev24_snow = wrf.getvar(
                    prev24_ncfile, "SNOW", timeidx=prev24_time_index
                )  # units is kg m-2 which is equal to 1mm of liquid h20
                prev24_total_snow = (
                    prev24_snow * 10 * 0.0393701
                )  # mm to inch snow using 10:1 ratio
            twentyfour_hour_snow = total_snow - prev24_total_snow
        else:
            twentyfour_hour_snow = np.zeros_like(to_np(total_snow))

        # Get the latitude and longitude coordinates and the cartopy projection for the data
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

        (
            lats_np,
            lons_np,
            twentyfour_hour_snow_np,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            to_np(twentyfour_hour_snow),
        )

        # Create a figure and axis using the cartopy projection
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Set the map extent
        ax.set_extent(
            [
                lons_np.min() - extent_adjustment,
                lons_np.max() + extent_adjustment,
                lats_np.min() - extent_adjustment,
                lats_np.max() + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        # Cartopy land feature
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])

        # Adding features
        for feature in features:
            add_feature(ax, *feature)

        # Cities
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)

        # Gridlines with labels (canonical helper)
        gl = add_latlon_gridlines(ax)
        _ = gl

        ax.tick_params(labelsize=12, width=2)
        _ = label_adjustment  # retained variable, consistent with playbook patterns

        # 0 to 42 Inch per 24 hour Inch interval colorbar
        Snow_levels = np.array(
            [0.50, 1, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30, 33, 36, 42]
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
            twentyfour_hour_snow_np,
            levels=Snow_levels,
            cmap=snow_map,
            norm=snow_norm,
            extend="max",
            transform=crs.PlateCarree(),
        )

        # Colorbar for snow
        cbar = fig.colorbar(
            Snow_contour,
            ax=ax,
            orientation="vertical",
            shrink=0.8,
            pad=0.05,
            ticks=Snow_levels,
        )

        cbar.set_label("24-hour Total Snow Precipitation (Inch)", fontsize=14)
        cbar.ax.set_yticklabels([f"{level:.2f}" for level in Snow_levels])

        # Add titles to the plot
        plt.title(
            "Weather Research and Forecasting Model\n"
            f"Average Grid Spacing:{avg_dx_km}x{avg_dy_km}km\n"
            "24-hour Total Snow Precipitation (Inch)\n"
            "10:1 Ratio",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {earliest_dt:%Y-%m-%d %H:%M UTC}\n{valid_dt:%Y-%m-%d %H:%M} UTC",
            loc="right",
            fontsize=13,
        )

        # Save the figure as a .png file (v9 filename timestamp rule)
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_24hr_Water_equivalent_Snow_Inch_{fname_time}.png"

        image_folder = os.path.join(path_figures, "Images")
        if not os.path.isdir(image_folder):
            os.mkdir(image_folder)

        plt.savefig(os.path.join(image_folder, file_out), bbox_inches="tight", dpi=250)
        plt.close(fig)


###############################################################################
# Frame discovery: handle multi-file and multi-time setups (v9 canonical)
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
# Main script entry point
###############################################################################
if __name__ == "__main__":

    # Check if the correct arguments were provided
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\nFor example: script_name.py /home/WRF/test/em_real d01\n"
        )
        sys.exit()

    # Define the path where the netcdf files are and the domain to be used
    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    path_figures = "wrf_SFC_24hr_Water_equivalent_Snow_Inch_figures"

    # Create a directory for saving the figures if it doesn't exist
    if not os.path.isdir(path_figures):
        os.mkdir(path_figures)

    # Create 'Images' folder inside the folder with PNG files
    image_folder = os.path.join(path_figures, "Images")
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)

    # Create 'Animation' folder inside the folder with PNG files
    animation_folder = os.path.join(path_figures, "Animation")
    if not os.path.isdir(animation_folder):
        os.mkdir(animation_folder)

    # Loop through each WRF output file and create a plot
    ncfile_paths = sorted(glob.glob(path_wrf + "/wrfout_" + domain + "*"))
    if not ncfile_paths:
        sys.exit(f"No wrfout files found in {path_wrf} matching wrfout_{domain}*")

    # Build list of frames (file, time_index)
    frames = discover_frames(ncfile_paths)
    if not frames:
        sys.exit("No timesteps found in provided WRF files.")

    ###############################################################################
    # Process each WRF output frame in parallel
    ###############################################################################
    # Build argument list, pairing each frame with the one 24 time steps earlier
    args_list = []
    for idx, (ncfile_path, time_index) in enumerate(frames):
        if idx >= 24:
            prev24_ncfile_path, prev24_time_index = frames[idx - 24]
        else:
            prev24_ncfile_path, prev24_time_index = (None, None)

        args_list.append(
            (
                ncfile_path,
                time_index,
                prev24_ncfile_path,
                prev24_time_index,
                domain,
                path_figures,
            )
        )

    max_workers = min(4, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    ###############################################################################
    # Build an animated GIF (if multiple .png frames are found)
    ###############################################################################
    # Sort the .png files by date order and create a .gif file from the sorted .png files
    print("SFC Snow Plots Complete")

    png_files = [
        f
        for f in os.listdir(os.path.join(path_figures, "Images"))
        if f.endswith(".png")
    ]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        sys.exit(0)

    # Filenames contain YYYYMMDDHHMMSS, so simple sort is chronological
    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files")

    duration = (
        800  # Set the duration (in milliseconds) between frames in the animated GIF
    )
    images = []
    for filename in png_files_sorted:
        filepath = os.path.join(image_folder, filename)
        images.append(Image.open(filepath))

    # If fewer than 24 images, exit the script
    if len(images) < 24:
        sys.exit(
            "The wrf run is less than 24hours and cannot make 24hr total accumulation."
        )

    # Otherwise, start at the 24th image in the list
    images = images[23:]

    gif_file_out = "wrf_" + domain + "_24-hour_Total_Snow_Inch.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)
    images[0].save(
        gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0
    )

    print("SFC 24hr Snow Depth Inches Plots Complete")
