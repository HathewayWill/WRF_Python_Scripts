#!/usr/bin/env python3
"""
SFC_RH.py

Plot WRF sea level pressure (hPa), 10-m wind barbs (knots),
and surface relative humidity (%) on a Cartopy map.

This script can handle:
    * Multiple wrfout_<domain>* files, each with one or more timesteps.
    * A single wrfout file containing many timesteps.

It does NOT assume the domain is static:
    * For each timestep, lat/lon, grid spacing, and extent are
      recomputed from the WRF fields. This automatically works
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
from wrf import ALL_TIMES, to_np

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
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


###############################################################################
# Process one (file, time_index) frame
###############################################################################
def process_frame(args):
    ncfile_path, time_index, domain, path_figures = args

    # Read the WRF output file for this frame
    with Dataset(ncfile_path) as ncfile:

        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%H:%M:%SZ %Y-%m-%d}")

        # Get the variables
        slp = wrf.getvar(ncfile, "slp", timeidx=time_index)
        u10 = wrf.getvar(ncfile, "U10", timeidx=time_index)
        v10 = wrf.getvar(ncfile, "V10", timeidx=time_index)
        rh2m = wrf.getvar(ncfile, "rh2", timeidx=time_index)

        u10_knots = u10 * 1.94384449
        v10_knots = v10 * 1.94384449

        # Convert wind speed from m/s to knots
        wind_speed_knots = np.sqrt(u10_knots**2 + v10_knots**2)

        # Get the latitude and longitude points
        lats, lons = wrf.latlon_coords(slp)

        # Get the cartopy mapping object
        cart_proj = wrf.get_cartopy(slp)

        # Compute grid spacing, extent padding, label adjustment (v9 canonical)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,
        ) = compute_grid_and_spacing(lats, lons)

        # Smooth the sea level pressure field
        slp_t = slp[:, :]
        smooth_slp = gaussian_filter(to_np(slp_t), sigma=5.0)

        # Convert fields to numpy for consistent continuity handling
        rh2m_np = to_np(rh2m)
        u10_knots_np = to_np(u10_knots)
        v10_knots_np = to_np(v10_knots)

        # Dateline continuity and polar masking (v9 canonical helper)
        (
            lats_np,
            lons_np,
            smooth_slp,
            rh2m_np,
            u10_knots_np,
            v10_knots_np,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            smooth_slp,
            rh2m_np,
            u10_knots_np,
            v10_knots_np,
        )

        # Create a plot using cartopy and matplotlib
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

        # Cities (canonical helper)
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)

        # Add gridlines and labels (canonical helper)
        gl = add_latlon_gridlines(ax)

        # Add the high and low pressure markers
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

        # Calculate contour levels of RH
        rh2m_levels = np.arange(0, 100, 10)

        rh2m_color_table = (
            np.array(
                [
                    (140, 81, 10),
                    (191, 129, 45),
                    (223, 194, 125),
                    (246, 232, 195),
                    (245, 245, 245),
                    (230, 245, 208),
                    (184, 225, 134),
                    (127, 188, 65),
                    (77, 146, 33),
                    (61, 116, 26),
                ],
                np.float32,
            )
            / 255.0
        )

        rh2m_map = plt.matplotlib.colors.ListedColormap(rh2m_color_table[:-1])
        rh2m_map.set_over(rh2m_color_table[-1])
        rh2m_norm = plt.matplotlib.colors.BoundaryNorm(rh2m_levels, rh2m_map.N)

        # Contour interval adjustment
        if avg_dx_km >= 9 or avg_dy_km >= 9:
            contour_interval = 4
        else:
            contour_interval = 2

        # Set the fixed start and end values for SLP contours
        SLP_start = 870
        SLP_end = 1090

        # Calculate contour levels using the fixed contour interval and start/end values
        SLP_levels = np.arange(SLP_start, SLP_end, contour_interval)

        # Plot the SLP contours
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

        # Add the RH contour lines to the plot:
        RH = ax.contourf(
            lons_np,
            lats_np,
            rh2m_np,
            levels=rh2m_levels,
            cmap=rh2m_map,
            norm=rh2m_norm,
            extend="max",
            alpha=0.75,
            transform=crs.PlateCarree(),
        )

        # Label the contour lines with the value at the midpoint
        plt.clabel(SLP_contours, inline=True, fontsize=12, fmt="%d", colors="k")

        # Adding wind barbs
        lons_plot = lons_np
        lats_plot = lats_np

        # Determine the shape (grid size) of the 10 m wind data
        ny, nx = np.shape(u10_knots_np)

        # Desired number of barbs to display across the grid
        desired_barbs = 15

        # Calculate the spacing (density) of barbs along the x and y directions
        barb_density_x = nx // desired_barbs
        barb_density_y = ny // desired_barbs

        # Use the maximum density to ensure consistent spacing across the grid
        barb_density = max(barb_density_x, barb_density_y)

        # Ensure barb density is at least 1 to avoid indexing errors
        if barb_density < 1:
            barb_density = 1

        # Compute the brightness of the isotach colors based on luminance formula
        # This is used to differentiate light and dark regions for the barbs
        color_brightness = np.dot(
            rh2m_color_table, [0.299, 0.587, 0.114]
        )  # Weights for RGB to brightness

        # Define a threshold to distinguish between light and dark regions
        brightness_threshold = 0.4

        # Normalize the RH values to match the RH color map
        norm = plt.Normalize(vmin=rh2m_levels[0], vmax=rh2m_levels[-1])
        rh2m_normalized = norm(rh2m_np)

        # Interpolate brightness values for regions with rh2m
        brightness_map = np.interp(
            rh2m_normalized, np.linspace(0, 1, len(color_brightness)), color_brightness
        )

        # Create masks for regions with dark and light backgrounds based on the brightness threshold
        dark_region = brightness_map <= brightness_threshold
        light_region = brightness_map > brightness_threshold

        # Create a mask for regions where RH is outside the contour range
        outside_contour_mask = (
            (to_np(rh2m_np) < rh2m_levels[0])
            | (to_np(rh2m_np) > rh2m_levels[-1])
            | np.isnan(to_np(rh2m_np))
        )

        # Create a mask for regions inside the contour range (complement of the outside mask)
        inside_contour_mask = ~outside_contour_mask

        # Apply barb density slicing to the boolean masks
        inside_contour_light_region = (inside_contour_mask & light_region)[
            ::barb_density, ::barb_density
        ]
        inside_contour_dark_region = (inside_contour_mask & dark_region)[
            ::barb_density, ::barb_density
        ]
        outside_contour = outside_contour_mask[::barb_density, ::barb_density]

        lons_ds = lons_plot[::barb_density, ::barb_density]
        lats_ds = lats_plot[::barb_density, ::barb_density]
        u_ds = u10_knots_np[::barb_density, ::barb_density]
        v_ds = v10_knots_np[::barb_density, ::barb_density]

        # Plot barbs in regions inside the contour range and light regions
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

        # Plot barbs in regions inside the contour range and dark regions
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

        # Plot barbs in regions outside the contour range
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

        # Add colorbar with label
        cbar2 = fig.colorbar(RH, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
        cbar2.set_label("Surface Relative Humidity (%)", fontsize=14)

        # Add title to image
        plt.title(
            f"Weather Research and Forecasting Model\nAverage Grid Spacing:{avg_dx_km}x{avg_dy_km}km\nSea Level Pressure (hPa)\nWind Barbs (knots)\nRelative Humidity (%)",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        # Defining the name of the figure (v9 filename rule: YYYYMMDDHHMMSS)
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_slp_wind_RH_{fname_time}.png"

        # Save the figure at 150dpi
        plt.savefig(
            os.path.join(path_figures, "Images", file_out), bbox_inches="tight", dpi=150
        )

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

    # Defining directory where the images will be saved
    path_figures = "wrf_SFC_RH_figures"

    # Create a directory for saving the figures if it doesn't exist
    image_folder = os.path.join(path_figures, "Images")
    animation_folder = os.path.join(path_figures, "Animation")

    for folder in (path_figures, image_folder, animation_folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    ###############################################################################
    # Process each WRF output file/time in parallel (frame semantics)
    ###############################################################################
    # Loop through each WRF output file and create a plot
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

    ###############################################################################
    # Build an animated GIF (if multiple .png frames are found)
    ###############################################################################
    print("SFC Plots Complete")

    # Get a list of all .png files in the directory
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        sys.exit(0)

    # Filenames contain YYYYMMDDHHMMSS, so simple sort is chronological
    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files")

    # Set the duration (in milliseconds) between frames in the animated GIF
    duration = 800

    # Create a list of Image objects from the sorted PNG files
    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]

    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        sys.exit(0)

    # Define the name of the GIF file with the domain name variable
    gif_file_out = "wrf_" + domain + "_SLP_WIND_RH.gif"

    # Save the list of Image objects as an animated GIF
    gif_path = os.path.join(animation_folder, gif_file_out)
    images[0].save(
        gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0
    )

    print("GIF generation complete.")
