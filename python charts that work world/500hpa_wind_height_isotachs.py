#!/usr/bin/env python3
"""
WRF 500 hPa Wind, Heights, and Isotachs (Multicore, Golden-style)
"""

# =============================================================================
# 1. Imports
# =============================================================================

import glob

# Standard library
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

# Third-party
import numpy as np
import wrf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
from netCDF4 import Dataset
from PIL import Image
from scipy.ndimage import gaussian_filter
from wrf import ALL_TIMES, to_np

# =============================================================================
# Warning suppression
# =============================================================================

# Ignore unnecessary warnings that clutter the terminal
warnings.filterwarnings("ignore")


# =============================================================================
# 2. Map features & cities
# =============================================================================

# Convenience wrapper to add a Natural Earth feature to a Cartopy axis.
# Fallback: parse valid time from the wrfout filename if metadata is missing.
# Get the valid time for a given timestep, preferring WRF metadata.
# Compute lat/lon in numpy form, average grid spacing (km), and map padding.
# Golden helper, called per-frame (safe for moving nests).
# Add labeled latitude/longitude gridlines to a Cartopy GeoAxes.
# Subset and thin cities based on population and grid resolution, then plot them.
# Golden pattern replacing GeoDataFrame.append loops.
# Discover (file, time_index) pairs for all available wrfout files.
# Supports:
# - Many files with 1 timestep each
# - Single file with many timesteps
# - Mixed scenarios


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


# =============================================================================
# 2. Map features & cities
# =============================================================================

# List of features to add (same content as your original)
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

# Load cities once per process
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


# =============================================================================
# 5. Discover frames (multi-file, multi-time)
# =============================================================================


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


# =============================================================================
# 6. Plot configuration (isotachs)
# =============================================================================

# Isotach thresholds in knots (unchanged)
wind_speed_ranges = np.array(
    [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 110, 120]
)

# Custom RGB colormap for isotachs (unchanged)
color_map_rgb = (
    np.array(
        [
            [16, 63, 120],
            [34, 94, 168],
            [29, 145, 192],
            [65, 182, 196],
            [127, 205, 187],
            [180, 215, 158],
            [223, 255, 158],
            [255, 255, 166],
            [255, 232, 115],
            [255, 196, 0],
            [255, 170, 0],
            [255, 89, 0],
            [255, 0, 0],
            [168, 0, 0],
            [110, 0, 0],
            [255, 190, 232],
            [255, 115, 223],
        ],
        np.float32,
    )
    / 255.0
)

# First 16 colors map to the 16 in-range bins; last color is for "over"
isotach_cmap = plt.matplotlib.colors.ListedColormap(color_map_rgb[:-1])
isotach_cmap.set_over(color_map_rgb[-1])

isotach_norm = plt.matplotlib.colors.BoundaryNorm(
    wind_speed_ranges, isotach_cmap.N, clip=False
)
# =============================================================================
# 7. Worker function: process a single frame
# =============================================================================


def process_frame(args):
    """
    Process a single (file, time_index) pair and save a PNG.

    Parameters
    ----------
    args : tuple
        (ncfile_path, time_index, domain, path_figures)
    """
    ncfile_path, time_index, domain, path_figures = args

    # Open the WRF file and get variables for this timestep
    with Dataset(ncfile_path) as nc:
        valid_dt = get_valid_time(nc, ncfile_path, time_index)

        # Progress log (requested format)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        u = wrf.getvar(nc, "ua", timeidx=time_index)
        v = wrf.getvar(nc, "va", timeidx=time_index)
        p = wrf.getvar(nc, "pressure", timeidx=time_index)
        z = wrf.getvar(nc, "z", units="dm", timeidx=time_index)

        # Interpolate to 500 hPa
        u_500 = wrf.interplevel(u, p, 500)
        v_500 = wrf.interplevel(v, p, 500)
        z_500 = wrf.interplevel(z, p, 500)
        z_500 = gaussian_filter(to_np(z_500), sigma=2)

        # Convert wind components to knots and compute scalar wind speed
        u_500_knots = u_500 * 1.94384449
        v_500_knots = v_500 * 1.94384449
        wind_speed_knots = np.sqrt(u_500_knots**2 + v_500_knots**2)

        u_500_knots_np = to_np(u_500_knots)
        v_500_knots_np = to_np(v_500_knots)
        wind_speed_knots_np = to_np(wind_speed_knots)

        # Geometry for this frame (safe for moving nests)
        lats, lons = wrf.latlon_coords(u_500)

        lats_np, lons_np, avg_dx_km, avg_dy_km, extent_adjustment, _ = (
            compute_grid_and_spacing(lats, lons)
        )

        # -------------------------------------------------------------------------
        # Domain continuity & polar masking (v9 canonical helper)
        # -------------------------------------------------------------------------
        (
            lats_np,
            lons_np,
            z_500,
            wind_speed_knots_np,
            u_500_knots_np,
            v_500_knots_np,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            z_500,
            wind_speed_knots_np,
            u_500_knots_np,
            v_500_knots_np,
        )

        cart_proj = wrf.get_cartopy(u_500)

        # -------------------------------------------------------------------------
        # Create figure and map background
        # -------------------------------------------------------------------------
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Set extent with padding
        ax.set_extent(
            [
                lons_np.min() - extent_adjustment,
                lons_np.max() + extent_adjustment,
                lats_np.min() - extent_adjustment,
                lats_np.max() + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        # Land base
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])

        # Extra map features
        for feat in features:
            add_feature(ax, *feat)

        # Cities + gridlines
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)
        add_latlon_gridlines(ax)

        # -------------------------------------------------------------------------
        # 500 hPa height contours
        # -------------------------------------------------------------------------
        z500_start = 400
        z500_end = 900
        contour_interval = 2
        height_levels = np.arange(z500_start, z500_end, contour_interval)

        height_contours = ax.contour(
            lons_np,
            lats_np,
            z_500,
            levels=height_levels,
            colors="black",
            linewidths=1,
            transform=crs.PlateCarree(),
        )
        ax.clabel(height_contours, inline=True, fontsize=11, fmt="%1.0f")

        # -------------------------------------------------------------------------
        # 500 hPa isotachs (filled wind speed)
        # -------------------------------------------------------------------------
        isotach_contours = ax.contourf(
            lons_np,
            lats_np,
            wind_speed_knots_np,
            levels=wind_speed_ranges,
            cmap=isotach_cmap,
            norm=isotach_norm,
            transform=crs.PlateCarree(),
            extend="max",
        )

        # -------------------------------------------------------------------------
        # Wind barbs with brightness-aware coloring (same logic as your script)
        # -------------------------------------------------------------------------
        ny, nx = u_500_knots_np.shape
        desired_barbs = 15

        barb_density_x = max(nx // desired_barbs, 1)
        barb_density_y = max(ny // desired_barbs, 1)
        barb_density = max(barb_density_x, barb_density_y)

        # Compute brightness of isotach colors
        color_brightness = np.dot(color_map_rgb, [0.299, 0.587, 0.114])
        brightness_threshold = 0.4

        norm_speed = plt.Normalize(
            vmin=wind_speed_ranges[0], vmax=wind_speed_ranges[-1]
        )
        isotachs_normalized = norm_speed(wind_speed_knots_np)
        brightness_map = np.interp(
            isotachs_normalized,
            np.linspace(0, 1, len(color_brightness)),
            color_brightness,
        )

        dark_region = brightness_map <= brightness_threshold
        light_region = brightness_map > brightness_threshold

        outside_contour_mask = (
            (wind_speed_knots_np < wind_speed_ranges[0])
            | (wind_speed_knots_np > wind_speed_ranges[-1])
            | np.isnan(wind_speed_knots_np)
        )
        inside_contour_mask = ~outside_contour_mask

        inside_contour_light_region = (inside_contour_mask & light_region)[
            ::barb_density, ::barb_density
        ]
        inside_contour_dark_region = (inside_contour_mask & dark_region)[
            ::barb_density, ::barb_density
        ]
        outside_contour = outside_contour_mask[::barb_density, ::barb_density]

        # Barbs in light isotach regions (black)
        ax.barbs(
            lons_np[::barb_density, ::barb_density][inside_contour_light_region],
            lats_np[::barb_density, ::barb_density][inside_contour_light_region],
            u_500_knots_np[::barb_density, ::barb_density][inside_contour_light_region],
            v_500_knots_np[::barb_density, ::barb_density][inside_contour_light_region],
            length=6,
            sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
            linewidth=0.8,
            color="black",
            transform=crs.PlateCarree(),
        )

        # Barbs in dark isotach regions (light gray)
        ax.barbs(
            lons_np[::barb_density, ::barb_density][inside_contour_dark_region],
            lats_np[::barb_density, ::barb_density][inside_contour_dark_region],
            u_500_knots_np[::barb_density, ::barb_density][inside_contour_dark_region],
            v_500_knots_np[::barb_density, ::barb_density][inside_contour_dark_region],
            length=6,
            sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
            linewidth=0.8,
            color="lightgray",
            transform=crs.PlateCarree(),
        )

        # Barbs outside isotach range (always black)
        ax.barbs(
            lons_np[::barb_density, ::barb_density][outside_contour],
            lats_np[::barb_density, ::barb_density][outside_contour],
            u_500_knots_np[::barb_density, ::barb_density][outside_contour],
            v_500_knots_np[::barb_density, ::barb_density][outside_contour],
            length=6,
            sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
            linewidth=0.8,
            color="black",
            transform=crs.PlateCarree(),
        )

        # -------------------------------------------------------------------------
        # Colorbar and titles
        # -------------------------------------------------------------------------
        cbar = plt.colorbar(
            isotach_contours,
            ax=ax,
            orientation="vertical",
            pad=0.05,
            shrink=0.8,
            ticks=wind_speed_ranges,
        )
        cbar.set_label("Isotachs (knots)")
        cbar.ax.set_yticklabels([f"{level:.0f}" for level in wind_speed_ranges])

        plt.title(
            "Weather Research and Forecasting Model\n"
            f"Average Grid Spacing:{avg_dx_km}x{avg_dy_km}km\n"
            "Wind Barbs at 500 hPa (knots)\n"
            "Isotachs (knots)\n"
            "500 hPa Geopotential Heights (dm)",
            loc="left",
            fontsize=13,
        )

        plt.title(
            f"Valid: {valid_dt:%H:%M:%S}Z {valid_dt:%Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        # -------------------------------------------------------------------------
        # Save PNG (timestamp-based filename for clean chronological sort)
        # -------------------------------------------------------------------------
        image_folder = os.path.join(path_figures, "Images")
        os.makedirs(image_folder, exist_ok=True)

        ts = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_wind_500hPa_{ts}.png"

        plt.savefig(
            os.path.join(image_folder, file_out),
            bbox_inches="tight",
            dpi=100,
        )

        plt.close(fig)


# =============================================================================
# 8. Main script: multiprocessing + GIF
# =============================================================================


def main():
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example: script_name.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    path_figures = "wrf_500hPa_Isotachs_Wnd_Hgt"
    image_folder = os.path.join(path_figures, "Images")
    animation_folder = os.path.join(path_figures, "Animation")

    for folder in (path_figures, image_folder, animation_folder):
        os.makedirs(folder, exist_ok=True)

    # Collect wrfout files
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print(f"No wrfout files found in {path_wrf} for domain {domain}")
        sys.exit(1)

    # Discover all (file, time_index) frames
    frames = discover_frames(ncfile_paths)

    # Build argument list for multiprocessing
    args_list = [
        (ncfile_path, time_index, domain, path_figures)
        for (ncfile_path, time_index) in frames
    ]

    max_workers = min(4, len(args_list)) if args_list else 1

    # Process frames in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("Wind barbs and isotachs at 500 hPa plot generation complete.")

    # ------------------------------------------------------------------
    # Build GIF from PNG frames
    # ------------------------------------------------------------------
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    # Filenames contain YYYYMMDDHHMMSS, so alphabetical sort is chronological
    png_files_sorted = sorted(png_files)

    if not png_files_sorted:
        print("No PNG files were found for GIF creation; skipping animation.")
        return

    print("Creating .gif file from sorted .png files")

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]

    duration_ms = 800
    gif_file_out = f"wrf_{domain}_500hPa_WIND_Hgt_Isotachs.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")


if __name__ == "__main__":
    main()
