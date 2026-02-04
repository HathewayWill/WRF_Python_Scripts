#!/usr/bin/env python3
"""
SFC_SLP_Wind_TD_degF.py

Plot WRF sea-level pressure (hPa), 10-m wind barbs (knots),
and 2-m dewpoint temperature (°F) on a Cartopy map.

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
from wrf import ALL_TIMES, to_np  # ALL_TIMES imported for consistency, even if unused

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


# Extract a valid time from a WRF output filename as a fallback.
# Handles:
#     wrfout_d01_YYYY-MM-DD_HH:MM:SS
#     wrfout_d01_YYYY-MM-DD_HH_MM_SS
# Falls back to standard index slicing, then to file mtime.
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


# Get the valid time for a given time index from the WRF file.
# Preferred: wrf.extract_times (model metadata). Fallback: parse from filename.
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


# Given 2D latitude/longitude arrays, compute:
#     lats_np, lons_np       : numpy arrays
#     avg_dx_km, avg_dy_km   : average grid spacing (km)
#     extent_adjustment      : padding for map extent
#     label_adjustment       : offset for labels (H/L, etc.)
# Recomputed per frame → safe for static and moving nests.
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


# Add latitude/longitude gridlines with consistent styling and labels.
# Call after setting map extent.
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


# Subset and thin cities, then plot them.
# - Subset to domain extent.
# - Sort by POP_MAX, keep top 150.
# - Thin by min distance (in degrees) based on resolution.
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
# Natural Earth features (v9 canonical – verbatim, order-locked comments intact)
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

# Deviation Justification (features)
# Deviation type: features
# What changed (exactly): admin_2_counties is enabled (uncommented) vs the v9 canonical example.
# Why it is technically required: Preserve original script’s boundary layer content.
# Why v9 canonical cannot satisfy this case: Canonical example shows counties commented out; this product requires them enabled.
# Evidence: N/A (visual layer preservation)
# Scope: SFC_SLP_Wind_TD_degF.py (this product only)
# Rollback plan: Comment out admin_2_counties entry to match v9 example exactly.

###############################################################################
# Cities (module scope)
###############################################################################
## Load populated places (cities) once per worker process.
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


###############################################################################
# SFC SLP + 10 m wind + 2 m Td (°F) plotting for one (file, time_index) frame
###############################################################################
def process_frame(args):
    """
    Process a single frame: one file and one time index.

    Steps:
        * Read WRF variables (slp, U10, V10, td2) at this time.
        * Smooth SLP and Td fields.
        * Compute wind speed in knots.
        * Build map (features, cities, gridlines).
        * Plot:
            - SLP contours (hPa).
            - 2 m Td (°F) as filled contours.
            - 10 m wind barbs (knots), density and color adapted
              to Td shading brightness.
        * Save a PNG file named with the valid time.
    """
    ncfile_path, time_index, domain, path_figures = args

    # Open the WRF file for this frame
    with Dataset(ncfile_path) as ncfile:

        # Get valid time as a datetime object
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # -------------------------------------------------------------------------
        # Physics block: WRF variable access and core diagnostics (immutable)
        # -------------------------------------------------------------------------
        slp = wrf.getvar(ncfile, "slp", timeidx=time_index)
        u10 = wrf.getvar(ncfile, "U10", timeidx=time_index)
        v10 = wrf.getvar(ncfile, "V10", timeidx=time_index)
        td2m = gaussian_filter(
            wrf.getvar(ncfile, "td2", units="degF", timeidx=time_index),
            sigma=1.0,
        )

        u10_knots = u10 * 1.94384449
        v10_knots = v10 * 1.94384449

        # Convert wind speed from m/s to knots
        wind_speed_knots = np.sqrt(u10_knots**2 + v10_knots**2)

        # Get the latitude and longitude points
        lats, lons = wrf.latlon_coords(slp)

        # -------------------------------------------------------------------------
        # Grid spacing, extent, and projection (moving-nest safe)
        # -------------------------------------------------------------------------
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,
        ) = compute_grid_and_spacing(lats, lons)

        # Smooth SLP and add H/L markers
        smooth_slp = gaussian_filter(to_np(slp), sigma=5.0)

        # -------------------------------------------------------------------------
        # Dateline continuity and polar masking (v9 canonical helper)
        # -------------------------------------------------------------------------
        u10_knots_np = to_np(u10_knots)
        v10_knots_np = to_np(v10_knots)
        td2m_np = to_np(td2m)

        (
            lats_np,
            lons_np,
            smooth_slp,
            td2m_np,
            u10_knots_np,
            v10_knots_np,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            smooth_slp,
            td2m_np,
            u10_knots_np,
            v10_knots_np,
        )

        # Get the cartopy mapping object from a WRF field
        cart_proj = wrf.get_cartopy(slp)

        # Create a plot using cartopy and matplotlib
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Set the map extent (with padding based on grid spacing)
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
        for feature in features:
            add_feature(ax, *feature)

        # Add cities and gridlines
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)
        add_latlon_gridlines(ax)

        # -------------------------------------------------------------------------
        # Smooth SLP and add H/L markers
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
        # Dewpoint (°F) filled contours
        # -------------------------------------------------------------------------
        td2m_levels = np.arange(-60, 121, 5)

        color_map_rgb = (
            np.array(
                [
                    [145, 0, 63],
                    [206, 18, 86],
                    [231, 41, 138],
                    [223, 101, 176],
                    [255, 115, 223],
                    [255, 190, 232],
                    [255, 255, 255],
                    [218, 218, 235],
                    [188, 189, 220],
                    [158, 154, 200],
                    [117, 107, 177],
                    [84, 39, 143],
                    [13, 0, 125],
                    [13, 61, 156],
                    [0, 102, 194],
                    [41, 158, 255],
                    [74, 199, 255],
                    [115, 215, 255],
                    [173, 255, 255],
                    [48, 207, 194],
                    [0, 153, 150],
                    [18, 87, 87],
                    [6, 109, 44],
                    [49, 163, 84],
                    [116, 196, 118],
                    [161, 217, 155],
                    [211, 255, 190],
                    [255, 255, 179],
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
            / 255.0
        )

        td2m_map = plt.matplotlib.colors.ListedColormap(color_map_rgb)
        td2m_norm = plt.matplotlib.colors.BoundaryNorm(td2m_levels, td2m_map.N)

        TD = ax.contourf(
            lons_np,
            lats_np,
            td2m_np,
            levels=td2m_levels,
            cmap=td2m_map,
            norm=td2m_norm,
            extend="both",
            transform=crs.PlateCarree(),
        )

        # -------------------------------------------------------------------------
        # SLP contours (fixed 2 hPa interval from 870 to 1090 hPa)
        # -------------------------------------------------------------------------
        contour_interval = 2
        SLP_start = 870
        SLP_end = 1090
        SLP_levels = np.arange(SLP_start, SLP_end, contour_interval)

        SLP_contours = ax.contour(
            lons_np,
            lats_np,
            smooth_slp,
            levels=SLP_levels,
            colors="k",
            linewidths=1.0,
            transform=crs.PlateCarree(),
        )
        ax.clabel(SLP_contours, inline=True, fontsize=12, fmt="%d", colors="k")

        # -------------------------------------------------------------------------
        # Wind barbs: density and color adapted to Td brightness
        # -------------------------------------------------------------------------
        ny, nx = np.shape(u10_knots_np)

        desired_barbs = 15
        barb_density_x = max(nx // desired_barbs, 1)
        barb_density_y = max(ny // desired_barbs, 1)
        barb_density = max(barb_density_x, barb_density_y)

        # Compute brightness of colormap colors
        color_brightness = np.dot(color_map_rgb, [0.299, 0.587, 0.114])
        brightness_threshold = 0.4

        norm_td = plt.Normalize(vmin=td2m_levels[0], vmax=td2m_levels[-1])
        td2m_normalized = norm_td(td2m_np)

        brightness_map = np.interp(
            td2m_normalized,
            np.linspace(0, 1, len(color_brightness)),
            color_brightness,
        )

        dark_region = brightness_map <= brightness_threshold
        light_region = brightness_map > brightness_threshold

        outside_contour_mask = (
            (to_np(td2m_np) < td2m_levels[0])
            | (to_np(td2m_np) > td2m_levels[-1])
            | np.isnan(to_np(td2m_np))
        )
        inside_contour_mask = ~outside_contour_mask

        inside_light = (inside_contour_mask & light_region)[
            ::barb_density, ::barb_density
        ]
        inside_dark = (inside_contour_mask & dark_region)[
            ::barb_density, ::barb_density
        ]
        outside_contour = outside_contour_mask[::barb_density, ::barb_density]

        lons_ds = lons_np[::barb_density, ::barb_density]
        lats_ds = lats_np[::barb_density, ::barb_density]
        u_ds = u10_knots_np[::barb_density, ::barb_density]
        v_ds = v10_knots_np[::barb_density, ::barb_density]

        # Inside Td range, light background -> black barbs
        ax.barbs(
            lons_ds[inside_light],
            lats_ds[inside_light],
            u_ds[inside_light],
            v_ds[inside_light],
            length=6,
            sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
            linewidth=0.8,
            color="black",
            transform=crs.PlateCarree(),
        )

        # Inside Td range, dark background -> light gray barbs
        ax.barbs(
            lons_ds[inside_dark],
            lats_ds[inside_dark],
            u_ds[inside_dark],
            v_ds[inside_dark],
            length=6,
            sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
            linewidth=0.8,
            color="lightgray",
            transform=crs.PlateCarree(),
        )

        # Outside Td contour range -> black barbs
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
        cbar2 = fig.colorbar(TD, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
        cbar2.set_label("Surface Dewpoint Temperature (°F)", fontsize=14)

        plt.title(
            "Weather Research and Forecasting Model\n"
            f"Average Grid Spacing: {avg_dx_km} x {avg_dy_km} km\n"
            "Sea Level Pressure (hPa)\n"
            "Wind Barbs (knots)\n"
            "Surface (2 m) Dewpoint Temperature (°F)",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        # -------------------------------------------------------------------------
        # Save PNG using valid time for filename (for chronological GIF sorting)
        # -------------------------------------------------------------------------
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_slp_wind_TD_degF_{fname_time}.png"

        images_folder = os.path.join(path_figures, "Images")
        if not os.path.isdir(images_folder):
            os.mkdir(images_folder)

        plt.savefig(
            os.path.join(images_folder, file_out),
            bbox_inches="tight",
            dpi=250,
        )

        plt.close(fig)


###############################################################################
# Frame Discovery (v9 canonical)
###############################################################################
# Discover all (file, time_index) combinations.
# Supports:
#     * Many wrfout_<domain>* files with one or more Time steps.
#     * A single wrfout file with multiple Time steps.
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
            "    SFC_SLP_Wind_TD_degF.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # -------------------------------------------------------------------------
    # Prepare output directories
    # -------------------------------------------------------------------------
    path_figures = "wrf_SFC_TD_degF_figures"
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

    print("SFC SLP + Wind + Td (°F) plot generation complete.")

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

    gif_file_out = f"wrf_{domain}_SLP_WIND_TD_degF.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
