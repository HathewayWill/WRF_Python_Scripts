#!/usr/bin/env python3
"""
850hPa_T_Wind_Hgt.py

Plot WRF 850-hPa wind barbs (knots), temperature (°C),
and 850-hPa geopotential height (dm) on a Cartopy map.

This script can handle:
    * Multiple wrfout_<domain>* files, each with one or more timesteps.
    * A single wrfout file containing many timesteps.

It does NOT assume the domain is static:
    * For each timestep, lat/lon, grid spacing, and extent are
      recomputed from the WRF fields. This automatically works
      for both static nests and moving/vortex-following nests.

-------------------------------------------------------------------------------
Preserved documentation from the pre-golden script (moved out of canonical defs)
-------------------------------------------------------------------------------

Add a Natural Earth feature to a Cartopy axis.

Extract a valid time from a WRF output filename as a fallback.

Handles filenames of the form:
    wrfout_d01_YYYY-MM-DD_HH:MM:SS
    wrfout_d01_YYYY-MM-DD_HH_MM_SS

If parsing fails, uses file modification time.

Get the valid time for a given time index from the WRF file.

Preferred: use wrf.extract_times (uses model time metadata).
Fallback: parse from wrfout filename.

Given 2D latitude/longitude arrays, compute:

    lats_np, lons_np       : numpy arrays
    avg_dx_km, avg_dy_km   : average grid spacing (km)
    extent_adjustment      : padding for map extent
    label_adjustment       : offset for labels (if needed)

This is recomputed per timestep, which is safe for both static and
moving/vortex-following nests.

Add latitude/longitude gridlines with consistent styling and labels.

Call this after setting the map extent.

Subset and thin cities, then plot them on the map.

- Subsets cities to the current domain extent.
- Sorts by POP_MAX and keeps the top 150.
- Thins them so they are at least a minimum distance apart.
- Plots city markers and labels.

Discover all (file, time_index) combinations to plot.

This supports:
    * Many wrfout_<domain>* files, each with one or more Time steps.
    * A single wrfout file containing multiple Time steps.

Returns
-------
frames : list of tuples
    Each tuple is (ncfile_path, time_index).
"""

###############################################################################
# Imports (clean, ordered)
###############################################################################
import glob

###############################################################################
# Standard library imports
###############################################################################
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

###############################################################################
# Third-party imports
###############################################################################
import numpy as np
import wrf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
from netCDF4 import Dataset
from PIL import Image
from scipy.ndimage import gaussian_filter
from wrf import ALL_TIMES, to_np  # ALL_TIMES imported for consistency, though unused

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
# 850 hPa T / wind / height plotting for one (file, time_index) frame
###############################################################################
def process_frame(args):
    """
    Process a single frame: one file and one time index.

    Steps:
        * Read WRF variables (ua, va, temp, z, pressure) at this time.
        * Interpolate to 850 hPa.
        * Smooth 850-hPa height and temperature.
        * Convert wind to knots.
        * Build map (features, cities, gridlines).
        * Plot:
            - 850-hPa height (dm) as contours.
            - 850-hPa temperature (°C) as filled contours.
            - 0°C isotherm as blue contour.
            - 850-hPa wind barbs (knots), density and color adapted
              to background brightness.
        * Save a PNG file named with the valid time.
    """
    ncfile_path, time_index, domain, path_figures = args

    # Open the WRF file for this frame
    with Dataset(ncfile_path) as ncfile:
        # Get valid time as a datetime object
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # -------------------------------------------------------------------------
        # Get base WRF variables at this time index
        # -------------------------------------------------------------------------
        u = wrf.getvar(ncfile, "ua", timeidx=time_index)  # m/s
        v = wrf.getvar(ncfile, "va", timeidx=time_index)  # m/s
        t = wrf.getvar(ncfile, "temp", timeidx=time_index)  # K
        p = wrf.getvar(ncfile, "pressure", timeidx=time_index)  # hPa
        z = wrf.getvar(
            ncfile, "z", timeidx=time_index, units="dm"
        )  # geopotential height (dm)

        # -------------------------------------------------------------------------
        # Interpolate to 850 hPa
        # -------------------------------------------------------------------------
        t_850 = wrf.vinterp(
            ncfile, t, "pressure", [850], extrapolate=True, squeeze=True, meta=True
        ).squeeze()
        u_850 = wrf.vinterp(
            ncfile, u, "pressure", [850], extrapolate=True, squeeze=True, meta=True
        ).squeeze()
        v_850 = wrf.vinterp(
            ncfile, v, "pressure", [850], extrapolate=True, squeeze=True, meta=True
        ).squeeze()
        z_850 = wrf.vinterp(
            ncfile,
            z,
            "pressure",
            [850],
            field_type="z",
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()

        # Smooth 850-hPa height (dm)
        z_850_smooth = gaussian_filter(to_np(z_850), sigma=2.0)

        # Smooth temperature, convert to Celsius
        t_850_smooth = gaussian_filter(to_np(t_850), sigma=1.0)
        t_850_c = t_850_smooth - 273.15  # °C
        t_850_c_np = t_850_c  # already numpy

        # Convert U & V wind components to knots (from m/s)
        u_850_knots = u_850 * 1.94384449
        v_850_knots = v_850 * 1.94384449

        u_850_knots_np = to_np(u_850_knots)
        v_850_knots_np = to_np(v_850_knots)

        # -------------------------------------------------------------------------
        # Get lat/lon and grid spacing (static or moving nests)
        # -------------------------------------------------------------------------
        lats, lons = wrf.latlon_coords(u_850)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            _label_adjustment,  # not used here, but returned for consistency
        ) = compute_grid_and_spacing(lats, lons)

        # -------------------------------------------------------------------------
        # Dateline continuity and polar masking (v9 canonical helper)
        # -------------------------------------------------------------------------
        (
            lats_np,
            lons_np,
            z_850_smooth,
            t_850_c_np,
            u_850_knots_np,
            v_850_knots_np,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            z_850_smooth,
            t_850_c_np,
            u_850_knots_np,
            v_850_knots_np,
        )

        # -------------------------------------------------------------------------
        # Set up figure and Cartopy projection
        # -------------------------------------------------------------------------
        cart_proj = wrf.get_cartopy(u_850)  # WRF native projection

        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Map extent with padding based on grid spacing
        ax.set_extent(
            [
                lons_np.min() - extent_adjustment,
                lons_np.max() + extent_adjustment,
                lats_np.min() - extent_adjustment,
                lats_np.max() + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        # -------------------------------------------------------------------------
        # Add land, coastlines, political boundaries, and cities
        # -------------------------------------------------------------------------
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])

        for feature in features:
            add_feature(ax, *feature)

        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)

        gl = add_latlon_gridlines(ax)

        # -------------------------------------------------------------------------
        # 850-hPa height contours (dm)
        # -------------------------------------------------------------------------
        z850_start = 60  # dm
        z850_end = 200  # dm
        contour_interval = 10  # dm

        height_levels = np.arange(z850_start, z850_end, contour_interval)

        height_contours = ax.contour(
            lons_np,
            lats_np,
            z_850_smooth,
            levels=height_levels,
            colors="black",
            linestyles="solid",
            transform=crs.PlateCarree(),
        )
        ax.clabel(height_contours, inline=True, fontsize=10, fmt="%1.0f")

        # -------------------------------------------------------------------------
        # Temperature color map and filled contours
        # -------------------------------------------------------------------------
        temp_levels = np.arange(-50, 50, 2)

        color_map_rgb = (
            np.array(
                [
                    [145, 0, 63],
                    [192, 13, 80],
                    [219, 30, 114],
                    [228, 59, 149],
                    [225, 102, 179],
                    [250, 112, 216],
                    [255, 161, 228],
                    [255, 215, 241],
                    [248, 248, 251],
                    [220, 220, 236],
                    [196, 197, 224],
                    [173, 172, 210],
                    [147, 142, 194],
                    [116, 105, 176],
                    [90, 53, 150],
                    [44, 17, 132],
                    [13, 20, 135],
                    [11, 65, 159],
                    [1, 96, 189],
                    [26, 138, 233],
                    [54, 175, 255],
                    [81, 202, 255],
                    [113, 214, 255],
                    [157, 244, 255],
                    [110, 231, 224],
                    [35, 192, 182],
                    [0, 150, 147],
                    [14, 99, 98],
                    [11, 99, 61],
                    [21, 128, 58],
                    [57, 167, 88],
                    [109, 192, 114],
                    [145, 209, 142],
                    [182, 233, 170],
                    [220, 255, 187],
                    [254, 255, 179],
                    [255, 241, 164],
                    [254, 222, 138],
                    [254, 198, 95],
                    [253, 171, 43],
                    [253, 146, 56],
                    [252, 102, 49],
                    [242, 58, 36],
                    [219, 22, 29],
                    [181, 2, 37],
                    [143, 0, 38],
                    [110, 0, 50],
                    [77, 0, 60],
                    [40, 0, 40],
                ],
                np.float32,
            )
            / 255.0
        )

        temp_map = plt.matplotlib.colors.ListedColormap(color_map_rgb)
        temp_norm = plt.matplotlib.colors.BoundaryNorm(temp_levels, temp_map.N)

        # 0°C isotherm as a blue line
        zero_isotherm = ax.contour(
            lons_np,
            lats_np,
            t_850_c_np,
            levels=[0],
            colors="blue",
            linestyles="solid",
            transform=crs.PlateCarree(),
        )
        ax.clabel(zero_isotherm, inline=True, fontsize=10, fmt="%1.0f°C")

        # Filled temperature contours
        contour_filled = ax.contourf(
            lons_np,
            lats_np,
            t_850_c_np,
            levels=temp_levels,
            cmap=temp_map,
            norm=temp_norm,
            extend="both",
            transform=crs.PlateCarree(),
        )

        # Colorbar for temperature
        cbar = plt.colorbar(
            contour_filled,
            ax=ax,
            orientation="vertical",
            pad=0.05,
            label="Temperature (°C)",
        )

        # -------------------------------------------------------------------------
        # Wind barbs: density and color adapted to temperature background brightness
        # -------------------------------------------------------------------------
        ny, nx = np.shape(u_850_knots_np)
        desired_barbs = 15

        barb_density_x = max(nx // desired_barbs, 1)
        barb_density_y = max(ny // desired_barbs, 1)
        barb_density = max(barb_density_x, barb_density_y)

        # Brightness of each color in the temp colormap
        color_brightness = np.dot(color_map_rgb, [0.299, 0.587, 0.114])
        brightness_threshold = 0.4

        # Normalize temperature into [0,1] over the chosen contour range
        norm = plt.Normalize(vmin=temp_levels[0], vmax=temp_levels[-1])
        temp_normalized = norm(t_850_c_np)

        # Map normalized temperatures to brightness values
        brightness_map = np.interp(
            temp_normalized,
            np.linspace(0, 1, len(color_brightness)),
            color_brightness,
        )

        # Masks for light/dark regions
        dark_region = brightness_map <= brightness_threshold
        light_region = brightness_map > brightness_threshold

        # Mask for regions outside the temperature contour range or NaNs
        outside_contour_mask = (
            (t_850_c_np < temp_levels[0])
            | (t_850_c_np > temp_levels[-1])
            | np.isnan(t_850_c_np)
        )
        inside_contour_mask = ~outside_contour_mask

        # Downsample masks according to barb density
        inside_light = (inside_contour_mask & light_region)[
            ::barb_density, ::barb_density
        ]
        inside_dark = (inside_contour_mask & dark_region)[
            ::barb_density, ::barb_density
        ]
        outside_contour = outside_contour_mask[::barb_density, ::barb_density]

        # Convenience slices
        lons_ds = lons_np[::barb_density, ::barb_density]
        lats_ds = lats_np[::barb_density, ::barb_density]
        u_ds = u_850_knots_np[::barb_density, ::barb_density]
        v_ds = v_850_knots_np[::barb_density, ::barb_density]

        # Barbs on light background -> black barbs
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

        # Barbs on dark background -> light gray barbs
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

        # Barbs outside the filled temperature range -> always black
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
        # Titles and saving
        # -------------------------------------------------------------------------
        plt.title(
            f"Weather Research and Forecasting Model\n"
            f"Average Grid Spacing: {avg_dx_km} x {avg_dy_km} km\n"
            f"Wind Barbs at 850 hPa (knots)\n"
            f"Temperature (°C)\n"
            f"850 hPa Geopotential Heights (dm)",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        # Use valid_dt for filename timestamp, keeps sort = time order
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_wind_850hPa_{fname_time}.png"

        plt.savefig(
            os.path.join(path_figures, "Images", file_out),
            bbox_inches="tight",
            dpi=150,
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
            "    850hPa_T_Wind_Hgt.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # -------------------------------------------------------------------------
    # Prepare output directories
    # -------------------------------------------------------------------------
    path_figures = "wrf_850hPa_T_Wind_Hgt"
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

    # -------------------------------------------------------------------------
    # Process frames in parallel using ProcessPoolExecutor
    # -------------------------------------------------------------------------
    max_workers = min(4, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("850 hPa plot generation complete.")

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

    gif_file_out = f"wrf_{domain}_850hPa_WIND_TEMP_Hgt.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
