#!/usr/bin/env python3
"""
Tropical Surface Gusts (mph) with SLP and 10 m wind

Playbook v3:
- Supports multiple wrfout_<domain>* files, each with one or more timesteps.
- Supports a single wrfout file containing many timesteps.
- Safe for static and moving/vortex-following nests:
  * For each frame, lat/lon, grid spacing, extent, and city thinning are recomputed.

Physics:
- Identical to original script:
  * Same WRF variables (slp, U10, V10, ua, va, PBLH, height_agl).
  * Same PBLH-capped gust parameterization (cap at 1000 m, random offset).
  * Same Gaussian smoothing (sigma=1) for gusts.
  * Same conversion of gusts to mph (m/s * 2.23694).
  * Same SLP contour levels (870–1090 hPa, interval 4).
  * Same gust contour levels (mph) and colormap.
  * Same barb brightness logic.
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
# Natural Earth features (order preserved; commented-out options intact)
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
# Frame processing: one (file, time_index) pair
###############################################################################
def process_frame(args):
    """
    Process a single frame (one file, one time_index):

      * Compute tropical surface gusts (mph) with original parameterization.
      * Plot:
          - SLP contours (hPa) + H/L markers
          - Max tropical gust potential (mph) shaded
          - 10 m wind barbs (knots) with brightness-aware color
          - Cities, coastlines, boundaries, lat/lon grid
      * Save PNG named with valid time.
    """
    ncfile_path, time_index, domain, path_figures = args

    image_folder = os.path.join(path_figures, "Images")

    # Open the WRF file for this frame
    with Dataset(ncfile_path) as ncfile:
        # Determine valid time
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # -------------------------------------------------------------------------
        # Retrieve WRF variables for this time (physics unchanged)
        # -------------------------------------------------------------------------
        slp = wrf.getvar(ncfile, "slp", timeidx=time_index, meta=True)
        u10 = wrf.getvar(ncfile, "U10", timeidx=time_index, meta=True)
        v10 = wrf.getvar(ncfile, "V10", timeidx=time_index, meta=True)
        u = wrf.getvar(ncfile, "ua", timeidx=time_index, meta=True)
        v = wrf.getvar(ncfile, "va", timeidx=time_index, meta=True)
        pblh = wrf.getvar(ncfile, "PBLH", timeidx=time_index, meta=True)
        height = wrf.getvar(
            ncfile, "height_agl", timeidx=time_index, meta=True, units="m"
        )

        # 10 m wind to knots (unchanged from original script)
        u10_knots = u10 * 1.94384449
        v10_knots = v10 * 1.94384449

        # Total 10 m wind speed (m/s)
        wspd_wdir10 = np.sqrt((u10**2) + (v10**2))

        # Wind speed at all levels (m/s)
        wspd_wdir = np.sqrt((u**2) + (v**2))

        # Adjust PBLH to cap at 1000 m (unchanged logic)
        adj_pblh = np.where(pblh < 1000, pblh, 1000)
        adj_pblh = np.where(adj_pblh <= height[0, :, :], height[0, :, :] + 1, adj_pblh)

        # Random offset to avoid exact equality with height field (original logic)
        tolerance = 1e-6
        while True:
            random_offset = np.random.uniform(0, 1, adj_pblh.shape)
            adj_pblh_with_offset = adj_pblh + random_offset
            if not np.any(np.abs(adj_pblh_with_offset - height[0, :, :]) < tolerance):
                adj_pblh = adj_pblh_with_offset
                break

        # Interpolate wind speed to capped PBLH (vPBL in parameterization)
        wspd_pblh = wrf.interplevel(wspd_wdir, height, adj_pblh)

        # Weight factor based on PBLH (unchanged physics)
        weight_k = 1 - (np.minimum(pblh, 1000) / 2000.0)

        # Surface gust speed (m/s) using original parameterization
        surface_gust_speed = wspd_wdir10 + ((wspd_pblh - wspd_wdir10) * weight_k)

        # Smooth and convert to mph (unchanged: m/s * 2.23694, sigma=1)
        gusts = gaussian_filter(surface_gust_speed * 2.23694, sigma=1)

        # -------------------------------------------------------------------------
        # Geometry: lat/lon, spacing, extent, cities, gridlines
        # -------------------------------------------------------------------------
        lats, lons = wrf.latlon_coords(slp)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,
        ) = compute_grid_and_spacing(lats, lons)

        # -------------------------------------------------------------------------
        # Derived fields used in plotting (ordering preserved; physics unchanged)
        # -------------------------------------------------------------------------
        smooth_slp = gaussian_filter(to_np(slp), sigma=5.0)

        u10_knots_np = to_np(u10_knots)
        v10_knots_np = to_np(v10_knots)

        # -------------------------------------------------------------------------
        # Dateline continuity and polar masking (v9 canonical helper)
        # -------------------------------------------------------------------------
        (
            lats_np,
            lons_np,
            smooth_slp,
            gusts,
            u10_knots_np,
            v10_knots_np,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            smooth_slp,
            gusts,
            u10_knots_np,
            v10_knots_np,
        )

        cart_proj = wrf.get_cartopy(slp)

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

        # Base land feature
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])

        # Natural Earth features
        for feature in features:
            add_feature(ax, *feature)

        # Cities and gridlines
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)
        gl = add_latlon_gridlines(ax)

        # -------------------------------------------------------------------------
        # Sea-level pressure contours and H/L markers
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

        contour_interval = 4
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
        ax.clabel(SLP_contours, inline=1, fontsize=10, fmt="%1.0f")
        plt.clabel(SLP_contours, inline=True, fontsize=12, fmt="%d", colors="k")

        # -------------------------------------------------------------------------
        # Gust shaded field (mph, unchanged levels)
        # -------------------------------------------------------------------------
        wind_speed_ranges = np.array(
            [0, 10, 20, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
        )

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

        gust_map = plt.matplotlib.colors.ListedColormap(color_map_rgb[:-1])
        gust_map.set_over(color_map_rgb[-1])
        gust_norm = plt.matplotlib.colors.BoundaryNorm(wind_speed_ranges, gust_map.N)

        gust_contours = ax.contourf(
            lons_np,
            lats_np,
            to_np(gusts),
            levels=wind_speed_ranges,
            cmap=gust_map,
            norm=gust_norm,
            transform=crs.PlateCarree(),
            extend="max",
        )

        cbar = plt.colorbar(
            gust_contours, ax=ax, orientation="vertical", pad=0.1, shrink=0.75
        )
        cbar.set_label("Wind Gust (mph)")
        cbar.set_ticks(wind_speed_ranges)
        cbar.set_ticklabels([f"{level:.0f}" for level in wind_speed_ranges])

        # -------------------------------------------------------------------------
        # 10 m wind barbs, brightness-aware (unchanged logic)
        # -------------------------------------------------------------------------
        ny, nx = np.shape(u10_knots_np)
        desired_barbs = 15
        barb_density_x = max(nx // desired_barbs, 1)
        barb_density_y = max(ny // desired_barbs, 1)
        barb_density = max(barb_density_x, barb_density_y)

        # Brightness of colormap colors
        color_brightness = np.dot(color_map_rgb, [0.299, 0.587, 0.114])
        brightness_threshold = 0.4

        norm_for_barbs = plt.Normalize(
            vmin=wind_speed_ranges[0], vmax=wind_speed_ranges[-1]
        )
        gusts_normalized = norm_for_barbs(gusts)

        brightness_map = np.interp(
            gusts_normalized,
            np.linspace(0, 1, len(color_brightness)),
            color_brightness,
        )

        dark_region = brightness_map <= brightness_threshold
        light_region = brightness_map > brightness_threshold

        outside_contour_mask = (
            (to_np(gusts) < wind_speed_ranges[0])
            | (to_np(gusts) > wind_speed_ranges[-1])
            | np.isnan(to_np(gusts))
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

        # Black barbs on light background
        ax.barbs(
            lons_ds[inside_light],
            lats_ds[inside_light],
            u_ds[inside_light],
            v_ds[inside_light],
            length=6,
            sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
            linewidth=0.8,
            color="black",
            zorder=2,
            transform=crs.PlateCarree(),
        )

        # Light-gray barbs on dark background
        ax.barbs(
            lons_ds[inside_dark],
            lats_ds[inside_dark],
            u_ds[inside_dark],
            v_ds[inside_dark],
            length=6,
            sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
            linewidth=0.8,
            color="lightgray",
            zorder=2,
            transform=crs.PlateCarree(),
        )

        # Black barbs outside gust shading
        ax.barbs(
            lons_ds[outside_contour],
            lats_ds[outside_contour],
            u_ds[outside_contour],
            v_ds[outside_contour],
            length=6,
            sizes=dict(emptybarb=0.25, spacing=0.2, height=0.5),
            linewidth=0.8,
            color="black",
            zorder=2,
            transform=crs.PlateCarree(),
        )

        # -------------------------------------------------------------------------
        # Titles and save
        # -------------------------------------------------------------------------
        plt.title(
            "Weather Research and Forecasting Model\n"
            f"Average Grid Spacing:{avg_dx_km}x{avg_dy_km}km\n"
            "Sea Level Pressure (hPa)\n"
            "Wind Barbs (mph)\n"
            "Max Tropical Wind Gusts Potential (sigma=1 smoothed)",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_slp_wind_gust_ft_{fname_time}.png"

        plt.savefig(
            os.path.join(image_folder, file_out),
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
# Main script entry point
###############################################################################
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Parse CLI args
    # -------------------------------------------------------------------------
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example:\n"
            "    Tropical_SFC_Gusts_mph_v3.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # -------------------------------------------------------------------------
    # Output directories
    # -------------------------------------------------------------------------
    path_figures = "wrf_Tropical_SFC_Gusts_mph_figures"
    image_folder = os.path.join(path_figures, "Images")
    animation_folder = os.path.join(path_figures, "Animation")

    for folder in (path_figures, image_folder, animation_folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    # -------------------------------------------------------------------------
    # Find WRF output files
    # -------------------------------------------------------------------------
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print(f"No wrfout files found in {path_wrf} matching wrfout_{domain}*")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Discover frames and build args_list
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
    # Process frames in parallel
    # -------------------------------------------------------------------------
    max_workers = min(4, len(args_list)) if args_list else 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("SFC Plots Complete")

    # -------------------------------------------------------------------------
    # Build animated GIF from PNG frames
    # -------------------------------------------------------------------------
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        sys.exit(0)

    # Filenames include YYYYMMDDHHMMSS → simple sort is chronological
    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files")

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]
    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        sys.exit(0)

    gif_file_out = f"wrf_{domain}_SLP_WIND_Gust.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
