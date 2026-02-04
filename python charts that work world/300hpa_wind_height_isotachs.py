#!/usr/bin/env python3
"""
300hPa_Wind_Height_Isotachs_multicore_golden.py

Plot WRF 300-hPa geopotential height (dm), isotachs (knots), and wind barbs
on the native WRF map projection, using multiprocessing and building an
animated GIF.

Supports BOTH:
    * Multiple wrfout_<domain>* files (1 or more timesteps each), OR
    * A single wrfout file containing many timesteps.

Also works with static or moving nests (domain-following); the map extent and
city plotting are recomputed per frame, while grid spacing is treated as
constant for the domain/run.
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
# Canonical helper function block (v8 – contiguous, order-locked)
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
# 300 hPa isotach and wind plotting for one (file, time_index) frame
###############################################################################
def process_frame(args):
    """
    Process a single frame: one file and one time index.

    Steps:
        * Read WRF variables (ua, va, pressure, z) at this time.
        * Interpolate to 300 hPa.
        * Smooth heights, compute wind speed in knots.
        * Build map (features, cities, gridlines).
        * Plot:
            - 300 hPa heights (dm) as contours.
            - 300 hPa isotachs (knots) as filled contours.
            - 300 hPa wind barbs (knots), density adapted to grid size.
        * Save a PNG file named with the valid time.
    """
    ncfile_path, time_index, domain, path_figures = args

    # Worker-local Dataset (no sharing across processes)
    with Dataset(ncfile_path) as ncfile:
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        u = wrf.getvar(ncfile, "ua", timeidx=time_index)
        v = wrf.getvar(ncfile, "va", timeidx=time_index)
        p = wrf.getvar(ncfile, "pressure", timeidx=time_index)
        z = wrf.getvar(ncfile, "z", timeidx=time_index, units="dm")

        u_300 = wrf.interplevel(u, p, 300)
        v_300 = wrf.interplevel(v, p, 300)
        z_300 = wrf.interplevel(z, p, 300)
        z_300 = gaussian_filter(to_np(z_300), sigma=2.0)

        u_kn = to_np(u_300) * 1.94384449
        v_kn = to_np(v_300) * 1.94384449
        wind_speed_knots = np.sqrt(u_kn**2 + v_kn**2)

        lats, lons = wrf.latlon_coords(u_300)
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
            z_300,
            wind_speed_knots,
            u_kn,
            v_kn,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            z_300,
            wind_speed_knots,
            u_kn,
            v_kn,
        )

        cart_proj = wrf.get_cartopy(u_300)
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

        height_levels = np.arange(700, 1300, 5)
        contours = ax.contour(
            lons_np,
            lats_np,
            z_300,
            levels=height_levels,
            colors="black",
            linewidths=1.0,
            transform=crs.PlateCarree(),
        )
        ax.clabel(contours, inline=True, fontsize=11, fmt="%1.0f")

        # -------------------------------------------------------------------------
        # Isotachs (wind speed) filled contours
        # -------------------------------------------------------------------------

        wind_speed_ranges = np.array(
            [
                50,
                60,
                70,
                80,
                90,
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
                180,
                190,
                200,
                220,
            ]
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
                    [255, 115, 223],  # <- want this for max/over
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

        filled = ax.contourf(
            lons_np,
            lats_np,
            wind_speed_knots,
            levels=wind_speed_ranges,
            cmap=isotach_cmap,
            norm=isotach_norm,
            transform=crs.PlateCarree(),
            extend="max",
        )

        # -------------------------------------------------------------------------
        # Wind barbs: density and color adapted to isotach brightness
        # -------------------------------------------------------------------------

        ny, nx = u_kn.shape
        barb_density = max(nx // 15, ny // 15, 1)

        color_brightness = np.dot(color_map_rgb, [0.299, 0.587, 0.114])
        brightness_threshold = 0.4
        norm = plt.Normalize(vmin=wind_speed_ranges[0], vmax=wind_speed_ranges[-1])
        brightness_map = np.interp(
            norm(wind_speed_knots),
            np.linspace(0, 1, len(color_brightness)),
            color_brightness,
        )

        dark = brightness_map <= brightness_threshold
        light = ~dark

        outside = (
            (wind_speed_knots < wind_speed_ranges[0])
            | (wind_speed_knots > wind_speed_ranges[-1])
            | np.isnan(wind_speed_knots)
        )

        lons_ds = lons_np[::barb_density, ::barb_density]
        lats_ds = lats_np[::barb_density, ::barb_density]
        u_ds = u_kn[::barb_density, ::barb_density]
        v_ds = v_kn[::barb_density, ::barb_density]

        ax.barbs(
            lons_ds[light[::barb_density, ::barb_density]],
            lats_ds[light[::barb_density, ::barb_density]],
            u_ds[light[::barb_density, ::barb_density]],
            v_ds[light[::barb_density, ::barb_density]],
            length=6,
            linewidth=0.8,
            color="black",
            transform=crs.PlateCarree(),
        )

        ax.barbs(
            lons_ds[dark[::barb_density, ::barb_density]],
            lats_ds[dark[::barb_density, ::barb_density]],
            u_ds[dark[::barb_density, ::barb_density]],
            v_ds[dark[::barb_density, ::barb_density]],
            length=6,
            linewidth=0.8,
            color="lightgray",
            transform=crs.PlateCarree(),
        )

        ax.barbs(
            lons_ds[outside[::barb_density, ::barb_density]],
            lats_ds[outside[::barb_density, ::barb_density]],
            u_ds[outside[::barb_density, ::barb_density]],
            v_ds[outside[::barb_density, ::barb_density]],
            length=6,
            linewidth=0.8,
            color="black",
            transform=crs.PlateCarree(),
        )

        cbar = plt.colorbar(
            filled,
            ax=ax,
            orientation="vertical",
            pad=0.05,
            shrink=0.8,
            ticks=wind_speed_ranges,
        )
        cbar.set_label("Isotachs (knots)")

        plt.title(
            "Weather Research and Forecasting Model\n"
            f"Average Grid Spacing: {avg_dx_km} x {avg_dy_km} km\n"
            "Wind Barbs at 300 hPa (knots)\n"
            "Isotachs (knots)\n"
            "300 hPa Geopotential Heights (dm)",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_wind_300hPa_{fname_time}.png"
        plt.savefig(
            os.path.join(path_figures, "Images", file_out),
            bbox_inches="tight",
            dpi=150,
        )

        plt.close(fig)


###############################################################################
# Frame discovery (v9 canonical)
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
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example:\n"
            "    300hPa_Wind_Height_Isotachs_multicore_golden.py "
            "/home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    path_figures = "wrf_300hPa_Isotachs_Wnd_Hgt"
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
    args_list = [
        (ncfile_path, time_index, domain, path_figures)
        for (ncfile_path, time_index) in frames
    ]

    max_workers = min(4, len(args_list)) if args_list else 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("Wind barbs and isotachs at 300 hPa plot generation complete.")

    png_files = sorted(f for f in os.listdir(image_folder) if f.endswith(".png"))
    if not png_files:
        sys.exit(0)

    images = [Image.open(os.path.join(image_folder, fname)) for fname in png_files]

    gif_file_out = f"wrf_{domain}_300hPa_WIND_Hgt_Isotachs.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
