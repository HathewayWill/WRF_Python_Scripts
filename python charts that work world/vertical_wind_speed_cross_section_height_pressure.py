#!/usr/bin/env python3
"""
WRF_Cross_Section_WSPD_Height_Pressure.py

Plot a vertical cross section of wind speed (kt) and wind barbs between two
lat/lon points, for all times in one or more wrfout_<domain> files.

v3 playbook aligned + updated with the newer inset-map + polar-projection fixes:
    * Supports multiple wrfout files AND multiple timesteps per file.
    * One frame = one (file, time_index) pair.
    * Uses netCDF4 + wrf-python only (no xarray).
    * Time handled via wrf.extract_times with filename fallback (RH/v9 pattern).
    * Grid spacing computed with metpy from wrf.latlon_coords (moving-nest safe).
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
import matplotlib.ticker as ticker
import metpy.calc as mpcalc
import numpy as np
import wrf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
from netCDF4 import Dataset
from PIL import Image
from scipy.ndimage import gaussian_filter
from wrf import to_np

warnings.filterwarnings("ignore")


###############################################################################
# Time helpers (MATCH RH/v9 pattern)
###############################################################################
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

    # Fallback: slice based on common wrfout pattern
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
        # Last resort: file mtime (UTC)
        return datetime.utcfromtimestamp(os.path.getmtime(path))


def get_valid_time(ncfile: Dataset, ncfile_path: str, time_index: int) -> datetime:
    """
    Preferred: wrf.extract_times (model metadata).
    Fallback: parse from filename.
    (This is the same normalization logic as your RH/v9 script.)
    """
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
# Grid / spacing helpers (moving-nest safe)
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


def add_latlon_gridlines(ax, draw_labels=True):
    gl = ax.gridlines(
        crs=crs.PlateCarree(),
        draw_labels=draw_labels,
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


###############################################################################
# Map features
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
]

###############################################################################
# Cities (module scope)
###############################################################################
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


def plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km, max_cities=10):
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

        if len(selected_rows) >= max_cities:
            break

    if not selected_rows:
        return

    filtered_cities = gpd.GeoDataFrame(selected_rows).set_geometry("geometry")

    for city_name, loc in zip(filtered_cities.NAME, filtered_cities.geometry):
        ax.plot(
            loc.x,
            loc.y,
            marker="o",
            markersize=4,
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
            fontsize=7,
            color="black",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", alpha=0.4),
            clip_on=True,
        )


###############################################################################
# Dateline continuity + polar mask helper (from your newer script)
###############################################################################
def handle_domain_continuity_and_polar_mask(lats_np, lons_np, *fields):
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


def _validate_point(lat, lon, name="Point"):
    if not np.isfinite(lat) or not np.isfinite(lon):
        raise ValueError(f"{name} has non-finite lat/lon: {lat}, {lon}")
    if not (-90 <= lat <= 90):
        raise ValueError(f"{name} latitude out of range: {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"{name} longitude out of range: {lon}")


###############################################################################
# Data utilities
###############################################################################
def fill_missing_data(cross_section_filled):
    for i in range(cross_section_filled.shape[-1]):
        column_vals = cross_section_filled[:, i]
        valid_indices = np.transpose((column_vals > -200).nonzero())
        if valid_indices.size == 0:
            continue
        first_idx = int(valid_indices[0])
        cross_section_filled[0:first_idx, i] = cross_section_filled[first_idx, i]


###############################################################################
# Frame discovery: multi-file + multi-time
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
# Per-frame worker
###############################################################################
def process_frame(args):
    (
        ncfile_path,
        time_index,
        domain,
        cross1_lat,
        cross1_lon,
        cross2_lat,
        cross2_lon,
        path_figures,
    ) = args

    image_folder = os.path.join(path_figures, "Images")

    with Dataset(ncfile_path) as ncfile:
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # Variables
        wspd_all = wrf.getvar(ncfile, "uvmet_wspd_wdir", timeidx=time_index, units="kt")
        wspd = wspd_all[0, :]  # wind speed (kt)

        ht = wrf.getvar(ncfile, "z", timeidx=time_index, units="m")
        ter = wrf.getvar(ncfile, "ter", timeidx=time_index, units="m")
        p = wrf.getvar(ncfile, "pres", timeidx=time_index, units="hPa")
        u = wrf.getvar(ncfile, "ua", timeidx=time_index, units="kt")
        v = wrf.getvar(ncfile, "va", timeidx=time_index, units="kt")

        # Grid and spacing
        lats, lons = wrf.latlon_coords(ter)
        lons_np_raw = to_np(lons)
        lon_span_raw = float(np.nanmax(lons_np_raw) - np.nanmin(lons_np_raw))
        dateline_crossing = lon_span_raw > 180.0

        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            _label_adjustment,
        ) = compute_grid_and_spacing(lats, lons)

        lats_np, lons_np = handle_domain_continuity_and_polar_mask(lats_np, lons_np)

        min_lat = float(np.nanmin(lats_np))
        max_lat = float(np.nanmax(lats_np))
        min_lon = float(np.nanmin(lons_np))
        max_lon = float(np.nanmax(lons_np))
        mean_lat = float(np.nanmean(lats_np))

        # Cross section endpoints
        start_point = wrf.CoordPair(lat=cross1_lat, lon=cross1_lon)
        end_point = wrf.CoordPair(lat=cross2_lat, lon=cross2_lon)

        # Cross section interpolation
        u_cross = wrf.vertcross(
            u,
            ht,
            wrfin=ncfile,
            start_point=start_point,
            end_point=end_point,
            latlon=True,
            meta=True,
        )
        v_cross = wrf.vertcross(
            v,
            ht,
            wrfin=ncfile,
            start_point=start_point,
            end_point=end_point,
            latlon=True,
            meta=True,
        )
        wspd_cross = wrf.vertcross(
            wspd,
            ht,
            wrfin=ncfile,
            start_point=start_point,
            end_point=end_point,
            latlon=True,
            meta=True,
        )

        vertical_coord = to_np(wspd_cross.coords["vertical"])

        # Fill and smooth
        wspd_cross_filled = np.ma.copy(to_np(wspd_cross))
        fill_missing_data(wspd_cross_filled)
        wspd_cross_smoothed = gaussian_filter(wspd_cross_filled, sigma=1)

        # Plot
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = plt.axes()

        coord_pairs = to_np(wspd_cross.coords["xy_loc"])
        x_range = np.arange(coord_pairs.shape[0])

        wind_speed_ranges = np.array(
            [0, 10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200]
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

        isotach_map = plt.matplotlib.colors.ListedColormap(color_map_rgb)
        isotach_norm = plt.matplotlib.colors.BoundaryNorm(
            wind_speed_ranges, isotach_map.N
        )

        wspd_contours = ax.contourf(
            x_range,
            vertical_coord,
            wspd_cross_smoothed,
            wind_speed_ranges,
            cmap=isotach_map,
            norm=isotach_norm,
            extend="max",
            zorder=1,
        )

        cbar = plt.colorbar(wspd_contours, ax=ax, orientation="vertical", pad=0.05)
        cbar.set_label("Wind Speed (kt)", fontsize=12)
        cbar.set_ticks(wind_speed_ranges)
        cbar.set_ticklabels([str(int(vv)) for vv in wind_speed_ranges])

        # Terrain
        ter_line = wrf.interpline(
            ter, wrfin=ncfile, start_point=start_point, end_point=end_point
        )
        ax.plot(x_range, to_np(ter_line), color="darkgrey", linewidth=2, zorder=2)
        ax.fill_between(x_range, 0, to_np(ter_line), color="darkgrey")

        # Wind barbs
        horizontal_step = 5
        vertical_step = 5
        ax.barbs(
            x_range[::horizontal_step],
            vertical_coord[::vertical_step],
            to_np(u_cross[::vertical_step, ::horizontal_step]),
            to_np(v_cross[::vertical_step, ::horizontal_step]),
            length=6,
            barbcolor="black",
            flagcolor="black",
            linewidth=0.5,
            clip_on=False,
        )

        # X axis ticks/labels
        tick_interval = 2
        x_ticks = np.arange(0, coord_pairs.shape[0], tick_interval)
        x_labels = [pair.latlon_str(fmt="{:.2f}°, {:.2f}°") for pair in coord_pairs]
        x_labels_subset = [x_labels[i] for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels_subset, rotation=90, fontsize=8)

        # Y axis height
        max_height = float(np.max(to_np(wspd_cross["vertical"])))
        ax.set_ylim([0, max_height])
        y_ticks = np.arange(0, max_height + 1, 2000)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{int(y)}m" for y in y_ticks])
        ax.set_xlabel("Latitude/Longitude", fontsize=12)
        ax.set_ylabel("Height", fontsize=12)

        # Secondary Y axis pressure
        ax2 = ax.twinx()
        ax2.set_yscale("symlog")
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax2.set_yticks(np.linspace(100, 1000, 10))
        p_np = to_np(p)
        ax2.set_ylim(p_np.max(), p_np.min())
        ax2.set_ylabel("Pressure (hPa)", fontsize=12)

        # Titles (with start/end)
        plt.title(
            f"Weather Research and Forecasting Model\n"
            f"Average Grid Spacing: {avg_dx_km} x {avg_dy_km} km\n"
            f"Wind Speed (kt)\n"
            f"Start: ({cross1_lat:.4f}, {cross1_lon:.4f})\n"
            f"End:   ({cross2_lat:.4f}, {cross2_lon:.4f})",
            loc="left",
            fontsize=13,
        )
        plt.title(f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}", loc="right", fontsize=13)

        # Inset map: polar-aware + WRF projection
        cart_proj = wrf.get_cartopy(ter)
        polar_domain = (abs(min_lat) > 70.0) or (abs(max_lat) > 70.0)
        if polar_domain:
            inset_proj = (
                crs.NorthPolarStereo() if mean_lat >= 0.0 else crs.SouthPolarStereo()
            )
            draw_labels = False
        else:
            inset_proj = cart_proj
            draw_labels = True

        map_ax = plt.axes([0.83, 0.60, 0.15, 0.22], projection=inset_proj)
        map_ax.set_extent(
            [
                min_lon - extent_adjustment,
                max_lon + extent_adjustment,
                min_lat - extent_adjustment,
                max_lat + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        map_ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])
        for feature in features:
            add_feature(map_ax, *feature)

        add_latlon_gridlines(map_ax, draw_labels=draw_labels)
        plot_cities(map_ax, lons_np, lats_np, avg_dx_km, avg_dy_km, max_cities=10)

        # Dateline wrap endpoints for plotting if needed
        def _wrap_lon(lon):
            return lon + 360.0 if lon < 0.0 else lon

        plot_lon1 = _wrap_lon(cross1_lon) if dateline_crossing else cross1_lon
        plot_lon2 = _wrap_lon(cross2_lon) if dateline_crossing else cross2_lon

        map_ax.plot(
            [plot_lon1, plot_lon2],
            [cross1_lat, cross2_lat],
            marker="o",
            markersize=4,
            color="red",
            transform=crs.PlateCarree(),
        )

        map_ax.text(
            plot_lon1,
            cross1_lat,
            "Start point",
            transform=crs.PlateCarree(),
            fontsize=9,
            color="black",
            ha="right",
            va="bottom",
        )
        map_ax.text(
            plot_lon2,
            cross2_lat,
            "End point",
            transform=crs.PlateCarree(),
            fontsize=9,
            color="black",
            ha="left",
            va="bottom",
        )

        # Save
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_WSPD_Cross_Section_Height_{fname_time}.png"
        out_path = os.path.join(image_folder, file_out)

        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()

        return out_path


###############################################################################
# GIF creation
###############################################################################
def create_gif(path_figures, domain):
    image_folder = os.path.join(path_figures, "Images")
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF creation. Skipping GIF step.")
        return

    png_files_sorted = sorted(png_files)
    print("Creating GIF from sorted PNG files")

    animation_folder = os.path.join(path_figures, "Animation")
    os.makedirs(animation_folder, exist_ok=True)

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]
    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        return

    gif_file_out = f"wrf_{domain}_Cross_Section_WSPD_Height.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")


###############################################################################
# Main
###############################################################################
def main():
    if len(sys.argv) not in (3, 7):
        print(
            "\nUsage:\n"
            "  script.py <path_wrf> <domain>\n"
            "  script.py <path_wrf> <domain> <lat1> <lon1> <lat2> <lon2>\n"
        )
        sys.exit(2)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    if len(sys.argv) == 7:
        cross1_lat = float(sys.argv[3])
        cross1_lon = float(sys.argv[4])
        cross2_lat = float(sys.argv[5])
        cross2_lon = float(sys.argv[6])
    else:
        while True:
            try:
                cross1_lat = float(
                    input("Enter first latitude (e.g., 30.889): ").strip()
                )
                cross1_lon = float(
                    input("Enter first longitude (e.g., -98.996): ").strip()
                )
                break
            except ValueError:
                print("Invalid input. Use decimal format (e.g., 30.889803, -98.996).")

        while True:
            try:
                cross2_lat = float(
                    input("Enter second latitude (e.g., 30.889): ").strip()
                )
                cross2_lon = float(
                    input("Enter second longitude (e.g., -98.996): ").strip()
                )
                break
            except ValueError:
                print("Invalid input. Use decimal format (e.g., 30.889803, -98.996).")

    _validate_point(cross1_lat, cross1_lon, "Start point")
    _validate_point(cross2_lat, cross2_lon, "End point")

    print(f"Start Point: [{cross1_lat} : {cross1_lon}]")
    print(f"End Point:   [{cross2_lat} : {cross2_lon}]")

    path_figures = "WRF_Cross_Section_WSPD_Height_Pressure"
    os.makedirs(os.path.join(path_figures, "Images"), exist_ok=True)
    os.makedirs(os.path.join(path_figures, "Animation"), exist_ok=True)

    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print("No WRF output files found matching pattern.")
        sys.exit(1)

    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    args_list = [
        (
            ncfile_path,
            time_index,
            domain,
            cross1_lat,
            cross1_lon,
            cross2_lat,
            cross2_lon,
            path_figures,
        )
        for (ncfile_path, time_index) in frames
    ]

    max_workers = min(4, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("Cross-section wind speed plots complete.")
    create_gif(path_figures, domain)


if __name__ == "__main__":
    main()
