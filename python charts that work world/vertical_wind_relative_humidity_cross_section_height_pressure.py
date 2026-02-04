#!/usr/bin/env python3
"""
WRF_Cross_Section_RH_Height_Pressure.py

Plot a vertical cross section of Relative Humidity (%) and wind barbs
between two lat/lon points, for all times in one or more wrfout_<domain> files.

v9 playbook aligned:
    * Supports multiple wrfout files AND multiple timesteps per file.
    * One frame = one (file, time_index) pair.
    * Uses netCDF4 + wrf-python only (no xarray).
    * Time handled via wrf.extract_times with filename fallback.
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
from wrf import ALL_TIMES, to_np

###############################################################################
# Warning suppression
###############################################################################
warnings.filterwarnings("ignore")

###############################################################################
# Canonical helper function block (v9 – contiguous)
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


def plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km, max_cities=5):
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
            fontsize=7,
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


def _validate_point(lat, lon, name="Point"):
    """Fail fast on bad CLI/typed input before it hits Fortran/wrf.vertcross."""
    if not np.isfinite(lat) or not np.isfinite(lon):
        raise ValueError(f"{name} has non-finite lat/lon: {lat}, {lon}")
    if not (-90 <= lat <= 90):
        raise ValueError(f"{name} latitude out of range: {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"{name} longitude out of range: {lon}")


def fill_missing_data(cross_section_filled):
    """
    For each vertical column, fill values above the first valid level
    with the first valid value, to avoid gaps near the top.
    """
    for i in range(cross_section_filled.shape[-1]):
        column_vals = cross_section_filled[:, i]
        valid_indices = np.transpose((column_vals > -200).nonzero())
        if valid_indices.size == 0:
            continue
        first_idx = int(valid_indices[0])
        cross_section_filled[0:first_idx, i] = cross_section_filled[first_idx, i]


def create_gif(path_figures, domain):
    image_folder = os.path.join(path_figures, "Images")

    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF creation. Skipping GIF step.")
        return

    # Filenames contain YYYYMMDDHHMMSS, so simple sort is chronological
    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files")

    animation_folder = os.path.join(path_figures, "Animation")
    if not os.path.isdir(animation_folder):
        os.mkdir(animation_folder)

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]

    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        return

    gif_file_out = f"wrf_{domain}_Cross_Section_RH_Height.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")


def main():
    # ----------------------------------------------------------------------
    # ARGUMENTS
    # ----------------------------------------------------------------------
    if len(sys.argv) not in (3, 7):
        print(
            "\nUsage:\n"
            "  script_name.py <path_wrf> <domain>\n"
            "  script_name.py <path_wrf> <domain> <lat1> <lon1> <lat2> <lon2>\n"
            "\nExample:\n"
            "  script_name.py /home/WRF/test/em_real d01 47.6061 -122.3328 47.6101 -122.2015\n"
        )
        sys.exit(2)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # ----------------------------------------------------------------------
    # GET CROSS-SECTION ENDPOINTS
    # ----------------------------------------------------------------------
    if len(sys.argv) == 7:
        cross1_lat = float(sys.argv[3])
        cross1_lon = float(sys.argv[4])
        cross2_lat = float(sys.argv[5])
        cross2_lon = float(sys.argv[6])
    else:
        while True:
            try:
                cross1_lat = float(
                    input(
                        "Enter first latitude in decimal format for start point "
                        "(e.g., 30.889): "
                    ).strip()
                )
                cross1_lon = float(
                    input(
                        "Enter first longitude in decimal format for start point "
                        "(e.g., -98.996): "
                    ).strip()
                )
                break
            except ValueError:
                print(
                    "Invalid input. Please enter latitude/longitude in decimal "
                    "format (e.g., 30.889803, -98.996)."
                )

        while True:
            try:
                cross2_lat = float(
                    input(
                        "Enter second latitude in decimal format for end point "
                        "(e.g., 30.889): "
                    ).strip()
                )
                cross2_lon = float(
                    input(
                        "Enter second longitude in decimal format for end point "
                        "(e.g., -98.996): "
                    ).strip()
                )
                break
            except ValueError:
                print(
                    "Invalid input. Please enter latitude/longitude in decimal "
                    "format (e.g., 30.889803, -98.996)."
                )

    _validate_point(cross1_lat, cross1_lon, "Start point")
    _validate_point(cross2_lat, cross2_lon, "End point")

    print(f"Start Point: [{cross1_lat} : {cross1_lon}]")
    print(f"End Point:   [{cross2_lat} : {cross2_lon}]")

    # ----------------------------------------------------------------------
    # OUTPUT DIRECTORIES
    # ----------------------------------------------------------------------
    path_figures = "WRF_Cross_Section_RH_Height_Pressure"

    if not os.path.isdir(path_figures):
        os.mkdir(path_figures)

    image_folder = os.path.join(path_figures, "Images")
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)

    # ----------------------------------------------------------------------
    # FIND ALL WRF OUTPUT FILES
    # ----------------------------------------------------------------------
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print(f"No WRF output files found in {path_wrf} matching wrfout_{domain}*")
        sys.exit(1)

    # ----------------------------------------------------------------------
    # DISCOVER ALL (file, time_index) FRAMES (v9 canonical)
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # PROCESS ALL FRAMES IN PARALLEL (v9 required pattern)
    # ----------------------------------------------------------------------
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("Cross Section RH plot generation complete.")

    # ----------------------------------------------------------------------
    # GIF CREATION
    # ----------------------------------------------------------------------
    create_gif(path_figures, domain)


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
# Per-frame worker (runs in each process)
###############################################################################
def process_frame(args):
    """
    Worker function that computes the vertical WND/RH cross section for a single
    (file, time_index) and saves a PNG.

    One frame = one (ncfile_path, time_index) pair.
    """
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

    # Output folder for PNGs
    image_folder = os.path.join(path_figures, "Images")

    # Open the WRF file (per-worker, per-frame; no shared Dataset objects)
    with Dataset(ncfile_path) as ncfile:
        # Get valid time from metadata (preferred) or filename
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # ------------------------------------------------------------------
        # LOAD NEEDED VARIABLES (physics unchanged)
        # ------------------------------------------------------------------
        ht = wrf.getvar(ncfile, "z", timeidx=time_index, units="m")  # height
        ter = wrf.getvar(ncfile, "ter", timeidx=time_index, units="m")  # terrain
        p = wrf.getvar(ncfile, "pres", timeidx=time_index, units="hPa")  # pressure
        u = wrf.getvar(ncfile, "ua", timeidx=time_index, units="kt")  # U wind
        v = wrf.getvar(ncfile, "va", timeidx=time_index, units="kt")  # V wind
        rh = wrf.getvar(ncfile, "rh", timeidx=time_index)  # Relative humidity (%)

        # ------------------------------------------------------------------
        # GRID SPACING & DOMAIN EXTENT (moving-nest safe)
        # ------------------------------------------------------------------
        # Use ter (2D) for lat/lon, which is the same grid as u/v/rh.
        lats, lons = wrf.latlon_coords(ter)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            _label_adjustment,
        ) = compute_grid_and_spacing(lats, lons)

        # ------------------------------------------------------------------
        # Dateline continuity and polar masking (v9 canonical helper)
        # ------------------------------------------------------------------
        lats_np, lons_np = handle_domain_continuity_and_polar_mask(lats_np, lons_np)

        min_lat = float(np.nanmin(lats_np))
        max_lat = float(np.nanmax(lats_np))
        min_lon = float(np.nanmin(lons_np))
        max_lon = float(np.nanmax(lons_np))
        mean_lat = float(np.nanmean(lats_np))

        # ------------------------------------------------------------------
        # DEFINE START AND END POINTS FOR CROSS SECTION
        # ------------------------------------------------------------------
        start_point = wrf.CoordPair(lat=cross1_lat, lon=cross1_lon)
        end_point = wrf.CoordPair(lat=cross2_lat, lon=cross2_lon)

        # ------------------------------------------------------------------
        # INTERPOLATE VARIABLES ALONG CROSS SECTION (physics unchanged)
        # ------------------------------------------------------------------
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
        rh_cross = wrf.vertcross(
            rh,
            ht,
            wrfin=ncfile,
            start_point=start_point,
            end_point=end_point,
            latlon=True,
            meta=True,
        )

        # Vertical coordinate (height in meters)
        vertical_coord_rh = to_np(rh_cross.coords["vertical"])

        # ------------------------------------------------------------------
        # FILL MISSING DATA & SMOOTHING (unchanged)
        # ------------------------------------------------------------------
        rh_cross_filled = np.ma.copy(to_np(rh_cross))
        fill_missing_data(rh_cross_filled)

        sigma = 1  # Gaussian smoothing sigma
        rh_cross_smoothed = gaussian_filter(rh_cross_filled, sigma=sigma)

        # ------------------------------------------------------------------
        # PLOTTING: RH CROSS SECTION
        # ------------------------------------------------------------------
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = plt.axes()

        coord_pairs = to_np(rh_cross.coords["xy_loc"])
        x_range = np.arange(coord_pairs.shape[0])

        # RH filled contours (same levels and color table)
        rh_levels = np.arange(0, 100, 10)
        rh_color_table = (
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
                ],
                np.float32,
            )
            / 255.0
        )
        rh_map = plt.matplotlib.colors.ListedColormap(rh_color_table)
        rh_norm = plt.matplotlib.colors.BoundaryNorm(rh_levels, rh_map.N)

        rh_contours = ax.contourf(
            x_range,
            vertical_coord_rh,
            rh_cross_smoothed,
            rh_levels,
            cmap=rh_map,
            norm=rh_norm,
            extend="both",
            zorder=1,
        )

        # Colorbar
        cbar = plt.colorbar(rh_contours, ax=ax, orientation="vertical", pad=0.05)
        cbar.set_label("Relative Humidity (%)", fontsize=12)

        # ------------------------------------------------------------------
        # TERRAIN LINE
        # ------------------------------------------------------------------
        ter_line = wrf.interpline(
            ter, wrfin=ncfile, start_point=start_point, end_point=end_point
        )

        ax.plot(x_range, to_np(ter_line), color="darkgrey", linewidth=2, zorder=2)
        ax.fill_between(x_range, 0, to_np(ter_line), color="darkgrey")

        # ------------------------------------------------------------------
        # WIND BARBS (U, V) WITH SUBSAMPLING (unchanged)
        # ------------------------------------------------------------------
        horizontal_step = 5
        vertical_step = 5

        ax.barbs(
            x_range[::horizontal_step],
            vertical_coord_rh[::vertical_step],
            to_np(u_cross[::vertical_step, ::horizontal_step]),
            to_np(v_cross[::vertical_step, ::horizontal_step]),
            length=6,
            barbcolor="black",
            flagcolor="black",
            linewidth=0.5,
            clip_on=False,
        )

        # ------------------------------------------------------------------
        # X-AXIS TICKS/LABELS WITH LAT/LON
        # ------------------------------------------------------------------
        tick_interval = 2
        x_ticks = np.arange(0, coord_pairs.shape[0], tick_interval)
        x_labels = []

        for pair in coord_pairs:
            label = pair.latlon_str(fmt="{:.2f}°, {:.2f}°")
            x_labels.append(label)

        x_labels_subset = [x_labels[i] for i in x_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels_subset, rotation=90, fontsize=8)

        # ------------------------------------------------------------------
        # Y-AXIS HEIGHT (m)
        # ------------------------------------------------------------------
        max_height = np.max(to_np(rh_cross["vertical"]))
        ax.set_ylim([0, max_height])

        y_ticks = np.arange(0, max_height + 1, 2000)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{int(y)}m" for y in y_ticks])

        ax.set_xlabel("Latitude / Longitude", fontsize=12)
        ax.set_ylabel("Height", fontsize=12)

        # ------------------------------------------------------------------
        # SECONDARY Y-AXIS: PRESSURE (hPa)
        # ------------------------------------------------------------------
        ax2 = ax.twinx()
        ax2.set_yscale("symlog")
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax2.set_yticks(np.linspace(100, 1000, 10))

        p_np = to_np(p)
        ax2.set_ylim(p_np.max(), p_np.min())
        ax2.set_ylabel("Pressure (hPa)", fontsize=12)

        # ------------------------------------------------------------------
        # TITLES
        # ------------------------------------------------------------------
        plt.title(
            f"Weather Research and Forecasting Model\n"
            f"Average Grid Spacing: {avg_dx_km}x{avg_dy_km} km\n"
            f"Relative Humidity (%)\n"
            f"Start: ({cross1_lat:.4f}, {cross1_lon:.4f})\n"
            f"End:   ({cross2_lat:.4f}, {cross2_lon:.4f})",
            loc="left",
            fontsize=13,
        )
        plt.title(f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}", loc="right", fontsize=13)

        # ------------------------------------------------------------------
        # MAP INSET (TOP RIGHT)
        # ------------------------------------------------------------------
        cart_proj = wrf.get_cartopy(ter)

        polar_domain = (abs(min_lat) > 70.0) or (abs(max_lat) > 70.0)
        if polar_domain:
            inset_proj = (
                crs.NorthPolarStereo() if mean_lat >= 0.0 else crs.SouthPolarStereo()
            )
        else:
            inset_proj = cart_proj

        # Shift right + slightly smaller
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

        # Gridlines (canonical helper; labels may be suppressed on non-PlateCarree axes)
        gl_inset = add_latlon_gridlines(map_ax)

        # Cities on the mini map (canonical function)
        plot_cities(map_ax, lons_np, lats_np, avg_dx_km, avg_dy_km)

        # Cross-section line
        map_ax.plot(
            [cross1_lon, cross2_lon],
            [cross1_lat, cross2_lat],
            marker="o",
            markersize=4,
            color="red",
            transform=crs.PlateCarree(),
        )

        map_ax.text(
            cross1_lon,
            cross1_lat,
            "Start point",
            transform=crs.PlateCarree(),
            fontsize=11,
            color="black",
            ha="right",
            va="bottom",
        )
        map_ax.text(
            cross2_lon,
            cross2_lat,
            "End point",
            transform=crs.PlateCarree(),
            fontsize=11,
            color="black",
            ha="left",
            va="bottom",
        )

        # ------------------------------------------------------------------
        # SAVE FIGURE
        # ------------------------------------------------------------------
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_RH_Cross_Section_Height_{fname_time}.png"
        out_path = os.path.join(image_folder, file_out)

        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()

        return out_path


###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    main()
