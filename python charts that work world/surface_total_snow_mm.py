#!/usr/bin/env python3
"""
SFC total accumulated snow (inches) from WRF snow height.

Computes:
    * Total snow depth from SNOWH (m) converted to inches.
    * Accumulated snow over the forecast period defined by:
        earliest model time and latest model time across all wrfout_<domain>* files.

Plots:
    * Total accumulated snow (inches) at the LAST valid time only.

Physics / diagnostics:
    - snow = wrf.getvar(ncfile, "SNOWH", timeidx=time_index)
    - total_snow_inches = snow * 39.3700787402  (m → inches)
    - total_forecast_snow = total_snow_last - total_snow_first
"""

###############################################################################
# Imports (clean, ordered)
###############################################################################
import glob
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor  # kept for possible extensions
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
from scipy.ndimage import gaussian_filter  # not used for snow, kept for parity
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
# Process one accumulated-snow product (start frame vs end frame)
###############################################################################
def process_frame(args):
    """
    Produce one PNG:
        - Accumulation is computed from the earliest valid time frame and latest
          valid time frame across all discovered frames.
        - Plot is rendered at the END frame only.

    Physics / diagnostics unchanged:
        snow = wrf.getvar(ncfile, "SNOWH", timeidx=time_index)
        total_snow_inches = snow * 39.3700787402  (m → inches)
        total_forecast_snow = total_snow_last - total_snow_first
    """
    (
        first_ncfile_path,
        first_time_index,
        last_ncfile_path,
        last_time_index,
        domain,
        path_figures,
    ) = args

    # -------------------------------------------------------------------------
    # Snow physics: first vs last timeframe only
    # -------------------------------------------------------------------------
    with Dataset(first_ncfile_path) as nc_first:
        start_valid_dt = get_valid_time(nc_first, first_ncfile_path, first_time_index)
        snow_first = wrf.getvar(nc_first, "SNOWH", timeidx=first_time_index)  # m
        total_snow_first = to_np(snow_first * 39.3700787402)  # inches

    with Dataset(last_ncfile_path) as nc_last:
        end_valid_dt = get_valid_time(nc_last, last_ncfile_path, last_time_index)
        snow_last = wrf.getvar(nc_last, "SNOWH", timeidx=last_time_index)  # m
        total_snow_last = to_np(snow_last * 39.3700787402)  # inches

        # Accumulated snow over forecast period
        total_forecast_snow = total_snow_last - total_snow_first

        print(f"Accumulation start time: {start_valid_dt:%Y/%m/%d %H:%M:%S} UTC")
        print(f"Accumulation end time:   {end_valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # ---------------------------------------------------------------------
        # Geometry and projection from last timeframe (moving-nest safe)
        #
        # IMPORTANT:
        #   Keep 'snow_last' as a WRF var (metadata-bearing) so get_cartopy works.
        # ---------------------------------------------------------------------
        lats, lons = wrf.latlon_coords(snow_last)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,  # unused but kept for consistency
        ) = compute_grid_and_spacing(lats, lons)

        # Dateline continuity and polar masking (v9 canonical helper)
        lats_np, lons_np, total_forecast_snow = handle_domain_continuity_and_polar_mask(
            lats_np, lons_np, total_forecast_snow
        )

        cart_proj = wrf.get_cartopy(snow_last)

        # ---------------------------------------------------------------------
        # Figure and map setup
        # ---------------------------------------------------------------------
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Base land + features
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])
        for feature in features:
            add_feature(ax, *feature)

        # Map extent with padding
        ax.set_extent(
            [
                lons_np.min() - extent_adjustment,
                lons_np.max() + extent_adjustment,
                lats_np.min() - extent_adjustment,
                lats_np.max() + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        # Cities & gridlines
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)
        add_latlon_gridlines(ax)
        ax.tick_params(labelsize=12, width=2)

        # ---------------------------------------------------------------------
        # Contours and color map for total snow (unchanged)
        # ---------------------------------------------------------------------
        Snow_levels = np.array(
            [0.50, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200]
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
            total_forecast_snow,
            levels=Snow_levels,
            cmap=snow_map,
            norm=snow_norm,
            extend="max",
            transform=crs.PlateCarree(),
        )

        # Colorbar for snow
        cbar = fig.colorbar(
            Snow_contour, ax=ax, orientation="vertical", shrink=0.8, pad=0.05
        )
        cbar.set_label("Total Snow (Inch)", fontsize=14)
        cbar.set_ticks(Snow_levels)
        cbar.set_ticklabels([f"{level:.2f}" for level in Snow_levels])

        # ---------------------------------------------------------------------
        # Time labeling and output
        # ---------------------------------------------------------------------
        start_year = start_valid_dt.strftime("%Y")
        start_month = start_valid_dt.strftime("%m")
        start_day = start_valid_dt.strftime("%d")
        start_hour = start_valid_dt.strftime("%H")
        start_minute = start_valid_dt.strftime("%M")

        end_year = end_valid_dt.strftime("%Y")
        end_month = end_valid_dt.strftime("%m")
        end_day = end_valid_dt.strftime("%d")
        end_hour = end_valid_dt.strftime("%H")
        end_minute = end_valid_dt.strftime("%M")

        # Titles (same spirit as original)
        plt.title(
            "Weather Research and Forecasting Model\n"
            f"Average Grid Spacing:{avg_dx_km}x{avg_dy_km}km\n"
            "Total Snow (Inch)\n"
            "Model Snow Height Difference",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid:\n"
            f"{start_hour}:{start_minute}Z {start_year}-{start_month}-{start_day}\n"
            f"{end_hour}:{end_minute}Z {end_year}-{end_month}-{end_day}",
            loc="right",
            fontsize=10,
        )

        # v9 naming: wrf_<domain>_<product>_YYYYMMDDHHMMSS.png
        fname_time = end_valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_total_snow_inch_{fname_time}.png"

        plt.savefig(
            os.path.join(path_figures, "Images", file_out),
            bbox_inches="tight",
            dpi=250,
        )
        plt.close(fig)

    print("SFC Total Snow Inch plot complete")


###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example: surface_total_snow_inch_slp_isotherm.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # Output directories
    path_figures = "wrf_SFC_Total_Snow_Inch_figures"
    image_folder = os.path.join(path_figures, "Images")

    for folder in (path_figures, image_folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    ###########################################################################
    # Find WRF files and discover all frames
    ###########################################################################
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print(f"No wrfout files found in {path_wrf} matching wrfout_{domain}*")
        sys.exit(0)

    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    ###########################################################################
    # Determine earliest and latest valid times (first and last timeframe)
    ###########################################################################
    frame_times = []
    for ncfile_path, time_index in frames:
        with Dataset(ncfile_path) as nc:
            valid_dt = get_valid_time(nc, ncfile_path, time_index)
        frame_times.append((ncfile_path, time_index, valid_dt))

    # First and last timeframe by actual valid time
    first_ncfile_path, first_time_index, start_valid_dt = min(
        frame_times, key=lambda x: x[2]
    )
    last_ncfile_path, last_time_index, end_valid_dt = max(
        frame_times, key=lambda x: x[2]
    )

    args_list = [
        (
            first_ncfile_path,
            first_time_index,
            last_ncfile_path,
            last_time_index,
            domain,
            path_figures,
        )
    ]

    # Required multiprocessing pattern (even for a single product)
    max_workers = min(1, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("SFC Total Snow Inch plot generation complete.")
