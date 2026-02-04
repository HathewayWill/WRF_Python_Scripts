#!/usr/bin/env python3
"""
Total accumulated precipitation (inches) from WRF surface fields.

Computes:
    * Total precip from RAINC + RAINNC + RAINSH (mm → inches)
    * Accumulated precip over the forecast period defined by:
        first wrfout_<domain>* file and last wrfout_<domain>* file.

Plots:
    * Total accumulated precipitation (inches) on a Cartopy map.
    * Map setup is moving-nest safe (geometry re-computed for the last file).

Physics / diagnostics are unchanged from the original script.
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
from scipy.ndimage import gaussian_filter
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
# Total accumulated precip plotting for one (start_frame, end_frame) product
###############################################################################
def process_frame(args):
    """
    Process a single accumulated-precip product.

    Frame semantics for this product:
        - One output PNG generated from one (start_path, start_time_index) and
          one (end_path, end_time_index) pair.
        - Map geometry is derived from the end frame (moving-nest safe).

    Physics identical:
        - total_rain = (RAINC + RAINNC + RAINSH) * 0.0393701 (mm → inches)
        - accumulated = end_total_rain - start_total_rain
    """
    (
        start_ncfile_path,
        start_time_index,
        end_ncfile_path,
        end_time_index,
        domain,
        path_figures,
    ) = args

    # -------------------------------------------------------------------------
    # Compute total rain at start and end frames (physics identical)
    # -------------------------------------------------------------------------
    with Dataset(start_ncfile_path) as nc_start:
        rainc_start = wrf.getvar(nc_start, "RAINC", timeidx=start_time_index)
        rainnc_start = wrf.getvar(nc_start, "RAINNC", timeidx=start_time_index)
        rainsh_start = wrf.getvar(nc_start, "RAINSH", timeidx=start_time_index)
        start_total_rain = to_np(
            (rainc_start + rainnc_start + rainsh_start) * 0.0393701
        )

        start_valid_dt = get_valid_time(nc_start, start_ncfile_path, start_time_index)

    with Dataset(end_ncfile_path) as nc_end:
        rainc_end = wrf.getvar(nc_end, "RAINC", timeidx=end_time_index)
        rainnc_end = wrf.getvar(nc_end, "RAINNC", timeidx=end_time_index)
        rainsh_end = wrf.getvar(nc_end, "RAINSH", timeidx=end_time_index)
        end_total_rain = to_np((rainc_end + rainnc_end + rainsh_end) * 0.0393701)

        end_valid_dt = get_valid_time(nc_end, end_ncfile_path, end_time_index)

        # ---------------------------------------------------------------------
        # Plotting fields derived from the end frame (moving-nest safe)
        # ---------------------------------------------------------------------
        temp = wrf.getvar(nc_end, "T2", timeidx=end_time_index)

        # Option B merge: single, consistent time index (no ALL_TIMES)
        temp3d = wrf.getvar(nc_end, "temp", units="degC", timeidx=end_time_index)

        temp_850 = wrf.vinterp(
            nc_end,
            temp3d,
            "pressure",
            [850],
            field_type="tc",
            extrapolate=True,
            squeeze=True,
            meta=True,
            timeidx=end_time_index,
        )
        temp_850 = np.squeeze(temp_850, axis=0)

        # Lat/lon and grid spacing (recomputed per frame)
        lats, lons = wrf.latlon_coords(temp)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,  # currently unused but kept for consistency
        ) = compute_grid_and_spacing(lats, lons)

        # Accumulated precip over forecast period (physics identical)
        total_forecast_rain = end_total_rain - start_total_rain

        # Smooth temp fields (physics kept even though not plotted)
        smooth_temp = gaussian_filter(to_np(temp), sigma=1.0)
        smooth_temp_850 = gaussian_filter(to_np(temp_850), sigma=1.0)

        # ---------------------------------------------------------------------
        # Dateline continuity and polar masking (v9 canonical helper)
        # ---------------------------------------------------------------------
        (
            lats_np,
            lons_np,
            total_forecast_rain,
            smooth_temp,
            smooth_temp_850,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            total_forecast_rain,
            smooth_temp,
            smooth_temp_850,
        )

        cart_proj = wrf.get_cartopy(temp)

        # Figure and axis
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Extent with padding (per-frame)
        ax.set_extent(
            [
                lons_np.min() - extent_adjustment,
                lons_np.max() + extent_adjustment,
                lats_np.min() - extent_adjustment,
                lats_np.max() + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        # Land and other features
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])
        for feature in features:
            add_feature(ax, *feature)

        # Cities & gridlines (canonical helpers)
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)
        add_latlon_gridlines(ax)

        ax.tick_params(labelsize=12, width=2)

        # ---------------------------------------------------------------------
        # Contours and color map for total precip (unchanged)
        # ---------------------------------------------------------------------
        # 24-hour precip (in) contours
        precip_levels = np.array(
            [
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                1.00,
                1.50,
                2.00,
                3.00,
                4.00,
                6.00,
                8.00,
                12.00,
                15.00,
            ]
        )

        color_map_rgb = (
            np.array(
                [
                    [199, 233, 192],
                    [161, 217, 155],
                    [116, 196, 118],
                    [49, 163, 83],
                    [0, 109, 44],
                    [255, 250, 138],
                    [255, 204, 79],
                    [254, 141, 60],
                    [252, 78, 42],
                    [214, 26, 28],
                    [173, 0, 38],
                    [112, 0, 38],
                    [59, 0, 48],
                    [76, 0, 115],
                    [255, 219, 255],
                ],
                np.float32,
            )
            / 255.0
        )
        rain_map = plt.matplotlib.colors.ListedColormap(color_map_rgb[:-1])
        rain_map.set_over(color_map_rgb[-1])
        rain_norm = plt.matplotlib.colors.BoundaryNorm(
            precip_levels, rain_map.N, clip=False
        )

        precip_contour = ax.contourf(
            lons_np,
            lats_np,
            total_forecast_rain,
            levels=precip_levels,
            cmap=rain_map,
            norm=rain_norm,
            extend="max",
            transform=crs.PlateCarree(),
        )

        # Colorbar
        cbar = fig.colorbar(
            precip_contour,
            ax=ax,
            orientation="vertical",
            shrink=0.8,
            pad=0.05,
            ticks=precip_levels,
        )
        cbar.set_label("Total Precipitation (in)", fontsize=14)
        cbar.ax.set_yticklabels([f"{level:.2f}" for level in precip_levels])

        # ---------------------------------------------------------------------
        # Titles and output
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

        plt.title(
            f"Weather Research and Forecasting Model\n"
            f"Average Grid Spacing:{avg_dx_km}x{avg_dy_km}km\n"
            f"Total Precipitation (Inch)",
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

        # v9 naming: wrf_<domain>_<product>_YYYYMMDDHHMMSS.png (timestamp from valid_dt)
        fname_time = end_valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_total_precip_{fname_time}.png"

        plt.savefig(
            os.path.join(path_figures, "Images", file_out),
            bbox_inches="tight",
            dpi=250,
        )
        plt.close(fig)

    print("SFC PRECIP Plots Complete")


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
    # Argument parsing
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example: script_name.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # Output directories
    path_figures = "wrf_SFC_Total_Precip_Inch_figures"
    image_folder = os.path.join(path_figures, "Images")

    for folder in (path_figures, image_folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    # Find all WRF output files for this domain
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print(f"No wrfout files found in {path_wrf} matching wrfout_{domain}*")
        sys.exit(0)

    # Build frames across mixed WRF output conventions (v9)
    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    # Start/end frames define the accumulation period (multi-file, multi-time safe)
    start_ncfile_path, start_time_index = frames[0]
    end_ncfile_path, end_time_index = frames[-1]

    args_list = [
        (
            start_ncfile_path,
            start_time_index,
            end_ncfile_path,
            end_time_index,
            domain,
            path_figures,
        )
    ]

    # Required multiprocessing pattern (even for a single product)
    max_workers = min(1, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass
