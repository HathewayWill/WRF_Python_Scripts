#!/usr/bin/env python3
"""
850hPa_TempAdvection_Height_WND_SPD_DIR.py

Plot WRF 850-hPa temperature advection (°C / 3 h),
850-hPa geopotential height (dm),
and 850-hPa wind barbs (knots).

Example:
    python 850hpa_temp_advection_height_wind_speed_dir.py \
        /home/workhorse/WRF_Intel/WRF-4.7.1/run/ d01

This script follows the 250 hPa "golden" playbook:
    * Supports multiple wrfout_<domain>* files, each with one or more timesteps.
    * Supports single wrfout files with many timesteps.
    * One frame = one (file, time_index) pair.
    * Geometry (lat/lon, dx/dy, extent, cities) recomputed each frame,
      safe for static and moving nests.
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
# 850-hPa temperature advection plotting for one (file, time_index) frame
###############################################################################
def process_frame(args):
    ncfile_path, time_index, domain, path_figures = args

    with Dataset(ncfile_path) as ncfile:
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        u = wrf.getvar(ncfile, "ua", timeidx=time_index)
        v = wrf.getvar(ncfile, "va", timeidx=time_index)
        t = wrf.getvar(ncfile, "temp", timeidx=time_index)
        z = wrf.getvar(ncfile, "z", timeidx=time_index, units="dm")

        t850 = wrf.vinterp(ncfile, t, "pressure", [850], extrapolate=True, squeeze=True)
        u850 = wrf.vinterp(ncfile, u, "pressure", [850], extrapolate=True, squeeze=True)
        v850 = wrf.vinterp(ncfile, v, "pressure", [850], extrapolate=True, squeeze=True)
        z850 = wrf.vinterp(
            ncfile, z, "pressure", [850], field_type="z", extrapolate=True, squeeze=True
        )

        tmpk = gaussian_filter(np.squeeze(to_np(t850)), sigma=1.5)
        uwnd = gaussian_filter(np.squeeze(to_np(u850)), sigma=2.0)
        vwnd = gaussian_filter(np.squeeze(to_np(v850)), sigma=2.0)
        hgt = gaussian_filter(np.squeeze(to_np(z850)), sigma=2.0)

        lats, lons = wrf.latlon_coords(t850)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,
        ) = compute_grid_and_spacing(lats, lons)

        (
            lats_np,
            lons_np,
            tmpk,
            uwnd,
            vwnd,
            hgt,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            tmpk,
            uwnd,
            vwnd,
            hgt,
        )

        dx_m = np.mean(mpcalc.lat_lon_grid_deltas(lons_np, lats_np)[0]).to("meter")
        dy_m = np.mean(mpcalc.lat_lon_grid_deltas(lons_np, lats_np)[1]).to("meter")

        T = tmpk * units.kelvin
        ny, nx = T.shape
        if ny < 2 or nx < 2:
            return

        dTdx = np.gradient(T, axis=1) / dx_m
        dTdy = np.gradient(T, axis=0) / dy_m
        tadv = -(uwnd * units("m/s")) * dTdx - (vwnd * units("m/s")) * dTdy
        tadv = gaussian_filter(
            (tadv * 3 * units.hour).to("delta_degC").magnitude, sigma=2.5
        )

        cart_proj = wrf.get_cartopy(t850)
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

        cf = ax.contourf(
            lons_np,
            lats_np,
            tadv,
            np.arange(-12, 13, 1),
            cmap="bwr",
            extend="both",
            transform=crs.PlateCarree(),
        )
        plt.colorbar(cf, ax=ax, pad=0.01, label="°C / 3 h")

        cs = ax.contour(
            lons_np,
            lats_np,
            hgt,
            np.arange(60, 200, 10),
            colors="black",
            transform=crs.PlateCarree(),
        )
        plt.clabel(cs, fmt="%d")

        ax.barbs(
            lons_np[::6, ::6],
            lats_np[::6, ::6],
            uwnd[::6, ::6] * 1.94384449,
            vwnd[::6, ::6] * 1.94384449,
            transform=crs.PlateCarree(),
        )

        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)

        add_latlon_gridlines(ax)

        plt.title(
            f"Weather Research and Forecasting Model\n"
            f"Average Grid Spacing: {avg_dx_km} x {avg_dy_km} km\n"
            f"Temperature Advection at 850 hPa (°C / 3 h)\n"
            f"Wind Barbs at 850 hPa (knots)\n"
            f"850 hPa Geopotential Heights (dm)",
            loc="left",
            fontsize=13,
        )

        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        fname = valid_dt.strftime("%Y%m%d%H%M%S")
        plt.savefig(
            os.path.join(
                path_figures, "Images", f"wrf_{domain}_850hPa_TADV_{fname}.png"
            ),
            bbox_inches="tight",
            dpi=150,
        )

        plt.close(fig)


###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "Usage: python 850hpa_temp_advection_height_wind_speed_dir.py <path_wrf> <domain>"
        )

    path_wrf, domain = sys.argv[1], sys.argv[2]

    path_figures = "wrf_850hPa_TempAdvection"
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
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    args_list = [(p, t, domain, path_figures) for (p, t) in frames]

    max_workers = min(4, len(args_list)) if args_list else 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    if not png_files:
        sys.exit("No PNGs generated; skipping GIF.")

    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files")

    images = [Image.open(os.path.join(image_folder, f)) for f in png_files_sorted]
    if not images:
        sys.exit("No images loaded for GIF creation. Skipping GIF step.")

    images[0].save(
        os.path.join(animation_folder, f"wrf_{domain}_850hPa_TADV.gif"),
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print("850-hPa temperature advection processing complete.")
