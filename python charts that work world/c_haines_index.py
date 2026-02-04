#!/usr/bin/env python3
"""
CHaines_Index.py

Compute and plot the continuous Haines Index (C-HAINES) from WRF output.

Physics/diagnostics are preserved exactly from the original:
    - temp/td fields in degC
    - wrf.vinterp to 850 and 700 hPa (temp) and 850 hPa (dewpoint)
    - Mills & McCaw (2010) C-HAINES formulation
    - Gaussian smoothing (sigma=1)
    - Color levels, colormap bins, and colorbar labeling

Structure follows WRF Plotting Playbook – v9:
    * Multiple wrfout_<domain>* files, each with one or more timesteps.
    * A single wrfout file containing many timesteps.
    * Safe for static and moving/vortex-following nests:
        - lat/lon, grid spacing, and extent recomputed per frame.
    * One frame = one (file, time_index) pair.
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
import matplotlib.colors as mcolors
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
# C-HAINES physics (unchanged)
###############################################################################
def c_haines_index(T850, T700, Td850):
    """
    Compute the mid-level continuous Haines Index (C-HAINES) as described by
    Mills and McCaw (2010), eqns (1)–(5).

    Inputs:
      T850   : Temperature at 850 hPa (°C)
      T700   : Temperature at 700 hPa (°C)
      Td850  : Dewpoint temp at 850 hPa (°C)

    Steps:
      1) Compute dewpoint depression: DPD = T850 - Td850
      2) Cap DPD at 30°C if it exceeds that
      3) CA = 0.5 * (T850 - T700) - 2
      4) CB = (1/3)*(DPD) - 1
         If CB > 5, reduce slope: CB = 5 + (CB - 5)/2
      5) CH = CA + CB, with a lower limit of 0

    Returns:
      A float >= 0 representing the C-HAINES value.
      Typical range 0–~13 for hot, dry conditions.
    """
    # Calculate dewpoint depression
    DPD = T850 - Td850
    # Cap at 30°C
    if DPD > 30.0:
        DPD = 30.0

    # Stability component
    CA = 0.5 * (T850 - T700) - 2.0

    # Humidity component
    CB = (1.0 / 3.0) * DPD - 1.0
    if CB > 5.0:
        CB = 5.0 + (CB - 5.0) / 2.0

    # Sum up
    CH = CA + CB
    if CH < 0:
        CH = 0.0

    return CH


###############################################################################
# C-HAINES plotting for one (file, time_index) frame
###############################################################################
def process_frame(args):
    """
    Process a single frame: one file and one time index.

    Physics preserved:
        * temp, td in degC
        * wrf.vinterp to 850/700 hPa (temp) and 850 hPa (dewpoint)
        * Mills & McCaw C-HAINES formula
        * Gaussian smoothing (sigma=1)
        * Levels, colormap, and colorbar labeling

    Structure:
        * Moving-nest-safe geometry via compute_grid_and_spacing.
        * Cities via plot_cities.
        * Gridlines via add_latlon_gridlines.
        * PNG named with valid time (YYYYMMDDHHMMSS).
    """
    ncfile_path, time_index, domain, path_figures = args

    # Open the WRF file for this frame
    with Dataset(ncfile_path) as ncfile:

        # Get valid time as a datetime object
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # -------------------------------------------------------------------------
        # Retrieve temperature fields at 850 and 700 hPa and dewpoint at 850 hPa
        # (Physics block kept identical: getvar + vinterp usage)
        # -------------------------------------------------------------------------
        t = wrf.getvar(ncfile, "temp", timeidx=time_index, units="degC")
        td = wrf.getvar(ncfile, "td", timeidx=time_index, units="degC")

        T850 = wrf.vinterp(
            ncfile, t, "pressure", [850], extrapolate=True, squeeze=True, meta=True
        ).squeeze()
        T700 = wrf.vinterp(
            ncfile, t, "pressure", [700], extrapolate=True, squeeze=True, meta=True
        ).squeeze()
        Td850 = wrf.vinterp(
            ncfile, td, "pressure", [850], extrapolate=True, squeeze=True, meta=True
        ).squeeze()

        # -------------------------------------------------------------------------
        # Lat/Lon and grid spacing (safe for static and moving nests)
        # -------------------------------------------------------------------------
        lats, lons = wrf.latlon_coords(T850)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            _label_adjustment,  # not needed here but kept for consistency
        ) = compute_grid_and_spacing(lats, lons)

        # Get the cartopy mapping object
        cart_proj = wrf.get_cartopy(T850)

        # -------------------------------------------------------------------------
        # Compute C-HAINES (physics identical to original)
        # -------------------------------------------------------------------------
        ny, nx = T850.shape
        CH_2D = np.zeros((ny, nx), dtype=np.float32)

        for j in range(ny):
            for i in range(nx):
                t850_val = T850[j, i].item()
                t700_val = T700[j, i].item()
                td850_val = Td850[j, i].item()

                ch_val = c_haines_index(t850_val, t700_val, td850_val)
                CH_2D[j, i] = ch_val

        # Smooth the result using a small Gaussian filter (sigma=1)
        CH_2D_smoothed = gaussian_filter(CH_2D, sigma=1)

        # -------------------------------------------------------------------------
        # Dateline continuity and polar masking (v9 canonical helper)
        # -------------------------------------------------------------------------
        (
            lats_np,
            lons_np,
            CH_2D_smoothed,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            CH_2D_smoothed,
        )

        # -------------------------------------------------------------------------
        # Create figure and axis
        # -------------------------------------------------------------------------
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Map extent: slightly larger than model domain
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
        # Base land feature
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])

        # Additional Natural Earth features
        for feat in features:
            add_feature(ax, *feat)

        # Plot cities (filtered and thinned) for this domain/time
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)

        # Add lat/lon gridlines with labels
        gl = add_latlon_gridlines(ax)

        # -------------------------------------------------------------------------
        # Plot the C-HAINES (continuous) – physics preserved
        # -------------------------------------------------------------------------
        # Define the levels to cover 0 up to ~14 in ~2-unit intervals
        levels = [0, 2, 4, 6, 8, 10, 12, 14]

        # Colormap for those intervals, with 7 color bins
        colors = [
            "darkgreen",  # 0 - 2
            "lime",  # 2 - 4
            "yellow",  # 4 - 6
            "orange",  # 6 - 8
            "red",  # 8 - 10
            "firebrick",  # 10 - 12
            "purple",  # 12 - 14
        ]
        cmap = mcolors.ListedColormap(colors)

        # Normalizer that matches the "levels"
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        ch_contour = ax.contourf(
            lons_np,
            lats_np,
            CH_2D_smoothed,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=crs.PlateCarree(),
        )

        # Colorbar
        cbar = plt.colorbar(
            ch_contour, ax=ax, orientation="vertical", shrink=0.8, pad=0.05
        )
        cbar.set_label("Continuous Haines Index (C-HAINES)")

        # Place the ticks in the midpoints of each bin
        midpoints = [1, 3, 5, 7, 9, 11, 13]
        cbar.set_ticks(midpoints)
        cbar.set_ticklabels(["0–2", "2–4", "4–6", "6–8", "8–10", "10–12", "12–13+"])

        # -------------------------------------------------------------------------
        # Titles and saving the figure
        # -------------------------------------------------------------------------
        plt.title(
            f"Weather Research and Forecasting Model\n"
            f"Avg Grid Spacing: {avg_dx_km} x {avg_dy_km} km\n"
            f"Continuous Haines Index (Smoothed, sigma=1)",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        # Use a timestamp from valid_dt for filename, keeps GIF sort correct
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_CHaines_Index_{fname_time}.png"

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
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example:\n"
            "    CHaines_Index.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    # Define the path where the netcdf files are and the domain to be used
    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # -------------------------------------------------------------------------
    # Prepare output directories
    # -------------------------------------------------------------------------
    path_figures = "wrf_CHaines_Index"
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

    # Build argument list for the worker function
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

    print("C-Haines Index plot generation complete.")

    # -------------------------------------------------------------------------
    # Build animated GIF from the sorted PNG files
    # -------------------------------------------------------------------------
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        sys.exit(0)

    # Filenames contain YYYYMMDDHHMMSS, so simple sort is chronological
    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files...")

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]
    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        sys.exit(0)

    gif_file_out = f"wrf_{domain}_CHaines_Index.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
