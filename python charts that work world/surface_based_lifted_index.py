#!/usr/bin/env python3
"""
Surface_Layer_LI_multicore_v3.py

Plot WRF surface-layer Lifted Index (K), computed using a moist-adiabatic
parcel profile launched from the lowest layer above the surface.

This script can handle:
    * Multiple wrfout_<domain>* files, each with one or more timesteps.
    * A single wrfout file containing many timesteps.
    * Mixed layouts.

For each (file, time_index) frame:
    * All geometry (lat/lon, grid spacing, extent, cities) is recomputed.
    * No geometry is cached across frames, so this is safe for static and
      moving/vortex-following nests.
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
from wrf import ALL_TIMES, to_np  # ALL_TIMES kept for consistency with other scripts

###############################################################################
# Warning suppression
###############################################################################
warnings.filterwarnings("ignore")

###############################################################################
# Canonical helper function block (v9 – contiguous, order-locked)
#
# Notes:
# - Canonical helpers below are verbatim per playbook (no code altered).
# - Additional non-canonical helpers (LI computation) are included here to keep
#   all helper defs contiguous per playbook structure rule.
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


def _compute_li_block(
    j_start,
    j_end,
    pres_np,
    temp_np,
    dew_np,
    psfc_np,
    t500_np,
    lowest_layer_dp,
):
    """
    Worker function to compute LI for a block of latitude rows [j_start, j_end).

    Physics matches original implementation.
    """
    nz, ny, nx = pres_np.shape
    ny_block = j_end - j_start
    li_block = np.full((ny_block, nx), np.nan, dtype=np.float32)

    for local_j, j in enumerate(range(j_start, j_end)):
        for i in range(nx):
            p_col = pres_np[:, j, i]
            T_col = temp_np[:, j, i]
            Td_col = dew_np[:, j, i]
            psfc_here = psfc_np[j, i]

            # Convert masked arrays to plain ndarrays with NaNs where masked
            if np.ma.isMaskedArray(p_col):
                p_vals = p_col.filled(np.nan)
            else:
                p_vals = np.array(p_col, dtype=float)

            if np.ma.isMaskedArray(T_col):
                T_vals = T_col.filled(np.nan)
            else:
                T_vals = np.array(T_col, dtype=float)

            if np.ma.isMaskedArray(Td_col):
                Td_vals = Td_col.filled(np.nan)
            else:
                Td_vals = np.array(Td_col, dtype=float)

            # Skip bad/missing columns
            if (
                np.any(np.isnan(p_vals))
                or np.any(np.isnan(T_vals))
                or np.any(np.isnan(Td_vals))
                or np.isnan(psfc_here)
            ):
                continue

            # Attach units for temperature/dewpoint
            T_q = T_vals * units.kelvin
            Td_q = Td_vals * units.kelvin

            # Lowest `lowest_layer_dp` hPa above the surface
            layer_mask = (p_vals <= psfc_here) & (p_vals >= psfc_here - lowest_layer_dp)

            if np.any(layer_mask):
                T0_q = T_q[layer_mask].mean()
                Td0_q = Td_q[layer_mask].mean()
            else:
                # Fallback: use lowest model level
                T0_q = T_q[0]
                Td0_q = Td_q[0]

            # Sort pressure from high -> low (required by parcel_profile)
            sort_idx = np.argsort(p_vals)[::-1]  # largest p -> smallest
            p_sorted_vals = p_vals[sort_idx]
            p_sorted = p_sorted_vals * units.hectopascal

            # Compute parcel profile along this sorted pressure column
            parcel_T = mpcalc.parcel_profile(p_sorted, T0_q, Td0_q)  # K

            # Interpolate parcel T to 500 hPa
            # np.interp expects x ascending, so flip arrays (low -> high)
            p_asc = p_sorted.magnitude[::-1]  # hPa ascending
            T_parcel_asc = parcel_T.to("kelvin").magnitude[::-1]  # K

            # Require that 500 hPa be within the pressure range
            if (500.0 < p_asc.min()) or (500.0 > p_asc.max()):
                continue

            T_parcel_500 = np.interp(500.0, p_asc, T_parcel_asc)  # K
            T_env_500 = t500_np[j, i]  # K

            # LI = T_env(500) - T_parcel(500)
            li_block[local_j, i] = T_env_500 - T_parcel_500

    return j_start, j_end, li_block


def _compute_li_block_wrapper(args):
    return _compute_li_block(*args)


def lifted_index(
    pressure,
    temperature,
    dewpoint,
    psfc_hpa,
    T500,
    smooth_sigma=2.0,
    lowest_layer_dp=25.0,
):
    """
    Compute Lifted Index (LI) using full vertical profiles and a moist-adiabatic
    parcel lift.

    LI = T_env(500 hPa) - T_parcel(500 hPa)

    Physics and parameters match the original implementation.
    """
    # Convert to numpy (may be masked arrays)
    pres_np = to_np(pressure)  # (nz, ny, nx), hPa
    temp_np = to_np(temperature)  # (nz, ny, nx), K
    dew_np = to_np(dewpoint)  # (nz, ny, nx), K
    psfc_np = to_np(psfc_hpa)  # (ny, nx), hPa

    # Environmental 500-hPa temperature field (optionally smoothed)
    t500_np = to_np(T500)  # (ny, nx), K
    if smooth_sigma is not None:
        t500_np = gaussian_filter(t500_np, sigma=smooth_sigma)

    nz, ny, nx = pres_np.shape
    li_np = np.full((ny, nx), np.nan, dtype=np.float32)

    # Use fixed 4 workers as in original
    n_workers = 4

    # Determine row blocks for each worker to reduce overhead
    rows_per_worker = max(1, ny // n_workers)
    blocks = []
    start = 0
    while start < ny:
        end = min(start + rows_per_worker, ny)
        blocks.append((start, end))
        start = end

    args_list = [
        (
            j_start,
            j_end,
            pres_np,
            temp_np,
            dew_np,
            psfc_np,
            t500_np,
            lowest_layer_dp,
        )
        for (j_start, j_end) in blocks
    ]

    # Parallelize over latitude row blocks (executor.map pattern)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for j_start, j_end, li_block in executor.map(
            _compute_li_block_wrapper, args_list
        ):
            li_np[j_start:j_end, :] = li_block

    return li_np


###############################################################################
# Natural Earth features (module scope)
###############################################################################
###############################################################################
# Map features, Natural Earth configuration, and cities
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

###############################################################################
# Cities (module scope)
###############################################################################
## Load populated places (cities) once per worker process.
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


###############################################################################
# Frame processing: one (file, time_index) pair
###############################################################################
def process_frame(args):
    """
    Process a single frame: one file and one time index.

    Steps:
        * Read WRF variables (pressure, temp, dewpoint, geopt, PSFC) at this time.
        * Vertically interpolate to get 500-hPa T and Z (T500, Z500).
        * Compute lifted index using moist-adiabatic parcel profile.
        * Smooth LI and plot contours on a Cartopy map.
        * Save a PNG file named with the valid time (for GIF sorting).
    """
    ncfile_path, time_index, domain, path_figures = args

    # Open the WRF file for this frame
    with Dataset(ncfile_path) as ncfile:

        # Get valid time as a datetime object
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # -------------------------------------------------------------------------
        # Get required variables (physics / arguments preserved, plus timeidx)
        # -------------------------------------------------------------------------
        pressure = wrf.getvar(ncfile, "pressure", timeidx=time_index)  # hPa
        temperature = wrf.getvar(ncfile, "temp", timeidx=time_index, units="K")  # K
        dewpoint = wrf.getvar(ncfile, "td", timeidx=time_index, units="K")  # K
        geopotential_height = wrf.getvar(ncfile, "geopt", timeidx=time_index)  # m2 s-2
        psfc_pa = wrf.getvar(ncfile, "PSFC", timeidx=time_index)  # Pa
        psfc = psfc_pa / 100.0  # hPa
        try:
            psfc.attrs["units"] = "hPa"
        except Exception:
            pass

        # 500-hPa environmental fields (vertical interpolation physics unchanged)
        Z500 = wrf.vinterp(
            ncfile,
            geopotential_height,
            "pressure",
            [500],
            field_type="ght",
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()  # m2 s-2

        T500 = wrf.vinterp(
            ncfile,
            temperature,
            "pressure",
            [500],
            field_type="tk",
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()  # K

        # Coordinates & projection from T500
        lats, lons = wrf.latlon_coords(T500)
        cart_proj = wrf.get_cartopy(T500)

        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,  # currently unused but kept for consistency
        ) = compute_grid_and_spacing(lats, lons)

        # -------------------------------------------------------------------------
        # Calculate the Lifted Index (moist-adiabatic, profile-based LI)
        # -------------------------------------------------------------------------
        LI_raw = lifted_index(
            pressure=pressure,
            temperature=temperature,
            dewpoint=dewpoint,
            psfc_hpa=psfc,
            T500=T500,
            smooth_sigma=2.0,
            lowest_layer_dp=25.0,  # same as original (comment there said 50 mb)
        )

        # Smooth LI to clean up grid-scale noise (physics: n=9, passes=5 preserved)
        LI_smooth = mpcalc.smooth_n_point(LI_raw, n=9, passes=5)

        # -------------------------------------------------------------------------
        # Dateline continuity and polar masking (v9 canonical helper)
        # -------------------------------------------------------------------------
        LI_smooth_np = to_np(LI_smooth)
        lats_np, lons_np, LI_smooth_np = handle_domain_continuity_and_polar_mask(
            lats_np, lons_np, LI_smooth_np
        )

        # -------------------------------------------------------------------------
        # Create plot
        # -------------------------------------------------------------------------
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Map extent (resolution-aware padding)
        ax.set_extent(
            [
                lons_np.min() - extent_adjustment,
                lons_np.max() + extent_adjustment,
                lats_np.min() - extent_adjustment,
                lats_np.max() + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        # Land + features
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])
        for feature in features:
            add_feature(ax, *feature)

        # Cities & gridlines
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)
        add_latlon_gridlines(ax)

        # LI contours (use smoothed LI field)
        LI_levels = np.arange(-10, 4, 1)
        LI_contour = ax.contour(
            lons_np,
            lats_np,
            LI_smooth_np,
            levels=LI_levels,
            colors="black",
            linestyles="solid",
            linewidths=1,
            transform=crs.PlateCarree(),
        )
        plt.clabel(
            LI_contour,
            inline=True,
            fontsize=11,
            fmt="%d",
            colors="black",
            manual=False,
            inline_spacing=10,
        )

        # Titles
        plt.title(
            "Weather Research and Forecasting Model\n"
            f"Average Grid Spacing:{avg_dx_km}x{avg_dy_km}km\n"
            "Surface Lifted Index (K)",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        # Save PNG with valid_dt-based timestamp for GIF sorting
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_Surface_Layer_LI_{fname_time}.png"

        image_folder = os.path.join(path_figures, "Images")
        plt.savefig(os.path.join(image_folder, file_out), bbox_inches="tight", dpi=250)

        plt.close(fig)


###############################################################################
# Frame Discovery (v9 canonical)
###############################################################################
###############################################################################
# Frame discovery: handle multi-file and multi-time setups
#
# Discover all (file, time_index) combinations.
#
# Supports:
#     * Many wrfout_<domain>* files with one or more Time steps.
#     * A single wrfout file with multiple Time steps.
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
    # CLI arguments
    # -------------------------------------------------------------------------
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example: Surface_Layer_LI_multicore_v3.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # -------------------------------------------------------------------------
    # Output directories (Images + Animation)
    # -------------------------------------------------------------------------
    path_figures = "Surface_Layer_LI"
    image_folder = os.path.join(path_figures, "Images")
    animation_folder = os.path.join(path_figures, "Animation")

    for folder in (path_figures, image_folder, animation_folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    # -------------------------------------------------------------------------
    # Discover frames (multi-file + multi-time)
    # -------------------------------------------------------------------------
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print(f"No wrfout files found in {path_wrf} matching wrfout_{domain}*")
        sys.exit(0)

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

    print("Surface-layer Lifted Index plot generation complete.")

    # -------------------------------------------------------------------------
    # Create GIF animation from sorted PNG files (timestamped filenames)
    # -------------------------------------------------------------------------
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        sys.exit(0)

    # Filenames contain YYYYMMDDHHMMSS → simple sort is chronological
    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files")

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]
    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        sys.exit(0)

    gif_file_out = f"wrf_{domain}_Surface_Layer_LI_animation.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
