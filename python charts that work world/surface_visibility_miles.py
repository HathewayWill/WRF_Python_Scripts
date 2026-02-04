#!/usr/bin/env python3
"""
Surface visibility (miles) from WRF output using AFWA-style vis_diagnostics logic.

AFWA method pieces:
  - Hydrometeor extinction from rain + graupel + snow
  - Haze / fog visibility from RH and q2m
  - Total visibility = min(hydrometeor, haze)  (no dust by default)
  - Alpha (shape parameter) field for haze/hydrometeor regime

Playbook v3:
  - Supports multiple wrfout_<domain>* files with multiple Time steps.
  - Each frame = one (file, time_index) pair.
  - Handles static and moving nests (geometry recomputed per frame).
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
from wrf import ALL_TIMES, to_np  # ALL_TIMES included for consistency with v3 pattern

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


def afwa_visibility_from_wrf(ncfile: Dataset, time_index: int):
    """
    Compute AFWA-style surface visibility for a single (file, time_index) frame.

    Returns:
        vis_total_m (2D np.ndarray): total visibility (meters), smoothed
        vis_alpha   (2D np.ndarray): AFWA alpha (shape parameter)
    """
    # --- Basic fields (wrf-python getvar, physics unchanged) ---
    psfc_da = wrf.getvar(ncfile, "PSFC", timeidx=time_index)  # Pa
    t2_da = wrf.getvar(ncfile, "T2", timeidx=time_index)  # K
    rh2m_da = wrf.getvar(ncfile, "rh2", timeidx=time_index)  # %
    q2m_da = wrf.getvar(ncfile, "Q2", timeidx=time_index)  # kg/kg

    # 10 m winds
    u10_da = wrf.getvar(ncfile, "U10", timeidx=time_index)
    v10_da = wrf.getvar(ncfile, "V10", timeidx=time_index)

    # 3D hydrometeors at this time
    q_rain_3d_da = wrf.getvar(ncfile, "QRAIN", timeidx=time_index)
    q_snow_3d_da = wrf.getvar(ncfile, "QSNOW", timeidx=time_index)
    q_grau_3d_da = wrf.getvar(ncfile, "QGRAUP", timeidx=time_index)

    # Use lowest model level (index 0)
    q_rain_da = q_rain_3d_da[0, :, :]
    q_snow_da = q_snow_3d_da[0, :, :]
    q_grau_da = q_grau_3d_da[0, :, :]

    # Precipitable water (same diagnostic, but per time index)
    try:
        pwater_da = wrf.getvar(ncfile, "pw", timeidx=time_index)
    except Exception:
        # crude fallback, same shape as psfc
        pwater_da = psfc_da * 0.0 + 20.0

    # Wind at 125 m AGL using WRF-Python if available
    wind125m_da = None
    try:
        height_agl_da = wrf.getvar(ncfile, "height_agl", timeidx=time_index)
        u_phy_da = wrf.getvar(ncfile, "ua", timeidx=time_index)
        v_phy_da = wrf.getvar(ncfile, "va", timeidx=time_index)

        u125_da = wrf.interplevel(u_phy_da, height_agl_da, 125.0)
        v125_da = wrf.interplevel(v_phy_da, height_agl_da, 125.0)
        wind125m_da = np.sqrt(u125_da**2 + v125_da**2)
    except Exception:
        wind125m_da = None

    # ----------------------
    # Convert to NumPy arrays
    # ----------------------
    psfc = to_np(psfc_da)
    t2 = to_np(t2_da)
    rh2m = to_np(rh2m_da)
    q2m = to_np(q2m_da)
    u10 = to_np(u10_da)
    v10 = to_np(v10_da)
    wind10m = np.sqrt(u10**2 + v10**2)

    q_rain = to_np(q_rain_da)
    q_snow = to_np(q_snow_da)
    q_grau = to_np(q_grau_da)
    pwater = to_np(pwater_da)

    if wind125m_da is not None:
        wind125m = to_np(wind125m_da)
    else:
        wind125m = wind10m  # fallback as in original spirit

    # ----------------------
    # AFWA vis_diagnostics (unchanged physics)
    # ----------------------
    visfactor = 3.912  # Koschmieder constant

    # Air density from psfc and T2
    R = 287.05  # J/(kg·K)
    rho = psfc / (R * t2)  # kg/m^3

    # Hydrometeor extinction coeff (m^-1)
    br = 1.1 * (1000.0 * rho * (q_rain + q_grau)) ** 0.75
    bs = 10.36 * (1000.0 * rho * q_snow) ** 0.78
    hydro_extcoeff = (br + bs) / 1000.0  # m^-1

    # Dust extinction (assumed zero here; AFWA uses 5-bin dust from WRF-Chem)
    dust_extcoeff = np.zeros_like(hydro_extcoeff)

    # Haze/fog visibility (AFWA updated algorithm)
    q2m_safe = np.where(q2m > 0.0, q2m, np.nan)
    vis_haze = np.full_like(q2m_safe, 999999.0, dtype=float)

    mask_q = ~np.isnan(q2m_safe)

    q2_local = q2m_safe[mask_q]
    rh_local = rh2m[mask_q]

    vis_haze_local = (
        1500.0 * (105.0 - rh_local) * (5.0 / np.minimum(1000.0 * q2_local, 5.0))
    )
    vis_haze[mask_q] = vis_haze_local

    # AFWA alpha term for haze (Weibull shape parameter)
    alpha_haze = np.full_like(q2m_safe, 3.6, dtype=float)
    if np.any(mask_q):
        pw_local = pwater[mask_q]
        w125_local = wind125m[mask_q]

        alpha_local = (
            0.1
            + pw_local / 25.0
            + w125_local / 3.0
            + (100.0 - rh_local) / 10.0
            + 1.0 / (1000.0 * q2_local)
        )
        alpha_local = np.minimum(alpha_local, 3.6)
        alpha_haze[mask_q] = alpha_local

    # Hydrometeor + dust visibility
    extcoeff = hydro_extcoeff + dust_extcoeff
    vis_hydlith = np.full_like(extcoeff, 999999.0, dtype=float)
    mask_ext = extcoeff > 0.0
    vis_hydlith[mask_ext] = np.minimum(visfactor / extcoeff[mask_ext], 999999.0)

    # Final AFWA combined visibility + alpha
    vis_total = np.empty_like(extcoeff, dtype=float)
    vis_alpha = np.empty_like(extcoeff, dtype=float)

    mask_hydro_better = vis_hydlith < vis_haze
    mask_haze_better = ~mask_hydro_better

    vis_total[mask_hydro_better] = vis_hydlith[mask_hydro_better]
    vis_alpha[mask_hydro_better] = 3.6  # Gaussian-like when hydrometeors dominate

    vis_total[mask_haze_better] = vis_haze[mask_haze_better]
    vis_alpha[mask_haze_better] = alpha_haze[mask_haze_better]

    # Smooth and return (meters)
    vis_total_smoothed = gaussian_filter(vis_total, sigma=1)

    return vis_total_smoothed, vis_alpha


def create_gif(image_folder, animation_folder, domain):
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        return

    png_files_sorted = sorted(png_files)
    print("Creating .gif file from sorted .png files")

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]
    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        return

    gif_file_out = f"wrf_{domain}_Surface_Visibility_miles_AFWA.gif"
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
# Frame processing: AFWA visibility for one (file, time_index)
###############################################################################
def process_frame(args):
    """
    Process a single frame: one file and one time index.

    Steps:
        * Compute AFWA visibility (m) and alpha.
        * Convert to miles.
        * Build map (features, cities, gridlines).
        * Plot total visibility (miles) as filled contours.
        * Save a PNG with valid-time-based filename.
    """
    ncfile_path, time_index, domain, path_figures = args

    # Open the WRF file for this frame
    with Dataset(ncfile_path) as ncfile:
        # Determine valid time
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Processing: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # AFWA visibility (meters)
        vis_total_m, vis_alpha = afwa_visibility_from_wrf(ncfile, time_index)

        # Convert to miles
        vis_total_miles = vis_total_m * 0.000621371

        # Geometry and projection from a surface field
        psfc_da = wrf.getvar(ncfile, "PSFC", timeidx=time_index)
        lats, lons = wrf.latlon_coords(psfc_da)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,  # not used here but kept for consistency
        ) = compute_grid_and_spacing(lats, lons)

        # Dateline continuity and polar masking (v9 canonical helper)
        lats_np, lons_np, vis_total_miles, vis_alpha = (
            handle_domain_continuity_and_polar_mask(
                lats_np,
                lons_np,
                vis_total_miles,
                vis_alpha,
            )
        )

        cart_proj = wrf.get_cartopy(psfc_da)

        # -------------------------------------------------------------------------
        # Figure and map setup
        # -------------------------------------------------------------------------
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

        # Cities & gridlines
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)
        add_latlon_gridlines(ax)
        ax.tick_params(labelsize=12, width=2)

        # -------------------------------------------------------------------------
        # Visibility ranges and colormap (miles)
        # -------------------------------------------------------------------------
        visibility_ranges = [
            0,
            0.25,
            0.5,
            0.75,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ]

        visibility_map = plt.cm.get_cmap("Greys_r")
        visibility_norm = plt.matplotlib.colors.BoundaryNorm(
            visibility_ranges, visibility_map.N
        )

        cs = ax.contourf(
            lons_np,
            lats_np,
            vis_total_miles,
            levels=visibility_ranges,
            cmap=visibility_map,
            norm=visibility_norm,
            transform=crs.PlateCarree(),
        )
        cbar = plt.colorbar(
            cs,
            ax=ax,
            orientation="vertical",
            pad=0.05,
            shrink=0.8,
            ticks=visibility_ranges,
        )
        cbar.set_label("Total Visibility (Miles)")
        cbar.ax.set_yticklabels([f"{lvl:.2f}" for lvl in visibility_ranges])

        # -------------------------------------------------------------------------
        # Titles and saving
        # -------------------------------------------------------------------------
        plt.title(
            "Weather Research and Forecasting Model\n"
            f"Average Grid Spacing: {avg_dx_km}x{avg_dy_km} km\n"
            "AFWA Visibility (Miles)\n"
            "Min of Hydrometeor & Haze Visibility",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_Surface_Visibility_miles_AFWA_{fname_time}.png"

        images_folder = os.path.join(path_figures, "Images")
        plt.savefig(
            os.path.join(images_folder, file_out),
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
            "For example: script_name.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # Output directories (v3: Images + Animation)
    path_figures = "wrf_Surface_Visibility_miles_AFWA"
    image_folder = os.path.join(path_figures, "Images")
    animation_folder = os.path.join(path_figures, "Animation")

    for folder in (path_figures, image_folder, animation_folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    # Find WRF files
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print(f"No wrfout files found in {path_wrf} matching wrfout_{domain}*")
        sys.exit(0)

    # Discover frames
    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    # Build args list for all frames
    args_list = [
        (ncfile_path, time_index, domain, path_figures)
        for (ncfile_path, time_index) in frames
    ]

    # Process in parallel
    max_workers = min(4, len(args_list)) if args_list else 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("AFWA Surface Visibility (miles) plot generation complete.")

    # GIF from all PNGs
    create_gif(image_folder, animation_folder, domain)
