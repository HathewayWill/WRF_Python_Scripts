#!/usr/bin/env python3
"""
Mixed_Layer_LI_multicore_v3.py

Plot WRF Mixed-Layer Lifted Index (LI; K) on a Cartopy map.

This script can handle:
    * Multiple wrfout_<domain>* files, each with one or more timesteps.
    * A single wrfout file containing many timesteps.

It does NOT assume the domain is static:
    * For each timestep, lat/lon, grid spacing, and extent are
      recomputed from the WRF fields. This automatically works
      for both static nests and moving/vortex-following nests.

Pattern:
    * Discover all (file, time_index) frames.
    * For each frame, compute Mixed-Layer LI and save one PNG.
    * Assemble all PNGs into a GIF sorted by valid time.
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


def lifted_index(
    pressure,
    temperature,
    dewpoint,
    psfc_hpa,
    T500,
    smooth_sigma=2.0,
    lowest_layer_dp=20.0,
):
    """
    Compute Lifted Index (LI) using full vertical profiles and a moist-adiabatic
    parcel lift.

    LI = T_env(500 hPa) - T_parcel(500 hPa)

    This is the same algorithm as in the original script, but implemented purely
    with NumPy arrays (no xarray / metpy.xarray).

    Parameters
    ----------
    pressure : WRF field
        3D full pressure [hPa], shape (nz, ny, nx).
    temperature : WRF field
        3D temperature [K], shape (nz, ny, nx).
    dewpoint : WRF field
        3D dewpoint temperature [K], shape (nz, ny, nx).
    psfc_hpa : WRF field
        2D surface pressure [hPa], shape (ny, nx).
    T500 : WRF field
        2D environmental temperature at 500 hPa [K], shape (ny, nx).
    smooth_sigma : float or None
        Sigma for Gaussian smoothing (grid points) applied to T500.
    lowest_layer_dp : float
        Pressure depth [hPa] above the surface over which to average T and Td
        for the starting parcel (e.g., 20 hPa).

    Returns
    -------
    li_np : np.ndarray
        2D field of Lifted Index [K], same horizontal shape as T500.
    """
    # Convert to numpy (may be masked arrays)
    pres_np = to_np(pressure)  # (nz, ny, nx)
    temp_np = to_np(temperature)  # (nz, ny, nx)
    dew_np = to_np(dewpoint)  # (nz, ny, nx)
    psfc_np = to_np(psfc_hpa)  # (ny, nx)

    # Environmental 500-hPa temperature field (optionally smoothed)
    t500_np = to_np(T500)  # (ny, nx)
    if smooth_sigma is not None:
        t500_np = gaussian_filter(t500_np, sigma=smooth_sigma)

    nz, ny, nx = pres_np.shape
    li_np = np.full((ny, nx), np.nan, dtype=np.float32)

    for j in range(ny):
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
            li_np[j, i] = T_env_500 - T_parcel_500

    return li_np


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
# Frame processing: one (file, time_index) → one PNG
###############################################################################
def process_frame(args):
    """
    Process a single frame: one file and one time index.

    Steps:
        * Read WRF variables at this time.
        * Compute Mixed-Layer Lifted Index (LI).
        * Smooth LI to clean grid-scale noise.
        * Build map (features, cities, gridlines).
        * Plot LI contours.
        * Save a PNG file named with the valid time.
    """
    ncfile_path, time_index, domain, path_figures = args

    # Open the WRF file for this frame
    with Dataset(ncfile_path) as ncfile:

        # Valid time from WRF metadata or filename fallback
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting Mixed-Layer LI: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # -------------------------------------------------------------------------
        # Get required 3D / 2D fields for LI (physics unchanged)
        # -------------------------------------------------------------------------
        # 3D pressure [hPa]
        pressure = wrf.getvar(ncfile, "pressure", timeidx=time_index)  # hPa

        # 3D temperature [K]
        temperature = wrf.getvar(ncfile, "temp", timeidx=time_index, units="K")  # K

        # 3D dewpoint temperature [K] from WRF diagnostic "td"
        dewpoint = wrf.getvar(ncfile, "td", timeidx=time_index, units="K")  # K

        # 3D geopotential (m^2 s^-2)
        geopotential_height = wrf.getvar(
            ncfile, "geopt", timeidx=time_index
        )  # m^2 s^-2

        # Surface pressure [Pa] -> [hPa]
        psfc_pa = wrf.getvar(ncfile, "PSFC", timeidx=time_index)  # Pa
        psfc = psfc_pa / 100.0  # hPa
        psfc.attrs["units"] = "hPa"

        # Environmental 500-hPa fields (vinterp physics unchanged)
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

        # Lat/lon and projection from a WRF field
        lats, lons = wrf.latlon_coords(T500)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,  # kept for consistency, not currently used
        ) = compute_grid_and_spacing(lats, lons)

        # -------------------------------------------------------------------------
        # Dateline continuity and polar masking (v9 canonical helper)
        # -------------------------------------------------------------------------
        (
            lats_np,
            lons_np,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
        )

        cart_proj = wrf.get_cartopy(T500)

        # -------------------------------------------------------------------------
        # Compute Lifted Index (moist-adiabatic, profile-based LI)
        # -------------------------------------------------------------------------
        LI_raw = lifted_index(
            pressure=pressure,
            temperature=temperature,
            dewpoint=dewpoint,
            psfc_hpa=psfc,
            T500=T500,
            smooth_sigma=2.0,
            lowest_layer_dp=20.0,  # same as original: lowest 20 hPa mixed layer
        )

        # Smooth LI to clean up grid-scale noise (same parameters as original)
        LI_smooth = mpcalc.smooth_n_point(LI_raw, n=9, passes=5)

        # Apply continuity/masking consistently to LI (v9 helper is field-agnostic)
        (
            lats_np,
            lons_np,
            LI_smooth,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            LI_smooth,
        )

        # -------------------------------------------------------------------------
        # Create plot with Cartopy / Matplotlib
        # -------------------------------------------------------------------------
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Map extent with resolution-based padding
        ax.set_extent(
            [
                lons_np.min() - extent_adjustment,
                lons_np.max() + extent_adjustment,
                lats_np.min() - extent_adjustment,
                lats_np.max() + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        # Land + other Natural Earth features
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])

        for feature in features:
            add_feature(ax, *feature)

        # Cities (subset + thinning)
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)

        # Lat/lon gridlines (canonical helper)
        gl = add_latlon_gridlines(ax)

        # -------------------------------------------------------------------------
        # LI contours (use smoothed LI field)
        # -------------------------------------------------------------------------
        LI_levels = np.arange(-10, 4, 1)

        LI_contour = ax.contour(
            lons_np,
            lats_np,
            LI_smooth,
            levels=LI_levels,
            colors="black",
            linestyles="solid",
            linewidths=1.0,
            transform=crs.PlateCarree(),
        )

        plt.clabel(
            LI_contour,
            inline=True,
            fontsize=11,
            fmt="%d",
            colors="black",
            inline_spacing=10,
        )

        # -------------------------------------------------------------------------
        # Titles and saving
        # -------------------------------------------------------------------------
        plt.title(
            f"Weather Research and Forecasting Model\n"
            f"Average Grid Spacing: {avg_dx_km} x {avg_dy_km} km\n"
            f"Mixed-Layer Lifted Index (K)",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        # Filename uses valid_dt timestamp → correct GIF sort by simple alphabetical order
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_Mixed_Layer_LI_{fname_time}.png"

        plt.savefig(
            os.path.join(path_figures, "Images", file_out),
            bbox_inches="tight",
            dpi=250,
        )

        plt.close(fig)


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
            "For example:\n"
            "    Mixed_Layer_LI_multicore_v3.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # -------------------------------------------------------------------------
    # Output directories (Images + Animation)
    # -------------------------------------------------------------------------
    path_figures = "Mixed_Layer_LI"
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
    # Build list of frames (file, time_index)
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
    # Process frames in parallel using ProcessPoolExecutor
    # -------------------------------------------------------------------------
    max_workers = min(4, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("Mixed-Layer Lifted Index plot generation complete.")

    # -------------------------------------------------------------------------
    # Create GIF animation (simple alphabetical sort; timestamps embedded)
    # -------------------------------------------------------------------------
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        sys.exit(0)

    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files")

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]
    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        sys.exit(0)

    gif_file_out = f"wrf_{domain}_Mixed_Layer_LI.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
