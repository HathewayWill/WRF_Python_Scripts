#!/usr/bin/env python3
"""
WRF Haines Index (2D) – v3

Plot a 2D Haines Index field from WRF output on a Cartopy map.

Supports:
    * Multiple wrfout_<domain>* files, each with one or more timesteps.
    * A single wrfout file containing many timesteps.
    * Mixed layouts (some files with 1 time, some with many).

It does NOT assume the domain is static:
    * For each frame (file, time_index), lat/lon, grid spacing, and map
      extent are recomputed from the WRF fields. This is safe for both
      static nests and moving/vortex-following nests.

Physics / diagnostics:
    * Haines index definition and calculation, vertical interpolation
      (wrf.vinterp to 950/850/700/500 hPa), Gaussian smoothing (sigma=1),
      and discrete levels are unchanged from the original script.
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
    """
    Add a Natural Earth feature to a Cartopy axis.
    """
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
    """
    Extract a valid time from a WRF output filename as a fallback.

    Handles:
        wrfout_d01_YYYY-MM-DD_HH:MM:SS
        wrfout_d01_YYYY-MM-DD_HH_MM_SS

    Falls back to standard index slicing, then to file mtime.
    """
    base = os.path.basename(path)

    # Regex: handles ':' or '_' between time parts
    match = re.search(r"wrfout_.*?_(\d{4}-\d{2}-\d{2})_(\d{2}[:_]\d{2}[:_]\d{2})", base)
    if match:
        date_str = match.group(1)
        time_str = match.group(2).replace("_", ":")
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H:%M:%S")
        except Exception:
            pass

    # Fallback: slice based on standard wrfout naming pattern
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
        # Last resort: file modification time
        return datetime.utcfromtimestamp(os.path.getmtime(path))


def get_valid_time(ncfile: Dataset, ncfile_path: str, time_index: int) -> datetime:
    """
    Get the valid time for a given time index from the WRF file.

    Preferred: wrf.extract_times (model metadata).
    Fallback: parse from filename.
    """
    try:
        valid = wrf.extract_times(ncfile, timeidx=time_index)

        # wrf.extract_times may return a numpy array or scalar; normalize to Python datetime
        if isinstance(valid, np.ndarray):
            valid = valid.item()

        # If it's still a numpy datetime64, convert to Python datetime
        if isinstance(valid, np.datetime64):
            valid = valid.astype("datetime64[ms]").tolist()

        if isinstance(valid, datetime):
            return valid
    except Exception:
        pass

    # Fallback: filename-based parsing
    return parse_valid_time_from_wrf_name(ncfile_path)


def compute_grid_and_spacing(lats, lons):
    """
    Given 2D latitude/longitude arrays, compute:

        lats_np, lons_np       : numpy arrays
        avg_dx_km, avg_dy_km   : average grid spacing (km)
        extent_adjustment      : padding for map extent
        label_adjustment       : offset for labels (if needed)

    Recomputed per frame → safe for static and moving nests.
    """
    lats_np = to_np(lats)
    lons_np = to_np(lons)

    dx, dy = mpcalc.lat_lon_grid_deltas(lons_np, lats_np)

    dx_km = dx.to(units.kilometer)
    dy_km = dy.to(units.kilometer)

    dx_km_rounded = np.round(dx_km.magnitude, decimals=2)
    dy_km_rounded = np.round(dy_km.magnitude, decimals=2)

    avg_dx_km = round(np.mean(dx_km_rounded), 2)
    avg_dy_km = round(np.mean(dy_km_rounded), 2)

    # Map extent padding based on resolution
    if avg_dx_km >= 9 or avg_dy_km >= 9:
        extent_adjustment = 0.50
    elif 3 < avg_dx_km < 9 or 3 < avg_dy_km < 9:
        extent_adjustment = 0.25
    else:
        extent_adjustment = 0.15

    # Label offset (if needed later)
    if avg_dx_km >= 9 or avg_dy_km >= 9:
        label_adjustment = 0.35
    elif 3 < avg_dx_km < 9 or 3 < avg_dy_km < 9:
        label_adjustment = 0.20
    else:
        label_adjustment = 0.15

    return lats_np, lons_np, avg_dx_km, avg_dy_km, extent_adjustment, label_adjustment


def add_latlon_gridlines(ax):
    """
    Add latitude/longitude gridlines with consistent styling and labels.

    Call after setting map extent.
    """
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
    """
    Subset and thin cities, then plot them.

    - Subset to domain extent.
    - Sort by POP_MAX, keep top 150.
    - Thin by min distance (in degrees) based on resolution.
    """
    plot_extent = [
        lons_np.min(),
        lons_np.max(),
        lats_np.min(),
        lats_np.max(),
    ]

    cities_within_extent = cities.cx[
        plot_extent[0] : plot_extent[1], plot_extent[2] : plot_extent[3]
    ]

    sorted_cities = cities_within_extent.sort_values(
        by="POP_MAX", ascending=False
    ).head(150)

    if sorted_cities.empty:
        return

    # Resolution-based min distance
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
            clip_on=True,
            transform=crs.PlateCarree(),
        )
        ax.text(
            loc.x,
            loc.y,
            city_name,
            transform=crs.PlateCarree(),
            clip_on=True,
            ha="center",
            va="bottom",
            fontsize=11,
            color="black",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", alpha=0.4),
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
# Haines Index physics (unchanged)
###############################################################################
def haines_index(
    elevation_m,
    T_950=None,
    T_850=None,
    T_700=None,
    T_500=None,
    td_850=None,
    td_700=None,
):
    """
    Compute the Haines Index (HI) given:
      - elevation_m: surface elevation in meters
      - T_950, T_850, T_700, T_500: temperatures at 950, 850, 700, 500 hPa (°C)
      - td_850, td_700: dewpoints at 850, 700 hPa (°C)
      - Haines, D. A., 1988: A lower atmosphere severity index for wildland fires.

    Depending on elevation:
       High Elevation:    > 3000 ft (> 914 m)
         Stability = T700 - T500
         Moisture  = T700 - td700
       Mid Elevation:     1000–3000 ft (305–914 m)
         Stability = T850 - T700
         Moisture  = T850 - td850
       Low Elevation:     < 1000 ft (< 305 m)
         Stability = T950 - T850
         Moisture  = T850 - td850

    Returns:
        Haines Index (integer 2–6), or None if required data is missing.
    """
    # 1. Elevation category
    ft_per_meter = 3.28084
    elevation_ft = elevation_m * ft_per_meter

    if elevation_ft > 3000:
        elev_category = "high"
    elif elevation_ft < 1000:
        elev_category = "low"
    else:
        elev_category = "mid"

    # 2. Stability term
    if elev_category == "high":
        if T_700 is None or T_500 is None:
            return None
        stability_diff = T_700 - T_500
    elif elev_category == "mid":
        if T_850 is None or T_700 is None:
            return None
        stability_diff = T_850 - T_700
    else:  # low
        if T_950 is None or T_850 is None:
            return None
        stability_diff = T_950 - T_850

    def classify_stability(elev_cat, diff):
        if elev_cat == "high":
            # <18 => 1, 18–21 => 2, >21 => 3
            if diff < 18:
                return 1
            elif 18 <= diff <= 21:
                return 2
            else:
                return 3
        elif elev_cat == "mid":
            # <6 => 1, 6–10 => 2, >10 => 3
            if diff < 6:
                return 1
            elif 6 <= diff <= 10:
                return 2
            else:
                return 3
        else:  # low
            # <4 => 1, 4–7 => 2, >7 => 3
            if diff < 4:
                return 1
            elif 4 <= diff <= 7:
                return 2
            else:
                return 3

    A = classify_stability(elev_category, stability_diff)

    # 3. Moisture term
    if elev_category == "high":
        if T_700 is None or td_700 is None:
            return None
        moisture_diff = T_700 - td_700
    else:
        if T_850 is None or td_850 is None:
            return None
        moisture_diff = T_850 - td_850

    def classify_moisture(elev_cat, diff):
        if elev_cat == "high":
            # <15 => 1, 15–20 => 2, >20 => 3
            if diff < 15:
                return 1
            elif 15 <= diff <= 20:
                return 2
            else:
                return 3
        elif elev_cat == "mid":
            # <6 => 1, 6–12 => 2, >12 => 3
            if diff < 6:
                return 1
            elif 6 <= diff <= 12:
                return 2
            else:
                return 3
        else:  # low
            # <6 => 1, 6–9 => 2, >9 => 3
            if diff < 6:
                return 1
            elif 6 <= diff <= 9:
                return 2
            else:
                return 3

    B = classify_moisture(elev_category, moisture_diff)

    # 4. Final Haines Index
    HI = A + B
    return HI


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
# Load populated places (cities) once per worker process
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


###############################################################################
# Process a single (file, time_index) frame
###############################################################################
def process_frame(args):
    """
    Process a single frame: one file and one time index.

    Steps:
        * Read WRF variables (T, Td, terrain) at this time.
        * Interpolate to 950/850/700/500 hPa (wrf.vinterp).
        * Compute discrete Haines Index (unchanged physics).
        * Apply sigma=1 Gaussian smoothing.
        * Build map (features, cities, lat/lon gridlines).
        * Plot smoothed Haines Index as filled contours.
        * Save PNG with timestamp from valid_dt.
    """
    ncfile_path, time_index, domain, path_figures = args

    ncfile = Dataset(ncfile_path)

    try:
        # Valid time (metadata → filename fallback)
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # ---------------------------------------------------------------------
        # Get WRF variables at this time index
        # ---------------------------------------------------------------------
        t = wrf.getvar(ncfile, "temp", timeidx=time_index, units="degC")
        td = wrf.getvar(ncfile, "td", timeidx=time_index, units="degC")
        ter = wrf.getvar(ncfile, "ter", timeidx=time_index, units="m")

        # Interpolate T and Td to needed pressure levels (unchanged)
        t_950 = wrf.vinterp(
            ncfile,
            t,
            "pressure",
            [950],
            timeidx=time_index,
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()
        t_850 = wrf.vinterp(
            ncfile,
            t,
            "pressure",
            [850],
            timeidx=time_index,
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()
        t_700 = wrf.vinterp(
            ncfile,
            t,
            "pressure",
            [700],
            timeidx=time_index,
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()
        t_500 = wrf.vinterp(
            ncfile,
            t,
            "pressure",
            [500],
            timeidx=time_index,
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()

        td_850 = wrf.vinterp(
            ncfile,
            td,
            "pressure",
            [850],
            timeidx=time_index,
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()
        td_700 = wrf.vinterp(
            ncfile,
            td,
            "pressure",
            [700],
            timeidx=time_index,
            extrapolate=True,
            squeeze=True,
            meta=True,
        ).squeeze()

        # Lat/lon and projection (per frame, moving-nest safe)
        lats, lons = wrf.latlon_coords(td)
        cart_proj = wrf.get_cartopy(td)

        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,  # not used here but kept for consistency
        ) = compute_grid_and_spacing(lats, lons)

        # ---------------------------------------------------------------------
        # Compute discrete Haines Index for each grid point (unchanged physics)
        # ---------------------------------------------------------------------
        ny, nx = to_np(ter).shape
        HI_2D = np.zeros((ny, nx), dtype=np.float32)

        for j in range(ny):
            for i in range(nx):
                elev_m = ter[j, i].item()
                t950_val = t_950[j, i].item()
                t850_val = t_850[j, i].item()
                t700_val = t_700[j, i].item()
                t500_val = t_500[j, i].item()
                td850_val = td_850[j, i].item()
                td700_val = td_700[j, i].item()

                hi_val = haines_index(
                    elev_m,
                    T_950=t950_val,
                    T_850=t850_val,
                    T_700=t700_val,
                    T_500=t500_val,
                    td_850=td850_val,
                    td_700=td700_val,
                )
                if hi_val is None:
                    hi_val = np.nan
                HI_2D[j, i] = hi_val

        # Gaussian smoothing (sigma=1) – unchanged
        HI_2D_smoothed = gaussian_filter(HI_2D, sigma=1)

        # ---------------------------------------------------------------------
        # Dateline continuity and polar masking (v9 canonical helper)
        # ---------------------------------------------------------------------
        (
            lats_np,
            lons_np,
            HI_2D_smoothed,
        ) = handle_domain_continuity_and_polar_mask(
            lats_np,
            lons_np,
            HI_2D_smoothed,
        )

        # ---------------------------------------------------------------------
        # Figure and map setup
        # ---------------------------------------------------------------------
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=cart_proj)

        # Map extent: slightly padded around model domain
        ax.set_extent(
            [
                lons_np.min() - extent_adjustment,
                lons_np.max() + extent_adjustment,
                lats_np.min() - extent_adjustment,
                lats_np.max() + extent_adjustment,
            ],
            crs=crs.PlateCarree(),
        )

        # Base land feature
        ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])

        # Additional Natural Earth features
        for feature in features:
            add_feature(ax, *feature)

        # Plot cities for this frame
        plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km)

        # Add lat/lon gridlines
        add_latlon_gridlines(ax)

        # ---------------------------------------------------------------------
        # Haines Index filled contours (levels 2–6 with half-step boundaries)
        # ---------------------------------------------------------------------
        levels = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

        # Custom colormap: 2–6 → darkgreen, lime, yellow, orange, red
        colors = ["darkgreen", "lime", "yellow", "orange", "red"]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        haines_contour = ax.contourf(
            lons_np,
            lats_np,
            HI_2D_smoothed,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=crs.PlateCarree(),
        )

        # Colorbar
        cbar = plt.colorbar(
            haines_contour, ax=ax, orientation="vertical", shrink=0.8, pad=0.04
        )
        cbar.set_label("Haines Index")
        cbar.set_ticks([2, 3, 4, 5, 6])
        cbar.set_ticklabels(
            ["2 - Very Low", "3 - Very Low", "4 - Low", "5 - Moderate", "6 - High"]
        )

        # Titles
        plt.title(
            "Weather Research and Forecasting Model\n"
            f"Average Grid Spacing: {avg_dx_km} x {avg_dy_km} km\n"
            "Haines Index (Smoothed using sigma=1 Gaussian filter)",
            loc="left",
            fontsize=13,
        )
        plt.title(
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            loc="right",
            fontsize=13,
        )

        # ---------------------------------------------------------------------
        # Save PNG – filename uses valid_dt for chronological sorting
        # ---------------------------------------------------------------------
        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        image_folder = os.path.join(path_figures, "Images")
        file_out = f"wrf_{domain}_Haines_Index_{fname_time}.png"

        plt.savefig(
            os.path.join(image_folder, file_out),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)

    except Exception as e:
        print(f"ERROR in {os.path.basename(ncfile_path)} (t={time_index}): {e}")
        try:
            plt.close("all")
        except Exception:
            pass
    finally:
        ncfile.close()


###############################################################################
# Frame discovery: multi-file + multi-time (v9 canonical)
###############################################################################
def discover_frames(ncfile_paths):
    """
    Discover all (file, time_index) combinations.

    Supports:
        * Many wrfout_<domain>* files with one or more Time steps.
        * A single wrfout file with multiple Time steps.
    """
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
    # Parse command-line arguments
    # -------------------------------------------------------------------------
    if len(sys.argv) != 3:
        print(
            "\nEnter the two required arguments: path_wrf and domain\n"
            "For example:\n"
            "    Haines_Index_v3.py /home/WRF/test/em_real d01\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]

    # -------------------------------------------------------------------------
    # Prepare output directories
    # -------------------------------------------------------------------------
    path_figures = "wrf_Haines_Index"
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
    # Discover all (file, time_index) frames
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
    # Process frames in parallel
    # -------------------------------------------------------------------------
    max_workers = min(4, len(args_list)) if args_list else 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("Haines Index plots completed.")

    # -------------------------------------------------------------------------
    # Build an animated GIF from sorted PNG files
    # -------------------------------------------------------------------------
    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        sys.exit(0)

    # Filenames contain YYYYMMDDHHMMSS, so simple sort is chronological
    png_files_sorted = sorted(png_files)

    print("Creating .gif file from sorted .png files")

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]
    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        sys.exit(0)

    gif_file_out = f"wrf_{domain}_Haines_Index.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")
