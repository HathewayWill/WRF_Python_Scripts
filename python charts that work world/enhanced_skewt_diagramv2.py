#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v5.3 (hybrid precip-type with microphysics + energy + Tw_sfc + SWEAT + ConvT)

Enhanced WRF sounding toolkit:
  * Skew-T / hodograph / hazard / map
  * Parcel thermodynamics and environment summary
  * Shear / SRH / storm-relative wind table
  * Multi-time rendering + animated GIF
  * 20×16 logical layout grid with labeled major + quarter subgrid

Structural v3 updates:
  * Multi-file + multi-time support via discover_frames(...)
  * One frame = one (file, time_index) pair
  * Valid time from WRF metadata (wrf.extract_times), filename as fallback
  * PNG filenames use YYYYMMDDHHMMSS timestamps for GIF ordering

All meteorological computations, indices, and precip-type logic are unchanged.
"""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import glob
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib as mpl
from matplotlib.path import Path
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import wrf
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import ScalarFormatter
from metpy.plots import SkewT
from metpy.units import units
from netCDF4 import Dataset
from PIL import Image

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================


# Optional Cartopy map panel (projection)
try:
    import cartopy.crs as ccrs

    _CARTOPY_OK = True
except Exception:
    _CARTOPY_OK = False

warnings.filterwarnings("ignore")
mpl.rcParams["figure.constrained_layout.use"] = False

# =============================================================================
# GLOBAL CONFIG / CONSTANTS
# =============================================================================

SHOW_LAYOUT_GRID = False

MAX_CITIES_ON_MAP = 30  # Max city labels on the map panel

# Global table font-size configuration (tweak here to change all tables at once)
TABLE_FONT_SIZE = 11
TABLE_HEADER_FONT_SIZE = 12

# =============================================================================
# PARTIAL-THICKNESS NOMOGRAM BOUNDS (authoritative)
# =============================================================================
# x-axis: 850–700 hPa thickness (m)
# y-axis: 1000–850 hPa thickness (m)
PT_X_MIN, PT_X_MAX = 1525.0, 1560.0
PT_Y_MIN, PT_Y_MAX = 1281.0, 1315.0

# =============================================================================
# TIME HANDLING HELPERS (v3 canonical)
# =============================================================================


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


# =============================================================================
# LAYOUT HELPER / DEBUG GRID
# =============================================================================


def draw_layout_grid(
    fig, ncols=20, nrows=16, subdiv=4, label_every=1, label_subdivisions=True
):
    """Draw a labeled layout grid in figure coordinates (0–1)."""
    if not SHOW_LAYOUT_GRID:
        return

    ax_grid = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=1000)
    ax_grid.set_xlim(0, 1)
    ax_grid.set_ylim(0, 1)
    ax_grid.set_axis_off()
    ax_grid.patch.set_alpha(0.0)

    total_col_steps = ncols * subdiv
    for step in range(total_col_steps + 1):
        x = step / total_col_steps
        is_integer = step % subdiv == 0
        major_idx = step // subdiv

        if is_integer:
            if major_idx in (0, ncols):
                lw = 1.4
                alpha = 0.9
            else:
                lw = 0.9
                alpha = 0.8
            color = "gray"
        else:
            lw = 0.5
            alpha = 0.70
            color = "darkgray"

        ax_grid.axvline(x, color=color, linewidth=lw, alpha=alpha)

        if is_integer and (major_idx % label_every == 0):
            ax_grid.text(
                x,
                -0.01,
                f"{major_idx}",
                ha="center",
                va="top",
                fontsize=9,
                color="black",
            )
            ax_grid.text(
                x,
                1.01,
                f"{major_idx}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )
        elif (not is_integer) and label_subdivisions:
            frac = (step % subdiv) / subdiv
            val = major_idx + frac
            ax_grid.text(
                x,
                -0.01,
                f"{val:.2f}",
                ha="center",
                va="top",
                fontsize=7,
                color="dimgray",
            )
            ax_grid.text(
                x,
                1.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="dimgray",
            )

    total_row_steps = nrows * subdiv
    for step in range(total_row_steps + 1):
        y = step / total_row_steps
        is_integer = step % subdiv == 0
        major_idx = step // subdiv

        if is_integer:
            if major_idx in (0, nrows):
                lw = 1.4
                alpha = 0.9
            else:
                lw = 0.9
                alpha = 0.8
            color = "gray"
        else:
            lw = 0.3
            alpha = 0.5
            color = "lightgray"

        ax_grid.axhline(y, color=color, linewidth=lw, alpha=alpha)

        if is_integer and (major_idx % label_every == 0):
            ax_grid.text(
                -0.01,
                y,
                f"{major_idx}",
                ha="right",
                va="center",
                fontsize=9,
                color="black",
            )
            ax_grid.text(
                1.01,
                y,
                f"{major_idx}",
                ha="left",
                va="center",
                fontsize=9,
                color="black",
            )
        elif (not is_integer) and label_subdivisions:
            frac = (step % subdiv) / subdiv
            val = major_idx + frac
            ax_grid.text(
                -0.01,
                y,
                f"{val:.2f}",
                ha="right",
                va="center",
                fontsize=7,
                color="dimgray",
            )
            ax_grid.text(
                1.01,
                y,
                f"{val:.2f}",
                ha="left",
                va="center",
                fontsize=7,
                color="dimgray",
            )


def grid_span(col_left, row_bottom, col_right, row_top, ncols=20, nrows=16):
    """Convert grid *edges* to a Matplotlib rect in figure coordinates."""
    left = col_left / ncols
    bottom = row_bottom / nrows
    width = (col_right - col_left) / ncols
    height = (row_top - row_bottom) / nrows
    return [left, bottom, width, height]


def get_default_panel_layout():
    """Define panel layout for the figure using a 20×16 logical grid."""
    return {
        "title": grid_span(1, 14, 19, 15),
        "skewt": grid_span(1, 4, 11, 14),
        "hodograph": grid_span(11, 7, 19, 14),
        "hazard": grid_span(11, 3.50, 14, 7),
        "Surface": grid_span(14, 3.50, 16, 7),
        "map": grid_span(16, 3.50, 19, 7),
        "storm_motion": grid_span(17.5, 7, 19, 9),
        "parcel_table": grid_span(1, 0.5, 7, 3.50),
        "env_summary": grid_span(7, 0.5, 11, 3.50),
        "shear_table": grid_span(11, 0.5, 19, 3.50),
    }


def add_panel_frame(fig, rect, color="red", lw=1.3):
    """Draw rectangle around panel in figure coords (debug)."""
    if not SHOW_LAYOUT_GRID:
        return

    fig.add_artist(
        Rectangle(
            (rect[0], rect[1]),
            rect[2],
            rect[3],
            transform=fig.transFigure,
            fill=False,
            edgecolor=color,
            linewidth=lw,
            zorder=900,
        )
    )


# =============================================================================
# PLOTTING HELPERS
# =============================================================================


def plot_skewt_wind_barbs(skew_plot, pressure, u_wind, v_wind):
    """Plot wind barbs along right side of Skew-T aligned with standard pressure ticks."""
    standard_levels = (
        np.array(
            [
                1013,
                1000,
                950,
                900,
                850,
                800,
                750,
                700,
                650,
                600,
                550,
                500,
                450,
                400,
                350,
                300,
                250,
                200,
                150,
                100,
            ],
            dtype=float,
        )
        * units.hPa
    )

    p_min = np.min(pressure)
    p_max = np.max(pressure)
    mask = (standard_levels >= p_min) & (standard_levels <= p_max)
    barb_levels = standard_levels[mask]
    if barb_levels.size == 0:
        return

    idx = mpcalc.resample_nn_1d(pressure, barb_levels)
    u_at = u_wind[idx]
    v_at = v_wind[idx]
    blank = np.zeros(len(idx))

    skew_plot.plot_barbs(
        pressure=barb_levels,
        u=blank,
        v=blank,
        xloc=1.0,
        fill_empty=True,
        sizes=dict(emptybarb=0.075, width=0.18, height=0.4),
    )
    skew_plot.plot_barbs(
        pressure=barb_levels,
        u=u_at,
        v=v_at,
        xloc=1.0,
        fill_empty=True,
        sizes=dict(emptybarb=0.075, width=0.18, height=0.4),
        length=7,
    )


# =============================================================================
# SMALL GENERIC HELPERS
# =============================================================================


def is_finite_scalar(value) -> bool:
    """Return True if *all* elements of value are finite."""
    try:
        arr = np.asarray(getattr(value, "m", value))
    except Exception:
        arr = np.asarray(value)
    return np.isfinite(arr).all()


def quantity_to_scalar(q, default=np.nan):
    """Convert quantity/array to float scalar; default on failure."""
    try:
        if hasattr(q, "magnitude"):
            base_val = q.magnitude
        else:
            base_val = q
        return float(np.atleast_1d(base_val)[0])
    except Exception:
        return default


def fmt(q, fmt_str="{:.0f}", unit_str=""):
    """Format quantity/number into string with unit, or 'N/A' if not finite."""
    val = quantity_to_scalar(q)
    if not np.isfinite(val):
        return "N/A"
    return fmt_str.format(val) + unit_str


def has_buoyancy(cape, cin):
    """Return True if parcel has non-trivial CAPE or CIN."""
    cape_val = quantity_to_scalar(cape)
    cin_val = quantity_to_scalar(cin)
    has_cape = np.isfinite(cape_val) and np.abs(cape_val) > 0.1
    has_cin = np.isfinite(cin_val) and np.abs(cin_val) > 0.1
    return has_cape or has_cin


def layer_lapse_rate(temp_profile, height_profile, bottom_agl, top_agl):
    """Compute layer-mean lapse rate (K/km) between bottom_agl and top_agl (m)."""
    try:
        mask = (height_profile >= bottom_agl) & (height_profile <= top_agl)
        if not np.any(mask):
            return np.nan * (units.kelvin / units.kilometer)

        z_layer = height_profile[mask]
        t_layer = temp_profile[mask].to("kelvin")
        if len(z_layer) < 2:
            return np.nan * (units.kelvin / units.kilometer)

        t_bottom = t_layer[0]
        t_top = t_layer[-1]
        z_bottom = z_layer[0]
        z_top = z_layer[-1]
        lapse = -(t_bottom - t_top) / ((z_bottom - z_top).to("kilometer"))
        return lapse.to("K/km")
    except Exception:
        return np.nan * (units.kelvin / units.kilometer)


def ensure_directories_exist(*dirs):
    """Create any directories that don't already exist."""
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)


# =============================================================================
# TABLE FONT HELPER
# =============================================================================


def style_table_fonts(table, body_size=None, header_size=None):
    """
    Apply global font sizes to a Matplotlib table.

    Parameters
    ----------
    table : matplotlib.table.Table
        Table instance to style.
    body_size : int or float, optional
        Font size for body rows (defaults to TABLE_FONT_SIZE).
    header_size : int or float, optional
        Font size for header row (defaults to TABLE_HEADER_FONT_SIZE).
    """
    if body_size is None:
        body_size = TABLE_FONT_SIZE
    if header_size is None:
        header_size = TABLE_HEADER_FONT_SIZE

    for (row, col), cell in table.get_celld().items():
        txt = cell.get_text()
        if row == 0:
            txt.set_fontsize(header_size)
        else:
            txt.set_fontsize(body_size)


# =============================================================================
# KINEMATIC / THERMODYNAMIC LAYER HELPERS
# =============================================================================


def compute_layer_shear_and_srh(pressure, height, u_wind, v_wind, depth):
    """Compute bulk shear, SRH, mean wind, and storm-relative mean wind for a layer."""
    (u_rm, v_rm), _, _ = mpcalc.bunkers_storm_motion(pressure, u_wind, v_wind, height)

    *_, srh_val = mpcalc.storm_relative_helicity(
        height, u_wind, v_wind, depth=depth, storm_u=u_rm, storm_v=v_rm
    )

    u_shear, v_shear = mpcalc.bulk_shear(
        pressure, u_wind, v_wind, height=height, depth=depth
    )
    shear_speed = mpcalc.wind_speed(u_shear, v_shear).to("knot")
    shear_dir = mpcalc.wind_direction(u_shear, v_shear)

    mask = height <= depth
    if np.any(mask):
        u_mean = np.mean(u_wind[mask])
        v_mean = np.mean(v_wind[mask])
        mean_speed = mpcalc.wind_speed(u_mean, v_mean).to("knot")
        mean_dir = mpcalc.wind_direction(u_mean, v_mean)
        srw_u = u_mean - u_rm
        srw_v = v_mean - v_rm
        srw_speed = mpcalc.wind_speed(srw_u, srw_v).to("knot")
        srw_dir = mpcalc.wind_direction(srw_u, srw_v)
    else:
        mean_speed = np.nan * units.knot
        mean_dir = np.nan * units.deg
        srw_speed = np.nan * units.knot
        srw_dir = np.nan * units.deg

    return (shear_speed, shear_dir, srh_val, mean_speed, mean_dir, srw_speed, srw_dir)


def compute_layer_cape(pressure, temperature, dewpoint, height, top_height):
    """Compute CAPE for surface parcel, only up to top_height (e.g., 3 km)."""
    mask = height <= top_height
    if np.count_nonzero(mask) < 3:
        return np.nan * (units.joule / units.kilogram)

    p_layer = pressure[mask]
    t_layer = temperature[mask]
    td_layer = dewpoint[mask]
    try:
        parcel_prof = mpcalc.parcel_profile(p_layer, t_layer[0], td_layer[0])
        cape, _ = mpcalc.cape_cin(
            p_layer, t_layer, td_layer, parcel_profile=parcel_prof
        )
        return cape
    except Exception:
        return np.nan * (units.joule / units.kilogram)


def compute_dcape(pressure, temperature, dewpoint):
    """Compute downdraft CAPE (DCAPE) using MetPy's downdraft_cape."""
    try:
        p_hpa = pressure.to("hPa").magnitude
        if np.nanmax(p_hpa) <= 700.0 or np.nanmin(p_hpa) >= 500.0:
            return np.nan * (units.joule / units.kilogram)
        dcape_result = mpcalc.downdraft_cape(pressure, temperature, dewpoint)
        return dcape_result[0]
    except Exception:
        return np.nan * (units.joule / units.kilogram)


# =============================================================================
# PRESSURE / HEIGHT INTERPOLATION HELPERS
# =============================================================================


def _interp_height_at_pressure(p_hpa, z_m, target_p_hpa):
    """Interpolate height (m) at given pressure (hPa)."""
    p = np.asarray(p_hpa, dtype=float)
    z = np.asarray(z_m, dtype=float)

    if not np.isfinite(target_p_hpa):
        return np.nan
    if target_p_hpa < p.min() or target_p_hpa > p.max():
        return np.nan

    sort_idx = np.argsort(p)
    p_sorted = p[sort_idx]
    z_sorted = z[sort_idx]
    return float(np.interp(target_p_hpa, p_sorted, z_sorted))


def pressure_to_height_km(target_p_hpa, p_hpa, z_m):
    """Interpolate height (km) at given pressure (hPa) using the profile."""
    p = np.asarray(p_hpa, dtype=float)
    z = np.asarray(z_m, dtype=float)
    if not np.isfinite(target_p_hpa):
        return np.nan

    sort_idx = np.argsort(p)
    p_sorted = p[sort_idx]
    z_sorted = z[sort_idx]
    if target_p_hpa < p_sorted[0] or target_p_hpa > p_sorted[-1]:
        return np.nan

    z_target_m = np.interp(target_p_hpa, p_sorted, z_sorted)
    return z_target_m / 1000.0


# =============================================================================
# PARTIAL THICKNESS / PRECIP-TYPE LOGIC (unchanged physics)
# =============================================================================


def get_partial_thickness_polygon_defs():
    """
    Polygon definitions aligned to the reference implementation.

    Axes:
      x = 850–700 hPa thickness (m)
      y = 1000–850 hPa thickness (m)

    Notes:
      * Several vertices intentionally extend far outside the plotted window
        to emulate open-ended regions.
      * Do not clamp polygon vertices. Only clamp the test point.
    """
    x_far = 20000.0
    y_far = 20000.0
    x_floor = 0.0
    y_floor = 0.0

    snow = np.array([
        [x_floor, y_floor],
        [x_floor, 1537.375],
        [1500.0, 1303.0],
        [1532.0, 1298.0],
        [1537.0, 1290.0],
        [1542.0, 1278.0],
        [1545.0, 1260.0],
        [1755.0, y_floor],
    ])

    unknown = np.array([
        [x_floor, y_far],
        [x_floor, 1537.375],
        [1500.0, 1303.0],
        [1532.0, 1298.0],
        [1530.0, 1312.0],
        [1538.0, 1312.0],
        [1538.0, y_far],
    ])

    snow_rain = np.array([
        [1530.0, 1312.0],
        [1538.0, 1312.0],
        [1543.0, 1290.0],
        [1537.0, 1290.0],
        [1532.0, 1298.0],
    ])

    wintry_mix = np.array([
        [1538.0, 1312.0],
        [1550.0, 1310.0],
        [1550.0, 1290.0],
        [1543.0, 1290.0],
    ])

    rain = np.array([
        [1538.0, 1312.0],
        [1550.0, 1310.0],
        [1580.0, 1314.0],
        [x_far, 3770.0],
        [x_far, y_far],
        [1538.0, y_far],
    ])

    snow_sleet = np.array([
        [1537.0, 1290.0],
        [1550.0, 1290.0],
        [1550.0, 1282.0],
        [1562.0, 1260.0],
        [24742.0 / 11.0, y_floor],
        [1755.0, y_floor],
        [1545.0, 1260.0],
        [1542.0, 1278.0],
    ])

    sleet_fzra = np.array([
        [1550.0, 1293.0],
        [1580.0, 1290.0],
        [14480.0, y_floor],
        [24742.0 / 11.0, y_floor],
        [1562.0, 1260.0],
        [1550.0, 1282.0],
        [1550.0, 1290.0],
    ])

    fzra_sleet = np.array([
        [1550.0, 1293.0],
        [1580.0, 1290.0],
        [1850.0, 1263.0],
        [1580.0, 1299.0],
        [1550.0, 1303.0],
    ])

    fzra = np.array([
        [1550.0, 1310.0],
        [1580.0, 1314.0],
        [x_far, 3770.0],
        [x_far, y_floor],
        [14480.0, y_floor],
        [1850.0, 1263.0],
        [1580.0, 1299.0],
        [1550.0, 1303.0],
    ])

    return {
        "snow": snow,
        "unknown": unknown,
        "snow_rain": snow_rain,
        "wintry_mix": wintry_mix,
        "rain": rain,
        "snow_sleet": snow_sleet,
        "sleet_fzra": sleet_fzra,
        "fzra_sleet": fzra_sleet,
        "fzra": fzra,
    }


def classify_precip_type_partial_thickness(
    pressure_profile,
    height_profile,
    temperature_profile,
    rh_profile,
    return_thickness=False,
):
    """
    Partial-thickness precip-type classification (internal-only).

    x-axis: 850–700 hPa thickness (m)
    y-axis: 1000–850 hPa thickness (m)

    Uses climo polygons and a saturation test in 1000–700 hPa.
    """

    # Nomogram bounds used only to clamp the polygon test point
    X_MIN, X_MAX = PT_X_MIN, PT_X_MAX
    Y_MIN, Y_MAX = PT_Y_MIN, PT_Y_MAX

    p_hpa = pressure_profile.to("hPa").magnitude
    z_m = height_profile.to("m").magnitude

    low_level_bottom = _interp_height_at_pressure(p_hpa, z_m, 1000.0)
    boundary_level = _interp_height_at_pressure(p_hpa, z_m, 850.0)
    mid_level_top = _interp_height_at_pressure(p_hpa, z_m, 700.0)

    if (
        np.isnan(low_level_bottom)
        or np.isnan(boundary_level)
        or np.isnan(mid_level_top)
    ):
        thickness_850_700 = np.nan
        thickness_1000_850 = np.nan
    else:
        thickness_850_700 = mid_level_top - boundary_level
        thickness_1000_850 = boundary_level - low_level_bottom

    # Build closed Paths for robust point-in-polygon tests in (x, y) space
    poly_defs = get_partial_thickness_polygon_defs()

    def _as_closed_path(verts):
        v = np.asarray(verts, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2:
            return None
        if not np.allclose(v[0], v[-1]):
            v = np.vstack([v, v[0]])
        return Path(v)

    snow_path = _as_closed_path(poly_defs["snow"])
    unknown_path = _as_closed_path(poly_defs["unknown"])
    snow_rain_path = _as_closed_path(poly_defs["snow_rain"])
    wintry_mix_path = _as_closed_path(poly_defs["wintry_mix"])
    rain_path = _as_closed_path(poly_defs["rain"])
    snow_sleet_path = _as_closed_path(poly_defs["snow_sleet"])
    sleet_fzra_path = _as_closed_path(poly_defs["sleet_fzra"])
    fzra_sleet_path = _as_closed_path(poly_defs["fzra_sleet"])
    fzra_path = _as_closed_path(poly_defs["fzra"])

    # Saturation test (unchanged physics)
    is_saturated_anywhere = False
    if rh_profile is not None:
        max_contig_depth = 0.0 * units.hPa
        cur_contig_depth = 0.0 * units.hPa

        for top in np.arange(1000, 700, -50) * units.hPa:
            bottom = top - 50 * units.hPa
            within = (pressure_profile <= top) & (pressure_profile > bottom)
            if np.count_nonzero(within) <= 1:
                continue

            sel_p = pressure_profile[within]
            sel_rh = rh_profile[within]
            sel_z = height_profile[within]

            try:
                rh_layer = mpcalc.mean_pressure_weighted(
                    sel_p,
                    sel_rh,
                    height=sel_z,
                    bottom=sel_p[0],
                    depth=(sel_p[0] - sel_p[-1]),
                )[0].magnitude
            except Exception:
                rh_layer = np.nan

            if np.isfinite(rh_layer) and rh_layer >= 0.75:
                cur_contig_depth += 50.0 * units.hPa
                if cur_contig_depth > max_contig_depth:
                    max_contig_depth = cur_contig_depth
            else:
                cur_contig_depth = 0.0 * units.hPa

        if max_contig_depth >= 50.0 * units.hPa:
            is_saturated_anywhere = True

    if is_saturated_anywhere:
        if np.isnan(thickness_850_700) or np.isnan(thickness_1000_850):
            precip_type = "UNKNOWN"
        else:
            # Clamp only for the polygon test, do not alter returned thickness values
            x = float(np.clip(thickness_850_700, X_MIN, X_MAX))
            y = float(np.clip(thickness_1000_850, Y_MIN, Y_MAX))
            pt_point = (x, y)

            if snow_path is not None and snow_path.contains_point(pt_point):
                precip_type = "SNOW"
            elif unknown_path is not None and unknown_path.contains_point(pt_point):
                precip_type = "UNKNOWN"
            elif snow_rain_path is not None and snow_rain_path.contains_point(pt_point):
                precip_type = "SNOW+RAIN"
            elif wintry_mix_path is not None and wintry_mix_path.contains_point(pt_point):
                precip_type = "WINTRY MIX"
            elif rain_path is not None and rain_path.contains_point(pt_point):
                precip_type = "RAIN"
            elif snow_sleet_path is not None and snow_sleet_path.contains_point(pt_point):
                precip_type = "SNOW+SLEET"
            elif sleet_fzra_path is not None and sleet_fzra_path.contains_point(pt_point):
                precip_type = "SLEET+FREEZING RAIN"
            elif fzra_sleet_path is not None and fzra_sleet_path.contains_point(pt_point):
                precip_type = "FREEZING RAIN+SLEET"
            elif fzra_path is not None and fzra_path.contains_point(pt_point):
                precip_type = "FREEZING RAIN"
            else:
                precip_type = "UNKNOWN"
    else:
        precip_type = "NONE"

    # Thermal sanity checks (unchanged)
    try:
        t_c = temperature_profile.to("degC")
        z_m_q = height_profile.to("meter")
        t_vals = t_c.magnitude
        z_vals = z_m_q.magnitude
        t_max = np.nanmax(t_vals)
        mask_low = (z_vals >= 0.0) & (z_vals <= 2000.0)
        if np.any(mask_low):
            t_low_min = np.nanmin(t_vals[mask_low])
        else:
            t_low_min = np.nan
    except Exception:
        t_max = t_low_min = np.nan

    if is_saturated_anywhere:
        if np.isfinite(t_low_min) and t_low_min > 0.5:
            precip_type = "RAIN"
        elif np.isfinite(t_max) and t_max < -0.5:
            precip_type = "SNOW"

    if return_thickness:
        return precip_type, thickness_850_700, thickness_1000_850, is_saturated_anywhere
    else:
        return precip_type


def classify_precip_type_hybrid(
    pressure_profile,
    height_profile,
    temperature_profile,
    dewpoint_profile,
    rh_profile,
    qr_profile,
    qs_profile,
    qg_profile=None,
    qi_profile=None,
):
    """
    Hybrid precip-type classifier (unchanged physics).
    """
    precip_type_thick, thickness_850_700, thickness_1000_850, is_saturated = (
        classify_precip_type_partial_thickness(
            pressure_profile,
            height_profile,
            temperature_profile,
            rh_profile,
            return_thickness=True,
        )
    )

    try:
        Tw_sfc = mpcalc.wet_bulb_temperature(
            pressure_profile[0],
            temperature_profile[0],
            dewpoint_profile[0],
        ).to("degC")
    except Exception:
        Tw_sfc = np.nan * units.degC

    Tw_sfc_scalar = quantity_to_scalar(Tw_sfc)

    # Hydrometeor intent in lowest 5 km (unchanged)
    z_m = height_profile.to("m").magnitude

    def _max_q(q):
        if q is None:
            return 0.0
        q = np.asarray(q, dtype=float)
        mask = z_m <= 5000.0  # extended to 0–5 km AGL
        if not np.any(mask):
            return 0.0
        return float(np.nanmax(q[mask]))

    qr_max = _max_q(qr_profile)
    qs_max = _max_q(qs_profile)
    qg_max = _max_q(qg_profile)
    qi_max = _max_q(qi_profile)

    q_vals = np.array([qr_max, qs_max, qg_max, qi_max])
    labels = np.array(["RAIN", "SNOW", "GRAUPEL", "ICE"])

    if np.all(~np.isfinite(q_vals)):
        dominant_idx = 0
        dominant_q = 0.0
    else:
        dominant_idx = int(np.nanargmax(q_vals))
        dominant_q = q_vals[dominant_idx]

    q_thresh = 5e-8  # kg/kg (unchanged)
    if not np.isfinite(dominant_q) or dominant_q < q_thresh:
        hydro_group = "NONE"
        hydro_label = "NONE"
    else:
        hydro_label = labels[dominant_idx]
        if hydro_label == "RAIN":
            hydro_group = "LIQUID"
        elif hydro_label in ("SNOW", "ICE", "GRAUPEL"):
            hydro_group = "FROZEN"
        else:
            hydro_group = "FROZEN"

    # Simple warm/cold energy areas (unchanged)
    T_c = temperature_profile.to("degC")
    mask_layer = height_profile <= 3000.0 * units.m
    if np.count_nonzero(mask_layer) >= 2:
        T_sel = T_c[mask_layer].magnitude
        z_sel = height_profile[mask_layer].to("m").magnitude
        dz = np.diff(z_sel)
        if len(dz) > 0:
            T_mid = 0.5 * (T_sel[1:] + T_sel[:-1])
            A_plus = float(np.sum(np.clip(T_mid, 0.0, None) * dz))
            A_minus = float(np.sum(np.clip(-T_mid, 0.0, None) * dz))
        else:
            A_plus = 0.0
            A_minus = 0.0
    else:
        A_plus = 0.0
        A_minus = 0.0

    # Final precip decision (unchanged)
    if (not is_saturated) or hydro_group == "NONE":
        try:
            T_c_local = temperature_profile.to("degC")
            z_m_local = height_profile.to("meter")

            low_mask = (z_m_local >= 0 * units.m) & (z_m_local <= 2000 * units.m)
            if np.any(low_mask):
                T_low = T_c_local[low_mask].magnitude
                frac_below_freezing = np.mean(T_low <= 0.0)

                if rh_profile is not None:
                    rh_low = rh_profile[low_mask]
                    rh_low_mean = np.mean(rh_low.magnitude)
                else:
                    rh_low_mean = np.nan

                if frac_below_freezing > 0.7 and (
                    np.isnan(rh_low_mean) or rh_low_mean >= 0.75
                ):
                    precip_final = "SNOW"
                else:
                    precip_final = "NONE"
            else:
                precip_final = "NONE"
        except Exception:
            precip_final = "NONE"

        return (
            precip_final,
            precip_type_thick,
            thickness_850_700,
            thickness_1000_850,
            is_saturated,
            Tw_sfc,
        )

    warm_sfc = np.isfinite(Tw_sfc_scalar) and (Tw_sfc_scalar >= 1.0)
    cold_sfc = np.isfinite(Tw_sfc_scalar) and (Tw_sfc_scalar <= -1.0)

    if hydro_group == "LIQUID":
        if warm_sfc:
            precip_final = "RAIN"
        elif cold_sfc:
            precip_final = "FREEZING RAIN"
        else:
            precip_final = "RAIN/FREEZING RAIN"

        return (
            precip_final,
            precip_type_thick,
            thickness_850_700,
            thickness_1000_850,
            is_saturated,
            Tw_sfc,
        )

    small_A = A_plus < 500.0
    if small_A:
        if warm_sfc and not cold_sfc:
            precip_final = "RAIN/SNOW"
        else:
            precip_final = "SNOW"

        return (
            precip_final,
            precip_type_thick,
            thickness_850_700,
            thickness_1000_850,
            is_saturated,
            Tw_sfc,
        )

    ratio = A_minus / A_plus if A_plus > 0.0 else 0.0
    if ratio >= 0.75:
        precip_final = "SLEET"
    elif ratio >= 0.25:
        precip_final = "SLEET/FREEZING RAIN"
    else:
        if cold_sfc:
            precip_final = "FREEZING RAIN"
        elif warm_sfc:
            precip_final = "RAIN"
        else:
            precip_final = "RAIN/FREEZING RAIN"

    return (
        precip_final,
        precip_type_thick,
        thickness_850_700,
        thickness_1000_850,
        is_saturated,
        Tw_sfc,
    )


def draw_partial_thickness_nomogram(
    ax,
    thickness_850_700,
    thickness_1000_850,
    precip_type_thick=None,
    show_clamped=True,
):
    """
    Draw a partial-thickness nomogram that matches the classifier polygons.

    x-axis: 850–700 hPa thickness (m)
    y-axis: 1000–850 hPa thickness (m)
    """
    poly_defs = get_partial_thickness_polygon_defs()

    X_MIN, X_MAX = PT_X_MIN, PT_X_MAX
    Y_MIN, Y_MAX = PT_Y_MIN, PT_Y_MAX

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("850–700 hPa thickness (m)")
    ax.set_ylabel("1000–850 hPa thickness (m)")
    ax.grid(True, linewidth=0.6, alpha=0.6, linestyle="--")

    face = {
        "snow": (0.2, 0.4, 1.0, 0.10),
        "unknown": (0.0, 0.0, 0.0, 0.06),
        "snow_rain": (0.2, 0.8, 0.9, 0.10),
        "wintry_mix": (0.6, 0.6, 0.6, 0.10),
        "rain": (0.2, 0.9, 0.2, 0.10),
        "snow_sleet": (0.6, 0.2, 0.9, 0.10),
        "sleet_fzra": (1.0, 0.5, 0.0, 0.10),
        "fzra_sleet": (1.0, 0.3, 0.3, 0.10),
        "fzra": (1.0, 0.0, 0.0, 0.08),
    }

    edge = {
        "snow": "blue",
        "unknown": "black",
        "snow_rain": "teal",
        "wintry_mix": "gray",
        "rain": "green",
        "snow_sleet": "slategray",
        "sleet_fzra": "darkorange",
        "fzra_sleet": "crimson",
        "fzra": "red",
    }

    for name, verts in poly_defs.items():
        v = np.asarray(verts, dtype=float)
        ax.add_patch(
            Polygon(
                v,
                closed=True,
                facecolor=face.get(name, (0, 0, 0, 0.0)),
                edgecolor=edge.get(name, "black"),
                linewidth=1.0,
                zorder=1,
            )
        )

    labels = {
        "snow": (1527.0, 1287.0, "Snow", dict(ha="left", va="bottom")),
        "unknown": (1525.0, 1314.0, "Unknown", dict(ha="left", va="top")),
        "snow_rain": (1536.0, 1300.0, "Snow +\nRain", dict(ha="center", va="center")),
        "wintry_mix": (1545.0, 1300.0, "Wintry\nMix", dict(ha="center", va="center")),
        "rain": (1554.0, 1314.0, "Rain", dict(ha="right", va="top")),
        "snow_sleet": (1544.0, 1285.0, "Snow+\nSleet", dict(ha="center", va="center")),
        "sleet_fzra": (1555.0, 1285.0, "Sleet+\nFrz. Rain", dict(ha="center", va="center")),
        "fzra_sleet": (1555.0, 1295.0, "Frz. Rain+\nSleet", dict(ha="center", va="center")),
        "fzra": (1555.0, 1305.0, "Frz. Rain", dict(ha="center", va="center")),
    }
    for _, (x, y, text, kw) in labels.items():
        ax.text(x, y, text, fontsize=8, zorder=2, alpha=0.9, **kw)

    x_raw = float(thickness_850_700) if np.isfinite(thickness_850_700) else np.nan
    y_raw = float(thickness_1000_850) if np.isfinite(thickness_1000_850) else np.nan

    if np.isfinite(x_raw) and np.isfinite(y_raw):
        ax.plot(
            x_raw,
            y_raw,
            marker="o",
            markersize=5,
            linestyle="None",
            zorder=10,
            label="Raw",
        )

        if show_clamped:
            x_c = float(np.clip(x_raw, X_MIN, X_MAX))
            y_c = float(np.clip(y_raw, Y_MIN, Y_MAX))
            ax.plot(
                x_c,
                y_c,
                marker="*",
                markersize=9,
                linestyle="None",
                zorder=11,
                label="Clamped",
            )


    title = "Partial-thickness nomogram"
    if precip_type_thick:
        title += f"\nClassified: {precip_type_thick}"
    ax.set_title(title, fontsize=10)



# =============================================================================
# HAZARD CLASSIFICATION (unchanged physics)
# =============================================================================


def classify_hazard(
    sb_cape,
    ml_cape,
    mu_cape,
    srh_0to1,
    srh_0to3,
    shear_0to6,
    pwat,
    dcape,
    surface_temp_c,
    surface_dew_c,
    surface_u,
    surface_v,
    precip_type,
):
    """
    Very simple rule-based hazard classifier.
    Thresholds are tunable if desired.
    """
    cape_values = [
        quantity_to_scalar(sb_cape),
        quantity_to_scalar(ml_cape),
        quantity_to_scalar(mu_cape),
    ]
    cape_max = np.nanmax(cape_values)

    s1 = quantity_to_scalar(srh_0to1)
    s3 = quantity_to_scalar(srh_0to3)
    shear_6_val = quantity_to_scalar(shear_0to6.to("knot"))

    pwat_val = quantity_to_scalar(pwat)
    dcape_val = quantity_to_scalar(dcape)
    downdraft_wmax = (
        np.sqrt(np.maximum(dcape_val, 0.0)) * np.sqrt(2.0)
        if np.isfinite(dcape_val)
        else np.nan
    )

    try:
        surface_speed = mpcalc.wind_speed(surface_u, surface_v).to("kt")
        surface_speed_val = quantity_to_scalar(surface_speed)
    except Exception:
        surface_speed_val = np.nan

    try:
        surface_rh = mpcalc.relative_humidity_from_dewpoint(
            surface_temp_c, surface_dew_c
        )
        app_temp = mpcalc.apparent_temperature(
            surface_temp_c, surface_rh, mpcalc.wind_speed(surface_u, surface_v)
        ).to("degF")
        app_temp_val = quantity_to_scalar(app_temp)
    except Exception:
        app_temp_val = np.nan

    pt = (precip_type or "").upper()

    if "SNOW" in pt:
        if surface_speed_val >= 35:
            return "BLIZZARD"
        elif surface_speed_val >= 20:
            return "SNOW / BLOWING SNOW"
        else:
            return "SNOW"

    if cape_max >= 2000 and (s1 >= 150 or s3 >= 250) and shear_6_val >= 45:
        return "PDS TOR"
    if cape_max >= 1000 and (s1 >= 100 or s3 >= 150) and shear_6_val >= 35:
        return "TOR"
    if cape_max >= 500 and (s1 >= 75 or s3 >= 100) and shear_6_val >= 30:
        return "MRGL TOR"

    if dcape_val >= 1000 and downdraft_wmax >= 20:
        return "SVR WIND"
    if dcape_val >= 750 and downdraft_wmax >= 15:
        return "MRGL WIND"

    if cape_max >= 1500 and shear_6_val <= 25:
        return "HAIL"
    if cape_max >= 1000 and shear_6_val <= 20:
        return "MRGL HAIL"

    if pwat_val >= 2.0 and surface_speed_val <= 20:
        return "FLASH FLOOD"

    if app_temp_val >= 105:
        return "EXCESSIVE HEAT"
    if app_temp_val <= -20:
        return "WIND CHILL"

    if cape_max >= 800 and shear_6_val >= 40:
        return "SVR"
    if cape_max >= 300:
        return "MRGL SVR"
    if cape_max >= 100 and shear_6_val >= 20:
        return "PULSE"
    if cape_max >= 100:
        return "TSTM"

    return "NONE"


def hazard_color(label: str) -> str:
    """Map a hazard label string to a text color."""
    if not label:
        return "black"

    l = label.upper()

    if "PDS TOR" in l:
        return "magenta"
    if "TOR" in l:
        return "red"

    if "SVR WIND" in l or "SVR" in l or "HAIL" in l:
        return "red"
    if "MRGL" in l or "PULSE" in l or "TSTM" in l:
        return "darkorange"

    if "BLIZZARD" in l:
        return "purple"
    if "SNOW" in l or "WINTER" in l or "WIND CHILL" in l:
        return "dodgerblue"

    if "FLOOD" in l:
        return "blue"

    if "HEAT" in l:
        return "darkred"

    return "black"


# =============================================================================
# HODOGRAPH PLOTTING (unchanged)
# =============================================================================


def plot_hodograph(ax, height_profile, u_profile, v_profile, pressure_profile):
    """Draw hodograph with colored height segments and Bunkers storm motions."""
    u_kts = u_profile.to("knot").magnitude
    v_kts = v_profile.to("knot").magnitude
    z_m = height_profile.to("meter").magnitude

    mask = np.isfinite(u_kts) & np.isfinite(v_kts) & np.isfinite(z_m)
    if np.count_nonzero(mask) < 2:
        return

    u_kts = u_kts[mask]
    v_kts = v_kts[mask]
    z_m = z_m[mask]

    component_range = 100.0
    ax.cla()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-component_range, component_range)
    ax.set_ylim(-component_range, component_range)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    radii = np.arange(10, component_range + 1e-6, 10)
    theta = np.linspace(0, 2 * np.pi, 361)
    for r in radii:
        linestyle = "-" if (r % 20 == 0) else "--"
        alpha_val = 0.35 if (r % 20 == 0) else 0.2
        ax.plot(
            r * np.cos(theta),
            r * np.sin(theta),
            linestyle=linestyle,
            linewidth=1.0,
            alpha=alpha_val,
            color="black",
        )

    ax.axhline(0, linewidth=0.5, color="black", alpha=0.3)
    ax.axvline(0, linewidth=0.5, color="black", alpha=0.3)

    for r in range(20, int(component_range) + 1, 20):
        ax.text(r, 0, str(r), fontsize=11, ha="left", va="bottom", alpha=0.5)
        ax.text(0, r, str(r), fontsize=11, ha="left", va="bottom", alpha=0.5)

    def segment_color(z_mid_m):
        z_km = z_mid_m / 1000.0
        if z_km < 1.0:
            return "fuchsia"
        elif z_km < 3.0:
            return "firebrick"
        elif z_km < 6.0:
            return "limegreen"
        elif z_km < 9.0:
            return "goldenrod"
        elif z_km < 12.0:
            return "turquoise"
        else:
            return "darkgray"

    for i in range(1, len(u_kts)):
        z_mid = 0.5 * (z_m[i] + z_m[i - 1])
        col = segment_color(z_mid)
        ax.plot(
            [u_kts[i - 1], u_kts[i]],
            [v_kts[i - 1], v_kts[i]],
            color=col,
            linewidth=2.5,
            zorder=3,
        )

    heights_to_sample = [
        100,
        500,
        1000,
        1500,
        2000,
        3000,
        4000,
        5000,
        6000,
        8000,
        10000,
        12000,
    ]
    for h_m in heights_to_sample:
        window = 250.0
        m = (z_m >= (h_m - window)) & (z_m <= (h_m + window))
        if np.count_nonzero(m) == 0:
            continue

        u_mean = np.mean(u_kts[m])
        v_mean = np.mean(v_kts[m])
        if not (np.isfinite(u_mean) and np.isfinite(v_mean)):
            continue

        label_km = h_m / 1000.0
        lbl = f"{label_km:.1f}" if label_km < 1.0 else f"{label_km:.0f}"
        ax.text(
            u_mean,
            v_mean,
            lbl,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            alpha=0.7,
            zorder=4,
        )

    ax.scatter(0, 0, s=20, color="black", zorder=4)
    ax.text(
        0.0,
        0.0,
        "SFC",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
        alpha=0.8,
    )

    try:
        rm_vec, lm_vec, mean_vec = mpcalc.bunkers_storm_motion(
            pressure_profile, u_profile, v_profile, height_profile
        )
        vectors = [
            ("RM", rm_vec, "red"),
            ("LM", lm_vec, "navy"),
            ("MW", mean_vec, "black"),
        ]
        for label, vec, col in vectors:
            u_vec = vec[0].to("knot").magnitude
            v_vec = vec[1].to("knot").magnitude
            if not (np.isfinite(u_vec) and np.isfinite(v_vec)):
                continue

            ax.arrow(
                0.0,
                0.0,
                u_vec,
                v_vec,
                linewidth=2.0,
                color=col,
                alpha=0.5,
                length_includes_head=True,
                head_width=2.5,
                zorder=5,
            )
            ax.text(
                u_vec + 1.0,
                v_vec - 1.0,
                label,
                fontsize=10,
                fontweight="bold",
                color=col,
                alpha=0.8,
                ha="left",
                va="center",
                zorder=5,
            )
    except Exception:
        pass


# =============================================================================
# CARTOPY MAP FEATURE HELPERS (unchanged)
# =============================================================================


def add_feature(
    ax, category, scale, facecolor, edgecolor, linewidth, name, zorder=None, alpha=None
):
    """Thin wrapper around NaturalEarthFeature for convenience."""
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


map_features = [
    ("physical", "10m", cfeature.COLORS["land"], "black", 0.50, "minor_islands"),
    ("physical", "10m", "none", "black", 0.50, "coastline"),
    ("cultural", "10m", "none", "red", 0.80, "roads", 2),
    ("physical", "10m", cfeature.COLORS["water"], None, None, "ocean_scale_rank", -1),
    ("physical", "10m", cfeature.COLORS["water"], "lightgrey", 0.75, "lakes", 0),
    ("cultural", "10m", "none", "grey", 1.00, "admin_1_states_provinces", 2),
    ("cultural", "10m", "none", "black", 1.50, "admin_0_countries", 2),
]

try:
    cities = gpd.read_file(
        "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
    )
except Exception as e:
    print(f"Warning loading city shapefile: {e}")
    cities = gpd.GeoDataFrame()


# =============================================================================
# WRF DATA EXTRACTION HELPERS (updated for time_index)
# =============================================================================


def extract_vertical_profiles(wrf_handle, point_lat, point_lon, time_index):
    """Extract vertical column at given lat/lon from a WRF output file."""
    xy_idx = wrf.ll_to_xy(wrf_handle, point_lat, point_lon)

    p3 = wrf.getvar(wrf_handle, "pres", timeidx=time_index, units="hPa")
    t3 = wrf.getvar(wrf_handle, "temp", timeidx=time_index, units="degC")
    td3 = wrf.getvar(wrf_handle, "td", timeidx=time_index, units="degC")
    u3 = wrf.getvar(wrf_handle, "ua", timeidx=time_index, units="kt")
    v3 = wrf.getvar(wrf_handle, "va", timeidx=time_index, units="kt")
    z3 = wrf.getvar(wrf_handle, "height_agl", timeidx=time_index, units="m")

    try:
        qr3 = wrf.getvar(wrf_handle, "QRAIN", timeidx=time_index)
    except Exception:
        qr3 = None
    try:
        qs3 = wrf.getvar(wrf_handle, "QSNOW", timeidx=time_index)
    except Exception:
        qs3 = None
    try:
        qg3 = wrf.getvar(wrf_handle, "QGRAUP", timeidx=time_index)
    except Exception:
        qg3 = None
    try:
        qi3 = wrf.getvar(wrf_handle, "QICE", timeidx=time_index)
    except Exception:
        qi3 = None

    j, i = xy_idx[1], xy_idx[0]

    prs_profile = p3[:, j, i].values * units.hectopascal
    temp_profile = t3[:, j, i].values * units.degC
    dew_profile = td3[:, j, i].values * units.degC
    u_profile = u3[:, j, i].values * units.knot
    v_profile = v3[:, j, i].values * units.knot
    z_profile = z3[:, j, i].values * units.meter

    def _extract_q(q3):
        if q3 is None:
            return None
        return q3[:, j, i].values

    qr_profile = _extract_q(qr3)
    qs_profile = _extract_q(qs3)
    qg_profile = _extract_q(qg3)
    qi_profile = _extract_q(qi3)

    sort_idx = np.argsort(prs_profile.magnitude)[::-1]
    prs_profile = prs_profile[sort_idx]
    temp_profile = temp_profile[sort_idx]
    dew_profile = dew_profile[sort_idx]
    u_profile = u_profile[sort_idx]
    v_profile = v_profile[sort_idx]
    z_profile = z_profile[sort_idx]
    if qr_profile is not None:
        qr_profile = qr_profile[sort_idx]
    if qs_profile is not None:
        qs_profile = qs_profile[sort_idx]
    if qg_profile is not None:
        qg_profile = qg_profile[sort_idx]
    if qi_profile is not None:
        qi_profile = qi_profile[sort_idx]

    return (
        prs_profile,
        temp_profile,
        dew_profile,
        u_profile,
        v_profile,
        z_profile,
        qr_profile,
        qs_profile,
        qg_profile,
        qi_profile,
    )


# =============================================================================
# PANEL DRAWING HELPERS (unchanged)
# =============================================================================


def draw_title_panel(fig, layout, city_name, point_lat, point_lon, valid_dt):
    """Draw top title bar with model info and valid time."""
    ax_title = fig.add_axes(layout["title"])
    ax_title.set_axis_off()

    if valid_dt is not None:
        valid_str = valid_dt.strftime("Valid: %H:%M:%SZ %Y-%m-%d")
    else:
        valid_str = "Valid: Unknown Time"

    left_title = (
        "Weather Research and Forecasting Model\n"
        f"Model Skew-T Plot at {city_name} Lat: {point_lat:.2f}, Lon: {point_lon:.2f}\n"
        "Hodograph\n"
        "Severe Weather Parameters"
    )
    ax_title.text(
        0.0,
        0.95,
        left_title,
        fontsize=13,
        ha="left",
        va="top",
        transform=ax_title.transAxes,
    )
    ax_title.text(
        1.0,
        0.95,
        valid_str,
        fontsize=13,
        ha="right",
        va="top",
        transform=ax_title.transAxes,
    )


def draw_hazard_panel(
    fig,
    layout,
    sb_cape,
    ml_cape,
    mu_cape,
    srh_0to1,
    srh_0to3,
    shear_0to6,
    pwat,
    dcape,
    surface_temperature,
    surface_dewpoint,
    surface_u,
    surface_v,
    precip_type_final,
    precip_type_thickness,
    thickness_850_700,
    thickness_1000_850,
    is_saturated,
):
    """
    Draw hazard + precip-type panel using hybrid precip_type_final
    and thickness-based type as secondary note.
    """
    ax_hazard = fig.add_axes(layout["hazard"])
    ax_hazard.set_position(layout["hazard"])
    ax_hazard.set_xlim(0, 1)
    ax_hazard.set_ylim(0, 1)
    ax_hazard.axis("off")

    ax_hazard.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax_hazard.transAxes,
            edgecolor="black",
            facecolor="white",
            linewidth=1.0,
        )
    )

    hazard_label = classify_hazard(
        sb_cape,
        ml_cape,
        mu_cape,
        srh_0to1,
        srh_0to3,
        shear_0to6,
        pwat,
        dcape,
        surface_temperature,
        surface_dewpoint,
        surface_u,
        surface_v,
        precip_type_final,
    )
    hazard_color_val = hazard_color(hazard_label)

    ax_hazard.text(
        0.5,
        0.95,
        "Possible\nHazard Type:",
        ha="center",
        va="top",
        fontsize=12,
        color="black",
        fontweight="bold",
        transform=ax_hazard.transAxes,
    )
    ax_hazard.text(
        0.5,
        0.85,
        hazard_label,
        ha="center",
        va="top",
        fontsize=18,
        color=hazard_color_val,
        fontweight="bold",
        transform=ax_hazard.transAxes,
    )

    if (
        (not is_saturated)
        or (precip_type_final is None)
        or (precip_type_final.upper() == "NONE")
    ):
        precip_text = "NO PRECIP\n(no deep saturated column)"
    else:
        precip_text = precip_type_final
        if precip_type_thickness is not None:
            pt_thick_up = (precip_type_thickness or "").upper()
            pt_final_up = (precip_type_final or "").upper()
            if (
                pt_thick_up not in ("", "NONE", "UNKNOWN", "UNKNOWNUNKNOWN")
                and pt_thick_up != pt_final_up
            ):
                precip_text += f"\n(thickness: {precip_type_thickness})"

    ax_hazard.text(
        0.5,
        0.75,
        "Precip Type:",
        ha="center",
        va="top",
        fontsize=12,
        color="black",
        fontweight="bold",
        transform=ax_hazard.transAxes,
    )
    ax_hazard.text(
        0.5,
        0.70,
        precip_text,
        ha="center",
        va="top",
        fontsize=18,
        color="green",
        transform=ax_hazard.transAxes,
    )
    #Partial-Thickness grid box
    ax_nom = ax_hazard.inset_axes([0.06, 0.12, 0.88, 0.40])
    draw_partial_thickness_nomogram(
        ax_nom,
        thickness_850_700,
        thickness_1000_850,
        precip_type_thick=precip_type_thickness,
        show_clamped=True,
    )
    ax_nom.tick_params(labelsize=7)


# =============================================================================
# FRAME DISCOVERY (multi-file + multi-time)
# =============================================================================


def discover_frames(ncfile_paths):
    """
    Discover all (file, time_index) combinations to plot.

    Supports:
        * Many wrfout_<domain>* files, each with one or more Time steps.
        * A single wrfout file containing multiple Time steps.
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


# =============================================================================
# MAIN WRF PROCESSING FOR ONE FRAME
# =============================================================================


def process_frame(args):
    """
    Extract a sounding from one (file, time_index) and render the full figure.

    Parameters
    ----------
    args : tuple
        (ncfile_path, time_index, domain, city_name, point_lat, point_lon, path_figures)
    """
    (
        ncfile_path,
        time_index,
        domain,
        city_name,
        point_lat,
        point_lon,
        path_figures,
    ) = args

    wrf_handle = Dataset(ncfile_path)

    try:
        file_name = os.path.basename(ncfile_path)
        valid_dt = get_valid_time(wrf_handle, ncfile_path, time_index)

        if valid_dt is not None:
            print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")
        else:
            print(f"Plotting data: unknown time ({file_name}, t={time_index})")

        (
            prs_profile,
            temp_profile,
            dew_profile,
            u_profile,
            v_profile,
            z_profile,
            qr_profile,
            qs_profile,
            qg_profile,
            qi_profile,
        ) = extract_vertical_profiles(wrf_handle, point_lat, point_lon, time_index)

        rh = mpcalc.relative_humidity_from_dewpoint(temp_profile, dew_profile)

        try:
            pwat = mpcalc.precipitable_water(prs_profile, dew_profile).to("inch")
        except Exception:
            pwat = np.nan * units.inch

        surface_temperature = temp_profile[0]
        surface_dewpoint = dew_profile[0]
        surface_u = u_profile[0]
        surface_v = v_profile[0]

        try:
            ccl_p, ccl_t, conv_temp_c = mpcalc.ccl(
                prs_profile, temp_profile, dew_profile
            )
        except Exception:
            ccl_p = np.nan * units.hectopascal
            ccl_t = np.nan * units.degC
            conv_temp_c = np.nan * units.degC

        sb_parcel_profile = mpcalc.parcel_profile(
            prs_profile, temp_profile[0], dew_profile[0]
        ).to("degC")
        sb_cape, sb_cin = mpcalc.cape_cin(
            prs_profile, temp_profile, dew_profile, parcel_profile=sb_parcel_profile
        )
        sb_lcl_p, sb_lcl_t = mpcalc.lcl(prs_profile[0], temp_profile[0], dew_profile[0])
        sb_lfc_p, sb_lfc_t = mpcalc.lfc(
            prs_profile,
            temp_profile,
            dew_profile,
            parcel_temperature_profile=sb_parcel_profile,
        )
        sb_el_p, sb_el_t = mpcalc.el(
            prs_profile,
            temp_profile,
            dew_profile,
            parcel_temperature_profile=sb_parcel_profile,
        )

        ml_temp, ml_dew = mpcalc.mixed_layer(
            prs_profile, temp_profile, dew_profile, depth=100 * units.hPa
        )
        ml_pressure, _, _ = mpcalc.mixed_parcel(
            prs_profile, temp_profile, dew_profile, depth=100 * units.hPa
        )
        ml_parcel_profile = mpcalc.parcel_profile(prs_profile, ml_temp, ml_dew).to(
            "degC"
        )
        ml_cape, ml_cin = mpcalc.cape_cin(
            prs_profile, temp_profile, dew_profile, parcel_profile=ml_parcel_profile
        )
        ml_lcl_p, ml_lcl_t = mpcalc.lcl(ml_pressure, ml_temp, ml_dew)
        ml_lfc_p, ml_lfc_t = mpcalc.lfc(
            prs_profile,
            temp_profile,
            dew_profile,
            parcel_temperature_profile=ml_parcel_profile,
        )
        ml_el_p, ml_el_t = mpcalc.el(
            prs_profile,
            temp_profile,
            dew_profile,
            parcel_temperature_profile=ml_parcel_profile,
        )

        mu_pressure, mu_temp, mu_dew, _ = mpcalc.most_unstable_parcel(
            prs_profile, temp_profile, dew_profile, depth=300 * units.hPa
        )
        mu_parcel_profile = mpcalc.parcel_profile(prs_profile, mu_temp, mu_dew).to(
            "degC"
        )
        mu_cape, mu_cin = mpcalc.cape_cin(
            prs_profile, temp_profile, dew_profile, parcel_profile=mu_parcel_profile
        )
        mu_lcl_p, mu_lcl_t = mpcalc.lcl(mu_pressure, mu_temp, mu_dew)
        mu_lfc_p, mu_lfc_t = mpcalc.lfc(
            prs_profile,
            temp_profile,
            dew_profile,
            parcel_temperature_profile=mu_parcel_profile,
        )
        mu_el_p, mu_el_t = mpcalc.el(
            prs_profile,
            temp_profile,
            dew_profile,
            parcel_temperature_profile=mu_parcel_profile,
        )

        sb_cape_0_3 = compute_layer_cape(
            prs_profile, temp_profile, dew_profile, z_profile, 3 * units.km
        )
        ml_cape_0_3 = compute_layer_cape(
            prs_profile, temp_profile, dew_profile, z_profile, 3 * units.km
        )
        mu_cape_0_3 = compute_layer_cape(
            prs_profile, temp_profile, dew_profile, z_profile, 3 * units.km
        )

        k_index = mpcalc.k_index(prs_profile, temp_profile, dew_profile)
        total_totals_index = mpcalc.total_totals_index(
            prs_profile, temp_profile, dew_profile
        )

        if is_finite_scalar(sb_lcl_p):
            new_p = (
                np.append(
                    prs_profile.magnitude[prs_profile.magnitude > sb_lcl_p.magnitude],
                    sb_lcl_p.magnitude,
                )
                * units.hPa
            )
            new_t = (
                np.append(
                    temp_profile.magnitude[prs_profile.magnitude > sb_lcl_p.magnitude],
                    sb_lcl_t.magnitude,
                )
                * units.degC
            )
            lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)
        else:
            lcl_height = np.nan * units.meter

        lifted_index = mpcalc.lifted_index(prs_profile, temp_profile, sb_parcel_profile)
        if prs_profile[0] < 850 * units.mbar:
            showalter_index = np.nan
        else:
            showalter_index = mpcalc.showalter_index(
                prs_profile, temp_profile, dew_profile
            )

        try:
            wspd_profile = mpcalc.wind_speed(u_profile, v_profile).to("knot")
            wdir_profile = mpcalc.wind_direction(u_profile, v_profile)
            sweat_index = mpcalc.sweat_index(
                prs_profile,
                temp_profile,
                dew_profile,
                wspd_profile,
                wdir_profile,
            )
        except Exception:
            sweat_index = np.nan * units.dimensionless

        (rm_vec, lm_vec, mean_vec) = mpcalc.bunkers_storm_motion(
            prs_profile, u_profile, v_profile, z_profile
        )
        bunkers_rm_u, bunkers_rm_v = rm_vec
        bunkers_lm_u, bunkers_lm_v = lm_vec
        bunkers_mw_u, bunkers_mw_v = mean_vec

        *_, srh_0to1 = mpcalc.storm_relative_helicity(
            z_profile,
            u_profile,
            v_profile,
            depth=1 * units.km,
            storm_u=bunkers_rm_u,
            storm_v=bunkers_rm_v,
        )
        *_, srh_0to3 = mpcalc.storm_relative_helicity(
            z_profile,
            u_profile,
            v_profile,
            depth=3 * units.km,
            storm_u=bunkers_rm_u,
            storm_v=bunkers_rm_v,
        )
        *_, srh_0to6 = mpcalc.storm_relative_helicity(
            z_profile,
            u_profile,
            v_profile,
            depth=6 * units.km,
            storm_u=bunkers_rm_u,
            storm_v=bunkers_rm_v,
        )

        bunkers_speed = mpcalc.wind_speed(bunkers_rm_u, bunkers_rm_v).to("knot")
        bunkers_dir = mpcalc.wind_direction(bunkers_rm_u, bunkers_rm_v)

        bunkers_rm_speed = mpcalc.wind_speed(bunkers_rm_u, bunkers_rm_v).to("knot")
        bunkers_rm_dir = mpcalc.wind_direction(bunkers_rm_u, bunkers_rm_v)
        bunkers_lm_speed = mpcalc.wind_speed(bunkers_lm_u, bunkers_lm_v).to("knot")
        bunkers_lm_dir = mpcalc.wind_direction(bunkers_lm_u, bunkers_lm_v)
        bunkers_mw_speed = mpcalc.wind_speed(bunkers_mw_u, bunkers_mw_v).to("knot")
        bunkers_mw_dir = mpcalc.wind_direction(bunkers_mw_u, bunkers_mw_v)

        u_shear_3, v_shear_3 = mpcalc.bulk_shear(
            prs_profile, u_profile, v_profile, height=z_profile, depth=3 * units.km
        )
        shear_0to3 = mpcalc.wind_speed(u_shear_3, v_shear_3)

        u_shear_6, v_shear_6 = mpcalc.bulk_shear(
            prs_profile, u_profile, v_profile, height=z_profile, depth=6 * units.km
        )
        shear_0to6 = mpcalc.wind_speed(u_shear_6, v_shear_6)

        sig_tornado_param = mpcalc.significant_tornado(
            sb_cape, lcl_height, srh_0to3, shear_0to3
        )
        supercell_composite = mpcalc.supercell_composite(mu_cape, srh_0to3, shear_0to6)

        ml_cape_val = quantity_to_scalar(ml_cape)
        srh_0to1_val = quantity_to_scalar(srh_0to1)
        srh_0to3_val = quantity_to_scalar(srh_0to3)

        if np.isfinite(ml_cape_val) and np.isfinite(srh_0to1_val):
            ehi_0_1 = (ml_cape_val * srh_0to1_val) / 160000.0
        else:
            ehi_0_1 = np.nan

        if np.isfinite(ml_cape_val) and np.isfinite(srh_0to3_val):
            ehi_0_3 = (ml_cape_val * srh_0to3_val) / 160000.0
        else:
            ehi_0_3 = np.nan

        dcape = compute_dcape(prs_profile, temp_profile, dew_profile)

        (
            precip_type,
            precip_type_thick,
            thickness_850_700,
            thickness_1000_850,
            is_saturated,
            Tw_sfc,
        ) = classify_precip_type_hybrid(
            prs_profile,
            z_profile,
            temp_profile,
            dew_profile,
            rh,
            qr_profile,
            qs_profile,
            qg_profile,
            qi_profile,
        )

        # Figure + layout
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        fig.patch.set_facecolor("white")

        layout = get_default_panel_layout()
        for name, rect in layout.items():
            if name == "title":
                continue
            add_panel_frame(fig, rect)

        draw_title_panel(fig, layout, city_name, point_lat, point_lon, valid_dt)

        # ------------------------------------------------------------------
        # Skew-T panel (unchanged physics/plotting)
        # ------------------------------------------------------------------
        skew_plot = SkewT(fig, rotation=45, rect=layout["skewt"])
        skew_plot.ax.set_ylim(1070, 100)
        skew_plot.ax.set_xlim(-50, 50)
        skew_plot.ax.set_xlabel("Temperature ($^\\circ$C)")
        skew_plot.ax.set_ylabel("Pressure (hPa)")
        skew_plot.ax.set_facecolor("whitesmoke")

        if SHOW_LAYOUT_GRID:
            skew_plot.ax.add_patch(
                Rectangle(
                    (0.0, 0.0),
                    1.0,
                    1.0,
                    transform=skew_plot.ax.transAxes,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.3,
                    zorder=50,
                )
            )

        skew_plot.ax.set_yscale("log")
        skew_plot.ax.set_yticks(
            [1000, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100]
        )
        skew_plot.ax.get_yaxis().set_major_formatter(ScalarFormatter())
        skew_plot.ax.get_yaxis().set_minor_formatter(ScalarFormatter())
        skew_plot.ax.tick_params(which="both", direction="in")

        x1_vals = np.linspace(-100, 40, 8)
        x2_vals = np.linspace(-90, 50, 8)
        y_range = [1100, 50]
        for idx in range(0, 8):
            skew_plot.shade_area(
                y=y_range,
                x1=x1_vals[idx],
                x2=x2_vals[idx],
                color="gray",
                alpha=0.02,
                zorder=1,
            )

        z_m_arr = z_profile.to("meter").magnitude
        p_hpa_arr = prs_profile.to("hPa").magnitude
        sort_idx_h = np.argsort(z_m_arr)
        z_for_interp = z_m_arr[sort_idx_h]
        p_for_interp = p_hpa_arr[sort_idx_h]

        height_km_levels = [0, 1, 3, 6, 9, 12, 15]
        height_to_pressure = {}
        height_to_pressure[0] = p_for_interp[0]
        for h_km in height_km_levels[1:]:
            z_target = h_km * 1000.0
            if z_target < z_for_interp[0] or z_target > z_for_interp[-1]:
                continue
            height_to_pressure[h_km] = np.interp(z_target, z_for_interp, p_for_interp)

        bands = [
            (0, 1, "fuchsia"),
            (1, 3, "firebrick"),
            (3, 6, "limegreen"),
            (6, 9, "goldenrod"),
            (9, 12, "turquoise"),
            (12, 15, "darkgray"),
        ]
        for bottom_km, top_km, color in bands:
            if bottom_km in height_to_pressure and top_km in height_to_pressure:
                skew_plot.ax.axhspan(
                    height_to_pressure[top_km],
                    height_to_pressure[bottom_km],
                    xmin=0.0,
                    xmax=0.06,
                    facecolor=color,
                    alpha=0.16,
                    edgecolor=color,
                    linewidth=0.0,
                    zorder=0,
                )

        for h_km in height_km_levels:
            if h_km in height_to_pressure:
                p_here = height_to_pressure[h_km]
                if h_km == 0:
                    label_text = f"SFC: {p_here:.1f} hPa"
                else:
                    kft = h_km * 3.28084
                    label_text = f"{h_km} km / {kft:.2f} kft: {p_here:.1f} hPa"

                skew_plot.ax.text(
                    0.02,
                    p_here,
                    label_text,
                    fontsize=11,
                    ha="left",
                    va="center",
                    color="black",
                    transform=skew_plot.ax.get_yaxis_transform(),
                    zorder=5,
                )

        skew_plot.plot(prs_profile, temp_profile, "r", lw=2, label="TEMPERATURE")
        skew_plot.plot(prs_profile, dew_profile, "g", lw=2, label="DEWPOINT")
        plot_skewt_wind_barbs(skew_plot, prs_profile, u_profile, v_profile)

        skew_plot.ax.axvline(0 * units.degC, linestyle="--", color="blue", alpha=0.5)
        skew_plot.plot_dry_adiabats(lw=1, alpha=0.4)
        skew_plot.plot_moist_adiabats(lw=1, alpha=0.4)
        skew_plot.plot_mixing_lines(lw=1, alpha=0.4)

        if has_buoyancy(sb_cape, sb_cin):
            skew_plot.plot(
                prs_profile,
                sb_parcel_profile,
                "darkorange",
                lw=1.5,
                ls="--",
                label="SB PARCEL",
            )
        if has_buoyancy(mu_cape, mu_cin):
            skew_plot.plot(
                prs_profile,
                mu_parcel_profile,
                "red",
                lw=1.5,
                ls="--",
                label="MU PARCEL",
            )
        if has_buoyancy(ml_cape, ml_cin):
            skew_plot.plot(
                prs_profile,
                ml_parcel_profile,
                "gold",
                lw=1.5,
                ls="--",
                label="ML PARCEL",
            )

        try:
            if has_buoyancy(sb_cape, sb_cin):
                skew_plot.shade_cape(
                    prs_profile,
                    temp_profile.to("degC"),
                    sb_parcel_profile.to("degC"),
                    alpha=0.35,
                    label="SBCAPE",
                )
                skew_plot.shade_cin(
                    prs_profile,
                    temp_profile.to("degC"),
                    sb_parcel_profile.to("degC"),
                    dew_profile.to("degC"),
                    alpha=0.35,
                    label="SBCIN",
                )
            if has_buoyancy(mu_cape, mu_cin):
                skew_plot.shade_cape(
                    prs_profile,
                    temp_profile.to("degC"),
                    mu_parcel_profile.to("degC"),
                    alpha=0.25,
                    label="MUCAPE",
                )
                skew_plot.shade_cin(
                    prs_profile,
                    temp_profile.to("degC"),
                    mu_parcel_profile.to("degC"),
                    dew_profile.to("degC"),
                    alpha=0.25,
                    label="MUCIN",
                )
            if has_buoyancy(ml_cape, ml_cin):
                skew_plot.shade_cape(
                    prs_profile,
                    temp_profile.to("degC"),
                    ml_parcel_profile.to("degC"),
                    alpha=0.2,
                    label="MLCAPE",
                )
                skew_plot.shade_cin(
                    prs_profile,
                    temp_profile.to("degC"),
                    ml_parcel_profile.to("degC"),
                    dew_profile.to("degC"),
                    alpha=0.2,
                    label="MLCIN",
                )
        except Exception as e:
            print(f"Warning: CAPE/CIN shading failed for {file_name}: {e}")

        if is_finite_scalar(sb_lcl_p):
            skew_plot.plot(
                sb_lcl_p, sb_lcl_t, "ko", markerfacecolor="black", markersize=3
            )
            lcl_p_val = quantity_to_scalar(sb_lcl_p.to("hPa"))
            lcl_height_km = pressure_to_height_km(lcl_p_val, p_for_interp, z_for_interp)
            lcl_label = f"←LCL {lcl_p_val:.0f} hPa"
            if np.isfinite(lcl_height_km):
                lcl_label += f" / {lcl_height_km:.2f} km"

            skew_plot.ax.text(
                0.82,
                lcl_p_val,
                lcl_label,
                transform=skew_plot.ax.get_yaxis_transform(),
                fontsize=11,
                color="gray",
            )

        if is_finite_scalar(sb_lfc_p):
            skew_plot.plot(
                sb_lfc_p, sb_lfc_t, "ko", markerfacecolor="red", markersize=3
            )
            lfc_p_val = quantity_to_scalar(sb_lfc_p.to("hPa"))
            lfc_height_km = pressure_to_height_km(lfc_p_val, p_for_interp, z_for_interp)
            lfc_label = f"←LFC {lfc_p_val:.0f} hPa"
            if np.isfinite(lfc_height_km):
                lfc_label += f" / {lfc_height_km:.2f} km"

            skew_plot.ax.text(
                0.82,
                lfc_p_val,
                lfc_label,
                transform=skew_plot.ax.get_yaxis_transform(),
                fontsize=11,
                color="gray",
            )

        if is_finite_scalar(sb_el_p):
            skew_plot.plot(
                sb_el_p, sb_el_t, "ko", markerfacecolor="green", markersize=3
            )
            el_p_val = quantity_to_scalar(sb_el_p.to("hPa"))
            el_height_km = pressure_to_height_km(el_p_val, p_for_interp, z_for_interp)
            el_label = f"←EL {el_p_val:.0f} hPa"
            if np.isfinite(el_height_km):
                el_label += f" / {el_height_km:.2f} km"

            skew_plot.ax.text(
                0.82,
                el_p_val,
                el_label,
                transform=skew_plot.ax.get_yaxis_transform(),
                fontsize=11,
                color="gray",
            )

        if is_finite_scalar(ccl_p):
            ccl_p_val = quantity_to_scalar(ccl_p.to("hPa"))
            ccl_height_km = pressure_to_height_km(ccl_p_val, p_for_interp, z_for_interp)
            ccl_label = f"←CCL {ccl_p_val:.0f} hPa"
            if np.isfinite(ccl_height_km):
                ccl_label += f" / {ccl_height_km:.2f} km"

            skew_plot.ax.text(
                0.82,
                ccl_p_val,
                ccl_label,
                transform=skew_plot.ax.get_yaxis_transform(),
                fontsize=11,
                color="orange",
            )

        if is_finite_scalar(ccl_p) and is_finite_scalar(conv_temp_c):
            p_sfc = prs_profile[0]
            if quantity_to_scalar(ccl_p) < quantity_to_scalar(p_sfc):
                p_line_vals = (
                    np.linspace(
                        quantity_to_scalar(p_sfc.to("hPa")),
                        quantity_to_scalar(ccl_p.to("hPa")),
                        25,
                    )
                    * units.hectopascal
                )
                t_line = mpcalc.dry_lapse(p_line_vals, conv_temp_c)

                skew_plot.plot(
                    p_line_vals,
                    t_line.to("degC"),
                    color="orange",
                    linewidth=1.4,
                    linestyle="--",
                    label="Convective Temp",
                )
                skew_plot.ax.plot(
                    quantity_to_scalar(conv_temp_c.to("degC")),
                    quantity_to_scalar(p_sfc.to("hPa")),
                    marker="o",
                    markersize=6,
                    color="orange",
                    zorder=6,
                )
                skew_plot.ax.plot(
                    quantity_to_scalar(temp_profile[0].to("degC")),
                    quantity_to_scalar(p_sfc.to("hPa")),
                    marker="o",
                    markersize=6,
                    color="red",
                    zorder=6,
                )
                skew_plot.ax.plot(
                    quantity_to_scalar(dew_profile[0].to("degC")),
                    quantity_to_scalar(p_sfc.to("hPa")),
                    marker="o",
                    markersize=6,
                    color="green",
                    zorder=6,
                )
                skew_plot.ax.plot(
                    quantity_to_scalar(ccl_t.to("degC")),
                    quantity_to_scalar(ccl_p.to("hPa")),
                    marker="^",
                    markersize=6,
                    color="orange",
                    zorder=6,
                )

        temp_np = temp_profile.magnitude
        prs_np = prs_profile.magnitude
        z_np = z_profile.to("meter").magnitude
        zero_idx = int(np.argmin(np.abs(temp_np - 0.0)))
        freezing_pressure = prs_np[zero_idx]
        if np.isfinite(freezing_pressure):
            freezing_height_km = z_np[zero_idx] / 1000.0
            frz_label = (
                f"←FRZ {freezing_pressure:.0f} hPa / {freezing_height_km:.2f} km"
            )
            skew_plot.ax.text(
                0.82,
                freezing_pressure,
                frz_label,
                transform=skew_plot.ax.get_yaxis_transform(),
                fontsize=11,
                color="blue",
                alpha=0.4,
            )

        dgz_mask = (
            (temp_profile <= -12 * units.degC)
            & (temp_profile >= -18 * units.degC)
            & (rh >= 0.6)
        )
        if np.any(dgz_mask):
            dgz_p = prs_profile[dgz_mask].to("hPa").magnitude
            dgz_t = temp_profile[dgz_mask].to("degC").magnitude
            dgz_td = dew_profile[dgz_mask].to("degC").magnitude
            dgz_z = z_profile[dgz_mask].to("meter").magnitude

            dgz_bottom_p = dgz_p.max()
            dgz_top_p = dgz_p.min()
            dgz_bottom_z = dgz_z[dgz_p.argmax()]
            dgz_top_z = dgz_z[dgz_p.argmin()]
            dgz_rh_val = rh[dgz_mask].mean().magnitude * 100.0

            skew_plot.ax.fill_betweenx(
                dgz_p, dgz_td, dgz_t, color="dodgerblue", alpha=0.18, zorder=1
            )
            mid_p = 0.5 * (dgz_bottom_p + dgz_top_p)
            skew_plot.ax.text(
                -47,
                mid_p,
                (
                    "Dendritic Growth Zone\n"
                    f"{dgz_bottom_p:.1f}-{dgz_top_p:.1f} hPa\n"
                    f"AGL: {dgz_bottom_z/1000:.1f}-{dgz_top_z/1000:.1f} km\n"
                    f"RH≈{dgz_rh_val:.0f}%"
                ),
                fontsize=7,
                color="blue",
                ha="left",
                va="center",
            )

        # ------------------------------------------------------------------
        # Hodograph
        # ------------------------------------------------------------------
        ax_hodograph = fig.add_axes(layout["hodograph"])
        plot_hodograph(
            ax_hodograph,
            z_profile,
            u_profile,
            v_profile,
            prs_profile,
        )
        ax_hodograph.patch.set_alpha(0.0)
        if SHOW_LAYOUT_GRID:
            ax_hodograph.add_patch(
                Rectangle(
                    (0.0, 0.0),
                    1.0,
                    1.0,
                    transform=ax_hodograph.transAxes,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.3,
                    zorder=50,
                )
            )

        # ------------------------------------------------------------------
        # Storm motion text panel
        # ------------------------------------------------------------------
        ax_storm = fig.add_axes(layout["storm_motion"])
        ax_storm.set_position(layout["storm_motion"])
        ax_storm.set_xlim(0, 1)
        ax_storm.set_ylim(0, 1)
        ax_storm.axis("off")
        ax_storm.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=ax_storm.transAxes,
                edgecolor="black",
                facecolor="white",
                linewidth=0.8,
                zorder=1,
            )
        )

        mn_str = f"{fmt(bunkers_mw_speed, '{:.0f}', ' kt')} / {fmt(bunkers_mw_dir, '{:.0f}', '°')}"
        rm_str = f"{fmt(bunkers_rm_speed, '{:.0f}', ' kt')} / {fmt(bunkers_rm_dir, '{:.0f}', '°')}"
        lm_str = f"{fmt(bunkers_lm_speed, '{:.0f}', ' kt')} / {fmt(bunkers_lm_dir, '{:.0f}', '°')}"

        ax_storm.text(
            0.5,
            0.92,
            "Storm Motion",
            ha="center",
            va="top",
            fontsize=TABLE_HEADER_FONT_SIZE,
            fontweight="bold",
            transform=ax_storm.transAxes,
        )

        lines = [
            ("Mean:", mn_str, "black", 0.68),
            ("RM:", rm_str, "red", 0.48),
            ("LM:", lm_str, "navy", 0.28),
        ]
        for label, value, color, y in lines:
            ax_storm.text(
                0.02,
                y,
                label,
                ha="left",
                va="center",
                fontsize=TABLE_FONT_SIZE,
                fontweight="bold",
                color=color,
                transform=ax_storm.transAxes,
            )
            ax_storm.text(
                0.98,
                y,
                value,
                ha="right",
                va="center",
                fontsize=TABLE_FONT_SIZE,
                color=color,
                transform=ax_storm.transAxes,
            )

        # ------------------------------------------------------------------
        # Hazard panel
        # ------------------------------------------------------------------
        draw_hazard_panel(
            fig,
            layout,
            sb_cape,
            ml_cape,
            mu_cape,
            srh_0to1,
            srh_0to3,
            shear_0to6,
            pwat,
            dcape,
            surface_temperature,
            surface_dewpoint,
            surface_u,
            surface_v,
            precip_type,
            precip_type_thick,
            thickness_850_700,
            thickness_1000_850,
            is_saturated,
        )

        # ------------------------------------------------------------------
        # Map panel (Cartopy)
        # ------------------------------------------------------------------
        if _CARTOPY_OK:
            ax_map = fig.add_axes(layout["map"], projection=ccrs.PlateCarree())
            ax_map.set_position(layout["map"])
            ax_map.set_aspect("auto")
            ax_map.set_xmargin(0)
            ax_map.set_ymargin(0)

            try:
                ax_map.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])
                for feat in map_features:
                    add_feature(ax_map, *feat)
            except Exception:
                try:
                    ax_map.coastlines(resolution="50m", linewidth=0.6)
                except Exception:
                    ax_map.coastlines()
                try:
                    ax_map.add_feature(
                        cfeature.BORDERS.with_scale("50m"), linewidth=0.4
                    )
                except Exception:
                    ax_map.add_feature(cfeature.BORDERS, linewidth=0.4)
                try:
                    ax_map.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)
                except Exception:
                    pass

            lat_pad = 2.0
            lon_pad = 2.0
            min_lon = max(-180.0, point_lon - lon_pad)
            max_lon = min(180.0, point_lon + lon_pad)
            min_lat = max(-90.0, point_lat - lat_pad)
            max_lat = min(90.0, point_lat + lat_pad)

            ax_map.set_extent(
                [min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree()
            )

            try:
                if not cities.empty:
                    cities_extent = cities[
                        (cities["LONGITUDE"].between(min_lon, max_lon))
                        & (cities["LATITUDE"].between(min_lat, max_lat))
                    ]
                    if "POP_MAX" in cities_extent.columns:
                        cities_plot = cities_extent.nlargest(
                            MAX_CITIES_ON_MAP, "POP_MAX"
                        )
                    else:
                        cities_plot = cities_extent.head(MAX_CITIES_ON_MAP)

                    ax_map.scatter(
                        cities_plot["LONGITUDE"],
                        cities_plot["LATITUDE"],
                        transform=ccrs.PlateCarree(),
                        marker="o",
                        s=10,
                        color="black",
                        zorder=6,
                    )
                    for _, row in cities_plot.iterrows():
                        name = None
                        for col in ("NAME", "NAMEASCII", "NAME_EN"):
                            if col in cities_plot.columns and isinstance(row[col], str):
                                name = row[col]
                                break
                        if not name:
                            continue

                        ax_map.text(
                            row["LONGITUDE"],
                            row["LATITUDE"],
                            name,
                            transform=ccrs.PlateCarree(),
                            fontsize=7,
                            ha="left",
                            va="bottom",
                            color="black",
                            zorder=7,
                            bbox=dict(
                                boxstyle="round,pad=0.10",
                                facecolor="white",
                                alpha=0.8,
                                linewidth=0.0,
                            ),
                        )
            except Exception as e:
                print(f"City plotting failed: {e}")

            ax_map.scatter(
                point_lon,
                point_lat,
                transform=ccrs.PlateCarree(),
                color="gold",
                marker="*",
                s=60,
                zorder=8,
            )
        else:
            ax_map = fig.add_axes(layout["map"])
            ax_map.set_position(layout["map"])
            ax_map.axis("off")
            ax_map.add_patch(
                Rectangle(
                    (0.0, 0.0),
                    1.0,
                    1.0,
                    transform=ax_map.transAxes,
                    edgecolor="black",
                    facecolor="white",
                    linewidth=1.0,
                )
            )
            ax_map.text(
                0.5,
                0.6,
                "Map unavailable\n(Cartopy not installed)",
                ha="center",
                va="center",
                fontsize=9,
                transform=ax_map.transAxes,
            )
            ax_map.text(
                0.5,
                0.3,
                f"Lat: {point_lat:.2f}\nLon: {point_lon:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                transform=ax_map.transAxes,
            )

        # ------------------------------------------------------------------
        # Parcel table (unchanged)
        # ------------------------------------------------------------------
        ax_parcel_table = fig.add_axes(layout["parcel_table"])
        ax_parcel_table.set_position(layout["parcel_table"])
        ax_parcel_table.axis("off")

        header_parcel = [
            "Parcel",
            "CAPE (J/kg)",
            "CINH (J/kg)",
            "LCL (hPa)",
            "LFC (hPa)",
            "EL (hPa)",
            "0–3 km CAPE\n(J/kg)",
        ]
        parcel_rows = [
            [
                "Surface Based",
                fmt(sb_cape),
                fmt(sb_cin),
                fmt(sb_lcl_p),
                fmt(sb_lfc_p),
                fmt(sb_el_p),
                fmt(sb_cape_0_3),
            ],
            [
                "Mixed Layer",
                fmt(ml_cape),
                fmt(ml_cin),
                fmt(ml_lcl_p),
                fmt(ml_lfc_p),
                fmt(ml_el_p),
                fmt(ml_cape_0_3),
            ],
            [
                "Most Unstable",
                fmt(mu_cape),
                fmt(mu_cin),
                fmt(mu_lcl_p),
                fmt(mu_lfc_p),
                fmt(mu_el_p),
                fmt(mu_cape_0_3),
            ],
        ]
        parcel_table = ax_parcel_table.table(
            cellText=parcel_rows,
            colLabels=header_parcel,
            loc="center",
            cellLoc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        parcel_table.scale(1.0, 1.4)
        style_table_fonts(parcel_table)

        for (row_idx, col_idx), cell in parcel_table.get_celld().items():
            cell.set_linewidth(0.5)
            if row_idx == 0:
                cell.set_facecolor("lightgray")

        # ------------------------------------------------------------------
        # Environment summary table (unchanged physics)
        # ------------------------------------------------------------------
        ax_env_summary = fig.add_axes(layout["env_summary"])
        ax_env_summary.set_position(layout["env_summary"])
        ax_env_summary.axis("off")

        sfc_t_f = quantity_to_scalar(surface_temperature.to("degF"))
        sfc_td_f = quantity_to_scalar(surface_dewpoint.to("degF"))
        sfc_tw_f = (
            quantity_to_scalar(Tw_sfc.to("degF"))
            if is_finite_scalar(Tw_sfc)
            else np.nan
        )
        sfc_t_f_str = f"{sfc_t_f:.0f}°F" if np.isfinite(sfc_t_f) else "N/A"
        sfc_td_f_str = f"{sfc_td_f:.0f}°F" if np.isfinite(sfc_td_f) else "N/A"
        sfc_tw_f_str = f"{sfc_tw_f:.0f}°F" if np.isfinite(sfc_tw_f) else "N/A"

        if is_finite_scalar(conv_temp_c):
            conv_temp_f = conv_temp_c.to("degF")
            conv_temp_f_val = quantity_to_scalar(conv_temp_f)
        else:
            conv_temp_f_val = np.nan
        conv_temp_f_str = (
            f"{conv_temp_f_val:.0f}°F" if np.isfinite(conv_temp_f_val) else "N/A"
        )

        try:
            sfc_speed = mpcalc.wind_speed(surface_u, surface_v).to("kt")
            sfc_dir = mpcalc.wind_direction(surface_u, surface_v)
            sfc_spd_str = fmt(sfc_speed, "{:.0f}", " kt")
            sfc_dir_str = fmt(sfc_dir, "{:.0f}", "°")
            sfc_wind_str = f"{sfc_spd_str} / {sfc_dir_str}"
        except Exception:
            sfc_wind_str = "N/A"

        bunkers_spd_str = fmt(bunkers_speed, "{:.0f}", " kt")
        bunkers_dir_str = fmt(bunkers_dir, "{:.0f}", "°")
        bunkers_rm_str = f"{bunkers_spd_str} / {bunkers_dir_str}"

        dcape_scalar = quantity_to_scalar(dcape)
        if np.isfinite(dcape_scalar) and dcape_scalar > 0.0:
            wmax_ms = np.sqrt(2.0 * dcape_scalar) * (units.meter / units.second)
            wmax_kt = wmax_ms.to("kt")
            wmax_mph = wmax_ms.to("mph")
            wmax_str = (
                f"{fmt(wmax_ms, '{:.0f}', ' m/s')}  "
                f"({fmt(wmax_kt, '{:.0f}', ' kt')}, "
                f"{fmt(wmax_mph, '{:.0f}', ' mph')})"
            )
        else:
            wmax_str = "N/A"

        dcape_str = fmt(dcape, "{:.0f}", " J/kg")

        scp_str = fmt(supercell_composite, "{:.2f}")
        stp_str = fmt(sig_tornado_param, "{:.2f}")
        ehi_0_1_str = "N/A" if not np.isfinite(ehi_0_1) else f"{ehi_0_1:.2f}"
        ehi_0_3_str = "N/A" if not np.isfinite(ehi_0_3) else f"{ehi_0_3:.2f}"

        srh_0to1_str = fmt(srh_0to1, "{:.0f}", " m²/s²")
        srh_0to3_str = fmt(srh_0to3, "{:.0f}", " m²/s²")
        srh_0to6_str = fmt(srh_0to6, "{:.0f}", " m²/s²")

        header_env = ["Parameter", "Value"]
        env_rows = [
            ["PWAT", fmt(pwat, "{:.2f}", " in")],
            ["Total Totals (TT)", fmt(total_totals_index)],
            ["K Index", fmt(k_index)],
            ["Lifted Index", fmt(lifted_index)],
            ["Showalter Index", fmt(showalter_index)],
            ["SWEAT Index", fmt(sweat_index, "{:.0f}")],
            ["Supercell Composite (SCP)", scp_str],
            ["Significant Tornado Parameter (STP)", stp_str],
            ["EHI SFC–>1 km", ehi_0_1_str],
            ["EHI SFC–>3 km", ehi_0_3_str],
            ["DCAPE", dcape_str],
            ["Max downdraft wind", wmax_str],
        ]
        env_table = ax_env_summary.table(
            cellText=env_rows,
            colLabels=header_env,
            loc="center",
            cellLoc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        env_table.scale(1.0, 1.8)
        style_table_fonts(env_table)

        for (row_idx, col_idx), cell in env_table.get_celld().items():
            cell.set_linewidth(0.5)
            if row_idx == 0:
                cell.set_facecolor("lightgray")
                cell.get_text().set_fontweight("bold")

        # ------------------------------------------------------------------
        # Surface summary table
        # ------------------------------------------------------------------
        ax_surface = fig.add_axes(layout["Surface"])
        ax_surface.set_position(layout["Surface"])
        ax_surface.axis("off")

        header_surface = ["Surface", "Value"]
        surface_rows = [
            ["SFC T (°F)", sfc_t_f_str],
            ["SFC Td (°F)", sfc_td_f_str],
            ["SFC Tw (°F)", sfc_tw_f_str],
            ["Convective Temp (°F)", conv_temp_f_str],
            ["SFC Wind (kt/°)", sfc_wind_str],
        ]
        surface_table = ax_surface.table(
            cellText=surface_rows,
            colLabels=header_surface,
            loc="center",
            cellLoc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        surface_table.scale(1.0, 1.8)
        style_table_fonts(surface_table)

        for (row_idx, col_idx), cell in surface_table.get_celld().items():
            cell.set_linewidth(0.5)
            if row_idx == 0:
                cell.set_facecolor("lightgray")
                cell.get_text().set_fontweight("bold")
                continue
            if col_idx in (0, 1):
                label_text = surface_table[row_idx, 0].get_text().get_text()
                if label_text.startswith("SFC T (°F)"):
                    cell.get_text().set_color("red")
                elif label_text.startswith("SFC Td (°F)"):
                    cell.get_text().set_color("green")
                elif label_text.startswith("SFC Tw (°F)"):
                    cell.get_text().set_color("blue")
                elif label_text.startswith("Convective Temp (°F)"):
                    cell.get_text().set_color("orange")

        # ------------------------------------------------------------------
        # Shear / SRH table
        # ------------------------------------------------------------------
        ax_shear_table = fig.add_axes(layout["shear_table"])
        ax_shear_table.set_position(layout["shear_table"])
        ax_shear_table.axis("off")

        header_shear = [
            "Layer (AGL)",
            "Bulk Wind Diff\n(kt/°)",
            "Mean Layer Wind\n(kt/°)",
            "Storm-Rel Helicity\n(m²/s²)",
            "Storm-Rel Mean Wind\n(kt/°)",
            "Γ (°C/km)",
        ]

        def build_layer_row(label, depth_km):
            depth_val = depth_km * units.kilometer
            (
                shear_speed,
                shear_dir,
                srh_val,
                mean_speed,
                mean_dir,
                sr_mean_speed,
                sr_mean_dir,
            ) = compute_layer_shear_and_srh(
                prs_profile, z_profile, u_profile, v_profile, depth_val
            )

            bulk_str = f"{fmt(shear_speed, '{:.0f}')} / {fmt(shear_dir, '{:.0f}', '°')}"
            mn_str = f"{fmt(mean_speed, '{:.0f}')} / {fmt(mean_dir, '{:.0f}', '°')}"
            srw_str = (
                f"{fmt(sr_mean_speed, '{:.0f}')} / {fmt(sr_mean_dir, '{:.0f}', '°')}"
            )

            lapse_layer = layer_lapse_rate(
                temp_profile, z_profile, 0 * units.m, depth_val
            )
            lapse_str = fmt(lapse_layer, "{:.1f}", " °C/km")

            return [
                label,
                bulk_str,
                mn_str,
                fmt(srh_val),
                srw_str,
                lapse_str,
            ]

        shear_rows = [
            build_layer_row("SFC→500 m", 0.5),
            build_layer_row("SFC→1 km", 1.0),
            build_layer_row("SFC→3 km", 3.0),
            build_layer_row("SFC→6 km", 6.0),
            build_layer_row("SFC→9 km", 9.0),
        ]
        shear_table = ax_shear_table.table(
            cellText=shear_rows,
            colLabels=header_shear,
            loc="center",
            cellLoc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        shear_table.scale(1.0, 4.0)
        style_table_fonts(shear_table)

        for (row_idx, col_idx), cell in shear_table.get_celld().items():
            cell.set_linewidth(0.5)
            if row_idx == 0:
                cell.set_facecolor("lightgray")

        layer_color_map = {
            "SFC→500 m": "fuchsia",
            "SFC→1 km": "firebrick",
            "SFC→3 km": "limegreen",
            "SFC→6 km": "goldenrod",
            "SFC→9 km": "turquoise",
        }
        for row_idx in range(1, len(shear_rows) + 1):
            layer_label = shear_rows[row_idx - 1][0]
            color = layer_color_map.get(layer_label)
            if color is not None:
                cell = shear_table[row_idx, 0]
                cell.get_text().set_color(color)
                cell.get_text().set_fontweight("bold")

        skew_plot.ax.legend(loc="upper right", fontsize=8)
        draw_layout_grid(fig)

        # ------------------------------------------------------------------
        # Save figure: filename uses valid_dt so alphabetical == chronological
        # ------------------------------------------------------------------
        if valid_dt is not None:
            file_time_tag = valid_dt.strftime("%Y%m%d%H%M%S")
        else:
            file_time_tag = "unknown"

        image_folder = os.path.join(path_figures, "Images")
        file_out = f"wrf_{domain}_SkewT_LogP_{file_time_tag}.png"
        output_path = os.path.join(image_folder, file_out)

        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    except Exception as e:
        print(f"ERROR in {os.path.basename(ncfile_path)} (t={time_index}): {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None
    finally:
        wrf_handle.close()


# =============================================================================
# GIF CREATION (v3 style)
# =============================================================================


def create_gif(path_figures, image_folder, domain):
    """Build an animated GIF from all PNGs in image_folder."""
    animation_folder = os.path.join(path_figures, "Animation")
    if not os.path.isdir(animation_folder):
        os.mkdir(animation_folder)

    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    if not png_files:
        print("No PNG files found for GIF creation.")
        return

    png_files_sorted = sorted(png_files)
    print("Creating .gif file from sorted .png files")

    duration_ms = 800
    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]
    gif_file_out = f"wrf_{domain}_SkewT_LogP.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"GIF generation complete: {gif_path}")


# =============================================================================
# MAIN ENTRY POINT (v3 multi-file + multi-time)
# =============================================================================


def main():
    """
    Parse arguments, find WRF files & timesteps, generate Skew-T figures, and build GIF.

    Usage
    -----
    python v5.3.py /path/to/wrfout d01 CityName 35.0 -97.5
    """
    if len(sys.argv) != 6:
        print("\nUsage:\n" "  python v5.3.py /path/to/wrfout d01 CityName 35.0 -97.5\n")
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]
    city = sys.argv[3]

    try:
        lat = float(sys.argv[4])
        lon = float(sys.argv[5])
    except ValueError:
        print("Invalid latitude/longitude; please use decimal degrees.")
        sys.exit(1)

    path_figures = f"wrf_SkewT_LogP_{city}_Lat_{lat}_Long_{lon}"
    image_folder = os.path.join(path_figures, "Images")
    animation_folder = os.path.join(path_figures, "Animation")

    ensure_directories_exist(path_figures, image_folder, animation_folder)

    print(f"Latitude: {lat}, Longitude: {lon}")
    print(f"Output directory: {path_figures}")

    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print("No WRF output files found matching pattern.")
        return

    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        return

    print(
        f"Found {len(ncfile_paths)} wrfout files, "
        f"{len(frames)} total time steps. Rendering in parallel..."
    )

    args_list = [
        (ncfile_path, time_index, domain, city, lat, lon, path_figures)
        for (ncfile_path, time_index) in frames
    ]

    max_workers = min(4, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    create_gif(path_figures, image_folder, domain)


if __name__ == "__main__":
    main()

