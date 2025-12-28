#!/usr/bin/env python3
"""
Meteogram_DegF_multicore_v3.py

Generates a multi-panel meteogram from WRF output in *Imperial* units:
    1. Sea Level Pressure
    2. 2m Temperature & Dew Point (°F)
    3. 2m Relative Humidity (%)
    4. 10m Wind Speed (mph) & Direction (deg/cardinal)
    5. 1-hr Precipitation Rates (Rain & Snow, in/hr, side-by-side bars)
    6. Surface Downwelling Shortwave Radiation (W/m²)

This script can handle:
    * Multiple wrfout_<domain>* files, each with one or more timesteps.
    * A single wrfout file containing many timesteps.

It treats each (file, time_index) pair as a “frame” for sampling the
meteogram point, then aggregates all frames into a single meteogram PNG.
"""

import glob

###############################################################################
# Standard library imports
###############################################################################
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

###############################################################################
# Third-party imports
###############################################################################
import numpy as np

# WRF
import wrf
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator
from netCDF4 import Dataset
from wrf import ll_to_xy

# Quiet down noisy warnings (optional but standard for v3)
warnings.filterwarnings("ignore")


###############################################################################
# Time handling helpers (playbook v3)
###############################################################################
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


###############################################################################
# Wind helper functions (physics unchanged)
###############################################################################
def calculate_wind_direction(u, v):
    """
    Calculate wind direction (degrees) from U and V components.

    Parameters
    ----------
    u : float
        Zonal wind component (m/s).
    v : float
        Meridional wind component (m/s).

    Returns
    -------
    float
        Wind direction in degrees, where:
            0° / 360° = North
            90°       = East
            180°      = South
            270°      = West
    """
    wind_dir = (np.arctan2(-u, -v) * 180.0 / np.pi) % 360.0
    return wind_dir


def wind_direction_to_cardinal(degrees):
    """
    Map wind direction in degrees to the nearest cardinal direction.

    Parameters
    ----------
    degrees : float
        Wind direction (0–360 degrees).

    Returns
    -------
    str
        One of: "N", "NE", "E", "SE", "S", "SW", "W", "NW".
    """
    directions = [
        ("N", 0),
        ("NE", 45),
        ("E", 90),
        ("SE", 135),
        ("S", 180),
        ("SW", 225),
        ("W", 270),
        ("NW", 315),
    ]

    degrees = degrees % 360.0

    min_diff = 360.0
    closest_dir = "N"
    for direction, angle in directions:
        diff = abs(degrees - angle)
        if diff < min_diff:
            min_diff = diff
            closest_dir = direction
    return closest_dir


###############################################################################
# Worker: extract point data for one (file, time_index) frame
###############################################################################
def process_frame(args):
    """
    Read a single (file, time_index) from a WRF output file and extract
    instantaneous/cumulative parameters at the specified latitude/longitude.

    Returns data in a mix of metric (for precip accumulation) and
    imperial (for T, wind). 1-hr rates are computed later from temporal
    differences.
    """
    ncfile_path, time_index, latitude, longitude = args

    # ----------------------------------------------------------------------
    # Open file
    # ----------------------------------------------------------------------
    ncfile = Dataset(ncfile_path)

    # Valid time from metadata (preferred) or filename
    valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
    print(f"Extracting meteogram data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

    # ----------------------------------------------------------------------
    # Find grid indices for target lat/lon
    # Use timeidx=time_index for moving/vortex-following nests safety
    # ----------------------------------------------------------------------
    try:
        xy_loc = ll_to_xy(ncfile, latitude, longitude, timeidx=time_index)
        x, y = int(xy_loc[0]), int(xy_loc[1])
    except Exception as e:
        ncfile.close()
        raise RuntimeError(
            f"Error finding grid indices for lat={latitude}, lon={longitude} "
            f"in {ncfile_path}: {e}"
        )

    # ----------------------------------------------------------------------
    # Extract required variables (metric, then convert where needed)
    # Physics & diagnostics unchanged from original script.
    # ----------------------------------------------------------------------
    try:
        # Temperatures
        temp_c = wrf.getvar(ncfile, "T2", timeidx=time_index)[y, x] - 273.15  # °C
        dew_c = wrf.getvar(ncfile, "td2", timeidx=time_index)[y, x]  # °C

        # Humidity & pressure
        rh2 = wrf.getvar(ncfile, "rh2", timeidx=time_index)[y, x]  # %
        pressure = wrf.getvar(ncfile, "slp", timeidx=time_index)[y, x]  # hPa/mb

        # Radiation
        solar_rad = wrf.getvar(ncfile, "SWDOWN", timeidx=time_index)[y, x]  # W/m²

        # Cumulative rain (mm)
        rain = (
            wrf.getvar(ncfile, "RAINC", timeidx=time_index)[y, x]
            + wrf.getvar(ncfile, "RAINNC", timeidx=time_index)[y, x]
            + wrf.getvar(ncfile, "RAINSH", timeidx=time_index)[y, x]
        )

        # 10 m wind components (m/s)
        u10 = wrf.getvar(ncfile, "U10", timeidx=time_index)[y, x]
        v10 = wrf.getvar(ncfile, "V10", timeidx=time_index)[y, x]
        wind_speed_ms = np.sqrt(u10**2 + v10**2)  # m/s
        wind_dir_deg = calculate_wind_direction(u10, v10)

        # Snow water equivalent (mm) using 10:1 ratio
        snowh20 = wrf.getvar(ncfile, "SNOW", timeidx=time_index)[y, x] * 10.0

    except KeyError as e:
        ncfile.close()
        raise RuntimeError(f"Variable {e} not found in WRF file {ncfile_path}.")
    finally:
        ncfile.close()

    # ----------------------------------------------------------------------
    # Physical consistency tweak: Td should not exceed T when RH is 100%
    # ----------------------------------------------------------------------
    if dew_c > temp_c and rh2 == 100:
        dew_c = temp_c

    # ----------------------------------------------------------------------
    # Convert to imperial units where appropriate
    # ----------------------------------------------------------------------
    temp_f = (temp_c * 9.0 / 5.0) + 32.0
    dew_f = (dew_c * 9.0 / 5.0) + 32.0
    wind_speed_mph = wind_speed_ms * 2.23694  # m/s → mph
    pressure_mb = pressure  # hPa == mb

    # Precip (rain/snow) remains in mm here; rates and inches come later.

    return (
        valid_dt,
        float(temp_f),
        float(dew_f),
        float(rh2),
        float(pressure_mb),
        float(solar_rad),
        float(rain),
        float(wind_speed_mph),
        float(wind_dir_deg),
        float(snowh20),
    )


###############################################################################
# Frame discovery: handle multi-file and multi-time setups
###############################################################################
def discover_frames(ncfile_paths):
    """
    Discover all (file, time_index) combinations.

    Supports:
        * Many wrfout_<domain>* files with one or more Time steps.
        * A single wrfout file containing many Time steps.
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
# Main script
###############################################################################
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Command-line arguments
    # ----------------------------------------------------------------------
    if len(sys.argv) != 6:
        print(
            "Usage: python Meteogram_DegF_multicore_v3.py "
            "<path_to_WRF> <domain> <city> <latitude> <longitude>"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]  # Path to WRF files
    domain = sys.argv[2]  # WRF domain (e.g., d01, d02, ...)
    city = sys.argv[3]  # City name (used in title and output file)
    latitude = float(sys.argv[4])
    longitude = float(sys.argv[5])

    # ----------------------------------------------------------------------
    # Output directory
    # ----------------------------------------------------------------------
    output_dir = f"meteogram_{city}_imperial"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # Discover WRF output files
    # ----------------------------------------------------------------------
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print(f"No wrfout files found in {path_wrf} matching wrfout_{domain}*")
        sys.exit(1)

    # ----------------------------------------------------------------------
    # Discover frames (file, time_index)
    # ----------------------------------------------------------------------
    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    # Build args list for worker function
    args_list = [
        (ncfile_path, time_index, latitude, longitude)
        for (ncfile_path, time_index) in frames
    ]

    # ----------------------------------------------------------------------
    # Read all frames using multiprocessing
    # ----------------------------------------------------------------------
    max_workers = min(4, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_frame, args_list))

    # Ensure chronological order
    results.sort(key=lambda r: r[0])

    if not results:
        print("No data extracted for meteogram.")
        sys.exit(0)

    # ----------------------------------------------------------------------
    # Storage lists for time series
    # ----------------------------------------------------------------------
    time_points = []
    temp_list = []
    dew_point_list = []
    rh2_list = []
    pressure_list = []
    solar_rad_list = []
    rainfall_list = []
    rain_rate_list = []
    wind_speed_list = []
    wind_direction_list = []
    snowh20_rate_list = []
    cumulative_snowh20_list = []

    # Precip cumulative values for rate calculation
    previous_rain = 0.0
    previous_snowh20 = 0.0

    # mm → inches conversion
    mm_to_inch = 0.03937

    # ----------------------------------------------------------------------
    # Build time series from results
    # ----------------------------------------------------------------------
    for i, (
        time,
        temp_f,
        dew_f,
        rh2,
        pressure_mb,
        solar_rad,
        rain_cum_mm,
        wind_speed_mph,
        wind_dir_deg,
        snowh20_cum_mm,
    ) in enumerate(results):
        time_points.append(time)

        # 1-hr precip rates from cumulative (mm/hr)
        if i > 0:
            rain_rate_mm = max(0.0, rain_cum_mm - previous_rain)
            snowh20_rate_mm = max(0.0, snowh20_cum_mm - previous_snowh20)
        else:
            rain_rate_mm = 0.0
            snowh20_rate_mm = 0.0

        previous_rain = rain_cum_mm
        previous_snowh20 = snowh20_cum_mm

        # Store primary variables (imperial + raw mm accumulations)
        temp_list.append(np.round(temp_f, 1))
        dew_point_list.append(np.round(dew_f, 1))
        rh2_list.append(np.round(rh2, 1))
        pressure_list.append(np.round(pressure_mb, 1))
        solar_rad_list.append(np.round(solar_rad, 1))

        rainfall_list.append(np.round(rain_cum_mm, 1))  # cumulative rain (mm)
        rain_rate_list.append(np.round(rain_rate_mm, 1))  # mm/hr
        snowh20_rate_list.append(np.round(snowh20_rate_mm, 1))  # mm/hr
        cumulative_snowh20_list.append(np.round(snowh20_cum_mm, 1))

        wind_speed_list.append(np.round(wind_speed_mph, 1))
        wind_direction_list.append(np.round(wind_dir_deg, 1))

    # ----------------------------------------------------------------------
    # Convert precip rates from mm/hr to inches/hr
    # ----------------------------------------------------------------------
    rain_rate_inch = [np.round(rate * mm_to_inch, 2) for rate in rain_rate_list]
    snowh20_rate_inch = [np.round(rate * mm_to_inch, 2) for rate in snowh20_rate_list]

    ###############################################################################
    # Create figure and subplots
    ###############################################################################
    dpi = 400 
    fig, ax = plt.subplots(6, 1, figsize=(3840/dpi, 2160/dpi), dpi=dpi, sharex=True)
    fig.patch.set_facecolor("white")

    # ======================================================================
    # 1. Sea Level Pressure
    # ======================================================================
    ax[0].plot(
        time_points,
        pressure_list,
        label="Pressure (mb)",
        color="black",
        linewidth=2,
        marker="o",
    )
    ax[0].set_ylabel("Pressure (mb)")
    ax[0].set_title("Sea Level Pressure")
    ax[0].grid()

    ax[0].xaxis.set_minor_locator(AutoMinorLocator(2))
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    ax[0].minorticks_on()

    # ======================================================================
    # 2. Temperature and Dew Point (°F)
    # ======================================================================
    ax[1].plot(
        time_points,
        temp_list,
        label="Temperature (°F)",
        color="red",
        linewidth=2,
        marker="o",
    )
    ax[1].plot(
        time_points,
        dew_point_list,
        label="Dew Point (°F)",
        color="green",
        linewidth=2,
        marker="o",
    )
    ax[1].set_ylabel("Temperature (°F)")
    ax[1].set_title("2m Temperature (Red) and Dew Point (Green)")
    ax[1].grid()

    ax[1].xaxis.set_minor_locator(AutoMinorLocator(2))
    ax[1].yaxis.set_minor_locator(MultipleLocator(0.1))
    ax[1].minorticks_on()
    ax[1].legend(loc="best", fontsize=10)

    # ======================================================================
    # 3. Relative Humidity (%)
    # ======================================================================
    ax[2].bar(
        time_points,
        rh2_list,
        color="lightgreen",
        alpha=0.6,
        label="Relative Humidity (%)",
        width=0.02,
        edgecolor="black",
        linewidth=0.5,
    )
    ax[2].set_ylabel("Relative Humidity (%)")
    ax[2].set_title("2m Relative Humidity")
    ax[2].set_ylim(0, 100)
    ax[2].grid()

    ax[2].xaxis.set_minor_locator(AutoMinorLocator(2))
    ax[2].yaxis.set_minor_locator(MultipleLocator(10))
    ax[2].minorticks_on()

    # ======================================================================
    # 4. 10m Wind Speed (mph) and Direction (deg/cardinal)
    # ======================================================================
    bar_wind = ax[3].bar(
        time_points,
        wind_speed_list,
        color="lightgrey",
        label="Wind Speed (mph)",
        width=0.02,
        edgecolor="black",
        linewidth=0.5,
    )

    ax3_secondary = ax[3].twinx()
    scatter_dir = ax3_secondary.scatter(
        time_points,
        wind_direction_list,
        color="orange",
        label="Wind Direction (°)",
        s=30,
    )

    ax[3].set_ylabel("Wind Speed (mph)")
    ax3_secondary.set_ylabel("Wind Direction (° / Cardinal)")
    ax[3].set_title("10m Wind Speed and Wind Direction")
    ax[3].grid()

    def cardinal_direction_formatter(deg, pos):
        cardinal = wind_direction_to_cardinal(deg)
        return f"{int(deg)}° {cardinal}"

    ax3_secondary.yaxis.set_major_formatter(FuncFormatter(cardinal_direction_formatter))

    ax3_secondary.set_ylim(0, 360)
    ax3_secondary.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax3_secondary.set_yticklabels(
        [
            "0° N",
            "45° NE",
            "90° E",
            "135° SE",
            "180° S",
            "225° SW",
            "270° W",
            "315° NW",
            "360° N",
        ]
    )

    ax[3].xaxis.set_minor_locator(AutoMinorLocator(2))
    ax[3].yaxis.set_minor_locator(MultipleLocator(0.1))
    ax[3].minorticks_on()

    ax3_secondary.yaxis.set_minor_locator(MultipleLocator(10))
    ax3_secondary.minorticks_on()

    handles_primary, labels_primary = ax[3].get_legend_handles_labels()
    handles_secondary, labels_secondary = ax3_secondary.get_legend_handles_labels()
    ax[3].legend(
        handles_primary + handles_secondary,
        labels_primary + labels_secondary,
        loc="upper left",
        fontsize=10,
    )

    # ======================================================================
    # 5. 1-hr Precipitation Rates (Imperial, in/hr)
    # ======================================================================
    ax4_primary = ax[4]
    ax4_secondary = ax4_primary.twinx()

    time_nums = mdates.date2num(time_points)
    bar_width = 0.01
    offset = 0.006

    bar_rain = ax4_primary.bar(
        time_nums - offset,
        rain_rate_inch,
        color="green",
        label="1-hr Rain Rate (in/hr)",
        width=bar_width,
        edgecolor="black",
        linewidth=0.5,
    )

    bar_snow = ax4_secondary.bar(
        time_nums + offset,
        snowh20_rate_inch,
        color="cyan",
        label="1-hr Snow Rate (in/hr)",
        width=bar_width,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.5,
    )

    ax4_primary.set_ylabel("Rain Rate (in/hr)", color="black", fontsize=12)
    ax4_secondary.set_ylabel("Snow Rate (in/hr)", color="black", fontsize=12)
    ax4_primary.set_title("1-Hour Precipitation Rates", fontsize=14)

    ax4_primary.grid()
    ax4_primary.set_ylim(bottom=0.0)
    ax4_secondary.set_ylim(bottom=0.0)

    ax4_primary.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4_primary.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax4_primary.minorticks_on()

    ax4_secondary.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax4_secondary.minorticks_on()

    # Numeric labels above bars (only for values ≥ 0.01 in/hr)
    for rect in bar_rain:
        height = rect.get_height()
        if height >= 0.01:
            ax4_primary.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

    for rect in bar_snow:
        height = rect.get_height()
        if height >= 0.01:
            ax4_secondary.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

    h_p, l_p = ax4_primary.get_legend_handles_labels()
    h_s, l_s = ax4_secondary.get_legend_handles_labels()
    ax4_primary.legend(
        h_p + h_s,
        l_p + l_s,
        loc="upper left",
        fontsize=10,
    )

    # ======================================================================
    # 6. Solar Radiation
    # ======================================================================
    ax[5].plot(
        time_points,
        solar_rad_list,
        label="Solar Radiation (W/m²)",
        color="orange",
        linewidth=2,
        marker="o",
    )
    ax[5].set_ylabel("Solar Radiation (W/m²)")
    ax[5].set_title("Solar Radiation")
    ax[5].grid()

    ax[5].xaxis.set_minor_locator(AutoMinorLocator(2))
    ax[5].yaxis.set_minor_locator(MultipleLocator(10))
    ax[5].minorticks_on()
    ax[5].legend(loc="best", fontsize=10)

    # ======================================================================
    # Shared time axis formatting & figure title
    # ======================================================================
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=90)
    ax[-1].set_xlabel("UTC Time (Z)")

    plt.suptitle(
        f"Meteogram for {city} at {latitude}°, {longitude}°\n"
        f"Time Period: {time_points[0].strftime('%Y-%m-%d %H:%MZ')} "
        f"to {time_points[-1].strftime('%Y-%m-%d %H:%MZ')}",
        fontsize=18,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    # ----------------------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------------------
    output_file = os.path.join(output_dir, f"meteogram_{city}_imperial.png")
    plt.savefig(output_file, dpi=250, facecolor=fig.get_facecolor())
    plt.close()

    print(f"Meteogram saved to {output_file}")
