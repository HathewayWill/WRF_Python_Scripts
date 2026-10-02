#!/usr/bin/env python3
"""
Meteogram_DegC_moving_nest_fixed.py

Generates a multi-panel meteogram (metric units) from WRF output:
1. Sea Level Pressure
2. 2m Temperature & Dew Point
3. 2m Relative Humidity
4. 10m Wind Speed & Direction
5. 1-hr Precipitation Rates (Rain & Snow, side-by-side bars)
6. Solar Radiation

This script can handle:
    * Multiple wrfout_<domain>* files, each with one or more timesteps.
    * A single wrfout file containing many timesteps.

It treats each (file, time_index) combination as a frame, extracts
point data at a specified latitude/longitude, then aggregates to a
single meteogram PNG.
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
# Wind helpers (physics unchanged)
###############################################################################
def calculate_wind_direction(u, v):
    """
    Calculates the wind direction (degrees) from the U and V wind components.

    Returns:
        float: Wind direction in degrees, where 0°/360° is North,
               90° is East, 180° is South, and 270° is West.
    """
    wind_dir = (np.arctan2(-u, -v) * 180.0 / np.pi) % 360.0
    return wind_dir


def wind_direction_to_cardinal(degrees):
    """
    Converts a wind direction in degrees to one of the eight primary
    cardinal directions.

    Parameters:
        degrees (float): Wind direction in degrees (0 to 360).

    Returns:
        str: Cardinal direction ("N", "NE", "E", "SE", "S", "SW", "W", "NW").
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
# Moving-nest point lookup helpers
###############################################################################
def get_domain_shape(ncfile: Dataset):
    """Return unstaggered WRF horizontal shape as (ny, nx)."""
    ny = len(ncfile.dimensions["south_north"])
    nx = len(ncfile.dimensions["west_east"])
    return ny, nx


def moving_nest_ll_to_xy(ncfile: Dataset, latitude: float, longitude: float, time_index: int):
    """
    Convert a fixed lat/lon to the nearest unstaggered grid point for this
    timestep. This is the critical moving-nest step: the same lat/lon can map
    to a different i/j location as the nest moves.
    """
    xy_loc = ll_to_xy(
        ncfile,
        latitude,
        longitude,
        timeidx=time_index,
        as_int=True,
        meta=False,
    )

    x = int(xy_loc[0])
    y = int(xy_loc[1])

    ny, nx = get_domain_shape(ncfile)
    if x < 0 or x >= nx or y < 0 or y >= ny:
        return None

    return x, y


def safe_diff_rate(current_value, previous_value, current_time, previous_time):
    """
    Convert a cumulative-value difference into an hourly rate.

    For hourly WRF output this returns the original frame-to-frame difference.
    For 5, 15, or 30 minute output this normalizes the value to units per hour.
    Negative differences are treated as zero, which protects against restarts,
    cumulative-field resets, and occasional moving-nest nearest-grid jumps.
    """
    if previous_value is None or previous_time is None:
        return 0.0

    delta_hours = (current_time - previous_time).total_seconds() / 3600.0
    if delta_hours <= 0.0:
        return 0.0

    increment = current_value - previous_value
    if increment < 0.0:
        increment = 0.0

    return increment / delta_hours


def median_time_step_days(time_points):
    """Return a plotting bar width in Matplotlib date units, adapted to timestep."""
    if len(time_points) < 2:
        return 0.02

    time_nums = mdates.date2num(time_points)
    diffs = np.diff(np.sort(time_nums))
    diffs = diffs[diffs > 0.0]

    if len(diffs) == 0:
        return 0.02

    return max(0.003, min(0.03, float(np.median(diffs)) * 0.60))


###############################################################################
# Worker: extract meteogram sample for one (file, time_index) frame
###############################################################################
def process_frame(args):
    """
    Reads a single (file, time_index) from a WRF output file and extracts
    point data at the given latitude/longitude.

    Moving-nest behavior:
        * Converts lat/lon to x/y using the current time_index.
        * Checks bounds for the current moving domain.
        * Returns None when the requested point is outside the moving nest
          instead of stopping the entire meteogram.
    """
    ncfile_path, time_index, latitude, longitude = args

    with Dataset(ncfile_path) as ncfile:
        # Valid time from metadata (preferred) or filename
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Extracting meteogram data: {valid_dt:%H:%M:%SZ %Y-%m-%d}")

        # ------------------------------------------------------------------
        # Find grid indices for the given latitude and longitude.
        # This MUST use timeidx for moving/vortex-following nests.
        # ------------------------------------------------------------------
        try:
            xy = moving_nest_ll_to_xy(ncfile, latitude, longitude, time_index)
        except Exception as e:
            print(
                f"Skipping {valid_dt:%H:%M:%SZ %Y-%m-%d}: "
                f"could not locate lat={latitude}, lon={longitude} "
                f"in {os.path.basename(ncfile_path)}: {e}"
            )
            return None

        if xy is None:
            print(
                f"Skipping {valid_dt:%H:%M:%SZ %Y-%m-%d}: "
                f"lat={latitude}, lon={longitude} is outside the moving "
                f"{os.path.basename(ncfile_path)} domain."
            )
            return None

        x, y = xy

        # ------------------------------------------------------------------
        # Extract required variables from the WRF file.
        # ------------------------------------------------------------------
        try:
            # 2m Temperature in °C (T2 is K in WRF)
            temp = float(wrf.getvar(ncfile, "T2", timeidx=time_index)[y, x] - 273.15)

            # 2m Dew Point in °C
            dew_point = float(wrf.getvar(ncfile, "td2", timeidx=time_index)[y, x])

            # 2m Relative Humidity in %
            rh2 = float(wrf.getvar(ncfile, "rh2", timeidx=time_index)[y, x])

            # Sea level pressure (hPa)
            pressure = float(wrf.getvar(ncfile, "slp", timeidx=time_index)[y, x])

            # Solar radiation (W/m²)
            solar_rad = float(wrf.getvar(ncfile, "SWDOWN", timeidx=time_index)[y, x])

            # Cumulative rainfall (mm)
            rain = float(
                wrf.getvar(ncfile, "RAINC", timeidx=time_index)[y, x]
                + wrf.getvar(ncfile, "RAINNC", timeidx=time_index)[y, x]
                + wrf.getvar(ncfile, "RAINSH", timeidx=time_index)[y, x]
            )

            # 10m wind components (m/s)
            wind_u = float(wrf.getvar(ncfile, "U10", timeidx=time_index)[y, x])
            wind_v = float(wrf.getvar(ncfile, "V10", timeidx=time_index)[y, x])
            wind_speed = float(np.sqrt(wind_u**2 + wind_v**2))
            wind_direction = float(calculate_wind_direction(wind_u, wind_v))

            # Cumulative snow water equivalent (mm), multiplied by 10
            snowh20 = float(wrf.getvar(ncfile, "SNOW", timeidx=time_index)[y, x] * 10.0)

        except KeyError as e:
            raise RuntimeError(
                f"Variable {e} not found in WRF file {ncfile_path} "
                f"at time_index={time_index}."
            )

    # ----------------------------------------------------------------------
    # Ensure that Td <= T when RH is 100%.
    # ----------------------------------------------------------------------
    if dew_point > temp and rh2 >= 99.9:
        dew_point = temp

    # Return instantaneous values; rates will be computed later.
    return (
        valid_dt,
        temp,
        dew_point,
        rh2,
        pressure,
        solar_rad,
        rain,
        wind_speed,
        wind_direction,
        snowh20,
    )



###############################################################################
# Frame discovery: handle multi-file and multi-time setups
###############################################################################
def discover_frames(ncfile_paths):
    """
    Discover all (file, time_index) combinations.

    Supports:
        * Many wrfout_<domain>* files with one or more Time steps.
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


###############################################################################
# Main script entry point
###############################################################################
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Check command-line arguments
    # ----------------------------------------------------------------------
    if len(sys.argv) != 6:
        print(
            "Usage: python Meteogram_DegC_moving_nest_fixed.py "
            "<path_to_WRF> <domain> <city> <latitude> <longitude>"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]  # Path to WRF files
    domain = sys.argv[2]  # WRF domain (e.g., d01, d02, etc.)
    city = sys.argv[3]  # City name
    latitude = float(sys.argv[4])
    longitude = float(sys.argv[5])

    # ----------------------------------------------------------------------
    # Set up the output directory where the meteogram will be saved
    # ----------------------------------------------------------------------
    output_dir = f"meteogram_{city}_metric"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # Find all WRF output files for this domain
    # ----------------------------------------------------------------------
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print(f"No wrfout files found in {path_wrf} matching wrfout_{domain}*")
        sys.exit(0)

    # ----------------------------------------------------------------------
    # Discover all (file, time_index) frames
    # ----------------------------------------------------------------------
    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    # Build argument list for the worker function
    args_list = [
        (ncfile_path, time_index, latitude, longitude)
        for (ncfile_path, time_index) in frames
    ]

    # ----------------------------------------------------------------------
    # Extract point data in parallel
    # ----------------------------------------------------------------------
    max_workers = min(4, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        raw_results = list(executor.map(process_frame, args_list))

    skipped_frames = sum(result is None for result in raw_results)
    results = [result for result in raw_results if result is not None]

    if skipped_frames:
        print(
            f"Skipped {skipped_frames} frame(s) because the requested point "
            f"was outside the moving nest or could not be sampled."
        )

    # Sort results by valid time (safety)
    results.sort(key=lambda r: r[0])

    # ----------------------------------------------------------------------
    # Initialize lists to store data for each variable
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

    # Initialize previous cumulative rain/snow for rate calculation
    previous_time = None
    previous_rain = None
    previous_snowh20 = None

    # ----------------------------------------------------------------------
    # Aggregate instantaneous values and compute hourly rates from cumulative
    # precipitation fields. Hourly output is unchanged; sub-hourly output is
    # normalized to mm/hr.
    # ----------------------------------------------------------------------
    for i, (
        time,
        temp,
        dew_point,
        rh2,
        pressure,
        solar_rad,
        rain,
        wind_speed,
        wind_direction,
        snowh20,
    ) in enumerate(results):
        time_points.append(time)

        # Hourly rates from cumulative values (mm/hr)
        rain_rate = safe_diff_rate(rain, previous_rain, time, previous_time)
        snowh20_rate = safe_diff_rate(
            snowh20,
            previous_snowh20,
            time,
            previous_time,
        )

        previous_time = time
        previous_rain = rain
        previous_snowh20 = snowh20

        # Store rounded values (metric units)
        temp_list.append(np.round(temp, 1))
        dew_point_list.append(np.round(dew_point, 1))
        rh2_list.append(np.round(rh2, 1))
        pressure_list.append(np.round(pressure, 1))
        solar_rad_list.append(np.round(solar_rad, 1))

        rainfall_list.append(np.round(rain, 1))  # cumulative rain
        rain_rate_list.append(np.round(rain_rate, 1))  # rain rate (mm/hr)

        wind_speed_list.append(np.round(wind_speed, 1))
        wind_direction_list.append(np.round(wind_direction, 1))

        snowh20_rate_list.append(np.round(snowh20_rate, 1))  # snow rate (mm/hr)
        cumulative_snowh20_list.append(np.round(snowh20, 1))  # cumulative snow (mm)

    if not time_points:
        print("No data points extracted for meteogram.")
        sys.exit(0)

    ###########################################################################
    # Create the meteogram plot with 6 subplots (plotting unchanged)
    ###########################################################################
    fig, ax = plt.subplots(6, 1, figsize=(20, 24), dpi=250, sharex=True)
    fig.patch.set_facecolor("white")

    # ----------------------------------------------------------------------
    # 1. Sea Level Pressure
    # ----------------------------------------------------------------------
    ax[0].plot(
        time_points,
        pressure_list,
        label="Pressure (hPa)",
        color="black",
        linewidth=2,
        marker="o",
    )
    ax[0].set_ylabel("Pressure (hPa)")
    ax[0].set_title("Sea Level Pressure")
    ax[0].grid()

    ax[0].xaxis.set_minor_locator(AutoMinorLocator(2))
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    ax[0].minorticks_on()

    # ----------------------------------------------------------------------
    # 2. Temperature and Dew Point
    # ----------------------------------------------------------------------
    ax[1].plot(
        time_points,
        temp_list,
        label="Temperature (°C)",
        color="red",
        linewidth=2,
        marker="o",
    )
    ax[1].plot(
        time_points,
        dew_point_list,
        label="Dew Point (°C)",
        color="green",
        linewidth=2,
        marker="o",
    )
    ax[1].set_ylabel("Temperature (°C)")
    ax[1].set_title("2m Temperature (Red) and Dew Point (Green)")
    ax[1].grid()

    ax[1].xaxis.set_minor_locator(AutoMinorLocator(2))
    ax[1].yaxis.set_minor_locator(MultipleLocator(0.1))
    ax[1].minorticks_on()
    ax[1].legend(loc="best", fontsize=10)

    # ----------------------------------------------------------------------
    # 3. Relative Humidity
    # ----------------------------------------------------------------------
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
    ax[2].grid()
    ax[2].set_ylim(0, 100)

    ax[2].xaxis.set_minor_locator(AutoMinorLocator(2))
    ax[2].yaxis.set_minor_locator(MultipleLocator(10))
    ax[2].minorticks_on()

    # ----------------------------------------------------------------------
    # 4. Wind Speed and Wind Direction
    # ----------------------------------------------------------------------
    bar_ws = ax[3].bar(
        time_points,
        wind_speed_list,
        color="lightgrey",
        label="Wind Speed (m/s)",
        width=0.02,
        edgecolor="black",
        linewidth=0.5,
    )

    ax3_secondary = ax[3].twinx()
    scatter_wd = ax3_secondary.scatter(
        time_points,
        wind_direction_list,
        color="orange",
        label="Wind Direction (°)",
        s=30,
    )

    ax[3].set_ylabel("Wind Speed (m/s)")
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

    # ----------------------------------------------------------------------
    # 5. 1-Hour Precipitation Rates (Rain & Snow, side-by-side bars)
    # ----------------------------------------------------------------------
    ax4_primary = ax[4]
    ax4_secondary = ax4_primary.twinx()

    time_nums = mdates.date2num(time_points)
    bar_width = median_time_step_days(time_points)
    offset = bar_width * 0.60

    bar_rain = ax4_primary.bar(
        time_nums - offset,
        rain_rate_list,
        color="green",
        label="1-hr Rain Rate (mm/hr)",
        width=bar_width,
        edgecolor="black",
        linewidth=0.5,
    )

    bar_snow = ax4_secondary.bar(
        time_nums + offset,
        snowh20_rate_list,
        color="cyan",
        label="1-hr Snow Rate (mm/hr)",
        width=bar_width,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.5,
    )

    ax4_primary.set_ylabel("Rain Rate (mm/hr)", color="black", fontsize=12)
    ax4_secondary.set_ylabel("Snow Rate (mm/hr)", color="black", fontsize=12)
    ax4_primary.set_title("1-Hour Precipitation Rates", fontsize=14)
    ax4_primary.grid()

    ax4_primary.set_ylim(bottom=0)
    ax4_secondary.set_ylim(bottom=0)

    ax4_primary.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4_primary.yaxis.set_minor_locator(MultipleLocator(1))
    ax4_primary.minorticks_on()

    ax4_secondary.yaxis.set_minor_locator(MultipleLocator(1))
    ax4_secondary.minorticks_on()

    # Label rain bars
    for rect in bar_rain:
        height = rect.get_height()
        if height >= 0.1:
            ax4_primary.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
                color="black",
            )

    # Label snow bars
    for rect in bar_snow:
        height = rect.get_height()
        if height >= 0.1:
            ax4_secondary.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
                color="black",
            )

    handles_p, labels_p = ax4_primary.get_legend_handles_labels()
    handles_s, labels_s = ax4_secondary.get_legend_handles_labels()
    ax4_primary.legend(
        handles_p + handles_s,
        labels_p + labels_s,
        loc="upper left",
        fontsize=10,
    )

    # ----------------------------------------------------------------------
    # 6. Solar Radiation
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Shared x-axis formatting and final layout
    # ----------------------------------------------------------------------
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%SZ %Y-%m-%d"))
    plt.xticks(rotation=90)
    ax[-1].set_xlabel("UTC Time (Z)")

    plt.suptitle(
        f"Meteogram for {city} at {latitude}°, {longitude}°\n"
        f"Time Period: {time_points[0].strftime('%H:%M:%SZ %Y-%m-%d')} "
        f"to {time_points[-1].strftime('%H:%M:%SZ %Y-%m-%d')}",
        fontsize=18,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    # ----------------------------------------------------------------------
    # Save the meteogram to a file
    # ----------------------------------------------------------------------
    output_file = os.path.join(output_dir, f"meteogram_{city}_metric.png")
    plt.savefig(output_file, dpi=250, facecolor=fig.get_facecolor())
    plt.close()

    print(f"Meteogram saved to {output_file}")
