#!/usr/bin/env python3
"""
WRF_Back_Trajectory.py

===============================================================================
USER HELP
===============================================================================

Purpose
-------
This script calculates WRF-native back trajectories directly from WRF-ARW
wrfout files. It is designed for academic/research use where a consistent,
repeatable method is preferred over many run-time method switches.

The script:
  * finds WRF output files for the requested domain,
  * reads all available WRF output times,
  * automatically uses the latest available WRF time as the trajectory start,
  * automatically uses the maximum available back-trajectory length,
  * supports nested domains by falling back outward, for example d03 -> d02 -> d01,
  * calculates one or more back trajectories from one or more launch locations,
  * supports multiple starting heights in meters AGL,
  * writes one CSV per trajectory,
  * writes a combined CSV when multiple trajectories are requested,
  * creates a Cartopy map with a vertical parcel-path panel underneath.

Standard academic method
------------------------
The streamlined default method is:

  * Horizontal wind source:
      Raw WRF U and V variables, destaggered and rotated to earth-relative flow.

  * Vertical motion:
      Passive 3D parcel motion using WRF W.

  * Lower-boundary behavior:
      If a passive parcel descends below the lowest usable WRF mass level,
      it is placed on the local lowest WRF mass level and continues.

  * Time window:
      Start at the latest available WRF output time and integrate backward
      through the full available WRF period.

  * Nested domains:
      If the requested domain is an inner nest and the parcel exits it,
      sampling continues on the parent domain.

Most common command
-------------------
Run from any folder, while pointing --wrf-dir to the WRF run directory:

  python3 WRF_Back_Trajectory.py \
    d03 SanMarcos \
    29.8899 -97.9961 \
    --wrf-dir /home/workhorse/WRF_Intel/WRF-4.7.1/run \
    --height-levels-m 100,500,1000

Deeper vertical profile example
-------------------------------
  python3 WRF_Back_Trajectory.py \
    d03 SanMarcos \
    29.8899 -97.9961 \
    --wrf-dir /home/workhorse/WRF_Intel/WRF-4.7.1/run \
    --height-levels-m 50,100,250,500,750,1000,1500,2000,2500,3000,4000,5000

Multiple-location example
-------------------------
  python3 WRF_Back_Trajectory.py \
    d03 SanMarcos \
    29.8899 -97.9961 \
    --wrf-dir /home/workhorse/WRF_Intel/WRF-4.7.1/run \
    --height-levels-m 100,500,1000 \
    --extra-location Austin,30.2672,-97.7431 \
    --extra-location CorpusChristi,27.8006,-97.3964

CSV locations-file example
--------------------------
Create a CSV file like this:

  city,lat,lon
  SanMarcos,29.8899,-97.9961
  Austin,30.2672,-97.7431
  CorpusChristi,27.8006,-97.3964

Then run:

  python3 WRF_Back_Trajectory.py \
    d03 SanMarcos \
    29.8899 -97.9961 \
    --wrf-dir /home/workhorse/WRF_Intel/WRF-4.7.1/run \
    --height-levels-m 100,500,1000 \
    --locations-file locations.csv

Main user options
-----------------
Required positional arguments:

  domain
      WRF domain to start with, for example d01, d02, or d03.

  city
      Name used for labels and output filenames.

  lat lon
      Launch-point latitude and longitude.

Common optional arguments:

  --wrf-dir PATH
      Directory containing wrfout files.

  --height-levels-m LIST
      Comma-separated starting heights in meters AGL.
      Example: 100,500,1000

  --extra-location CITY,LAT,LON
      Add another launch point to the same run and map.
      Can be repeated.

  --locations-file FILE.csv
      CSV file containing city, lat, and lon columns.

  --start-time latest
      Default. Uses the latest WRF output time.
      You may also give an explicit UTC time, such as 2026-06-26_00:00:00.

  --back-hours max
      Default. Uses the maximum available back period before the selected start.
      You may also give a number, such as 6, 12, or 24.

  --dt-min 15
      Trajectory time step in minutes.

Output files
------------
The script writes outputs into:

  wrf_back_trajectories_<CityName>/

Typical outputs include:

  wrf_backtraj_<domain>_<city>_<time>_<height>mAGL_passive_w_modelclip_raw.csv
  wrf_backtraj_<domain>_MULTI_<time>_levels_<heights>mAGL_passive_w_modelclip_raw.csv
  wrf_backtraj_<domain>_MULTI_<time>_levels_<heights>mAGL_passive_w_modelclip_raw.png

Important CSV columns
---------------------
  city
      Launch location name.

  start_height_agl_m
      Requested starting height in meters AGL.

  age_hours_back
      Hours before the trajectory start time.

  lat, lon
      Parcel position.

  height_agl_m
      Parcel height AGL used by the integration.

  height_agl_used_m
      Actual model-interpolation height used at the sample point.

  wrf_domain_used
      Domain sampled at that step, for example d03, d02, or d01.

  u_east_mps, v_north_mps, w_mps
      Interpolated parcel wind components.

  clipped_below_model, clipped_above_model
      Whether the requested parcel height was outside the local model vertical range.

Requirements
------------
This script is intended to be run inside a Python environment that has:

  * wrf-python
  * netCDF4
  * numpy
  * pandas
  * matplotlib
  * cartopy
  * geopandas
  * metpy

In this project, that environment is usually activated with:

  conda activate wrf-python

Notes
-----
For inner domains, the map grid may show the parent domain grid spacing because
the map must be large enough to display the full trajectory after it leaves the
inner nest. The CSV records which WRF domain was used at every step.
"""



from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as crs
import cartopy.feature as cfeature
import geopandas as gpd
import metpy.calc as mpcalc
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
import pandas as pd
from netCDF4 import Dataset

try:
    import wrf
    from wrf import to_np
except Exception as exc:  # pragma: no cover, environment dependent
    wrf = None
    to_np = None
    _WRF_IMPORT_ERROR = exc
else:
    _WRF_IMPORT_ERROR = None

warnings.filterwarnings("ignore")

EARTH_RADIUS_M = 6_371_000.0
GRAVITY = 9.81
DEFAULT_TZ = "America/Chicago"
RAW_REQUIRED_VARIABLES = [
    "Times",
    "XLAT",
    "XLONG",
    "U",
    "V",
    "W",
    "PH",
    "PHB",
    "HGT",
    "SINALPHA",
    "COSALPHA",
]


@dataclass(frozen=True)
class FrameRef:
    """One WRF output frame."""

    path: str
    time_idx: int
    time_utc: datetime
    domain: str = ""


@dataclass
class VelocitySample:
    """Interpolated velocity and sampling diagnostics."""

    u_east_mps: float
    v_north_mps: float
    w_mps: float
    x: float
    y: float
    z_agl_requested_m: float
    z_agl_used_m: float
    terrain_m: float
    clipped_vertical: bool
    lower_frame_time_utc: datetime
    upper_frame_time_utc: datetime
    time_weight_upper: float
    field_source_used: str
    domain_used: str = ""
    z_agl_min_m: float = float("nan")
    z_agl_max_m: float = float("nan")
    clipped_below_model: bool = False
    clipped_above_model: bool = False


def require_wrf_python() -> None:
    """Fail with a direct message when wrf-python is unavailable."""
    if wrf is None:
        raise ImportError(
            "This script needs wrf-python for WRF projection transforms "
            "(ll_to_xy). Install/use the same WRF Python environment used by "
            "your Skew-T script."
        ) from _WRF_IMPORT_ERROR


def ensure_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_user_time_utc(text: str) -> datetime:
    """Parse a trajectory start time and treat naive times as UTC."""
    clean = text.strip().replace("_", "T")
    if clean.endswith("Z"):
        clean = clean[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(clean)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse --start-time '{text}'. Use YYYY-MM-DD_HH:MM:SS."
        ) from exc
    return ensure_utc(dt)


def parse_valid_time_from_wrf_name(path: str) -> datetime:
    """Fallback parser for standard wrfout filenames."""
    base = os.path.basename(path)
    match = re.search(
        r"wrfout_.*?_(\d{4}-\d{2}-\d{2})[_T](\d{2}[:_]\d{2}[:_]\d{2})",
        base,
    )
    if match:
        date_str = match.group(1)
        time_str = match.group(2).replace("_", ":")
        try:
            return datetime.strptime(
                f"{date_str}_{time_str}", "%Y-%m-%d_%H:%M:%S"
            ).replace(tzinfo=timezone.utc)
        except Exception:
            pass

    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)


def get_valid_time(ncfile: Dataset, ncfile_path: str, time_idx: int) -> datetime:
    """Read the valid WRF time using metadata first and filename fallback second."""
    require_wrf_python()
    try:
        valid = wrf.extract_times(ncfile, timeidx=time_idx)
        if isinstance(valid, np.ndarray):
            valid = valid.item()
        if isinstance(valid, np.datetime64):
            valid = valid.astype("datetime64[ms]").tolist()
        if isinstance(valid, datetime):
            return ensure_utc(valid)
    except Exception:
        pass
    return parse_valid_time_from_wrf_name(ncfile_path)


def build_wrfout_search_patterns(
    path_wrf: str, domain: str, file_glob: str | None = None, recursive: bool = False
) -> List[str]:
    """Build the exact wrfout search patterns used by this script."""
    if file_glob:
        return [os.path.expandvars(os.path.expanduser(file_glob))]

    root = Path(os.path.expandvars(os.path.expanduser(path_wrf))).resolve()
    if root.is_file():
        return [str(root)]

    patterns = [
        str(root / f"wrfout_{domain}*"),
        str(root / f"*wrfout_{domain}*"),
    ]
    if recursive:
        patterns.extend(
            [
                str(root / "**" / f"wrfout_{domain}*"),
                str(root / "**" / f"*wrfout_{domain}*"),
            ]
        )
    return patterns


def find_wrfout_files(
    path_wrf: str, domain: str, file_glob: str | None = None, recursive: bool = False
) -> List[str]:
    """Find wrfout files for one domain using a directory, file, or full glob."""
    files: List[str] = []
    for pattern in build_wrfout_search_patterns(path_wrf, domain, file_glob, recursive):
        if any(ch in pattern for ch in "*?["):
            files.extend(glob.glob(pattern, recursive=recursive))
        elif os.path.isfile(pattern):
            files.append(pattern)

    # Only keep real wrfout files for the requested domain unless a full file_glob
    # was provided. This prevents met_em, wrfinput, and WPS FILE:* products from
    # accidentally being used as trajectory input.
    out: List[str] = []
    for candidate in files:
        base = os.path.basename(candidate)
        if file_glob or base.startswith(f"wrfout_{domain}") or f"wrfout_{domain}" in base:
            out.append(os.path.abspath(candidate))
    return sorted(set(out))


def discover_frames(ncfile_paths: Iterable[str], domain: str = "") -> List[FrameRef]:
    """Return all WRF frames, including multi-Time files."""
    require_wrf_python()
    frames: List[FrameRef] = []
    for path in ncfile_paths:
        with Dataset(path) as nc:
            if "Time" in nc.dimensions:
                n_times = len(nc.dimensions["Time"])
            elif "Times" in nc.variables:
                n_times = nc.variables["Times"].shape[0]
            else:
                n_times = 1

            for t in range(n_times):
                frames.append(
                    FrameRef(path=path, time_idx=t, time_utc=get_valid_time(nc, path, t), domain=domain)
                )
    frames.sort(key=lambda f: f.time_utc)
    return frames


def domain_number(domain: str) -> int | None:
    """Return the numeric part of a WRF domain name such as d03."""
    match = re.fullmatch(r"d(\d+)", str(domain).strip().lower())
    if not match:
        return None
    return int(match.group(1))


def parent_domain_chain(domain: str) -> list[str]:
    """Return selected domain followed by its parent domains, for example d03 -> d03,d02,d01."""
    number = domain_number(domain)
    if number is None or number <= 1:
        return [domain]
    return [f"d{i:02d}" for i in range(number, 0, -1)]


def build_domain_frame_sets(
    path_wrf: str,
    requested_domain: str,
    file_glob: str | None,
    recursive: bool,
    use_parent_domains: bool,
) -> tuple[list[tuple[str, list[FrameRef], list[str]]], list[str]]:
    """Build ordered domain frame sets for inner-nest sampling with parent-domain fallback."""
    if file_glob:
        files = find_wrfout_files(path_wrf, requested_domain, file_glob=file_glob, recursive=recursive)
        frames = discover_frames(files, requested_domain) if files else []
        return [(requested_domain, frames, files)], [requested_domain]

    domains = parent_domain_chain(requested_domain) if use_parent_domains else [requested_domain]
    frame_sets: list[tuple[str, list[FrameRef], list[str]]] = []

    for domain in domains:
        files = find_wrfout_files(path_wrf, domain, file_glob=None, recursive=recursive)
        if not files:
            if domain == requested_domain:
                return [], domains
            print(f"NOTE: Parent-domain fallback skipped {domain}; no wrfout files were found.")
            continue
        frames = discover_frames(files, domain)
        if not frames:
            if domain == requested_domain:
                return [], domains
            print(f"NOTE: Parent-domain fallback skipped {domain}; no WRF frames were found.")
            continue
        frame_sets.append((domain, frames, files))

    return frame_sets, domains


def write_variable_inventory(first_wrfout: str, out_csv: Path) -> None:
    """Write all variables, dimensions, units, descriptions, and stagger flags."""
    rows = []
    with Dataset(first_wrfout) as nc:
        for name, var in nc.variables.items():
            rows.append(
                {
                    "name": name,
                    "dimensions": ",".join(var.dimensions),
                    "shape": "x".join(str(s) for s in var.shape),
                    "units": getattr(var, "units", ""),
                    "description": getattr(var, "description", ""),
                    "stagger": getattr(var, "stagger", ""),
                    "memory_order": getattr(var, "MemoryOrder", ""),
                    "coordinates": getattr(var, "coordinates", ""),
                }
            )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def write_required_variable_report(first_wrfout: str, out_csv: Path) -> None:
    """Write a focused report for the raw variables needed by this script."""
    rows = []
    with Dataset(first_wrfout) as nc:
        for name in RAW_REQUIRED_VARIABLES:
            if name in nc.variables:
                var = nc.variables[name]
                rows.append(
                    {
                        "name": name,
                        "present": True,
                        "dimensions": ",".join(var.dimensions),
                        "shape": "x".join(str(s) for s in var.shape),
                        "units": getattr(var, "units", ""),
                        "description": getattr(var, "description", ""),
                        "stagger": getattr(var, "stagger", ""),
                    }
                )
            else:
                rows.append(
                    {
                        "name": name,
                        "present": False,
                        "dimensions": "",
                        "shape": "",
                        "units": "",
                        "description": "",
                        "stagger": "",
                    }
                )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def validate_raw_variables(nc: Dataset) -> None:
    """Raise if raw fallback variables are missing."""
    missing = [name for name in RAW_REQUIRED_VARIABLES if name not in nc.variables]
    if missing:
        raise KeyError(
            "Raw WRF field source is missing required variable(s): " + ", ".join(missing)
        )


def to_float_array(value) -> np.ndarray:
    """Convert xarray, masked arrays, or netCDF data to float numpy arrays."""
    if to_np is not None:
        try:
            value = to_np(value)
        except Exception:
            pass
    if np.ma.isMaskedArray(value):
        value = np.ma.filled(value, np.nan)
    return np.asarray(value, dtype=float)


def read_time_variable(nc: Dataset, name: str, time_idx: int) -> np.ndarray:
    """Read a NetCDF variable, applying time_idx only when it has a Time dimension."""
    var = nc.variables[name]
    if var.dimensions and var.dimensions[0] == "Time":
        return to_float_array(var[time_idx, ...])
    return to_float_array(var[...])


def destagger_x(u_stag: np.ndarray) -> np.ndarray:
    return 0.5 * (u_stag[:, :, :-1] + u_stag[:, :, 1:])


def destagger_y(v_stag: np.ndarray) -> np.ndarray:
    return 0.5 * (v_stag[:, :-1, :] + v_stag[:, 1:, :])


def destagger_z(w_stag: np.ndarray) -> np.ndarray:
    return 0.5 * (w_stag[:-1, :, :] + w_stag[1:, :, :])


def xy_from_latlon(path: str, time_idx: int, lat: float, lon: float) -> Tuple[float, float]:
    """Convert lat/lon to fractional WRF mass-grid x/y coordinates."""
    require_wrf_python()
    with Dataset(path) as nc:
        try:
            xy = wrf.ll_to_xy(nc, lat, lon, timeidx=time_idx, as_int=False, meta=False)
        except TypeError:
            xy = wrf.ll_to_xy(nc, lat, lon, timeidx=time_idx)
    arr = np.asarray(xy, dtype=float)
    return float(arr[0]), float(arr[1])


def bilinear_column(field3d: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinearly interpolate a 3D mass-grid field in x/y to a vertical column."""
    nz, ny, nx = field3d.shape
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    if x0 < 0 or y0 < 0 or x1 >= nx or y1 >= ny:
        raise ValueError(
            "Point is outside the domain or too close to the edge for bilinear interpolation."
        )

    wx = x - x0
    wy = y - y0
    c00 = field3d[:, y0, x0]
    c10 = field3d[:, y0, x1]
    c01 = field3d[:, y1, x0]
    c11 = field3d[:, y1, x1]
    return (
        c00 * (1.0 - wx) * (1.0 - wy)
        + c10 * wx * (1.0 - wy)
        + c01 * (1.0 - wx) * wy
        + c11 * wx * wy
    )


def bilinear_2d(field2d: np.ndarray, x: float, y: float) -> float:
    """Bilinearly interpolate a 2D mass-grid field at x/y."""
    ny, nx = field2d.shape
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    if x0 < 0 or y0 < 0 or x1 >= nx or y1 >= ny:
        raise ValueError("Point is outside the domain edge for 2D interpolation.")
    wx = x - x0
    wy = y - y0
    return float(
        field2d[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + field2d[y0, x1] * wx * (1.0 - wy)
        + field2d[y1, x0] * (1.0 - wx) * wy
        + field2d[y1, x1] * wx * wy
    )


def interp_to_height(
    values_col: np.ndarray, z_col: np.ndarray, target_z_m: float
) -> Tuple[float, float, bool]:
    """Interpolate a column to target AGL height, clipping to valid model levels."""
    values = np.asarray(values_col, dtype=float)
    z = np.asarray(z_col, dtype=float)
    good = np.isfinite(values) & np.isfinite(z)
    values = values[good]
    z = z[good]

    if len(z) < 2:
        raise ValueError("Not enough valid vertical levels for interpolation.")

    order = np.argsort(z)
    z = z[order]
    values = values[order]
    z_used = float(np.clip(target_z_m, z[0], z[-1]))
    clipped = not math.isclose(z_used, target_z_m, rel_tol=0.0, abs_tol=1.0e-6)
    return float(np.interp(z_used, z, values)), z_used, clipped


def load_frame_fields_wrfpython(
    path: str, time_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int], str]:
    """Load earth-relative winds and AGL height using wrf-python diagnostics."""
    require_wrf_python()
    with Dataset(path) as nc:
        uvmet = to_float_array(wrf.getvar(nc, "uvmet", timeidx=time_idx, units="m s-1", meta=False))
        if uvmet.ndim != 4 or uvmet.shape[0] != 2:
            raise ValueError(f"Unexpected uvmet shape {uvmet.shape} in {path}")
        u_east = uvmet[0, :, :, :]
        v_north = uvmet[1, :, :, :]
        w_mass = to_float_array(wrf.getvar(nc, "wa", timeidx=time_idx, units="m s-1", meta=False))
        z_agl = to_float_array(wrf.getvar(nc, "height_agl", timeidx=time_idx, units="m", meta=False))
        terrain = read_time_variable(nc, "HGT", time_idx)

    return verify_field_shapes(u_east, v_north, w_mass, z_agl, terrain, "wrfpython")


def load_frame_fields_raw(
    path: str, time_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int], str]:
    """Load and derive mass-grid, earth-relative fields directly from raw wrfout variables."""
    with Dataset(path) as nc:
        validate_raw_variables(nc)
        u_grid = destagger_x(read_time_variable(nc, "U", time_idx))
        v_grid = destagger_y(read_time_variable(nc, "V", time_idx))
        w_mass = destagger_z(read_time_variable(nc, "W", time_idx))
        ph = read_time_variable(nc, "PH", time_idx)
        phb = read_time_variable(nc, "PHB", time_idx)
        terrain = read_time_variable(nc, "HGT", time_idx)
        sinalpha = read_time_variable(nc, "SINALPHA", time_idx)
        cosalpha = read_time_variable(nc, "COSALPHA", time_idx)

    z_w_msl = (ph + phb) / GRAVITY
    z_msl = destagger_z(z_w_msl)
    z_agl = z_msl - terrain[np.newaxis, :, :]

    # Rotate grid-relative mass-point winds to east/north winds.
    u_east = u_grid * cosalpha[np.newaxis, :, :] - v_grid * sinalpha[np.newaxis, :, :]
    v_north = v_grid * cosalpha[np.newaxis, :, :] + u_grid * sinalpha[np.newaxis, :, :]

    return verify_field_shapes(u_east, v_north, w_mass, z_agl, terrain, "raw")


def verify_field_shapes(
    u_east: np.ndarray,
    v_north: np.ndarray,
    w_mass: np.ndarray,
    z_agl: np.ndarray,
    terrain: np.ndarray,
    source_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int], str]:
    """Check that trajectory fields are all on the same mass grid."""
    if u_east.shape != v_north.shape or u_east.shape != w_mass.shape or u_east.shape != z_agl.shape:
        raise ValueError(
            "Wind and height fields do not share the same mass-grid shape: "
            f"u={u_east.shape}, v={v_north.shape}, w={w_mass.shape}, z_agl={z_agl.shape}."
        )
    if terrain.shape != u_east.shape[1:]:
        raise ValueError(
            f"Terrain shape {terrain.shape} does not match horizontal grid {u_east.shape[1:]}"
        )
    return u_east, v_north, w_mass, z_agl, terrain, u_east.shape, source_name


@lru_cache(maxsize=12)
def load_frame_fields(
    path: str, time_idx: int, field_source: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int], str]:
    """Load fields from wrf-python diagnostics, raw variables, or automatic fallback."""
    if field_source == "wrfpython":
        return load_frame_fields_wrfpython(path, time_idx)
    if field_source == "raw":
        return load_frame_fields_raw(path, time_idx)

    try:
        return load_frame_fields_wrfpython(path, time_idx)
    except Exception:
        return load_frame_fields_raw(path, time_idx)


def bracket_frames(frames: Sequence[FrameRef], t_utc: datetime) -> Tuple[FrameRef, FrameRef, float]:
    """Find lower/upper WRF frames bracketing t_utc and upper-frame weight."""
    if t_utc < frames[0].time_utc or t_utc > frames[-1].time_utc:
        raise ValueError(
            f"Trajectory time {t_utc.isoformat()} is outside WRF frame coverage "
            f"{frames[0].time_utc.isoformat()} to {frames[-1].time_utc.isoformat()}."
        )

    for frame in frames:
        if t_utc == frame.time_utc:
            return frame, frame, 0.0

    for idx in range(len(frames) - 1):
        lo = frames[idx]
        hi = frames[idx + 1]
        if lo.time_utc <= t_utc <= hi.time_utc:
            total = (hi.time_utc - lo.time_utc).total_seconds()
            if total <= 0:
                return lo, hi, 0.0
            return lo, hi, float((t_utc - lo.time_utc).total_seconds() / total)

    raise ValueError(f"Could not bracket time {t_utc.isoformat()}.")


def sample_velocity_at_frame(
    frame: FrameRef, lat: float, lon: float, z_agl_m: float, field_source: str
) -> VelocitySample:
    """Sample one WRF frame at a parcel lat/lon/height."""
    x, y = xy_from_latlon(frame.path, frame.time_idx, lat, lon)
    u3d, v3d, w3d, z3d, terrain2d, shape, source_used = load_frame_fields(
        frame.path, frame.time_idx, field_source
    )

    _nz, ny, nx = shape
    if x < 0 or y < 0 or x >= nx - 1 or y >= ny - 1:
        raise ValueError(
            f"Parcel left the WRF domain at lat={lat:.6f}, lon={lon:.6f}, x={x:.3f}, y={y:.3f}."
        )

    u_col = bilinear_column(u3d, x, y)
    v_col = bilinear_column(v3d, x, y)
    w_col = bilinear_column(w3d, x, y)
    z_col = bilinear_column(z3d, x, y)
    terrain_m = bilinear_2d(terrain2d, x, y)

    z_min = float(np.nanmin(z_col))
    z_max = float(np.nanmax(z_col))
    clipped_below = bool(z_agl_m < z_min)
    clipped_above = bool(z_agl_m > z_max)

    u, z_used_u, clip_u = interp_to_height(u_col, z_col, z_agl_m)
    v, z_used_v, clip_v = interp_to_height(v_col, z_col, z_agl_m)
    w, z_used_w, clip_w = interp_to_height(w_col, z_col, z_agl_m)
    z_used = float(np.mean([z_used_u, z_used_v, z_used_w]))

    return VelocitySample(
        u_east_mps=u,
        v_north_mps=v,
        w_mps=w,
        x=x,
        y=y,
        z_agl_requested_m=z_agl_m,
        z_agl_used_m=z_used,
        terrain_m=terrain_m,
        clipped_vertical=bool(clip_u or clip_v or clip_w),
        lower_frame_time_utc=frame.time_utc,
        upper_frame_time_utc=frame.time_utc,
        time_weight_upper=0.0,
        field_source_used=source_used,
        domain_used=frame.domain,
        z_agl_min_m=z_min,
        z_agl_max_m=z_max,
        clipped_below_model=clipped_below,
        clipped_above_model=clipped_above,
    )


def sample_velocity(
    frames: Sequence[FrameRef],
    t_utc: datetime,
    lat: float,
    lon: float,
    z_agl_m: float,
    field_source: str,
) -> VelocitySample:
    """Sample time-interpolated winds at parcel position."""
    lo, hi, weight_hi = bracket_frames(frames, t_utc)
    s_lo = sample_velocity_at_frame(lo, lat, lon, z_agl_m, field_source)

    if lo == hi:
        s_lo.lower_frame_time_utc = lo.time_utc
        s_lo.upper_frame_time_utc = hi.time_utc
        s_lo.time_weight_upper = 0.0
        return s_lo

    s_hi = sample_velocity_at_frame(hi, lat, lon, z_agl_m, field_source)
    wlo = 1.0 - weight_hi
    whi = weight_hi
    source_used = s_lo.field_source_used if s_lo.field_source_used == s_hi.field_source_used else "mixed"
    domain_used = s_lo.domain_used if s_lo.domain_used == s_hi.domain_used else f"{s_lo.domain_used}+{s_hi.domain_used}"

    return VelocitySample(
        u_east_mps=(s_lo.u_east_mps * wlo) + (s_hi.u_east_mps * whi),
        v_north_mps=(s_lo.v_north_mps * wlo) + (s_hi.v_north_mps * whi),
        w_mps=(s_lo.w_mps * wlo) + (s_hi.w_mps * whi),
        x=(s_lo.x * wlo) + (s_hi.x * whi),
        y=(s_lo.y * wlo) + (s_hi.y * whi),
        z_agl_requested_m=z_agl_m,
        z_agl_used_m=(s_lo.z_agl_used_m * wlo) + (s_hi.z_agl_used_m * whi),
        terrain_m=(s_lo.terrain_m * wlo) + (s_hi.terrain_m * whi),
        clipped_vertical=bool(s_lo.clipped_vertical or s_hi.clipped_vertical),
        lower_frame_time_utc=lo.time_utc,
        upper_frame_time_utc=hi.time_utc,
        time_weight_upper=weight_hi,
        field_source_used=source_used,
        domain_used=domain_used,
        z_agl_min_m=(s_lo.z_agl_min_m * wlo) + (s_hi.z_agl_min_m * whi),
        z_agl_max_m=(s_lo.z_agl_max_m * wlo) + (s_hi.z_agl_max_m * whi),
        clipped_below_model=bool(s_lo.clipped_below_model or s_hi.clipped_below_model),
        clipped_above_model=bool(s_lo.clipped_above_model or s_hi.clipped_above_model),
    )


def sample_velocity_with_domain_fallback(
    domain_frame_sets: Sequence[tuple[str, Sequence[FrameRef]]],
    t_utc: datetime,
    lat: float,
    lon: float,
    z_agl_m: float,
    field_source: str,
) -> VelocitySample:
    """Sample winds from the innermost available domain, falling back outward when needed."""
    last_domain = None
    last_error = None

    for domain, frames in domain_frame_sets:
        last_domain = domain
        try:
            sample = sample_velocity(frames, t_utc, lat, lon, z_agl_m, field_source)
            sample.domain_used = domain
            return sample
        except ValueError as exc:
            message = str(exc)
            if "Parcel left the WRF domain" in message:
                last_error = exc
                continue
            raise

    raise ValueError(
        "Parcel left all available WRF domains at "
        f"lat={lat:.6f}, lon={lon:.6f}. Last domain checked: {last_domain}. "
        f"Last error: {last_error}"
    )


def advance_position(
    lat: float,
    lon: float,
    z_agl_m: float,
    sample: VelocitySample,
    dt_seconds: float,
    vertical_mode: str,
    vertical_floor_m: float | None = None,
    surface_behavior: str = "floor",
) -> Tuple[float, float, float]:
    """Advance a parcel using earth-relative U/V and optional passive W."""
    lat_rad = math.radians(lat)
    cos_lat = max(math.cos(lat_rad), 1.0e-6)
    dlat_deg = (sample.v_north_mps * dt_seconds / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon_deg = (sample.u_east_mps * dt_seconds / (EARTH_RADIUS_M * cos_lat)) * (
        180.0 / math.pi
    )
    new_lat = lat + dlat_deg
    new_lon = lon + dlon_deg

    if vertical_mode == "passive_w":
        new_z = z_agl_m + sample.w_mps * dt_seconds
        if surface_behavior == "floor":
            floor = 0.0 if vertical_floor_m is None else max(0.0, float(vertical_floor_m))
            new_z = max(floor, new_z)
    else:
        new_z = z_agl_m

    return new_lat, new_lon, new_z


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2.0
    ) ** 2
    return (EARTH_RADIUS_M * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))) / 1000.0


def integrate_back_trajectory(
    domain_frame_sets: Sequence[tuple[str, Sequence[FrameRef]]],
    start_time_utc: datetime,
    start_lat: float,
    start_lon: float,
    start_height_agl_m: float,
    back_hours: float,
    dt_min: float,
    vertical_mode: str,
    field_source: str,
    vertical_floor_m: float | None = None,
    surface_behavior: str = "floor",
) -> pd.DataFrame:
    """Integrate a single parcel backward using midpoint integration."""
    if dt_min <= 0:
        raise ValueError("--dt-min must be positive.")
    if back_hours <= 0:
        raise ValueError("--back-hours must be positive.")
    if start_height_agl_m < 0:
        raise ValueError("--height-agl-m cannot be negative.")
    if vertical_floor_m is not None and vertical_floor_m < 0:
        raise ValueError("--vertical-floor-m cannot be negative.")
    if surface_behavior not in {"floor", "model_clip", "stop"}:
        raise ValueError("--surface-behavior must be floor, model_clip, or stop.")
    if vertical_floor_m is not None and (vertical_mode != "passive_w" or surface_behavior != "floor"):
        print("NOTE: --vertical-floor-m only affects --vertical-mode passive_w with --surface-behavior floor.")

    target_seconds = back_hours * 3600.0
    nominal_step_seconds = abs(dt_min) * 60.0
    elapsed_seconds = 0.0
    step = 0
    rows = []

    lat = float(start_lat)
    lon = float(start_lon)
    z_agl = float(start_height_agl_m)
    t = start_time_utc
    prev_lat = lat
    prev_lon = lon

    while True:
        sample0 = sample_velocity_with_domain_fallback(domain_frame_sets, t, lat, lon, z_agl, field_source)
        surface_limited = False
        trajectory_stop_reason = ""

        if vertical_mode == "passive_w" and sample0.clipped_below_model:
            surface_limited = True
            if surface_behavior == "model_clip":
                z_agl = sample0.z_agl_used_m
            elif surface_behavior == "stop":
                z_agl = sample0.z_agl_used_m
                trajectory_stop_reason = "lower_model_boundary"

        segment_distance_km = 0.0 if step == 0 else haversine_km(prev_lat, prev_lon, lat, lon)
        distance_from_start_km = haversine_km(start_lat, start_lon, lat, lon)
        rows.append(
            {
                "step": step,
                "age_hours_back": elapsed_seconds / 3600.0,
                "time_utc": t.isoformat().replace("+00:00", "Z"),
                "lat": lat,
                "lon": lon,
                "height_agl_m": z_agl,
                "height_agl_used_m": sample0.z_agl_used_m,
                "terrain_m": sample0.terrain_m,
                "height_msl_m": sample0.terrain_m + z_agl,
                "z_agl_min_model_m": sample0.z_agl_min_m,
                "z_agl_max_model_m": sample0.z_agl_max_m,
                "clipped_below_model": sample0.clipped_below_model,
                "clipped_above_model": sample0.clipped_above_model,
                "surface_limited_by_behavior": surface_limited,
                "trajectory_stop_reason": trajectory_stop_reason,
                "u_east_mps": sample0.u_east_mps,
                "v_north_mps": sample0.v_north_mps,
                "w_mps": sample0.w_mps,
                "wind_speed_horizontal_mps": math.hypot(sample0.u_east_mps, sample0.v_north_mps),
                "wrf_x": sample0.x,
                "wrf_y": sample0.y,
                "wrf_domain_used": sample0.domain_used,
                "segment_distance_km": segment_distance_km,
                "distance_from_start_km": distance_from_start_km,
                "vertical_clipped_to_model_levels": sample0.clipped_vertical,
                "lower_frame_time_utc": sample0.lower_frame_time_utc.isoformat().replace("+00:00", "Z"),
                "upper_frame_time_utc": sample0.upper_frame_time_utc.isoformat().replace("+00:00", "Z"),
                "time_weight_upper": sample0.time_weight_upper,
                "vertical_mode": vertical_mode,
                "surface_behavior": surface_behavior,
                "vertical_floor_m": vertical_floor_m,
                "field_source_used": sample0.field_source_used,
            }
        )

        if trajectory_stop_reason:
            break

        if elapsed_seconds >= target_seconds - 1.0e-6:
            break

        remaining = target_seconds - elapsed_seconds
        step_seconds = -min(nominal_step_seconds, remaining)

        half_t = t + timedelta(seconds=step_seconds / 2.0)
        mid_lat, mid_lon, mid_z = advance_position(
            lat, lon, z_agl, sample0, step_seconds / 2.0, vertical_mode, vertical_floor_m, surface_behavior
        )
        sample_mid = sample_velocity_with_domain_fallback(domain_frame_sets, half_t, mid_lat, mid_lon, mid_z, field_source)

        prev_lat, prev_lon = lat, lon
        lat, lon, z_agl = advance_position(
            lat, lon, z_agl, sample_mid, step_seconds, vertical_mode, vertical_floor_m, surface_behavior
        )
        t = t + timedelta(seconds=step_seconds)
        elapsed_seconds += abs(step_seconds)
        step += 1

    return pd.DataFrame(rows)


def safe_tag(text: str) -> str:
    """Make text safe for filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def add_local_time_columns(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    """Add a local-time column to the output table."""
    tz = ZoneInfo(tz_name)
    out = df.copy()
    t = pd.to_datetime(out["time_utc"], utc=True)
    out["time_local"] = t.dt.tz_convert(tz).astype(str)
    return out



###############################################################################
# Cartopy map method copied into this self-contained trajectory script
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


def plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km, plot_extent=None):
    if plot_extent is None:
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

    This function is field-agnostic: pass any number of fields, or none.
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
# Natural Earth features from the uploaded Cartopy chart method
###############################################################################
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
# Cities from the uploaded Cartopy chart method
###############################################################################
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


def nearest_frame_for_time(frames: Sequence[FrameRef], target_time_utc: datetime) -> FrameRef:
    """Return the WRF frame nearest to the trajectory start time for map setup."""
    target = ensure_utc(target_time_utc)
    return min(frames, key=lambda frame: abs((frame.time_utc - target).total_seconds()))


def read_plot_map_context(frames: Sequence[FrameRef], start_time_utc: datetime):
    """
    Read the WRF-native Cartopy projection, lat/lon grid, grid spacing, and map extent.

    This intentionally follows the same map setup pattern as the uploaded
    250-hPa Cartopy chart: read a WRF field, get lat/lon coordinates, compute
    grid spacing with MetPy, and use wrf.get_cartopy for the native WRF projection.
    The final plotted extent is calculated later from the trajectory bounds so
    the back track is viewable instead of forcing the full parent-domain view.
    """
    frame = nearest_frame_for_time(frames, start_time_utc)
    with Dataset(frame.path) as ncfile:
        # Terrain is a 2D WRF field with WRF projection metadata and lat/lon coords.
        grid = wrf.getvar(ncfile, "ter", timeidx=frame.time_idx)
        lats, lons = wrf.latlon_coords(grid)
        (
            lats_np,
            lons_np,
            avg_dx_km,
            avg_dy_km,
            extent_adjustment,
            label_adjustment,
        ) = compute_grid_and_spacing(lats, lons)
        (lats_np, lons_np) = handle_domain_continuity_and_polar_mask(lats_np, lons_np)
        cart_proj = wrf.get_cartopy(grid)

    return {
        "frame": frame,
        "lats_np": lats_np,
        "lons_np": lons_np,
        "avg_dx_km": avg_dx_km,
        "avg_dy_km": avg_dy_km,
        "extent_adjustment": extent_adjustment,
        "label_adjustment": label_adjustment,
        "cart_proj": cart_proj,
    }


def compute_trajectory_view_extent(
    df: pd.DataFrame,
    lons_np: np.ndarray,
    lats_np: np.ndarray,
    avg_dx_km: float,
    avg_dy_km: float,
    extent_adjustment: float,
) -> list[float]:
    """
    Build a map extent centered on the trajectory instead of the full WRF domain.

    The projection and base map still come from the WRF frame exactly like the
    250-hPa Cartopy plotting script. Only the extent changes: it is based on the
    back-trajectory path plus enough padding to keep the track, nearby cities,
    and direction labels readable. The extent is then clamped to the WRF domain.
    """
    traj_lons = np.asarray(df["lon"].values, dtype=float)
    traj_lats = np.asarray(df["lat"].values, dtype=float)

    lon_min = float(np.nanmin(traj_lons))
    lon_max = float(np.nanmax(traj_lons))
    lat_min = float(np.nanmin(traj_lats))
    lat_max = float(np.nanmax(traj_lats))

    domain_lon_min = float(np.nanmin(lons_np))
    domain_lon_max = float(np.nanmax(lons_np))
    domain_lat_min = float(np.nanmin(lats_np))
    domain_lat_max = float(np.nanmax(lats_np))

    # Keep a useful minimum window. Parent domains need a larger context window,
    # while nests can be tighter.
    if avg_dx_km >= 9 or avg_dy_km >= 9:
        min_lat_span = 1.60
        min_lon_span = 1.60
    elif 3 < avg_dx_km < 9 or 3 < avg_dy_km < 9:
        min_lat_span = 1.00
        min_lon_span = 1.00
    else:
        min_lat_span = 0.55
        min_lon_span = 0.55

    raw_lon_span = max(lon_max - lon_min, 0.01)
    raw_lat_span = max(lat_max - lat_min, 0.01)

    pad_lon = max(extent_adjustment, raw_lon_span * 0.25)
    pad_lat = max(extent_adjustment, raw_lat_span * 0.25)

    lon_span = max(raw_lon_span + 2.0 * pad_lon, min_lon_span)
    lat_span = max(raw_lat_span + 2.0 * pad_lat, min_lat_span)

    # Match the wide 3840x2160 figure shape so the plotted track is not cramped.
    center_lat = 0.5 * (lat_min + lat_max)
    cos_lat = max(math.cos(math.radians(center_lat)), 0.30)
    target_aspect = 16.0 / 9.0
    current_aspect = (lon_span * cos_lat) / lat_span
    if current_aspect < target_aspect:
        lon_span = (target_aspect * lat_span) / cos_lat
    elif current_aspect > target_aspect:
        lat_span = (lon_span * cos_lat) / target_aspect

    center_lon = 0.5 * (lon_min + lon_max)
    center_lat = 0.5 * (lat_min + lat_max)

    domain_lon_span = domain_lon_max - domain_lon_min
    domain_lat_span = domain_lat_max - domain_lat_min

    if lon_span >= domain_lon_span:
        view_lon_min, view_lon_max = domain_lon_min, domain_lon_max
    else:
        view_lon_min = center_lon - lon_span / 2.0
        view_lon_max = center_lon + lon_span / 2.0
        if view_lon_min < domain_lon_min:
            view_lon_max += domain_lon_min - view_lon_min
            view_lon_min = domain_lon_min
        if view_lon_max > domain_lon_max:
            view_lon_min -= view_lon_max - domain_lon_max
            view_lon_max = domain_lon_max

    if lat_span >= domain_lat_span:
        view_lat_min, view_lat_max = domain_lat_min, domain_lat_max
    else:
        view_lat_min = center_lat - lat_span / 2.0
        view_lat_max = center_lat + lat_span / 2.0
        if view_lat_min < domain_lat_min:
            view_lat_max += domain_lat_min - view_lat_min
            view_lat_min = domain_lat_min
        if view_lat_max > domain_lat_max:
            view_lat_min -= view_lat_max - domain_lat_max
            view_lat_max = domain_lat_max

    return [view_lon_min, view_lon_max, view_lat_min, view_lat_max]


def add_direction_arrows(ax, df: pd.DataFrame, n_arrows: int = 3) -> None:
    """Draw arrows along the plotted line pointing backward in time."""
    if len(df) < 4 or n_arrows <= 0:
        return
    transform = crs.PlateCarree()
    idxs = np.linspace(1, len(df) - 2, min(n_arrows, max(1, len(df) // 10)), dtype=int)
    for idx in idxs:
        lon0 = float(df["lon"].iloc[idx])
        lat0 = float(df["lat"].iloc[idx])
        lon1 = float(df["lon"].iloc[idx + 1])
        lat1 = float(df["lat"].iloc[idx + 1])
        ax.annotate(
            "",
            xy=(lon1, lat1),
            xytext=(lon0, lat0),
            xycoords=transform._as_mpl_transform(ax),
            textcoords=transform._as_mpl_transform(ax),
            arrowprops=dict(arrowstyle="->", linewidth=1.2, shrinkA=0, shrinkB=0),
            zorder=30,
        )


def compute_multi_trajectory_view_extent(
    traj_sets: Sequence[dict],
    lons_np: np.ndarray,
    lats_np: np.ndarray,
    avg_dx_km: float,
    avg_dy_km: float,
    extent_adjustment: float,
) -> list[float]:
    """Build a trajectory-centered view extent that includes all plotted trajectories."""
    frames = []
    for traj in traj_sets:
        df = traj["df"]
        frames.append(df[["lon", "lat"]].copy())
    merged = pd.concat(frames, ignore_index=True)
    return compute_trajectory_view_extent(
        df=merged,
        lons_np=lons_np,
        lats_np=lats_np,
        avg_dx_km=avg_dx_km,
        avg_dy_km=avg_dy_km,
        extent_adjustment=extent_adjustment,
    )


def parse_extra_location_spec(text: str) -> tuple[str, float, float]:
    """Parse one CITY,LAT,LON extra-location specification."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Could not parse extra location '{text}'. Use CITY,LAT,LON, for example Austin,30.2672,-97.7431"
        )
    city = parts[0]
    try:
        lat = float(parts[1])
        lon = float(parts[2])
    except ValueError as exc:
        raise ValueError(
            f"Could not parse latitude/longitude in extra location '{text}'."
        ) from exc
    return city, lat, lon


def load_launch_points(args: argparse.Namespace) -> list[dict]:
    """Build the full launch-point list from the main point plus optional extras."""
    points = [{"city": args.city, "lat": float(args.lat), "lon": float(args.lon)}]

    for item in args.extra_location or []:
        city, lat, lon = parse_extra_location_spec(item)
        points.append({"city": city, "lat": lat, "lon": lon})

    if args.locations_file:
        loc_path = Path(args.locations_file).expanduser().resolve()
        df_loc = pd.read_csv(loc_path)
        required = {"city", "lat", "lon"}
        missing = required - set(df_loc.columns.str.lower()) if hasattr(df_loc.columns, 'str') else set()
        if missing:
            # Try case-insensitive rename first.
            lower_map = {str(col).lower(): col for col in df_loc.columns}
            if not required.issubset(lower_map):
                raise ValueError(
                    f"Locations file must contain columns city, lat, lon. Missing: {sorted(required - set(lower_map))}"
                )
            df_loc = df_loc.rename(columns={lower_map['city']: 'city', lower_map['lat']: 'lat', lower_map['lon']: 'lon'})
        else:
            rename_map = {}
            for col in df_loc.columns:
                low = str(col).lower()
                if low in required:
                    rename_map[col] = low
            df_loc = df_loc.rename(columns=rename_map)
        for _, row in df_loc.iterrows():
            points.append({
                "city": str(row['city']).strip(),
                "lat": float(row['lat']),
                "lon": float(row['lon']),
            })

    deduped = []
    seen = set()
    for p in points:
        key = (p['city'], round(p['lat'], 6), round(p['lon'], 6))
        if key not in seen:
            deduped.append(p)
            seen.add(key)
    return deduped


def parse_height_values(args: argparse.Namespace) -> list[float]:
    """Return the requested starting parcel heights in meters AGL."""
    if args.height_levels_m:
        parts = [part for part in re.split(r"[,\s]+", args.height_levels_m.strip()) if part]
        if not parts:
            raise ValueError("--height-levels-m was supplied, but no heights were found.")
        heights = [float(part) for part in parts]
    else:
        heights = [float(args.height_agl_m)]

    cleaned = []
    seen = set()
    for height in heights:
        if height < 0:
            raise ValueError("Trajectory starting heights cannot be negative.")
        key = round(height, 6)
        if key not in seen:
            cleaned.append(height)
            seen.add(key)
    return cleaned


def format_height_summary(heights: Sequence[float]) -> str:
    """Make a compact title string for one or more trajectory starting heights."""
    unique_heights = []
    seen = set()
    for height in heights:
        key = round(float(height), 6)
        if key not in seen:
            unique_heights.append(float(height))
            seen.add(key)

    if len(unique_heights) == 1:
        return f"{unique_heights[0]:g} m AGL"
    return ", ".join(f"{height:g}" for height in unique_heights) + " m AGL"


def make_height_group_tag(heights: Sequence[float]) -> str:
    """Make a compact filename tag for one or more starting heights."""
    if len(heights) == 1:
        return safe_tag(f"{heights[0]:g}mAGL")
    joined = "-".join(f"{height:g}" for height in heights)
    return safe_tag(f"levels_{joined}mAGL")


def plot_vertical_profile_panel(ax, traj_sets: Sequence[dict], args: argparse.Namespace) -> None:
    """Plot parcel-height history for one or more trajectories."""
    for traj in traj_sets:
        df = traj['df']
        color = traj['color']
        ax.plot(
            df['age_hours_back'].values,
            df['height_agl_m'].values,
            color=color,
            linewidth=2.0,
            marker='o',
            markersize=2.8,
            label=traj.get('label', traj['city']),
        )
        ax.scatter(
            [df['age_hours_back'].iloc[0]],
            [df['height_agl_m'].iloc[0]],
            marker='*',
            s=80,
            color=color,
            edgecolor='black',
            linewidth=0.5,
            zorder=5,
        )
        ax.scatter(
            [df['age_hours_back'].iloc[-1]],
            [df['height_agl_m'].iloc[-1]],
            marker='x',
            s=50,
            color=color,
            linewidth=1.5,
            zorder=5,
        )
    ax.set_xlabel('Hours back')
    ax.set_ylabel('Height AGL (m)')
    ax.set_title('Vertical parcel path', fontsize=12, pad=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(0, max(float(traj['df']['age_hours_back'].max()) for traj in traj_sets))
    ymax = max(float(traj['df']['height_agl_m'].max()) for traj in traj_sets)
    ymin = min(float(traj['df']['height_agl_m'].min()) for traj in traj_sets)
    if math.isclose(ymax, ymin):
        pad = max(50.0, ymax * 0.1 if ymax > 0 else 50.0)
    else:
        pad = max(25.0, (ymax - ymin) * 0.10)
    ax.set_ylim(max(0.0, ymin - pad * 0.10), ymax + pad)
    ax.legend(loc='upper right', ncol=min(4, len(traj_sets)), fontsize=9)


def plot_trajectory_map(
    traj_sets: Sequence[dict],
    out_png: Path,
    title: str,
    args: argparse.Namespace,
    frames: Sequence[FrameRef],
    start_time_utc: datetime,
) -> None:
    """Plot one or more WRF back trajectories using the exact Cartopy method from the WRF chart."""
    context = read_plot_map_context(frames, start_time_utc)
    lats_np = context['lats_np']
    lons_np = context['lons_np']
    avg_dx_km = context['avg_dx_km']
    avg_dy_km = context['avg_dy_km']
    extent_adjustment = context['extent_adjustment']
    cart_proj = context['cart_proj']

    dpi = plt.rcParams.get('figure.dpi', 400)
    fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)

    # The map is the main panel. The parcel-height panel is always directly
    # below it. Explicit margins keep the title attached to the map and avoid
    # a large blank area at the top of the figure.
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.15, 1.15],
        left=0.035,
        right=0.985,
        bottom=0.060,
        top=0.925,
        hspace=0.210,
    )
    ax = fig.add_subplot(gs[0, 0], projection=cart_proj)
    ax_prof = fig.add_subplot(gs[1, 0])

    view_extent = compute_multi_trajectory_view_extent(
        traj_sets=traj_sets,
        lons_np=lons_np,
        lats_np=lats_np,
        avg_dx_km=avg_dx_km,
        avg_dy_km=avg_dy_km,
        extent_adjustment=extent_adjustment,
    )
    ax.set_extent(view_extent, crs=crs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS['land'])
    for feature in features:
        add_feature(ax, *feature)
    plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km, plot_extent=view_extent)
    add_latlon_gridlines(ax)

    data_crs = crs.PlateCarree()
    cmap = plt.cm.get_cmap('tab20', max(3, len(traj_sets)))

    for idx, traj in enumerate(traj_sets):
        df = traj['df']
        color = cmap(idx % cmap.N)
        traj['color'] = color
        ax.plot(
            df['lon'].values,
            df['lat'].values,
            marker='o',
            markersize=2.8,
            linewidth=2.2,
            color=color,
            transform=data_crs,
            zorder=20,
            label=traj.get('label', traj['city']),
        )
        ax.scatter(
            [df['lon'].iloc[0]],
            [df['lat'].iloc[0]],
            marker='*',
            s=160,
            color=color,
            edgecolor='black',
            linewidth=0.6,
            transform=data_crs,
            zorder=25,
        )
        ax.scatter(
            [df['lon'].iloc[-1]],
            [df['lat'].iloc[-1]],
            marker='x',
            s=90,
            color=color,
            linewidth=2.0,
            transform=data_crs,
            zorder=25,
        )
        add_direction_arrows(ax, df, n_arrows=3)

        show_age_labels = len(traj_sets) <= 2
        if show_age_labels:
            text_every = max(1, len(df) // 8)
            for _, row in df.iloc[::text_every].iterrows():
                ax.text(
                    row['lon'],
                    row['lat'],
                    f"{row['age_hours_back']:.1f}h",
                    fontsize=9,
                    color='black',
                    transform=data_crs,
                    zorder=35,
                    bbox=dict(boxstyle='round,pad=0.10', facecolor='white', alpha=0.60),
                    clip_on=True,
                )

        ax.text(
            df['lon'].iloc[0],
            df['lat'].iloc[0],
            f" {traj['city']}",
            transform=data_crs,
            ha='left',
            va='center',
            fontsize=11,
            color='black',
            bbox=dict(boxstyle='round,pad=0.10', facecolor='white', alpha=0.65),
            zorder=36,
            clip_on=True,
        )

    plot_vertical_profile_panel(ax_prof, traj_sets, args)

    location_names = []
    for traj in traj_sets:
        if traj['city'] not in location_names:
            location_names.append(traj['city'])
    loc_text = location_names[0] if len(location_names) == 1 else f"{len(location_names)} locations"
    height_text = format_height_summary([traj['start_height_agl_m'] for traj in traj_sets])
    mode_text = "3D passive vertical motion" if args.vertical_mode == "passive_w" else "fixed AGL"
    if args.vertical_mode == "passive_w":
        if args.surface_behavior == "floor":
            floor_text = "0" if args.vertical_floor_m is None else f"{args.vertical_floor_m:g}"
            mode_text += f" | surface: floor {floor_text} m AGL"
        elif args.surface_behavior == "model_clip":
            mode_text += " | surface: model clip"
        elif args.surface_behavior == "stop":
            mode_text += " | surface: stop"

    domain_text = getattr(args, "domain_chain_text", args.domain)
    map_domain = getattr(args, "map_domain", args.domain)

    title_left = (
        f"WRF back trajectories | {domain_text} | {loc_text}\n"
        f"Start: {start_time_utc:%HZ %Y-%m-%d} | Length: {args.back_hours:g} h back | "
        f"Heights: {height_text} | {mode_text}"
    )
    title_right = f"Map grid: {map_domain} {avg_dx_km} x {avg_dy_km} km"

    ax.set_title(title_left, loc='left', fontsize=13, pad=8)
    ax.set_title(title_right, loc='right', fontsize=13, pad=8)

    ax.legend(loc='upper right', title='Trajectories')
    plt.savefig(out_png, bbox_inches='tight', dpi=100)
    plt.close(fig)



def plot_trajectory(
    traj_sets: Sequence[dict],
    out_png: Path,
    title: str,
    args: argparse.Namespace,
    frames: Sequence[FrameRef],
    start_time_utc: datetime,
) -> None:
    """Write the WRF-native Cartopy map PNG, optionally with a vertical parcel-path panel."""
    plot_trajectory_map(traj_sets, out_png, title, args, frames, start_time_utc)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate WRF-native back trajectories using a streamlined academic workflow. "
            "Defaults use the latest WRF output time, the full available back period, "
            "3D passive vertical motion, raw WRF fields, and model-level lower-boundary clipping."
        ),
        epilog=(
            "Examples:\n"
            "  python3 WRF_Back_Trajectory.py d03 SanMarcos 29.8899 -97.9961 "
            "--wrf-dir /home/workhorse/WRF_Intel/WRF-4.7.1/run --height-levels-m 100,500,1000\n\n"
            "  python3 WRF_Back_Trajectory.py d03 SanMarcos 29.8899 -97.9961 "
            "--wrf-dir /home/workhorse/WRF_Intel/WRF-4.7.1/run "
            "--height-levels-m 50,100,250,500,750,1000,1500,2000,2500,3000,4000,5000\n\n"
            "For a full method description, read the USER HELP section at the top of this file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "positional",
        nargs="+",
        help=(
            "Either: domain city lat lon, with --wrf-dir supplied or current "
            "directory used; OR legacy form: path_wrf domain city lat lon."
        ),
    )
    parser.add_argument(
        "--wrf-dir",
        default=None,
        help=(
            "Directory containing wrfout_<domain>* files. This is the preferred "
            "way to point the script at WRF output while running the script from elsewhere."
        ),
    )
    parser.add_argument(
        "--file-glob",
        default=None,
        help=(
            "Optional full glob for wrfout files. Overrides --wrf-dir search. "
            "Example: '/home/workhorse/WRF/run/wrfout_d01_2026-06-25*'."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search under --wrf-dir recursively for wrfout files.",
    )
    parser.add_argument(
        "--no-parent-domain-fallback",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--start-time", default="latest", help="Start time in UTC, or latest/first to select from discovered WRF frames.")
    parser.add_argument("--back-hours", default="max", help="Hours backward to integrate, or max to use all available WRF time before the selected start time.")
    parser.add_argument("--dt-min", type=float, default=5.0, help="Trajectory integration time step in minutes.")
    parser.add_argument("--time-margin-min", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--height-agl-m", type=float, default=100.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--height-levels-m",
        default="100,500,1000",
        help=(
            "Comma-separated starting parcel heights in meters AGL. "
            "Default is 100,500,1000."
        ),
    )
    parser.add_argument("--vertical-mode", choices=["passive_w"], default="passive_w", help=argparse.SUPPRESS)
    parser.add_argument("--vertical-floor-m", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--surface-behavior", choices=["model_clip"], default="model_clip", help=argparse.SUPPRESS)
    parser.add_argument("--field-source", choices=["raw"], default="raw", help=argparse.SUPPRESS)
    parser.add_argument("--tz", default=DEFAULT_TZ, help="Local timezone column to add to CSV.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to wrf_back_trajectories_<city>.")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plot creation.")
    parser.add_argument("--inspect-vars", action="store_true", help="Write a full variable inventory and required-variable report.")
    parser.add_argument("--inspect-only", action="store_true", help="Only write inventories and frame coverage, then exit.")
    parser.add_argument(
        "--extra-location",
        action="append",
        default=None,
        help="Add another launch point to the same map using CITY,LAT,LON. Repeat as needed.",
    )
    parser.add_argument(
        "--locations-file",
        default=None,
        help="Optional CSV file with columns city, lat, lon for additional launch points.",
    )
    parser.add_argument(
        "--vertical-profile",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def parse_cli_args() -> argparse.Namespace:
    """Parse old and new command styles into one normalized namespace."""
    parser = build_arg_parser()
    args = parser.parse_args()
    pos = args.positional

    if len(pos) == 4:
        args.domain, args.city, lat_text, lon_text = pos
        args.path_wrf = args.wrf_dir or os.getcwd()
    elif len(pos) == 5:
        legacy_path_wrf, args.domain, args.city, lat_text, lon_text = pos
        args.path_wrf = args.wrf_dir or legacy_path_wrf
    else:
        parser.error(
            "Use either: domain city lat lon --wrf-dir /path/to/wrf/run, "
            "or legacy: /path/to/wrf/run domain city lat lon."
        )

    try:
        args.lat = float(lat_text)
        args.lon = float(lon_text)
    except ValueError as exc:
        parser.error(f"Latitude and longitude must be numeric: {lat_text} {lon_text}")
        raise exc

    args.path_wrf = os.path.abspath(os.path.expandvars(os.path.expanduser(args.path_wrf)))
    # Academic streamlined defaults. These are intentionally fixed to reduce
    # method ambiguity in routine runs.
    args.vertical_mode = "passive_w"
    args.surface_behavior = "model_clip"
    args.vertical_floor_m = None
    args.field_source = "raw"
    return args


def resolve_trajectory_time_window(
    args: argparse.Namespace,
    frames: Sequence[FrameRef],
) -> tuple[datetime, float]:
    """Resolve manual or file-aware trajectory timing from discovered WRF frames."""
    if not frames:
        raise ValueError("No WRF frames are available for time-window selection.")

    coverage_start = frames[0].time_utc
    coverage_end = frames[-1].time_utc

    if args.time_margin_min < 0:
        raise ValueError("--time-margin-min must be zero or positive.")

    margin = timedelta(minutes=float(args.time_margin_min))
    usable_start = coverage_start + margin
    usable_end = coverage_end - margin

    if usable_end <= usable_start:
        raise ValueError(
            "The requested --time-margin-min removes the entire available WRF time window."
        )

    start_text = str(args.start_time).strip().lower()
    if start_text in {"latest", "last", "end"}:
        start_time_utc = usable_end
    elif start_text in {"first", "earliest", "begin", "beginning", "start"}:
        start_time_utc = usable_start
    else:
        start_time_utc = parse_user_time_utc(str(args.start_time))

    back_text = str(args.back_hours).strip().lower()
    if back_text in {"max", "maximum", "all", "available", "full"}:
        back_hours = (start_time_utc - usable_start).total_seconds() / 3600.0
    else:
        try:
            back_hours = float(args.back_hours)
        except ValueError as exc:
            raise ValueError("--back-hours must be a number of hours or max.") from exc

    if back_hours <= 0:
        raise ValueError("--back-hours must be greater than zero after time-window resolution.")

    requested_end = start_time_utc - timedelta(hours=back_hours)

    print("Resolved trajectory time window:")
    print(f"  WRF coverage start: {coverage_start.isoformat().replace('+00:00', 'Z')}")
    print(f"  WRF coverage end:   {coverage_end.isoformat().replace('+00:00', 'Z')}")
    if margin.total_seconds() > 0:
        print(f"  Usable start:       {usable_start.isoformat().replace('+00:00', 'Z')}")
        print(f"  Usable end:         {usable_end.isoformat().replace('+00:00', 'Z')}")
        print(f"  Time margin:        {args.time_margin_min:g} min")
    print(f"  Trajectory start:   {start_time_utc.isoformat().replace('+00:00', 'Z')}")
    print(f"  Trajectory end:     {requested_end.isoformat().replace('+00:00', 'Z')}")
    print(f"  Back length:        {back_hours:.2f} h")

    return start_time_utc, back_hours


def main() -> None:
    args = parse_cli_args()
    require_wrf_python()

    # --start-time may be an explicit UTC time or a file-aware keyword such as latest/first.
    use_parent_domains = not args.no_parent_domain_fallback
    domain_frame_sets_full, requested_domain_chain = build_domain_frame_sets(
        args.path_wrf,
        args.domain,
        file_glob=args.file_glob,
        recursive=args.recursive,
        use_parent_domains=use_parent_domains,
    )

    if not domain_frame_sets_full:
        patterns = build_wrfout_search_patterns(
            args.path_wrf, args.domain, args.file_glob, args.recursive
        )
        print("ERROR: No WRF history files were found.", file=sys.stderr)
        print(f"  Domain requested: {args.domain}", file=sys.stderr)
        print(f"  Current directory: {os.getcwd()}", file=sys.stderr)
        print("  Search patterns tried:", file=sys.stderr)
        for pattern in patterns:
            print(f"    {pattern}", file=sys.stderr)
        print("\nUse --wrf-dir to point at the folder where wrf.exe wrote wrfout files, for example:", file=sys.stderr)
        print(
            "  --wrf-dir /home/workhorse/WRF_Intel/WRFV4.6.1/run",
            file=sys.stderr,
        )
        sys.exit(1)

    # The first set is the requested domain. Later sets are parents used only
    # when a parcel moves outside the inner nest.
    primary_domain, frames, wrf_files = domain_frame_sets_full[0]
    domain_frame_sets = [(domain, frame_list) for domain, frame_list, _files in domain_frame_sets_full]
    args.domain_chain_text = " → ".join(domain for domain, _frames in domain_frame_sets)
    args.map_domain = domain_frame_sets[-1][0] if domain_frame_sets else args.domain

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"wrf_back_trajectories_{safe_tag(args.city)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(wrf_files)} wrfout file(s) for {args.domain}.")
    print(f"WRF input directory: {args.path_wrf}")
    print(f"First wrfout file: {wrf_files[0]}")
    print(f"Last wrfout file:  {wrf_files[-1]}")

    if len(domain_frame_sets) > 1:
        fallback_text = ", ".join(domain for domain, _frames in domain_frame_sets)
        print(f"Parent-domain fallback enabled: {fallback_text}")
    elif use_parent_domains and domain_number(args.domain) and domain_number(args.domain) > 1:
        print("Parent-domain fallback requested, but only the selected domain is available.")

    if args.inspect_vars or args.inspect_only:
        inventory_csv = out_dir / f"wrf_{args.domain}_variable_inventory.csv"
        required_csv = out_dir / f"wrf_{args.domain}_trajectory_required_variables.csv"
        write_variable_inventory(wrf_files[0], inventory_csv)
        write_required_variable_report(wrf_files[0], required_csv)
        print(f"Variable inventory written to: {inventory_csv}")
        print(f"Trajectory required-variable report written to: {required_csv}")

    if not frames:
        print("ERROR: No WRF frames found.", file=sys.stderr)
        sys.exit(1)

    coverage_start = frames[0].time_utc
    coverage_end = frames[-1].time_utc
    print(f"Discovered {len(frames)} WRF frame(s).")
    print(
        "WRF time coverage: "
        f"{coverage_start.isoformat().replace('+00:00', 'Z')} to "
        f"{coverage_end.isoformat().replace('+00:00', 'Z')}"
    )
    if args.inspect_only:
        return

    try:
        start_time_utc, resolved_back_hours = resolve_trajectory_time_window(args, frames)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    requested_end = start_time_utc - timedelta(hours=resolved_back_hours)
    if (
        start_time_utc < coverage_start
        or start_time_utc > coverage_end
        or requested_end < coverage_start
        or requested_end > coverage_end
    ):
        print(
            "ERROR: Requested trajectory is outside WRF time coverage.\n"
            f"  Requested: {requested_end.isoformat()} to {start_time_utc.isoformat()}\n"
            f"  Available: {coverage_start.isoformat()} to {coverage_end.isoformat()}",
            file=sys.stderr,
        )
        sys.exit(1)

    args.back_hours = resolved_back_hours

    print("Trajectory method:")
    print("  Horizontal wind: raw WRF U/V rotated to earth-relative flow")
    print("  Vertical motion: WRF W, passive parcel height")
    print("  Lower boundary: local lowest WRF mass level, then continue")
    print("  Parent domains: enabled for nested runs")

    launch_points = load_launch_points(args)
    height_values = parse_height_values(args)
    traj_sets = []

    start_tag = safe_tag(start_time_utc.strftime("%Y%m%d_%H%M%SZ"))
    height_group_tag = make_height_group_tag(height_values)
    mode_tag = safe_tag(args.vertical_mode)
    if args.vertical_mode == "passive_w":
        if args.surface_behavior == "floor" and args.vertical_floor_m is not None:
            mode_tag = f"{mode_tag}_floor{safe_tag(f'{args.vertical_floor_m:g}m')}"
        elif args.surface_behavior == "model_clip":
            mode_tag = f"{mode_tag}_modelclip"
        elif args.surface_behavior == "stop":
            mode_tag = f"{mode_tag}_stop"
    source_tag = safe_tag(args.field_source)

    for point in launch_points:
        for start_height_agl_m in height_values:
            height_tag = safe_tag(f"{start_height_agl_m:g}mAGL")
            df = integrate_back_trajectory(
                domain_frame_sets=domain_frame_sets,
                start_time_utc=start_time_utc,
                start_lat=point['lat'],
                start_lon=point['lon'],
                start_height_agl_m=start_height_agl_m,
                back_hours=args.back_hours,
                dt_min=args.dt_min,
                vertical_mode=args.vertical_mode,
                field_source=args.field_source,
                vertical_floor_m=args.vertical_floor_m,
                surface_behavior=args.surface_behavior,
            )
            df = add_local_time_columns(df, args.tz)
            df.insert(0, 'city', point['city'])
            df.insert(1, 'start_height_agl_m', start_height_agl_m)
            csv_path = out_dir / f"wrf_backtraj_{args.domain}_{safe_tag(point['city'])}_{start_tag}_{height_tag}_{mode_tag}_{source_tag}.csv"
            df.to_csv(csv_path, index=False, float_format="%.6f")
            print(f"Trajectory CSV written to: {csv_path}")

            if len(height_values) == 1 and len(launch_points) == 1:
                label = point['city']
            elif len(height_values) == 1:
                label = point['city']
            elif len(launch_points) == 1:
                label = f"{start_height_agl_m:g} m"
            else:
                label = f"{point['city']} {start_height_agl_m:g} m"

            traj_sets.append({
                'city': point['city'],
                'label': label,
                'lat': point['lat'],
                'lon': point['lon'],
                'start_height_agl_m': start_height_agl_m,
                'df': df,
                'csv_path': csv_path,
            })

    if len(traj_sets) > 1:
        combined_df = pd.concat([traj['df'] for traj in traj_sets], ignore_index=True)
        combined_csv = out_dir / f"wrf_backtraj_{args.domain}_MULTI_{start_tag}_{height_group_tag}_{mode_tag}_{source_tag}.csv"
        combined_df.to_csv(combined_csv, index=False, float_format="%.6f")
        print(f"Combined trajectory CSV written to: {combined_csv}")

    if not args.no_plot:
        if len(traj_sets) == 1:
            png_path = traj_sets[0]['csv_path'].with_suffix('.png')
        else:
            png_path = out_dir / f"wrf_backtraj_{args.domain}_MULTI_{start_tag}_{height_group_tag}_{mode_tag}_{source_tag}.png"
        title = (
            f"WRF Back Trajectory {args.domain}\n"
            f"Start {start_time_utc.strftime('%Y-%m-%d %H:%MZ')}, "
            f"{format_height_summary(height_values)}, {args.back_hours:g} h back"
        )
        map_frames = domain_frame_sets[-1][1] if domain_frame_sets else frames
        plot_trajectory(traj_sets, png_path, title, args, map_frames, start_time_utc)
        print(f"Trajectory PNG written to: {png_path}")

    combined_for_checks = pd.concat([traj['df'] for traj in traj_sets], ignore_index=True)
    if bool(combined_for_checks.get("clipped_above_model", pd.Series(dtype=bool)).any()):
        print(
            "WARNING: Some requested heights were above the available WRF mass-level range. "
            "Check height_agl_used_m and z_agl_max_model_m in the CSV."
        )
    if bool(combined_for_checks.get("clipped_below_model", pd.Series(dtype=bool)).any()):
        if args.vertical_mode == "passive_w" and args.surface_behavior == "model_clip":
            print("NOTE: Some parcels reached the lower model boundary, used the local lowest WRF mass level, and continued.")
        elif args.vertical_mode == "passive_w" and args.surface_behavior == "stop":
            print("NOTE: At least one passive trajectory stopped at the lower model boundary.")
        else:
            print(
                "WARNING: Some requested heights were below the available WRF mass-level range. "
                "Check height_agl_used_m and z_agl_min_model_m in the CSV."
            )
    print("Run complete.")


if __name__ == "__main__":
    main()
