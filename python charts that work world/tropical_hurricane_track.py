#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hurricane_tracker_units.py

Unified WRF operational and data-agnostic storm tracking.
Retains tropical_hurricane_track.py's map, intensity, CSV, ATCF-style file,
optional best-track verification, opt-in parent-domain fallback, and WRF diagnostics.
Uses stormtrack.py's black track, coral motion arrows, white centre, and warm
neutral land, with the Saffir-Simpson wind-category colours from
tropical_surface_slp_wind_speed_mph_direction.py for track markers and intensity;
includes stormtrack's independent
centroid tracker, pressure readers, temporal motion fit, and overlay API.

stormtrack component (MIT): Copyright attribution: SUMAILI NDEBA Bienvenu,
sumailib@gmail.com / sumaili.bienvenu@um6p.ma. See the original stormtrack.py
and its MIT LICENSE for complete terms. Other code derived from the supplied
WRF hurricane tracker.

Examples:
    python hurricane_tracker_independent_nests.py d02 --wrf-dir ~/WRF_Intel/WRF-4.7.1/run
    python hurricane_tracker_independent_nests.py d02 --wrf-dir /path/to/run --center-method stormtrack
    python hurricane_tracker_independent_nests.py generic --files '/data/*.nc' --init-lat 23 --init-lon -85
    python hurricane_tracker_independent_nests.py generic --files '/data/*.grib2' --grib-filter typeOfLevel=meanSea

Operational-style WRF hurricane/vortex tracker for static or moving nests.
Moving-nest coordinates are read independently at every file and time index.

Required:
    domain

Typical command:
    python3 WRF_Hurricane_Track.py d02 --wrf-dir /path/to/WRF/run

Optional first guess:
    python3 WRF_Hurricane_Track.py d02 --wrf-dir /path/to/WRF/run --init-lat 24.5 --init-lon -94.0

Default operational tracking:
    --max-jump-km 30 (trigger other methods; not a motion limit)
    --max-motion-speed-kt 65 (interval-scaled physical plausibility check)
    --center-coherence-km 90 (pressure/center spatial consistency)
    --prediction-tolerance-km 75 (motion-extrapolation uncertainty)
    --track-smooth-window 3
    --intensity-smooth-window 3
    --label-interval-hours 12

Outputs:
    * CSV with raw and smoothed centers
    * ATCF-style output
    * Track map
    * Intensity time series
    * Optional best-track comparison

Notes:
    Native WRF mode keeps each requested domain independent by default;
    --parent-fallback explicitly enables parent-domain recovery.
    Generic mode uses stormtrack's pressure centroid, accepts NetCDF/GRIB/Zarr,
    and sets unavailable upper-air diagnostics to NaN, not invented values.
    10 m model winds are written directly in knots; they are NOT converted to
    NHC-standard 1-minute sustained wind. Generic ATCF model ID is GEN.
"""

from __future__ import annotations

###############################################################################
# Imports
###############################################################################
import argparse
import glob
import math
import os
import re
import sys
import warnings
import xarray as xr
import matplotlib
matplotlib.use("Agg")
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# These dependencies are required only for the native WRF engine, or for
# optional cartographic detail. The generic engine works without them.
try:
    import cartopy.crs as crs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
except ImportError:
    crs = None
    cfeature = None
    LATITUDE_FORMATTER = LONGITUDE_FORMATTER = None
try:
    import geopandas as gpd
except ImportError:
    gpd = None
try:
    import metpy.calc as mpcalc
    from metpy.units import units
except ImportError:
    mpcalc = units = None
try:
    import wrf
    from wrf import ALL_TIMES, to_np
except ImportError:
    wrf = None
    ALL_TIMES = None
    to_np = np.asarray
try:
    from netCDF4 import Dataset
except ImportError:
    Dataset = None


###############################################################################
# Warning suppression
###############################################################################
# Do not suppress diagnostic warnings, especially missing fields.

###############################################################################
# Constants
###############################################################################
EARTH_RADIUS_KM = 6371.0
KNOTS_PER_MPS = 1.9438444924406046
MPH_PER_KNOT = 1.15078

GENERIC_ATCF_BASIN = "XX"
GENERIC_ATCF_NUMBER = "00"
GENERIC_ATCF_MODEL = "WRF"

# Background and motion colors from stormtrack.py; hurricane-track layout retained.
STORM_PALETTE = {
    "track": "#000000",
    "motion": "#d6604d",
    "land": "#e6e3dc",
    "water": "#ffffff",
    "center_face": "#ffffff",
    "best": "#727272",
    "grid": "gray",
    "pressure_cmap": "viridis_r",
}

# Exact Saffir-Simpson-style wind band thresholds and RGB colors from
# tropical_surface_slp_wind_speed_mph_direction.py (mph). The 220 mph plotting
# cap of that script is intentionally not a classification upper bound.
# These are visual wind-speed bins, NOT an official NHC intensity estimate:
# model grid-point 10-m maxima need not represent 1-minute sustained winds.
WIND_STRENGTHS = (
    ("Below Tropical Storm", -np.inf, 39.0, "#808080"),
    ("Tropical Storm", 39.0, 74.0, "#00FFFF"),
    ("Category 1", 74.0, 96.0, "#00FF00"),
    ("Category 2", 96.0, 111.0, "#FFFF00"),
    ("Category 3", 111.0, 130.0, "#FF8000"),
    ("Category 4", 130.0, 157.0, "#FF0000"),
    ("Category 5", 157.0, np.inf, "#FF00FF"),
)


def wind_strength(wind_mph: float) -> tuple[str, str]:
    """Return (wind-bin label, HEX color) with the reference script's breaks."""
    if not np.isfinite(wind_mph) or wind_mph < 0:
        return "Unknown", "#B0B0B0"
    for label, low, high, color in WIND_STRENGTHS:
        if low <= wind_mph < high:
            return label, color
    raise AssertionError("wind classification bounds must include all positive winds")


def strength_colors(wind_kt: np.ndarray) -> list[str]:
    """Color each center from its maximum model 10-m wind, not from SLP."""
    return [wind_strength(float(w) * MPH_PER_KNOT)[1] for w in wind_kt]


def wind_band_unit_label(low_mph: float, high_mph: float) -> str:
    """Show the *same* mph cutoffs in mph, m/s, and kt without changing bins.

    The <upper notation denotes a strict open upper boundary. Decimal values
    are display approximations; classification always uses original mph bounds.
    """
    def format_band(low: float, high: float, factor: float, precision: int, unit: str) -> str:
        def fmt(value: float) -> str:
            return f"{value * factor:.{precision}f}"
        if not np.isfinite(low):
            return f"<{fmt(high)} {unit}"
        if not np.isfinite(high):
            return f"≥{fmt(low)} {unit}"
        return f"{fmt(low)}–<{fmt(high)} {unit}"

    return "  |  ".join((
        format_band(low_mph, high_mph, 1., 0, "mph"),
        format_band(low_mph, high_mph, 1. / (KNOTS_PER_MPS * MPH_PER_KNOT), 1, "m/s"),
        format_band(low_mph, high_mph, 1. / MPH_PER_KNOT, 1, "kt"),
    ))


###############################################################################
# Canonical helper function block from WRF plotting scripts
###############################################################################
def add_feature(
    ax, category, scale, facecolor, edgecolor, linewidth, name, zorder=None, alpha=None
):
    if cfeature is None:
        return
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

    if mpcalc is None:
        # Same physical-distance calculation as stormtrack; no MetPy dependency.
        dxk = haversine_km(lats_np[:, :-1], lons_np[:, :-1], lats_np[:, 1:], lons_np[:, 1:])
        dyk = haversine_km(lats_np[:-1, :], lons_np[:-1, :], lats_np[1:, :], lons_np[1:, :])
        avg_dx_km = float(np.nanmean(dxk))
        avg_dy_km = float(np.nanmean(dyk))
        extent_adjustment = 0.50 if max(avg_dx_km, avg_dy_km) >= 9 else (0.25 if max(avg_dx_km, avg_dy_km) > 3 else 0.15)
        label_adjustment = 0.35 if max(avg_dx_km, avg_dy_km) >= 9 else (0.20 if max(avg_dx_km, avg_dy_km) > 3 else 0.15)
        return lats_np, lons_np, round(avg_dx_km, 2), round(avg_dy_km, 2), extent_adjustment, label_adjustment

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
    global cities
    cities = load_cities()
    if cities is None or crs is None:
        return
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
# Natural Earth features
###############################################################################
features = []
if cfeature is not None:
    features = [
        ("physical", "10m", STORM_PALETTE["land"], "black", 0.50, "minor_islands"),
        ("physical", "10m", "none", "black", 0.50, "coastline"),
        ("physical", "10m", STORM_PALETTE["water"], None, None, "ocean_scale_rank", -1),
        ("physical", "10m", STORM_PALETTE["water"], "lightgrey", 0.75, "lakes", 0),
        ("cultural", "10m", "none", "grey", 1.00, "admin_1_states_provinces", 2),
        ("cultural", "10m", "none", "black", 1.50, "admin_0_countries", 2),
    ]

###############################################################################
# Cities
###############################################################################
cities = None

def load_cities():
    """City data is optional; never download it during module import."""
    global cities
    if cities is not None:
        return cities
    if gpd is None:
        return None
    try:
        cities = gpd.read_file(
            "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
        )
    except Exception as exc:
        warnings.warn(f"Optional city layer unavailable: {exc}")
    return cities


###############################################################################
# Data classes
###############################################################################
@dataclass(frozen=True)
class FrameRef:
    domain: str
    path: str
    time_index: int
    time_utc: datetime


@dataclass
class FieldSet:
    domain: str
    path: str
    time_index: int
    time_utc: datetime
    lat: np.ndarray
    lon: np.ndarray
    slp_hpa: np.ndarray
    u10_mps: np.ndarray
    v10_mps: np.ndarray
    wspd10_mps: np.ndarray
    vort850: np.ndarray
    vort700: np.ndarray
    avg_dx_km: float
    avg_dy_km: float


@dataclass
class PointCandidate:
    lat: float
    lon: float
    value: float
    source: str
    valid: bool = True


###############################################################################
# General helpers
###############################################################################
def safe_tag(value: object) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return tag or "unknown"


def normalize_datetime(value: object, fallback: datetime | None = None) -> datetime:
    if isinstance(value, np.ndarray):
        value = value.item()

    if isinstance(value, np.datetime64):
        value = value.astype("datetime64[ms]").tolist()

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, bytes):
        dt = datetime.strptime(value.decode("utf-8"), "%Y-%m-%d_%H:%M:%S")
    elif isinstance(value, str):
        clean = value.replace("T", "_").replace("Z", "")
        dt = datetime.strptime(clean[:19], "%Y-%m-%d_%H:%M:%S")
    elif fallback is not None:
        dt = fallback
    else:
        raise ValueError(f"Could not parse datetime from {value!r}")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def parent_domain_chain(domain: str, use_parent_fallback: bool = False) -> list[str]:
    domain = domain.lower()
    match = re.fullmatch(r"d(\d+)", domain)

    if not match:
        return [domain]

    number = int(match.group(1))

    if not use_parent_fallback:
        return [f"d{number:02d}"]

    return [f"d{i:02d}" for i in range(number, 0, -1)]


def build_search_patterns(
    wrf_dir: str | None,
    domain: str,
    file_glob: str | None = None,
) -> list[str]:
    if file_glob:
        return [file_glob]

    base = Path(wrf_dir or ".").expanduser()

    return [
        str(base / f"wrfout_{domain}_*"),
        str(base / f"wrfout_{domain}:*"),
        str(base / f"wrfout_{domain}"),
    ]


def discover_frames_for_domain(
    domain: str,
    wrf_dir: str | None,
    file_glob: str | None = None,
) -> list[FrameRef]:
    paths: list[str] = []

    for pattern in build_search_patterns(wrf_dir, domain, file_glob):
        paths.extend(glob.glob(pattern))

    paths = sorted(set(paths))
    frames: list[FrameRef] = []

    for path in paths:
        if not os.path.isfile(path):
            continue

        fallback_time = parse_valid_time_from_wrf_name(path).replace(tzinfo=timezone.utc)

        try:
            with Dataset(path) as nc:
                try:
                    times = wrf.extract_times(
                        nc,
                        timeidx=ALL_TIMES,
                        meta=False,
                    )

                    if not isinstance(times, np.ndarray):
                        times = np.array([times])

                except Exception:
                    times = np.array([fallback_time])

                for time_index, time_value in enumerate(times):
                    time_utc = normalize_datetime(time_value, fallback=fallback_time)

                    frames.append(
                        FrameRef(
                            domain=domain,
                            path=path,
                            time_index=int(time_index),
                            time_utc=time_utc,
                        )
                    )

        except Exception as exc:
            print(f"WARNING: Could not inspect {path}: {exc}", file=sys.stderr)

    frames.sort(key=lambda f: (f.time_utc, f.path, f.time_index))
    return frames


def build_domain_frames(
    requested_domain: str,
    wrf_dir: str | None,
    file_glob: str | None,
    use_parent_fallback: bool = False,
) -> tuple[list[str], dict[str, dict[datetime, FrameRef]]]:
    chain = parent_domain_chain(requested_domain, use_parent_fallback)
    by_domain: dict[str, dict[datetime, FrameRef]] = {}

    for domain in chain:
        domain_glob = file_glob

        if file_glob and requested_domain != domain:
            domain_glob = re.sub(r"wrfout_d\d\d", f"wrfout_{domain}", file_glob)

        frames = discover_frames_for_domain(domain, wrf_dir, domain_glob)
        by_domain[domain] = {frame.time_utc: frame for frame in frames}

        if frames:
            print(f"Found {len(frames)} WRF frame(s) for {domain}.")
            print(f"  First: {frames[0].time_utc.isoformat().replace('+00:00', 'Z')}")
            print(f"  Last:  {frames[-1].time_utc.isoformat().replace('+00:00', 'Z')}")
        else:
            print(f"WARNING: No WRF frames found for {domain}.", file=sys.stderr)

    return chain, by_domain


###############################################################################
# WRF variable readers
###############################################################################
def read_var_time(nc: Dataset, name: str, time_index: int) -> np.ndarray:
    var = nc.variables[name]
    dims = var.dimensions

    # For vortex-following moving nests, XLAT/XLONG can change each Time.
    # Read the requested time slice, never the first frame's saved geometry.
    if "Time" in dims:
        selectors = [slice(None)] * len(dims)
        selectors[dims.index("Time")] = time_index
        return np.asarray(var[tuple(selectors)]).copy()

    return np.asarray(var[...]).copy()


def read_lat_lon(nc: Dataset, time_index: int) -> tuple[np.ndarray, np.ndarray]:
    lat = read_var_time(nc, "XLAT", time_index)
    lon = read_var_time(nc, "XLONG", time_index)

    return np.squeeze(lat), np.squeeze(lon)


def read_slp_hpa(nc: Dataset, time_index: int) -> np.ndarray:
    if "SLP" in nc.variables:
        return np.squeeze(read_var_time(nc, "SLP", time_index)).astype(float)

    slp = wrf.getvar(nc, "slp", timeidx=time_index)
    return np.asarray(to_np(slp), dtype=float)


def read_10m_wind(
    nc: Dataset,
    time_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "U10" not in nc.variables or "V10" not in nc.variables:
        raise RuntimeError("U10 and V10 are required for 10 m wind diagnostics.")

    u10 = np.squeeze(read_var_time(nc, "U10", time_index)).astype(float)
    v10 = np.squeeze(read_var_time(nc, "V10", time_index)).astype(float)
    wspd = np.hypot(u10, v10)

    return u10, v10, wspd


def read_pressure_level_vorticity(
    nc: Dataset,
    time_index: int,
    pressure_hpa: float,
) -> np.ndarray:
    pressure = wrf.getvar(nc, "pressure", timeidx=time_index)
    avo = wrf.getvar(nc, "avo", timeidx=time_index)
    vort = wrf.interplevel(avo, pressure, pressure_hpa)

    return np.asarray(to_np(vort), dtype=float)


def load_fields(frame: FrameRef) -> FieldSet:
    with Dataset(frame.path) as nc:
        lat, lon = read_lat_lon(nc, frame.time_index)
        (
            _,
            _,
            avg_dx_km,
            avg_dy_km,
            _,
            _,
        ) = compute_grid_and_spacing(lat, lon)
        slp_hpa = read_slp_hpa(nc, frame.time_index)
        u10_mps, v10_mps, wspd10_mps = read_10m_wind(nc, frame.time_index)

        # A moving WRF nest retains its grid indices but changes the geographic
        # positions of those indices over time. Check that all fields share the
        # CURRENT frame's geometry before diagnosing the storm location.
        for name, field in (("XLONG", lon), ("SLP", slp_hpa),
                            ("U10", u10_mps), ("V10", v10_mps)):
            if field.shape != lat.shape or lat.ndim != 2:
                raise ValueError(
                    f"{frame.path}, Time={frame.time_index}: "
                    f"XLAT shape {lat.shape} incompatible with {name} shape "
                    f"{field.shape}; cannot geolocate moving-nest centers."
                )

        try:
            vort850 = read_pressure_level_vorticity(nc, frame.time_index, 850.0)
        except Exception as exc:
            print(
                f"WARNING: 850 hPa vorticity unavailable for {frame.path}: {exc}",
                file=sys.stderr,
            )
            vort850 = np.full_like(slp_hpa, np.nan, dtype=float)

        try:
            vort700 = read_pressure_level_vorticity(nc, frame.time_index, 700.0)
        except Exception as exc:
            print(
                f"WARNING: 700 hPa vorticity unavailable for {frame.path}: {exc}",
                file=sys.stderr,
            )
            vort700 = np.full_like(slp_hpa, np.nan, dtype=float)

    return FieldSet(
        domain=frame.domain,
        path=frame.path,
        time_index=frame.time_index,
        time_utc=frame.time_utc,
        lat=np.asarray(lat, dtype=float),
        lon=np.asarray(lon, dtype=float),
        slp_hpa=np.asarray(slp_hpa, dtype=float),
        u10_mps=np.asarray(u10_mps, dtype=float),
        v10_mps=np.asarray(v10_mps, dtype=float),
        wspd10_mps=np.asarray(wspd10_mps, dtype=float),
        vort850=np.asarray(vort850, dtype=float),
        vort700=np.asarray(vort700, dtype=float),
        avg_dx_km=float(avg_dx_km),
        avg_dy_km=float(avg_dy_km),
    )


###############################################################################
# Geometry and tracking helpers
###############################################################################
def haversine_km(lat1, lon1, lat2, lon2):
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))

    return EARTH_RADIUS_KM * c


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)

    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    )

    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def edge_mask(shape: tuple[int, int], buffer_points: int) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)

    if buffer_points <= 0:
        return mask

    n_y, n_x = shape

    if 2 * buffer_points >= min(n_y, n_x):
        return mask

    mask[:buffer_points, :] = False
    mask[-buffer_points:, :] = False
    mask[:, :buffer_points] = False
    mask[:, -buffer_points:] = False

    return mask


def build_search_mask(
    fields: FieldSet,
    guess_lat: float | None,
    guess_lon: float | None,
    radius_km: float | None,
    edge_buffer_grid_points: int,
) -> np.ndarray:
    mask = edge_mask(fields.lat.shape, edge_buffer_grid_points)

    if guess_lat is not None and guess_lon is not None and radius_km is not None:
        dist = haversine_km(guess_lat, guess_lon, fields.lat, fields.lon)
        mask &= dist <= radius_km

    return mask


def smooth_field(field: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(field, dtype=float)

    if sigma <= 0:
        return arr

    valid = np.isfinite(arr)

    if not valid.any():
        return arr

    filled = np.where(valid, arr, np.nanmean(arr[valid]))
    smoothed = gaussian_filter(filled, sigma=float(sigma))
    smoothed[~valid] = np.nan

    return smoothed


def select_minimum(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
    source: str,
) -> PointCandidate:
    valid = mask & np.isfinite(field)

    if not valid.any():
        return PointCandidate(np.nan, np.nan, np.nan, source, valid=False)

    work = np.where(valid, field, np.nan)
    ij = np.unravel_index(np.nanargmin(work), work.shape)

    return PointCandidate(
        float(lat[ij]),
        float(lon[ij]),
        float(field[ij]),
        source,
        valid=True,
    )


def select_cyclonic_vorticity_extreme(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
    source: str,
    hemisphere: str,
) -> PointCandidate:
    valid = mask & np.isfinite(field)

    if not valid.any():
        return PointCandidate(np.nan, np.nan, np.nan, source, valid=False)

    work = np.where(valid, field, np.nan)

    if hemisphere == "south":
        ij = np.unravel_index(np.nanargmin(work), work.shape)
    else:
        ij = np.unravel_index(np.nanargmax(work), work.shape)

    return PointCandidate(
        float(lat[ij]),
        float(lon[ij]),
        float(field[ij]),
        source,
        valid=True,
    )


def infer_hemisphere(
    setting: str,
    guess_lat: float | None,
    pressure_center: PointCandidate,
) -> str:
    if setting in {"north", "south"}:
        return setting

    lat_ref = guess_lat

    if lat_ref is None or not np.isfinite(lat_ref):
        lat_ref = pressure_center.lat

    if np.isfinite(lat_ref) and lat_ref < 0:
        return "south"

    return "north"


def weighted_center(candidates: list[tuple[PointCandidate, float]]) -> PointCandidate:
    total = 0.0
    lat_sum = 0.0
    lon_sum = 0.0

    # Circular average prevents 179E and 179W from erroneously averaging to 0E.
    lon_sin, lon_cos = 0.0, 0.0
    for candidate, weight in candidates:
        if (
            candidate.valid
            and np.isfinite(candidate.lat)
            and np.isfinite(candidate.lon)
            and weight > 0
        ):
            total += weight
            lat_sum += candidate.lat * weight
            lon_sin += weight * math.sin(math.radians(candidate.lon))
            lon_cos += weight * math.cos(math.radians(candidate.lon))

    if total <= 0:
        return PointCandidate(np.nan, np.nan, np.nan, "weighted", valid=False)

    return PointCandidate(
        lat_sum / total,
        math.degrees(math.atan2(lon_sin, lon_cos)),
        np.nan,
        "weighted",
        valid=True,
    )


def calculate_rmw(
    fields: FieldSet,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    edge_buffer_grid_points: int,
) -> tuple[float, float, float, float]:
    mask = build_search_mask(
        fields,
        center_lat,
        center_lon,
        radius_km,
        edge_buffer_grid_points,
    )

    valid = mask & np.isfinite(fields.wspd10_mps)

    if not valid.any():
        return np.nan, np.nan, np.nan, np.nan

    work = np.where(valid, fields.wspd10_mps, np.nan)
    ij = np.unravel_index(np.nanargmax(work), work.shape)

    vmax_mps = float(fields.wspd10_mps[ij])
    vmax_kt = vmax_mps * KNOTS_PER_MPS
    rmw_km = float(
        haversine_km(
            center_lat,
            center_lon,
            float(fields.lat[ij]),
            float(fields.lon[ij]),
        )
    )

    return vmax_mps, vmax_kt, rmw_km, float(fields.slp_hpa[ij])


###############################################################################
# Vortex analysis
###############################################################################
def analyze_fields(
    fields: FieldSet,
    guess_lat: float | None,
    guess_lon: float | None,
    args: argparse.Namespace,
    unrestricted_first_search: bool = False,
) -> dict[str, object] | None:
    radius = None if unrestricted_first_search else float(args.search_radius_km)

    base_mask = build_search_mask(
        fields,
        guess_lat,
        guess_lon,
        radius,
        args.edge_buffer_grid_points,
    )

    if int(np.count_nonzero(base_mask)) < args.min_search_points:
        return None

    slp_smooth = smooth_field(fields.slp_hpa, args.slp_smooth_sigma)
    vort850_smooth = smooth_field(fields.vort850, args.vort_smooth_sigma)
    vort700_smooth = smooth_field(fields.vort700, args.vort_smooth_sigma)
    wind_smooth = smooth_field(fields.wspd10_mps, args.wind_smooth_sigma)

    pressure_center = select_minimum(
        slp_smooth,
        fields.lat,
        fields.lon,
        base_mask,
        "pressure",
    )

    if not pressure_center.valid:
        return None

    hemisphere = infer_hemisphere(args.hemisphere, guess_lat, pressure_center)

    vort850_center = select_cyclonic_vorticity_extreme(
        vort850_smooth,
        fields.lat,
        fields.lon,
        base_mask,
        "vorticity850",
        hemisphere,
    )

    vort700_center = select_cyclonic_vorticity_extreme(
        vort700_smooth,
        fields.lat,
        fields.lon,
        base_mask,
        "vorticity700",
        hemisphere,
    )

    wind_mask = build_search_mask(
        fields,
        pressure_center.lat,
        pressure_center.lon,
        args.wind_center_radius_km,
        args.edge_buffer_grid_points,
    )

    wind_center = select_minimum(
        wind_smooth,
        fields.lat,
        fields.lon,
        wind_mask,
        "wind10min",
    )

    # Calculate the independent center solutions once per WRF frame. A later
    # continuity check can cycle methods without rereading WRF or recomputing
    # the pressure, vorticity, and wind diagnostic fields.
    stormtrack_center, pressure_depth_hpa = pressure_deficit_centroid(
        slp_smooth, fields.lat, fields.lon, pressure_center, args.refine_km,
    )
    weighted = weighted_center([
        (pressure_center, args.weight_pressure),
        (vort850_center, args.weight_vort850),
        (vort700_center, args.weight_vort700),
        (wind_center, args.weight_wind),
    ])
    method_centers = {
        "weighted": weighted,
        "stormtrack": stormtrack_center,
        "pressure": pressure_center,
        "vorticity850": vort850_center,
    }
    preferred = method_centers[args.center_method]
    final_center = preferred if preferred.valid else pressure_center
    selected_depth = pressure_depth_hpa if final_center is stormtrack_center else np.nan

    jump_from_previous_km = np.nan

    if guess_lat is not None and guess_lon is not None:
        jump_from_previous_km = float(
            haversine_km(
                guess_lat,
                guess_lon,
                final_center.lat,
                final_center.lon,
            )
        )

    vmax_mps, vmax_kt, rmw_km, slp_at_vmax_hpa = calculate_rmw(
        fields,
        final_center.lat,
        final_center.lon,
        args.rmw_search_radius_km,
        args.edge_buffer_grid_points,
    )

    return {
        "time_utc": fields.time_utc.isoformat().replace("+00:00", "Z"),
        "domain_used": fields.domain,
        "source_file": fields.path,
        "time_index": fields.time_index,
        # Auditable time-dependent WRF nest geometry. Both corner and grid
        # midpoint are read from XLAT/XLONG at THIS forecast time.
        "grid_corner_00_lat": float(fields.lat[0, 0]),
        "grid_corner_00_lon": float(fields.lon[0, 0]),
        "grid_midpoint_lat": float(fields.lat[fields.lat.shape[0] // 2, fields.lat.shape[1] // 2]),
        "grid_midpoint_lon": float(fields.lon[fields.lon.shape[0] // 2, fields.lon.shape[1] // 2]),
        "avg_dx_km": fields.avg_dx_km,
        "avg_dy_km": fields.avg_dy_km,
        "center_method": args.center_method,
        "final_center_lat": final_center.lat,
        "final_center_lon": final_center.lon,
        "surface_pressure_center_lat": pressure_center.lat,
        "surface_pressure_center_lon": pressure_center.lon,
        "min_slp_hpa": float(pressure_center.value),
        "pressure_depth_hpa": selected_depth,
        "track_flag": ("weak" if np.isfinite(selected_depth) and selected_depth < args.min_depth
                       else "ok"),
        "vorticity850_center_lat": vort850_center.lat,
        "vorticity850_center_lon": vort850_center.lon,
        "vorticity850_value": vort850_center.value,
        "vorticity700_center_lat": vort700_center.lat,
        "vorticity700_center_lon": vort700_center.lon,
        "vorticity700_value": vort700_center.value,
        "wind10_center_lat": wind_center.lat,
        "wind10_center_lon": wind_center.lon,
        "wind10_center_value_mps": wind_center.value,
        "vmax10_mps": vmax_mps,
        "vmax10_kt": vmax_kt,
        "rmw_km": rmw_km,
        "slp_at_vmax_hpa": slp_at_vmax_hpa,
        "pressure_to_final_km": float(
            haversine_km(
                final_center.lat,
                final_center.lon,
                pressure_center.lat,
                pressure_center.lon,
            )
        ),
        "vort850_to_final_km": float(
            haversine_km(
                final_center.lat,
                final_center.lon,
                vort850_center.lat,
                vort850_center.lon,
            )
        )
        if vort850_center.valid
        else np.nan,
        "vort700_to_final_km": float(
            haversine_km(
                final_center.lat,
                final_center.lon,
                vort700_center.lat,
                vort700_center.lon,
            )
        )
        if vort700_center.valid
        else np.nan,
        "wind_to_final_km": float(
            haversine_km(
                final_center.lat,
                final_center.lon,
                wind_center.lat,
                wind_center.lon,
            )
        )
        if wind_center.valid
        else np.nan,
        "hemisphere_used": hemisphere,
        "search_radius_km": radius if radius is not None else np.nan,
        "jump_from_previous_km": jump_from_previous_km,
        "max_jump_exceeded": False,
        "used_as_fallback_after_jump_limit": False,
        "_method_centers": method_centers,
        "_stormtrack_depth_hpa": pressure_depth_hpa,
    }




def pressure_deficit_centroid(
    slp: np.ndarray, lat: np.ndarray, lon: np.ndarray,
    pressure_center: PointCandidate, refine_km: float,
) -> tuple[PointCandidate, float]:
    """Sub-grid centre and depth from stormtrack's pressure-deficit method.

    Uses a 90th-percentile environment in a distance-limited disc and
    dateline-safe local coordinates. Returns (centre, pressure depth in hPa).
    """
    if not pressure_center.valid:
        return pressure_center, float("nan")
    r = st_haversine_km(pressure_center.lat, pressure_center.lon, lat, lon)
    disc = np.isfinite(slp) & (r <= refine_km)
    if not np.any(disc):
        return pressure_center, float("nan")
    environment = float(np.nanpercentile(slp[disc], 90))
    deficit = np.where(disc, np.clip(environment - slp, 0, None), 0.0)
    weight = float(np.nansum(deficit))
    if weight > 0:
        x, y = st_to_plane(lat, lon, pressure_center.lat, pressure_center.lon)
        lat1, lon1 = st_from_plane(
            float(np.nansum(deficit * x) / weight),
            float(np.nansum(deficit * y) / weight),
            pressure_center.lat, pressure_center.lon,
        )
    else:
        lat1, lon1 = pressure_center.lat, pressure_center.lon
    return PointCandidate(lat1, lon1, pressure_center.value, "stormtrack"), environment - float(pressure_center.value)


def add_fitted_motion(df: pd.DataFrame, window_h: float) -> pd.DataFrame:
    """Add stormtrack local least-squares 2D motion without removing WRF columns."""
    if df.empty:
        return df
    quality = df.get('center_solution_accepted', pd.Series(True, index=df.index))
    quality = quality.fillna(False).astype(bool)
    st = pd.DataFrame({
        'time': pd.to_datetime(df['time_utc'], utc=True).dt.tz_localize(None),
        'lat': df['final_center_lat'], 'lon': df['final_center_lon'],
        # Suppress unreliable fixes from the motion-vector fit, but not CSV.
        'flag': np.where(quality, 'ok', 'lost'),
    })
    st = st_motion(st, window_h=window_h)
    for old, new in (("speed_ms", "fitted_speed_ms"), ("speed_kt", "fitted_speed_kt"),
                     ("heading_deg", "fitted_heading_deg"), ("u_ms", "fitted_u_ms"),
                     ("v_ms", "fitted_v_ms")):
        df[new] = st[old].to_numpy()
    return df

def candidate_jump(candidate: dict[str, object]) -> float:
    value = candidate.get("jump_from_previous_km", np.nan)

    try:
        return float(value)
    except Exception:
        return np.nan


def predict_next_center(
    rows: list[dict[str, object]], time_utc: datetime,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Predict the cyclone location using only previously credible fixes.

    Extrapolation uses an unwrapped local tangent plane, with an upper bound on
    inferred velocity. An unresolved fix is retained in the output, but is NOT
    allowed to redirect subsequent tracking. Missing/implausible motion falls
    back to persistence, explicitly labeled as such.
    """
    current = pd.Timestamp(time_utc)
    good = [r for r in rows if r.get('center_solution_accepted', True)]
    history = good if good else rows
    if not history:
        return dict(last_lat=args.init_lat, last_lon=args.init_lon,
                    predicted_lat=args.init_lat, predicted_lon=args.init_lon,
                    elapsed_hours=np.nan, prediction_reliable=False,
                    prediction_basis='first_guess' if args.init_lat is not None else 'unseeded')
    last = history[-1]
    lat, lon = float(last['final_center_lat']), float(last['final_center_lon'])
    elapsed_h = (current - pd.Timestamp(last['time_utc'])).total_seconds() / 3600.0
    result = dict(last_lat=lat, last_lon=lon, predicted_lat=lat, predicted_lon=lon,
                  elapsed_hours=elapsed_h, prediction_reliable=False,
                  prediction_basis='last_credible_center')
    if (getattr(args, 'no_motion_prediction', False) or len(good) < 2
            or not (0 < elapsed_h <= args.max_prediction_hours)):
        return result
    prior = good[-2]
    prev_h = (pd.Timestamp(last['time_utc']) - pd.Timestamp(prior['time_utc'])).total_seconds()/3600.0
    if not (0 < prev_h <= args.max_prediction_hours):
        return result
    px, py = st_to_plane(lat, lon,
                         float(prior['final_center_lat']), float(prior['final_center_lon']))
    past_speed_kt = float(np.hypot(px, py)) / prev_h / 1.852
    if not np.isfinite(past_speed_kt) or past_speed_kt > args.max_prediction_speed_kt:
        # Do not extrapolate a highly suspicious previous displacement.
        result['prediction_basis'] = 'past_motion_implausible'
        return result
    x0, y0 = st_to_plane(lat, lon, float(prior['final_center_lat']),
                         float(prior['final_center_lon']))
    # Take east/north displacements into a plane anchored at the last fix.
    result['predicted_lat'], result['predicted_lon'] = st_from_plane(
        float(x0 * elapsed_h / prev_h), float(y0 * elapsed_h / prev_h), lat, lon,
    )
    result['prediction_reliable'] = True
    result['prediction_basis'] = 'constant_velocity_two_credible_centers'
    return result


def choose_continuous_center(
    analysis: dict[str, object],
    fields: FieldSet,
    last_lat: float | None,
    last_lon: float | None,
    requested_method: str,
    jump_limit_km: float,
    previous_fix_exists: bool,
    args: argparse.Namespace,
    *,
    predicted_lat: float | None = None,
    predicted_lon: float | None = None,
    elapsed_hours: float | None = None,
    prediction_reliable: bool = False,
    prediction_basis: str = 'last_center',
) -> dict[str, object]:
    """Motion- and structure-aware selection across independent center methods.

    30 km is a REVIEW trigger, NOT a maximum permissible storm movement. If
    the preferred center exceeds the trigger, compare other centers against a
    predicted location, pressure/vorticity agreement, and an interval-scaled
    forward-speed screen. Keep legitimate fast motion when it agrees with the
    predicted trajectory; do not reward false stationary centers.

    A 'resolved' result is only a heuristic model-vortex association. No method
    proves the physical identity of the center without external verification.
    """
    options: dict[str, PointCandidate] = analysis['_method_centers']
    order = list(dict.fromkeys((requested_method, 'stormtrack', 'pressure',
                                'vorticity850', 'weighted')))
    valid = {name: options[name] for name in order if name in options and options[name].valid
             and np.isfinite([options[name].lat, options[name].lon]).all()}
    if not valid:
        raise ValueError('No valid center from any method')
    previous = (previous_fix_exists and last_lat is not None and last_lon is not None
                and np.isfinite([last_lat, last_lon]).all())
    use_prediction = (previous and predicted_lat is not None and predicted_lon is not None
                      and np.isfinite([predicted_lat, predicted_lon]).all())
    if not use_prediction:
        predicted_lat, predicted_lon = last_lat, last_lon
    dt = float(elapsed_hours) if elapsed_hours is not None else np.nan
    time_valid = np.isfinite(dt) and dt > 0
    # A time-aware speed screen is independent of the 30 km review trigger.
    max_travel = (args.max_motion_speed_kt * 1.852 * dt + args.motion_position_buffer_km
                  if time_valid else np.inf)
    prediction_tolerance = (args.prediction_tolerance_km +
                            max(dt - 1., 0.) * args.prediction_tolerance_growth_kmh
                            if time_valid else args.prediction_tolerance_km)
    pressure = options.get('pressure')
    pvalid = pressure is not None and pressure.valid and np.isfinite([pressure.lat, pressure.lon]).all()
    v850 = options.get('vorticity850')
    v700_lat = float(analysis.get('vorticity700_center_lat', np.nan))
    v700_lon = float(analysis.get('vorticity700_center_lon', np.nan))
    v850_valid = v850 is not None and v850.valid and np.isfinite([v850.lat, v850.lon]).all()
    v700_valid = np.isfinite([v700_lat, v700_lon]).all()
    vorticity_pressure_separation = []
    if pvalid:
        if v850_valid:
            vorticity_pressure_separation.append(float(haversine_km(
                pressure.lat, pressure.lon, v850.lat, v850.lon)))
        if v700_valid:
            vorticity_pressure_separation.append(float(haversine_km(
                pressure.lat, pressure.lon, v700_lat, v700_lon)))
    vortex_agrees = (not vorticity_pressure_separation or
                     min(vorticity_pressure_separation) <= args.vorticity_agreement_km)
    # If no available vorticity, document lack of independent confirmation.
    independent_vorticity_available = bool(vorticity_pressure_separation)
    diagnostics = {}
    for name, center in valid.items():
        step = float(haversine_km(last_lat, last_lon, center.lat, center.lon)) if previous else np.nan
        dev = (float(haversine_km(predicted_lat, predicted_lon, center.lat, center.lon))
               if use_prediction else np.nan)
        pdist = (float(haversine_km(pressure.lat, pressure.lon, center.lat, center.lon))
                 if pvalid else np.nan)
        # Low-level vorticity should support the same pressure circulation.
        # Avoid rewarding extreme vorticity centered far outside the low.
        compatible_vort = []
        if v850_valid and (not pvalid or float(haversine_km(pressure.lat, pressure.lon,
                                                           v850.lat, v850.lon)) <= args.vorticity_agreement_km):
            compatible_vort.append(float(haversine_km(center.lat, center.lon, v850.lat, v850.lon)))
        if v700_valid and (not pvalid or float(haversine_km(pressure.lat, pressure.lon,
                                                           v700_lat, v700_lon)) <= args.vorticity_agreement_km):
            compatible_vort.append(float(haversine_km(center.lat, center.lon, v700_lat, v700_lon)))
        near_vort = min(compatible_vort) if compatible_vort else np.nan
        # Penalties apply to spatial disagreement. A pressure+centroid duo is
        # not counted as independent vorticity confirmation.
        score = (dev if previous and np.isfinite(dev) else 0.0)
        if np.isfinite(pdist):
            score += 0.45 * max(0., pdist - 20.)
        if np.isfinite(near_vort):
            score += 0.10 * max(0., near_vort - 35.)
        if name == requested_method:
            score -= 5.0  # small stability preference; no absolute priority
        structure_ok = (not np.isfinite(pdist) or pdist <= args.center_coherence_km)
        motion_ok = not previous or not time_valid or step <= max_travel
        prediction_ok = (not prediction_reliable or not previous or
                         dev <= prediction_tolerance)
        diagnostics[name] = dict(step=step, deviation=dev, pressure_gap=pdist,
                                 vorticity_gap=near_vort, score=score,
                                 plausible=structure_ok and motion_ok and prediction_ok,
                                 structure_ok=structure_ok, motion_ok=motion_ok,
                                 prediction_ok=prediction_ok)
    preferred_jump = diagnostics.get(requested_method, {}).get('step', np.nan)
    preferred_dev = diagnostics.get(requested_method, {}).get('deviation', np.nan)
    # Preserve the requested method when the original displacement test passes
    # AND it is physically coherent. A strong prediction mismatch can trigger
    # reconsideration even for a deceptively stationary apparent center.
    fallback_trigger = bool(previous and jump_limit_km > 0 and
                            (not np.isfinite(preferred_jump) or preferred_jump > jump_limit_km))
    if previous and prediction_reliable and np.isfinite(preferred_dev) and preferred_dev > prediction_tolerance:
        fallback_trigger = True
    # A center that barely moves can still be the wrong vortex. If a credible
    # motion prediction strongly favors a coherent alternate method, review the
    # preferred center even though its raw displacement is under 30 km.
    if prediction_reliable and requested_method in diagnostics:
        preferred_score = diagnostics[requested_method]['score']
        better_alternates = [d['score'] for name, d in diagnostics.items()
                             if name != requested_method and d['plausible']]
        if better_alternates and min(better_alternates) + args.prediction_switch_margin_km < preferred_score:
            fallback_trigger = True
    if requested_method in diagnostics and not diagnostics[requested_method]['structure_ok']:
        fallback_trigger = True
    if not diagnostics.get(requested_method, {}).get('motion_ok', False):
        fallback_trigger = True
    plausible = [(d['score'], order.index(name), name) for name,d in diagnostics.items()
                 if d['plausible']]
    if not previous:
        # Initial fix is unvalidated if it was an unrestricted domain search.
        selected = requested_method if requested_method in valid else next(iter(valid))
        status = 'initial_unverified'
    elif not fallback_trigger and requested_method in diagnostics and diagnostics[requested_method]['plausible']:
        selected = requested_method
        status = 'preferred_plausible'
    elif plausible:
        selected = min(plausible)[2]
        status = 'motion_structure_fallback' if selected != requested_method else 'preferred_motion_verified'
    else:
        selected = min((d['score'] + (150. if not d['structure_ok'] else 0.) +
                        (150. if not d['motion_ok'] else 0.), order.index(name), name)
                       for name,d in diagnostics.items())[2]
        status = 'unresolved_no_plausible_center'
    chosen = diagnostics[selected]
    center = valid[selected]
    record = analysis.copy()
    record.pop('_method_centers', None)
    depth = float(record.pop('_stormtrack_depth_hpa', np.nan))
    new_lat, new_lon = float(center.lat), float(center.lon)
    insufficient_structure = bool(independent_vorticity_available and not vortex_agrees)
    shallow = bool(np.isfinite(depth) and depth < args.min_depth)
    unresolved = status == 'unresolved_no_plausible_center'
    # Insufficient independent structure lowers confidence but does not force a
    # false association with a parent domain.
    quality = ('low' if unresolved or shallow or insufficient_structure else
               ('high' if prediction_reliable and independent_vorticity_available
                else 'medium'))
    record.update({
        'center_method': selected,
        'center_method_requested': requested_method,
        'center_method_fallback_used': selected != requested_method,
        'center_method_selection_status': status,
        'center_method_jump_limit_km': jump_limit_km,
        'preferred_method_jump_km': preferred_jump,
        'method_center_distances_km': '; '.join(
            f'{name}={diagnostics[name]["step"]:.2f}' if previous and name in diagnostics
            else f'{name}=unavailable' if name not in diagnostics else f'{name}=initial'
            for name in order),
        'method_prediction_errors_km': '; '.join(
            f'{name}={diagnostics[name]["deviation"]:.2f}' if previous and name in diagnostics
            else f'{name}=unavailable' if name not in diagnostics else f'{name}=initial'
            for name in order),
        'selection_score_km': chosen['score'],
        'method_selection_scores': '; '.join(
            f'{name}={diagnostics[name]["score"]:.2f}' if name in diagnostics else f'{name}=unavailable'
            for name in order),
        'final_center_lat': new_lat, 'final_center_lon': new_lon,
        'jump_from_previous_km': chosen['step'],
        'predicted_center_lat': predicted_lat if use_prediction else np.nan,
        'predicted_center_lon': predicted_lon if use_prediction else np.nan,
        'prediction_error_km': chosen['deviation'],
        'prediction_reliable': bool(prediction_reliable),
        'prediction_basis': prediction_basis,
        'elapsed_hours_since_credible_center': dt,
        'max_plausible_travel_km': max_travel if np.isfinite(max_travel) else np.nan,
        'pressure_center_agreement_km': chosen['pressure_gap'],
        'vorticity_center_agreement_km': chosen['vorticity_gap'],
        'vorticity_pressure_min_separation_km': (min(vorticity_pressure_separation)
                                                  if vorticity_pressure_separation else np.nan),
        'vorticity_consistent_with_pressure': bool(vortex_agrees),
        'center_confidence': quality,
        'center_solution_accepted': not unresolved and not shallow and not insufficient_structure,
        'max_jump_exceeded': unresolved,
        'used_as_fallback_after_jump_limit': False,
        'motion_review_triggered': fallback_trigger,
        'center_unresolved': unresolved,
    })
    record['pressure_depth_hpa'] = depth
    record['track_flag'] = ('unresolved' if unresolved else
                            'weak' if shallow else
                            'structure_disagreement' if insufficient_structure else 'ok')
    # Wind radius and intensity diagnostics MUST match the selected center.
    if (not np.isclose(new_lat, analysis['final_center_lat']) or
            not np.isclose(new_lon, analysis['final_center_lon'])):
        vmax_ms, vmax_kt, rmw_km, slp_at_vmax = calculate_rmw(
            fields, new_lat, new_lon, args.rmw_search_radius_km,
            args.edge_buffer_grid_points,
        )
        record.update(vmax10_mps=vmax_ms, vmax10_kt=vmax_kt,
                      rmw_km=rmw_km, slp_at_vmax_hpa=slp_at_vmax)
    for _, lat_name, lon_name, column in (
        ('pressure', 'surface_pressure_center_lat', 'surface_pressure_center_lon', 'pressure_to_final_km'),
        ('vorticity850', 'vorticity850_center_lat', 'vorticity850_center_lon', 'vort850_to_final_km'),
        ('vorticity700', 'vorticity700_center_lat', 'vorticity700_center_lon', 'vort700_to_final_km'),
        ('wind10min', 'wind10_center_lat', 'wind10_center_lon', 'wind_to_final_km'),
    ):
        other_lat, other_lon = record[lat_name], record[lon_name]
        record[column] = float(haversine_km(new_lat, new_lon, other_lat, other_lon)) if np.isfinite([other_lat, other_lon]).all() else np.nan
    return record


def track_vortex(
    chain: list[str],
    by_domain: dict[str, dict[datetime, FrameRef]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    requested_frames = sorted(by_domain.get(args.domain, {}).values(), key=lambda f: f.time_utc)
    if not requested_frames:
        raise RuntimeError(f'No frames found for {args.domain}.')
    if args.max_times and args.max_times > 0:
        requested_frames = requested_frames[:args.max_times]
    print(f'Tracking {len(requested_frames)} frames: {requested_frames[0].time_utc} to '
          f'{requested_frames[-1].time_utc}')
    print('Domain search order:', ' -> '.join(chain))
    if args.init_lat is None:
        warnings.warn('No initial vortex location provided. An unrestricted first scan can '
                      'lock onto a different low in d01 and d02. Supply the same '
                      '--init-lat/--init-lon to both runs for fair comparison.')
    rows: list[dict[str, object]] = []
    for frame_number, primary_frame in enumerate(requested_frames):
        time_utc = primary_frame.time_utc
        prediction = predict_next_center(rows, time_utc, args)
        prior = rows[-1] if rows else None
        # Search the moving storm around the predicted position instead of
        # around the last center. The search width remains a user option.
        search_lat, search_lon = prediction['predicted_lat'], prediction['predicted_lon']
        unrestricted = frame_number == 0 and args.init_lat is None
        resolved, unresolved = [], []
        for domain in chain:
            frame = by_domain.get(domain, {}).get(time_utc)
            if frame is None:
                continue
            try:
                fields = load_fields(frame)
                analysis = analyze_fields(fields, search_lat, search_lon, args,
                                          unrestricted_first_search=unrestricted)
                if analysis is None:
                    continue
                candidate = choose_continuous_center(
                    analysis, fields, prediction['last_lat'], prediction['last_lon'],
                    args.center_method, args.max_jump_km,
                    previous_fix_exists=bool(rows), args=args,
                    predicted_lat=search_lat, predicted_lon=search_lon,
                    elapsed_hours=prediction['elapsed_hours'],
                    prediction_reliable=prediction['prediction_reliable'],
                    prediction_basis=prediction['prediction_basis'],
                )
            except Exception as exc:
                print(f'WARNING: {domain} failed at {time_utc}: {exc}', file=sys.stderr)
                continue
            if not candidate['center_solution_accepted']:
                # A low-confidence or shallow center may be improved by the
                # parent at the same model time. Preserve it as fallback.
                unresolved.append(candidate)
            else:
                resolved.append(candidate)
                # Prefer requested domain whenever its structure is credible.
                break
        if resolved:
            accepted = resolved[0]
        elif unresolved:
            # Retain an actual diagnosis and its low confidence label, never
            # fabricate a stationary replacement or a clipped 30 km step.
            accepted = min(unresolved, key=lambda r: float(r.get('selection_score_km', np.inf)))
            accepted['used_as_fallback_after_jump_limit'] = True
            warnings.warn(f'No high-confidence vortex center at {time_utc}; '
                          f'keeping a low-confidence calculated fix for review.')
        else:
            warnings.warn(f'No valid storm center at {time_utc}; leaving a time gap.')
            continue
        accepted['domain_requested'] = args.domain
        accepted['parent_domain_fallback_used'] = accepted['domain_used'] != args.domain
        if prior is not None:
            accepted['distance_from_immediately_prior_fix_km'] = float(haversine_km(
                prior['final_center_lat'], prior['final_center_lon'],
                accepted['final_center_lat'], accepted['final_center_lon']))
        else:
            accepted['distance_from_immediately_prior_fix_km'] = np.nan
        rows.append(accepted)
        print(f'{time_utc.isoformat().replace("+00:00","Z")} '
              f'{accepted["domain_used"]} '
              f'lat={accepted["final_center_lat"]:.3f} lon={accepted["final_center_lon"]:.3f} '
              f'method={accepted["center_method"]} '
              f'confidence={accepted["center_confidence"]} '
              f'pred_error={accepted["prediction_error_km"]:.1f}km '
              f'status={accepted["center_method_selection_status"]}')
    if not rows:
        raise RuntimeError('No vortex centers found.')
    df = pd.DataFrame(rows)
    df = add_motion_columns(df, lat_col='final_center_lat', lon_col='final_center_lon')
    df = add_smoothed_track_columns(df, window=args.track_smooth_window)
    df = add_smoothed_intensity_columns(df, window=args.intensity_smooth_window)
    df = add_fitted_motion(df, window_h=args.motion_window_h)
    return df


###############################################################################
# Post-processing
###############################################################################
def add_motion_columns(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    prefix: str = "",
) -> pd.DataFrame:
    out = df.copy()
    times = pd.to_datetime(out["time_utc"], utc=True)

    distance_col = f"{prefix}motion_distance_km"
    speed_col = f"{prefix}motion_speed_kt"
    direction_col = f"{prefix}motion_direction_deg"

    speeds = [np.nan]
    bearings = [np.nan]
    distances = [np.nan]

    for i in range(1, len(out)):
        dt_hours = (times.iloc[i] - times.iloc[i - 1]).total_seconds() / 3600.0

        lat1 = float(out.loc[out.index[i - 1], lat_col])
        lon1 = float(out.loc[out.index[i - 1], lon_col])
        lat2 = float(out.loc[out.index[i], lat_col])
        lon2 = float(out.loc[out.index[i], lon_col])

        dist_km = float(haversine_km(lat1, lon1, lat2, lon2))
        speed_kt = dist_km / dt_hours / 1.852 if dt_hours > 0 else np.nan
        bearing = bearing_degrees(lat1, lon1, lat2, lon2)

        distances.append(dist_km)
        speeds.append(speed_kt)
        bearings.append(bearing)

    out[distance_col] = distances
    out[speed_col] = speeds
    out[direction_col] = bearings

    return out


def centered_rolling_median(series: pd.Series, window: int) -> pd.Series:
    if window <= 1 or len(series) < 3:
        return series.copy()

    if window % 2 == 0:
        window += 1

    return series.rolling(window=window, center=True, min_periods=1).median()


def add_smoothed_track_columns(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    out = df.copy()

    out["raw_center_lat"] = out["final_center_lat"]
    out["raw_center_lon"] = out["final_center_lon"]

    # Never smooth through unresolved tracking gaps, because doing so can
    # fabricate a path between different model vortices. Isolate each suspect
    # center before and after the discontinuity.
    breaks = out.get('center_unresolved', pd.Series(False, index=out.index))
    breaks = breaks.fillna(False).astype(bool).to_numpy()
    block = np.cumsum(breaks.astype(int) + np.r_[False, breaks[:-1]].astype(int))
    lat = pd.to_numeric(out['final_center_lat'], errors='coerce')
    lon = pd.to_numeric(out['final_center_lon'], errors='coerce')
    out['smoothed_center_lat'] = lat.groupby(block).transform(
        lambda x: centered_rolling_median(x, window))
    def smooth_lon(chunk):
        ref = float(chunk.iloc[0])
        continuous = pd.Series(ref + st_wrap180(chunk.to_numpy(float) - ref), index=chunk.index)
        return pd.Series(st_wrap180(centered_rolling_median(continuous, window)), index=chunk.index)
    out['smoothed_center_lon'] = lon.groupby(block).transform(smooth_lon)

    out = add_motion_columns(
        out,
        lat_col="smoothed_center_lat",
        lon_col="smoothed_center_lon",
        prefix="smoothed_",
    )

    return out


def add_smoothed_intensity_columns(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    out = df.copy()

    out["min_slp_hpa_raw"] = out["min_slp_hpa"]
    out["vmax10_kt_raw"] = out["vmax10_kt"]

    out["min_slp_hpa_smoothed"] = centered_rolling_median(
        pd.to_numeric(out["min_slp_hpa"], errors="coerce"),
        window,
    )

    out["vmax10_kt_smoothed"] = centered_rolling_median(
        pd.to_numeric(out["vmax10_kt"], errors="coerce"),
        window,
    )

    # Preserve both raw and smoothed wind bins in the CSV. Both visualizations
    # use whichever intensity series is selected by --no-smoothed-output.
    for suffix, column in (("raw", "vmax10_kt"),
                           ("smoothed", "vmax10_kt_smoothed")):
        mph = pd.to_numeric(out[column], errors="coerce") * MPH_PER_KNOT
        out[f"vmax10_mph_{suffix}"] = mph
        out[f"vmax10_mps_{suffix}"] = pd.to_numeric(out[column], errors="coerce") / KNOTS_PER_MPS
        out[f"wind_strength_{suffix}"] = [wind_strength(float(x))[0] for x in mph]
        out[f"wind_color_{suffix}"] = [wind_strength(float(x))[1] for x in mph]

    # Explicit raw mph alias alongside the existing vmax10_mps and vmax10_kt.
    out["vmax10_mph"] = out["vmax10_mph_raw"]
    return out


def output_track_columns(df: pd.DataFrame, use_smoothed: bool) -> tuple[str, str]:
    if (
        use_smoothed
        and "smoothed_center_lat" in df.columns
        and "smoothed_center_lon" in df.columns
    ):
        return "smoothed_center_lat", "smoothed_center_lon"

    return "final_center_lat", "final_center_lon"


def output_intensity_columns(df: pd.DataFrame, use_smoothed: bool) -> tuple[str, str]:
    if (
        use_smoothed
        and "min_slp_hpa_smoothed" in df.columns
        and "vmax10_kt_smoothed" in df.columns
    ):
        return "min_slp_hpa_smoothed", "vmax10_kt_smoothed"

    return "min_slp_hpa", "vmax10_kt"


###############################################################################
# ATCF-style output and best-track comparison
###############################################################################
def format_atcf_lat(lat: float) -> str:
    hemi = "N" if lat >= 0 else "S"
    return f"{int(round(abs(lat) * 10)):03d}{hemi}"


def format_atcf_lon(lon: float) -> str:
    hemi = "E" if lon >= 0 else "W"
    return f"{int(round(abs(lon) * 10)):04d}{hemi}"


def write_generic_atcf(
    df: pd.DataFrame,
    path: Path,
    use_smoothed_output: bool,
) -> None:
    if df.empty:
        return

    lat_col, lon_col = output_track_columns(df, use_smoothed_output)
    slp_col, vmax_col = output_intensity_columns(df, use_smoothed_output)

    start_time = pd.to_datetime(df["time_utc"].iloc[0], utc=True)
    lines: list[str] = []

    for _, row in df.iterrows():
        # ATCF-style coordinates should not silently advertise an unresolved
        # or weak association as a verified storm fix. Keep these in the CSV.
        if 'center_solution_accepted' in df.columns and not bool(row['center_solution_accepted']):
            continue
        time_utc = pd.to_datetime(row["time_utc"], utc=True)
        tau = int(round((time_utc - start_time).total_seconds() / 3600.0))
        ymdh = time_utc.strftime("%Y%m%d%H")

        lat_text = format_atcf_lat(float(row[lat_col]))
        lon_text = format_atcf_lon(float(row[lon_col]))

        vmax = int(round(float(row[vmax_col]))) if np.isfinite(row[vmax_col]) else 0
        mslp = int(round(float(row[slp_col]))) if np.isfinite(row[slp_col]) else 0

        line = (
            f"{GENERIC_ATCF_BASIN}, {GENERIC_ATCF_NUMBER}, {ymdh}, 03, "
            f"{'GEN' if str(row['domain_used']) == 'generic' else GENERIC_ATCF_MODEL}, {tau:03d}, {lat_text}, {lon_text}, "
            f"{vmax:3d}, {mslp:4d}, XX,  34, NEQ,    0,    0,    0,    0"
        )

        lines.append(line)

    path.write_text("\n".join(lines) + "\n")


def parse_atcf_lat(value: str) -> float:
    value = value.strip().upper()
    hemi = value[-1]
    mag = float(value[:-1]) / 10.0

    return -mag if hemi == "S" else mag


def parse_atcf_lon(value: str) -> float:
    value = value.strip().upper()
    hemi = value[-1]
    mag = float(value[:-1]) / 10.0

    return -mag if hemi == "W" else mag


def read_best_track(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(path)

    if file_path.suffix.lower() in {".csv", ".txt"}:
        try:
            df = pd.read_csv(file_path)
            cols = {c.lower(): c for c in df.columns}

            if {"time_utc", "lat", "lon"}.issubset(cols):
                out = pd.DataFrame()
                out["time_utc"] = pd.to_datetime(df[cols["time_utc"]], utc=True)
                out["best_lat"] = pd.to_numeric(df[cols["lat"]], errors="coerce")
                out["best_lon"] = pd.to_numeric(df[cols["lon"]], errors="coerce")

                if "mslp_hpa" in cols:
                    out["best_mslp_hpa"] = pd.to_numeric(
                        df[cols["mslp_hpa"]],
                        errors="coerce",
                    )

                if "vmax_kt" in cols:
                    out["best_vmax_kt"] = pd.to_numeric(
                        df[cols["vmax_kt"]],
                        errors="coerce",
                    )

                return out.dropna(subset=["time_utc", "best_lat", "best_lon"])

        except Exception:
            pass

    rows = []

    for line in file_path.read_text(errors="ignore").splitlines():
        parts = [p.strip() for p in line.split(",")]

        if len(parts) < 10:
            continue

        try:
            time_utc = datetime.strptime(parts[2], "%Y%m%d%H").replace(
                tzinfo=timezone.utc
            )

            rows.append(
                {
                    "time_utc": time_utc,
                    "best_lat": parse_atcf_lat(parts[6]),
                    "best_lon": parse_atcf_lon(parts[7]),
                    "best_vmax_kt": float(parts[8]),
                    "best_mslp_hpa": float(parts[9]),
                }
            )

        except Exception:
            continue

    if not rows:
        raise ValueError(
            "Could not parse best-track file. Use CSV columns time_utc, lat, lon."
        )

    return pd.DataFrame(rows)


def compare_to_best_track(
    model_df: pd.DataFrame,
    best_track_path: str,
    use_smoothed_output: bool,
) -> pd.DataFrame:
    best = read_best_track(best_track_path)
    model = model_df.copy()

    lat_col, lon_col = output_track_columns(model, use_smoothed_output)
    slp_col, vmax_col = output_intensity_columns(model, use_smoothed_output)

    model["time_dt"] = pd.to_datetime(model["time_utc"], utc=True)
    best["time_dt"] = pd.to_datetime(best["time_utc"], utc=True)

    merged = pd.merge(
        model,
        best,
        on="time_dt",
        how="inner",
        suffixes=("", "_best"),
    )

    if merged.empty:
        return merged

    merged["track_error_km"] = [
        float(
            haversine_km(
                float(row[lat_col]),
                float(row[lon_col]),
                float(row["best_lat"]),
                float(row["best_lon"]),
            )
        )
        for _, row in merged.iterrows()
    ]

    merged["track_error_nm"] = merged["track_error_km"] / 1.852

    if "best_mslp_hpa" in merged.columns:
        merged["mslp_error_hpa"] = merged[slp_col] - merged["best_mslp_hpa"]

    if "best_vmax_kt" in merged.columns:
        merged["vmax_error_kt"] = merged[vmax_col] - merged["best_vmax_kt"]

    return merged


###############################################################################
# Plot title helpers
###############################################################################
def grid_spacing_from_track(df: pd.DataFrame) -> tuple[float, float]:
    if "avg_dx_km" in df.columns and "avg_dy_km" in df.columns:
        dx = pd.to_numeric(df["avg_dx_km"], errors="coerce").median()
        dy = pd.to_numeric(df["avg_dy_km"], errors="coerce").median()

        if np.isfinite(dx) and np.isfinite(dy):
            return round(float(dx), 2), round(float(dy), 2)

    return np.nan, np.nan


def valid_period_from_track(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    times = pd.to_datetime(df["time_utc"], utc=True)
    return times.iloc[0], times.iloc[-1]


def valid_title_text(df: pd.DataFrame) -> str:
    start_time, end_time = valid_period_from_track(df)

    if start_time == end_time:
        return f"Valid: {start_time:%H:%M:%SZ %Y-%m-%d}"

    return f"Valid: {start_time:%H:%M:%SZ %Y-%m-%d} to {end_time:%H:%M:%SZ %Y-%m-%d}"


def grid_spacing_title_text(df: pd.DataFrame) -> str:
    avg_dx_km, avg_dy_km = grid_spacing_from_track(df)

    if np.isfinite(avg_dx_km) and np.isfinite(avg_dy_km):
        return f"Average Grid Spacing: {avg_dx_km} x {avg_dy_km} km"

    return "Average Grid Spacing: unavailable"


###############################################################################
# Plotting
###############################################################################
def map_extent_arrays(
    df: pd.DataFrame, lat_col: str, lon_col: str, padding_degrees: float = 2.0,
):
    """Dateline-safe track extent with the original wide geographic context.

    Restores the earlier plotting view, which gave the surrounding coastlines,
    state boundaries, land shading, and ocean meaningful space on the map.
    The padding is adjustable, and it changes only plot framing, not tracking.
    """
    if not np.isfinite(padding_degrees) or padding_degrees < 0:
        raise ValueError("--map-padding-deg must be a finite nonnegative number")
    lats = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
    lons = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(lats) & np.isfinite(lons)
    if not finite.any():
        raise ValueError("No finite track centers available for mapping")
    reference = float(lons[np.flatnonzero(finite)[0]])
    lons = reference + st_wrap180(lons - reference)
    lat_range = max(float(np.ptp(lats[finite])), .2)
    lon_range = max(float(np.ptp(lons[finite])), .2)
    # The refined close-up (~0.55 degrees) cropped the Louisiana coast. The
    # earlier hurricane-track chart used a 2-degree minimum geographic margin.
    pad_lat = max(padding_degrees, lat_range * .20)
    pad_lon = max(padding_degrees, lon_range * .20)
    extent = [float(np.min(lons[finite]) - pad_lon), float(np.max(lons[finite]) + pad_lon),
              float(np.min(lats[finite]) - pad_lat), float(np.max(lats[finite]) + pad_lat)]
    llon = np.array([[extent[0], extent[1]], [extent[0], extent[1]]])
    llat = np.array([[extent[2], extent[2]], [extent[3], extent[3]]])
    return extent, llat, llon


def add_map_quality_columns(
    df: pd.DataFrame, max_speed_kt: float = 45., use_smoothed_output: bool = True
) -> pd.DataFrame:
    """Non-destructive display QC. Flag, never delete or relocate, large center jumps.

    A distance/time threshold indicates a suspect model-center association,
    not proof of an erroneous atmospheric trajectory. Retain all raw diagnostics.
    """
    out = df.copy()
    lat_col, lon_col = output_track_columns(out, use_smoothed_output)
    lats = pd.to_numeric(out[lat_col], errors="coerce").to_numpy(float)
    lons = pd.to_numeric(out[lon_col], errors="coerce").to_numpy(float)
    ts = pd.to_datetime(out["time_utc"], utc=True)
    speeds = np.full(len(out), np.nan)
    for i in range(1, len(out)):
        dt_hours = (ts.iloc[i] - ts.iloc[i-1]).total_seconds() / 3600.
        if dt_hours > 0 and np.isfinite([lats[i], lats[i-1], lons[i], lons[i-1]]).all():
            speeds[i] = (float(haversine_km(lats[i-1], lons[i-1], lats[i], lons[i]))
                         / (dt_hours * 1.852))
    flagged_by_tracker = np.zeros(len(out), dtype=bool)
    for flag_col in ("max_jump_exceeded", "used_as_fallback_after_jump_limit", "center_unresolved"):
        if flag_col in out:
            flagged_by_tracker |= out[flag_col].fillna(False).astype(bool).to_numpy()
    if 'track_flag' in out:
        flagged_by_tracker |= out['track_flag'].isin(['unresolved', 'structure_disagreement']).to_numpy()
    flagged = (np.isfinite(speeds) & (speeds > max_speed_kt)) | flagged_by_tracker
    if len(flagged):
        flagged[0] = False
    out["map_segment_motion_kt"] = speeds
    out["map_suspect_jump"] = flagged
    out["map_jump_threshold_kt"] = max_speed_kt
    return out


def _render_track_labels(fig, ax, lons, lats, times, label_interval_hours, xycoords):
    """Place short UTC date labels with deterministic overlap avoidance."""
    if label_interval_hours <= 0:
        return 0
    # Canvas positions are used only for collision-aware label placement.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer).padded(-6)
    taken = []
    written = 0
    t0 = times.iloc[0]
    # Short labels are deliberately designed for 00Z/12Z synoptic positions.
    offsets = [(12, 12), (-12, 12), (12, -12), (-12, -12),
               (24, 3), (-24, 3), (4, 25), (4, -28),
               (32, 20), (-32, -20), (32, -20), (-32, 20)]
    from matplotlib.transforms import Bbox
    for i, tm in enumerate(times):
        elapsed = (tm - t0).total_seconds() / 3600.
        if abs(elapsed / label_interval_hours - round(elapsed / label_interval_hours)) > 1e-3:
            continue
        if not np.isfinite([lons[i], lats[i]]).all():
            continue
        stamp = tm.strftime("%d %b\n%HZ")
        for ox, oy in offsets:
            ha = "left" if ox >= 0 else "right"
            va = "bottom" if oy >= 0 else "top"
            artist = ax.annotate(
                stamp, xy=(float(lons[i]), float(lats[i])), xycoords=xycoords,
                xytext=(ox, oy), textcoords="offset points", ha=ha, va=va,
                fontsize=8, linespacing=1.02, zorder=17, annotation_clip=True,
                bbox=dict(boxstyle="round,pad=.18", fc="white", ec="#888888",
                          lw=.45, alpha=.93),
                arrowprops=dict(arrowstyle="-", color="#757575", lw=.55,
                                alpha=.8, shrinkA=2, shrinkB=5),
            )
            bb = artist.get_window_extent(renderer).padded(3)
            contained = (bb.x0 >= axes_box.x0 and bb.x1 <= axes_box.x1
                         and bb.y0 >= axes_box.y0 and bb.y1 <= axes_box.y1)
            if contained and all(not bb.overlaps(old) for old in taken):
                taken.append(bb)
                written += 1
                break
            artist.remove()
    return written


def plot_track_map(
    df: pd.DataFrame, path: Path, best_df: pd.DataFrame | None, dpi: int,
    use_smoothed_output: bool, label_interval_hours: int,
    arrow_every: int = 6, show_cities: bool = False,
    map_jump_speed_kt: float = 45., map_padding_deg: float = 2.0,
) -> None:
    """Clear, category-colored hurricane track with labeled suspicious jumps.

    Colored legs indicate model max 10-m wind bins, not official 1-min intensity.
    Suspect discontinuities are shown as dashed connectors; no rows are removed.
    Direction arrows have fixed display lengths and DO NOT encode motion speed.
    """
    if df.empty:
        return
    if "map_suspect_jump" not in df or "map_segment_motion_kt" not in df:
        df = add_map_quality_columns(df, max_speed_kt=map_jump_speed_kt,
                                     use_smoothed_output=use_smoothed_output)
    lat_col, lon_col = output_track_columns(df, use_smoothed_output)
    _, wind_col = output_intensity_columns(df, use_smoothed_output)
    extent, lats_np, lons_np = map_extent_arrays(
        df, lat_col, lon_col, padding_degrees=map_padding_deg
    )
    base_lon = float(df[lon_col].iloc[0])
    lons = base_lon + st_wrap180(df[lon_col].to_numpy(float) - base_lon)
    lats = df[lat_col].to_numpy(float)
    wind_kt = pd.to_numeric(df[wind_col], errors="coerce").to_numpy(float)
    point_colors = strength_colors(wind_kt)
    suspect = df["map_suspect_jump"].fillna(False).astype(bool).to_numpy()

    # Reserve a real legend column, not an extra legend beyond the map border.
    fig = plt.figure(figsize=(14.1, 9.1), dpi=dpi, facecolor="white")
    geo = crs is not None
    if geo:
        central_lon = float(np.mean(lons[np.isfinite(lons)]))
        ax = fig.add_axes([.07, .13, .68, .70],
                          projection=crs.PlateCarree(central_longitude=central_lon))
        coords = dict(transform=crs.PlateCarree())
        label_xycoords = crs.PlateCarree()._as_mpl_transform(ax)
        ax.set_extent(extent, crs=crs.PlateCarree())
        # Original land/ocean contrast and coast detail. The generous framing
        # is important: features beyond the track bounding box stay visible.
        ax.set_facecolor(STORM_PALETTE["water"])
        ax.add_feature(cfeature.LAND, facecolor=STORM_PALETTE["land"], zorder=0)
        # Keep native hurricane-track cartography, but draw under the cyclone.
        for feature in features:
            add_feature(ax, *feature)
        if show_cities:
            plot_cities(ax, lons_np, lats_np, 10.0, 10.0)
        add_latlon_gridlines(ax)
    else:
        ax = fig.add_axes([.09, .13, .66, .70])
        coords = {}
        label_xycoords = "data"
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.grid(True, color="#888888", linestyle="--", alpha=.3, lw=.8)
        ax.set_aspect(1. / max(np.cos(np.deg2rad(float(np.nanmean(lats)))), 0.1))

    # Traces stop across suspect jumps, rather than implying a verified path.
    for i in range(1, len(df)):
        if not np.isfinite([lons[i-1], lons[i], lats[i-1], lats[i]]).all():
            continue
        if suspect[i]:
            ax.plot(lons[i-1:i+1], lats[i-1:i+1], ls="--", lw=1.5,
                    color="#666666", alpha=.8, zorder=4, **coords)
        else:
            ax.plot(lons[i-1:i+1], lats[i-1:i+1], lw=2.35,
                    color=point_colors[i], zorder=5, solid_capstyle="round", **coords)
    # Draw the larger review X UNDER the smaller category-color dots. Keeping
    # the arms visible around the dot conveys uncertainty without masking wind.
    if suspect.any():
        ax.scatter(lons[suspect], lats[suspect], marker="x", s=145,
                   color="#4b4b4b", linewidths=1.65, zorder=6, **coords)
    ax.scatter(lons, lats, c=point_colors, s=38, edgecolors="black",
               linewidths=.55, zorder=7, **coords)

    if best_df is not None and not best_df.empty:
        blons = base_lon + st_wrap180(best_df.best_lon.to_numpy(float) - base_lon)
        ax.plot(blons, best_df.best_lat, linestyle="--", marker="x",
                color=STORM_PALETTE["best"], linewidth=1.2,
                markersize=4, label="Best track", zorder=6, **coords)

    # Only a handful of small arrows. They convey direction, not magnitude.
    if arrow_every > 0 and {"fitted_u_ms", "fitted_v_ms"}.issubset(df.columns):
        possibilities = np.arange(0, len(df), max(arrow_every, 1))
        if len(possibilities) > 4:
            possibilities = possibilities[np.linspace(0, len(possibilities)-1, 4).round().astype(int)]
        uu = df.fitted_u_ms.to_numpy(float)
        vv = df.fitted_v_ms.to_numpy(float)
        for i in possibilities:
            # Skip uncertain vectors near model-center jumps.
            if suspect[max(0, i-1):min(len(suspect), i+2)].any():
                continue
            speed = np.hypot(uu[i], vv[i])
            if not np.isfinite(speed) or speed < .2 or speed > 25.:
                continue
            ux, uy = uu[i] / speed, vv[i] / speed
            ax.annotate("", xy=(float(lons[i]), float(lats[i])),
                        xycoords=label_xycoords,
                        xytext=(-ux * 18, -uy * 18), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-|>", lw=1.7,
                                        color=STORM_PALETTE["motion"],
                                        mutation_scale=12),
                        zorder=11)

    times = pd.to_datetime(df.time_utc, utc=True)
    _render_track_labels(fig, ax, lons, lats, times,
                         label_interval_hours, label_xycoords)

    # Scale and north symbol are unobtrusive and independent of the legend.
    ax.annotate("N", xy=(.94, .965), xytext=(.94, .895), xycoords="axes fraction",
                ha="center", va="center", fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="-|>", lw=1.3))
    lat_bar = extent[2] + (extent[3] - extent[2]) * .095
    lon_bar = extent[0] + (extent[1] - extent[0]) * .085
    span_km = max(float(st_haversine_km(lat_bar, extent[0], lat_bar, extent[1])), 1.)
    scale_km = min([25, 50, 100, 200, 300, 500, 1000],
                   key=lambda x: abs(x - span_km / 5.))
    dlon = scale_km / (111.32 * max(np.cos(np.radians(lat_bar)), .1))
    ax.plot([lon_bar, lon_bar + dlon], [lat_bar, lat_bar],
            color=STORM_PALETTE["track"], linewidth=2.5, zorder=11, **coords)
    ax.text(lon_bar + dlon / 2, lat_bar + (extent[3] - extent[2]) * .017,
            f"{scale_km} km", ha="center", fontsize=9, zorder=11, **coords)

    title_suffix = "Smoothed Center Track" if use_smoothed_output else "Raw Center Track"
    model_heading = ("Weather Research and Forecasting Model"
                     if (df.domain_used != "generic").any() else "Gridded Meteorological Model")
    fig.text(.065, .966, model_heading, ha="left", va="top", fontsize=17, weight="semibold")
    fig.text(.065, .919, f"Hurricane/Vortex Track  |  {title_suffix}",
             ha="left", va="top", fontsize=12)
    fig.text(.065, .886, grid_spacing_title_text(df),
             ha="left", va="top", fontsize=10, color="#444444")
    fig.text(.065, .042, valid_title_text(df), ha="left", va="bottom",
             fontsize=10, color="#333333")

    # Dedicated legend panel: model wind-speed bins in three unit systems.
    panel = fig.add_axes([.76, .20, .235, .64])
    panel.set_axis_off()
    panel.text(.03, .98, "10 m MODEL WIND", transform=panel.transAxes,
               va="top", fontsize=12, weight="bold")
    panel.text(.03, .92, "Category colors  |  mph, m/s, kt", transform=panel.transAxes,
               va="top", fontsize=9.0, color="#444444")
    from matplotlib.patches import Rectangle
    for i, (name, low, high, color) in enumerate(WIND_STRENGTHS):
        y = .842 - i * .090
        panel.add_patch(Rectangle((.025, y-.026), .075, .045, facecolor=color,
                                  edgecolor="black", lw=.6, transform=panel.transAxes))
        panel.text(.13, y+.015, name, va="center", fontsize=9.0,
                   weight="medium", transform=panel.transAxes)
        panel.text(.13, y-.019, wind_band_unit_label(low, high), va="center",
                   fontsize=6.8, color="#444444", transform=panel.transAxes)

    # The example uses a colored point laid over a gray X, matching the map.
    panel.scatter([.065], [.216], marker="x", s=130, c="#4b4b4b",
                  linewidths=1.7, zorder=1, transform=panel.transAxes, clip_on=False)
    panel.scatter([.065], [.216], s=50, c=["#00FFFF"], edgecolors="black",
                  linewidths=.6, zorder=2, transform=panel.transAxes, clip_on=False)
    panel.text(.13, .216, "Uncertain fix (X behind dot)", va="center",
               fontsize=8.4, transform=panel.transAxes)
    panel.plot([.025, .10], [.150, .150], transform=panel.transAxes,
               color="#666666", ls="--", lw=1.5)
    panel.text(.13, .150, "Review center jump", va="center",
               fontsize=8.7, transform=panel.transAxes)
    panel.annotate("", xy=(.10, .081), xytext=(.025, .081), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>", lw=1.7,
                                   color=STORM_PALETTE["motion"]))
    panel.text(.13, .081, "Motion heading", va="center",
               fontsize=8.7, transform=panel.transAxes)
    panel.text(.03, -.015, "Model grid-point 10 m winds.\nNot official 1-minute sustained intensity.",
               fontsize=8.1, color="#555555", va="top", transform=panel.transAxes)
    if suspect.any():
        fig.text(.78, .14, f"{int(np.count_nonzero(suspect))} suspect track legs\n"
                 f"(over {map_jump_speed_kt:g} kt or flagged)\nkept in CSV",
                 ha="left", va="top", color="#8a3535", fontsize=9)

    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_intensity(
    df: pd.DataFrame, path: Path, best_df: pd.DataFrame | None,
    dpi: int, use_smoothed_output: bool,
) -> None:
    """Two-panel operational chart with source-accurate wind-category colors."""
    if df.empty:
        return
    slp_col, vmax_col = output_intensity_columns(df, use_smoothed_output)
    times = pd.to_datetime(df.time_utc, utc=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), dpi=dpi,
                                    sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1], "hspace": .12})
    ax1.plot(times, df[slp_col], color=STORM_PALETTE["track"],
             marker="o", markersize=4, linewidth=1.6,
             markerfacecolor="white", markeredgecolor="black", label="Model min SLP")
    ax1.set_ylabel("Minimum sea-level pressure (hPa)")
    ax1.invert_yaxis()
    ax1.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax1.grid(True, alpha=.35)

    wind_values = pd.to_numeric(df[vmax_col], errors="coerce").to_numpy(float)
    wind_colors = strength_colors(wind_values)
    # Thin coral underlay ties the figure visually to the stormtrack palette;
    # individual line segments and square symbols show the wind category.
    ax2.plot(times, wind_values, color=STORM_PALETTE["motion"],
             linewidth=1.0, alpha=.65, label="Model max 10 m wind")
    for i in range(1, len(wind_values)):
        if np.isfinite(wind_values[i-1:i+1]).all():
            ax2.plot(times.iloc[i-1:i+1], wind_values[i-1:i+1],
                     color=wind_colors[i], linewidth=2.25, zorder=3)
    ax2.scatter(times, wind_values, c=wind_colors, marker="s", s=25,
                edgecolors="black", linewidths=.4, zorder=4)
    ax2.set_ylabel("Maximum 10 m wind (kt)")
    # Secondary metric scale uses the same data, not a second wind estimate.
    metric_axis = ax2.secondary_yaxis("right", functions=(
        lambda kt: np.asarray(kt) / KNOTS_PER_MPS,
        lambda mps: np.asarray(mps) * KNOTS_PER_MPS,
    ))
    metric_axis.set_ylabel("Maximum 10 m wind (m/s)")
    ax2.set_xlabel("Valid time (UTC)")
    ax2.grid(True, alpha=.35)
    # Very faint category bands are confined to the visible data range, to
    # avoid expanding a weak storm's y axis to the Category 5 threshold.
    finite_winds = wind_values[np.isfinite(wind_values)]
    if finite_winds.size:
        ymin, ymax = ax2.get_ylim()
        for _, low, high, color in WIND_STRENGTHS:
            lo = max(ymin, low / MPH_PER_KNOT)
            hi = min(ymax, high / MPH_PER_KNOT)
            if hi > lo:
                ax2.axhspan(lo, hi, facecolor=color, alpha=.08, zorder=0)

    if best_df is not None and not best_df.empty:
        bt = pd.to_datetime(best_df.time_utc, utc=True)
        if "best_mslp_hpa" in best_df:
            ax1.plot(bt, best_df.best_mslp_hpa, linestyle="--", marker="x",
                     color=STORM_PALETTE["best"], linewidth=1.3,
                     label="Best-track MSLP")
        if "best_vmax_kt" in best_df:
            ax2.plot(bt, best_df.best_vmax_kt, linestyle="--", marker="x",
                     color=STORM_PALETTE["best"], linewidth=1.3,
                     label="Best-track wind")
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    ax2.text(.99, .02, "Colors: model 10 m winds, not 1-minute sustained intensity",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
             bbox=dict(facecolor="white", alpha=.75, edgecolor="none"))
    title_suffix = "Smoothed Intensity" if use_smoothed_output else "Raw Intensity"
    model_heading = "Weather Research and Forecasting Model" if (df.domain_used != "generic").any() else "Gridded Meteorological Model"
    ax1.set_title(f"{model_heading}\n{grid_spacing_title_text(df)}\n"
                  "Hurricane/Vortex Intensity Time Series\n"
                  "Minimum Sea-Level Pressure (hPa)\nMaximum 10 m Wind (kt)\n"
                  f"{title_suffix}", loc="left", fontsize=13, pad=10)
    ax1.set_title(valid_title_text(df), loc="right", fontsize=13, pad=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def track_generic_files(args: argparse.Namespace) -> pd.DataFrame:
    """Track NetCDF, GRIB, Zarr, or WRF output using stormtrack's engine."""
    if not args.files:
        raise SystemExit("ERROR: Generic mode requires --files FILE_OR_GLOB [FILE_OR_GLOB ...].")
    paths = sorted({p for pattern in args.files for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit("ERROR: No input files matched --files patterns.")
    gf = dict(pair.split("=", 1) for pair in args.grib_filter.split(",")) if args.grib_filter else None
    frs = st_frames(paths, var=args.pressure_var, mask_terrain=args.mask_terrain, grib_filter=gf)
    frs = [f for f in frs if pd.notna(f.time)]
    if args.max_times and args.max_times > 0:
        frs = frs[:args.max_times]
    if not frs:
        raise RuntimeError("No frames with recognizable times. Supply valid CF times or timestamped filenames.")

    result = st_track(
        frs,
        first_guess=(args.init_lat, args.init_lon) if args.init_lat is not None else None,
        box=args.first_box, refine_km=args.refine_km, max_speed=args.max_speed,
        max_dev_speed=args.max_dev_speed, min_depth=args.min_depth,
        smooth_km=args.smooth_km, vmax_km=args.rmw_search_radius_km,
        max_misses=args.max_misses, edge=args.edge_buffer_grid_points,
    )
    result = st_motion(result, window_h=args.motion_window_h)
    if result.empty:
        raise RuntimeError("Stormtrack did not process any frames.")
    good = result[result.flag != "lost"].reset_index(drop=True)
    if good.empty:
        raise RuntimeError("No valid centres identified; adjust --init-lat/--init-lon or search options.")

    # Preserve both the native stormtrack fields and the hurricane CSV/ATCF API.
    # For generic datasets, no upper-air vorticity is inferred from unavailable data.
    rows = []
    available = {pd.Timestamp(f.time): f for f in frs}
    for _, r in good.iterrows():
        f = available.get(pd.Timestamp(r.time))
        dx, dy = (np.nan, np.nan)
        rmw = np.nan
        slp_at_vmax = np.nan
        if f is not None:
            try:
                _, _, dx, dy, _, _ = compute_grid_and_spacing(f.lat, f.lon)
            except Exception:
                pass
            if f.wspd is not None:
                dist = st_haversine_km(r.lat, r.lon, f.lat, f.lon)
                mask = np.isfinite(f.wspd) & (dist <= args.rmw_search_radius_km)
                if mask.any():
                    ind = np.unravel_index(np.nanargmax(np.where(mask, f.wspd, np.nan)), f.wspd.shape)
                    rmw = float(dist[ind])
                    slp_at_vmax = float(f.p[ind])
        rows.append({
            "time_utc": pd.Timestamp(r.time).isoformat() + "Z",
            "domain_used": "generic", "source_file": r.source,
            "time_index": np.nan, "avg_dx_km": dx, "avg_dy_km": dy,
            "center_method": "stormtrack", "final_center_lat": float(r.lat),
            "final_center_lon": float(r.lon),
            "surface_pressure_center_lat": float(r.grid_lat),
            "surface_pressure_center_lon": float(r.grid_lon),
            "min_slp_hpa": float(r.pmin_hpa),
            "pressure_depth_hpa": float(r.depth_hpa),
            "track_flag": str(r.flag),
            "vorticity850_center_lat": np.nan, "vorticity850_center_lon": np.nan,
            "vorticity850_value": np.nan,
            "vorticity700_center_lat": np.nan, "vorticity700_center_lon": np.nan,
            "vorticity700_value": np.nan,
            "wind10_center_lat": np.nan, "wind10_center_lon": np.nan,
            "wind10_center_value_mps": np.nan,
            "vmax10_mps": float(r.vmax_ms),
            "vmax10_kt": float(r.vmax_ms * KNOTS_PER_MPS),
            "rmw_km": rmw, "slp_at_vmax_hpa": slp_at_vmax,
            "pressure_to_final_km": float(st_haversine_km(r.grid_lat, r.grid_lon, r.lat, r.lon)),
            "vort850_to_final_km": np.nan, "vort700_to_final_km": np.nan,
            "wind_to_final_km": np.nan,
            "hemisphere_used": "south" if r.lat < 0 else "north",
            "search_radius_km": np.nan,
            "jump_from_previous_km": np.nan,
            "max_jump_exceeded": False,
            "used_as_fallback_after_jump_limit": False,
        })
    df = pd.DataFrame(rows)
    df = add_motion_columns(df, lat_col="final_center_lat", lon_col="final_center_lon")
    df = add_smoothed_track_columns(df, window=args.track_smooth_window)
    df = add_smoothed_intensity_columns(df, window=args.intensity_smooth_window)
    df = add_fitted_motion(df, window_h=args.motion_window_h)
    return df

###############################################################################
# Command line
###############################################################################
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track a WRF hurricane/vortex center from wrfout files. "
            "Pass d01/d02/d03 for native WRF diagnostics, or generic --files for NetCDF/GRIB/Zarr."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "domain",
        help=(
            "WRF domain to track first, such as d01, d02, or d03, or 'generic'. "
            "Backward-compatible wrapper calls may pass the WRF run directory here "
            "and the domain as the next positional argument."
        ),
    )

    parser.add_argument(
        "legacy_domain",
        nargs="?",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--wrf-dir",
        default=".",
        help="Directory containing wrfout files.",
    )

    parser.add_argument(
        "--file-glob",
        default=None,
        help="Optional explicit wrfout glob pattern.",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default is wrf_hurricane_track_<domain> or stormtrack_hurricane_track_generic.",
    )

    parser.add_argument(
        "--init-lat",
        type=float,
        default=None,
        help="Optional first-guess latitude.",
    )

    parser.add_argument(
        "--init-lon",
        type=float,
        default=None,
        help="Optional first-guess longitude.",
    )

    parser.add_argument(
        "--search-radius-km",
        type=float,
        default=250.0,
        help="Search radius after the first center is found.",
    )

    parser.add_argument(
        "--rmw-search-radius-km",
        type=float,
        default=250.0,
        help="Radius used to find maximum 10 m wind and RMW.",
    )

    parser.add_argument(
        "--wind-center-radius-km",
        type=float,
        default=150.0,
        help="Radius around pressure center used to find 10 m wind minimum.",
    )

    parser.add_argument(
        "--center-method",
        choices=["weighted", "pressure", "vorticity850", "stormtrack"],
        default="weighted",
    )

    parser.add_argument(
        "--hemisphere",
        choices=["auto", "north", "south"],
        default="auto",
    )

    parser.add_argument("--files", nargs="+", default=None,
                        help="Input NetCDF/GRIB/Zarr patterns for generic mode (quote globs).")
    parser.add_argument("--pressure-var", default=None,
                        help="Override sea-level-pressure variable for generic mode.")
    parser.add_argument("--grib-filter", default=None,
                        help="cfgrib filter, e.g. typeOfLevel=meanSea.")
    parser.add_argument("--first-box", nargs=4, type=float, default=None,
                        metavar=("LAT0", "LAT1", "LON0", "LON1"),
                        help="Generic mode: constrain first unrestricted search.")
    parser.add_argument("--mask-terrain", type=float, default=200,
                        help="Generic mode: mask terrain higher than this when reducing surface pressure (m).")
    parser.add_argument("--refine-km", type=float, default=150.0,
                        help="Pressure-deficit centroid search radius (km).")
    parser.add_argument("--max-speed", type=float, default=30.0,
                        help="Generic mode: assumed initial maximum motion speed (m/s).")
    parser.add_argument("--max-dev-speed", type=float, default=15.0,
                        help="Generic mode: speed uncertainty after motion known (m/s).")
    parser.add_argument("--min-depth", type=float, default=1.0,
                        help="Pressure-deficit depth threshold for weak flag (hPa).")
    parser.add_argument("--smooth-km", type=float, default=0.0,
                        help="Generic mode: Gaussian field smoothing radius (km).")
    parser.add_argument("--max-misses", type=int, default=2,
                        help="Generic mode: stop after this many consecutive lost centers.")
    parser.add_argument("--motion-window-h", type=float, default=3.0,
                        help="Half-window (hours) for the least-squares motion vector fit.")
    parser.add_argument("--arrow-every", type=int, default=6,
                        help="Draw up to four small directional arrows sampled every N centers (0 disables).")
    parser.add_argument("--map-jump-speed-kt", type=float, default=45.,
                        help="Flag map segments above this center motion speed (kt); preserve all CSV fixes.")
    parser.add_argument("--map-padding-deg", type=float, default=2.0,
                        help="Minimum lat/lon geographic padding around the track, in degrees. "
                             "Default restores the original wide map background and coastal context.")
    parser.add_argument("--cities", action="store_true",
                        help="Enable optional Natural Earth city labels (may download city dataset).")
    parser.add_argument("--slp-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--vort-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--wind-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--edge-buffer-grid-points", type=int, default=5)
    parser.add_argument("--min-search-points", type=int, default=25)

    # Treat different WRF domains as independent forecast experiments unless
    # the user explicitly opts into mixing their center candidates. Switching
    # parents is NOT required for moving / vortex-following nests.
    parent_options = parser.add_mutually_exclusive_group()
    parent_options.add_argument(
        "--parent-fallback", dest="parent_fallback", action="store_true",
        help="OPT IN to parent-domain centers (d03 -> d02 -> d01) when the requested domain lacks a credible center.",
    )
    parent_options.add_argument(
        "--no-parent-fallback", dest="parent_fallback", action="store_false",
        help="Track only the requested domain (default; retained for command compatibility).",
    )
    parser.set_defaults(parent_fallback=False)

    parser.add_argument(
        "--best-track",
        default=None,
        help="Optional best-track CSV or ATCF-style file.",
    )

    parser.add_argument(
        "--max-times",
        type=int,
        default=None,
        help="Optional maximum number of WRF times to process.",
    )

    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")

    parser.add_argument("--weight-pressure", type=float, default=0.45)
    parser.add_argument("--weight-vort850", type=float, default=0.25)
    parser.add_argument("--weight-vort700", type=float, default=0.20)
    parser.add_argument("--weight-wind", type=float, default=0.10)

    parser.add_argument(
        "--max-jump-km",
        type=float,
        default=30.0,
        help=(
            "Preferred displacement threshold (30 km) to trigger motion-aware comparison "
            "of stormtrack, pressure, vorticity850, and weighted candidates. This is a "
            "review trigger, NOT a physical movement cap. 0 disables the displacement trigger, "
            "but physical consistency checks remain active."
        ),
    )

    parser.add_argument('--max-motion-speed-kt', type=float, default=65.0,
                        help='Physical forward-speed screen in kt, scaled by elapsed hours; '
                             'not the 30 km fallback trigger.')
    parser.add_argument('--motion-position-buffer-km', type=float, default=15.,
                        help='Distance tolerance for motion speed screening (km).')
    parser.add_argument('--max-prediction-speed-kt', type=float, default=45.,
                        help='Maximum credible previous speed for motion extrapolation (kt).')
    parser.add_argument('--max-prediction-hours', type=float, default=6.,
                        help='Maximum interval for extrapolating storm movement (hours).')
    parser.add_argument('--prediction-tolerance-km', type=float, default=75.,
                        help='Allowed distance from predicted center before a candidate is suspect (km).')
    parser.add_argument('--prediction-tolerance-growth-kmh', type=float, default=20.,
                        help='Additional prediction error tolerance per hour after the first.')
    parser.add_argument('--prediction-switch-margin-km', type=float, default=20.,
                        help='How much better another credible method must score to override a nearby stationary center (km).')
    parser.add_argument('--center-coherence-km', type=float, default=90.,
                        help='Maximum offset of an accepted method center from minimum SLP (km).')
    parser.add_argument('--vorticity-agreement-km', type=float, default=120.,
                        help='Maximum distance between SLP and a supporting vorticity center (km).')
    parser.add_argument('--no-motion-prediction', action='store_true',
                        help='Disable prior-motion extrapolation but keep physical consistency checks.')
    parser.add_argument(
        "--track-smooth-window",
        type=int,
        default=3,
        help="Centered rolling-median window for smoothed track columns.",
    )

    parser.add_argument(
        "--intensity-smooth-window",
        type=int,
        default=3,
        help="Centered rolling-median window for plotted/output intensity columns.",
    )

    parser.add_argument(
        "--no-smoothed-output",
        action="store_true",
        help="Use raw centers and raw intensity for map and ATCF-style output.",
    )

    parser.add_argument(
        "--label-interval-hours",
        type=int,
        default=12,
        help="Track map label interval in hours (12 recommended). Use 0 to disable labels.",
    )

    return parser


def make_output_stem(domain: str, df: pd.DataFrame) -> str:
    start = pd.to_datetime(df["time_utc"].iloc[0], utc=True).strftime("%Y%m%d%H")
    end = pd.to_datetime(df["time_utc"].iloc[-1], utc=True).strftime("%Y%m%d%H")

    prefix = "stormtrack_hurricane_track" if domain == "generic" else "wrf_hurricane_track"
    return f"{prefix}_{safe_tag(domain)}_{start}_to_{end}"


def is_wrf_domain(value: object) -> bool:
    if not isinstance(value, str):
        return False

    return re.fullmatch(r"d\d+", value.strip().lower()) is not None


def normalize_cli_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """
    Support both command styles:

        python3 tropical_hurricane_track.py d02 --wrf-dir /path/to/WRF/run
        python3 tropical_hurricane_track.py /path/to/WRF/run d02

    The second form matches the existing wrapper convention used by the other
    WRF plotting scripts: <script> <wrf_run_dir> <domain>.
    """
    if args.legacy_domain is not None:
        first_arg = str(args.domain)
        second_arg = str(args.legacy_domain)

        if is_wrf_domain(second_arg):
            args.wrf_dir = first_arg
            args.domain = second_arg
        else:
            raise SystemExit(
                "ERROR: When two positional arguments are used, the second one "
                "must be a WRF domain such as d01, d02, or d03. "
                "Expected either: tropical_hurricane_track.py d02 --wrf-dir /path/to/run "
                "or: tropical_hurricane_track.py /path/to/run d02"
            )

    args.domain = str(args.domain).lower()

    if args.domain == "generic":
        return args

    if not is_wrf_domain(args.domain):
        raise SystemExit(
            "ERROR: Domain must look like d01, d02, or d03. "
            f"Received: {args.domain!r}"
        )

    return args



###############################################################################
# Embedded stormtrack engine (MIT, originally by SUMAILI NDEBA Bienvenu)
# Functions are prefixed st_ to avoid conflicts with WRF analysis helpers.
# The original pressure-centroid tracker, data readers, motion fit, plotting,
# and overlay API are retained; the latter is exposed as st_overlay().
###############################################################################
ST_R_KM = 6371.0
ST_G, ST_RD = 9.80665, 287.05

ST_NAMES = {
    "mslp": ["msl", "prmsl", "mslp", "slp", "pmsl", "mslet", "psl", "sea_level_pressure",
             "air_pressure_at_mean_sea_level", "air_pressure_at_sea_level"],
    "psfc": ["psfc", "sp", "ps", "surface_pressure", "pres_surface", "pressfc", "surface_air_pressure"],
    "hgt": ["hgt", "hgt_m", "orog", "terrain", "z_sfc", "surface_altitude", "hgtsfc", "orography"],
    "t2": ["t2", "t2m", "2t", "tmp2m", "air_temperature_2m"],
    "u10": ["u10", "10u", "ugrd10m", "u10m", "uas", "eastward_wind_10m"],
    "v10": ["v10", "10v", "vgrd10m", "v10m", "vas", "northward_wind_10m"],
    "lat": ["lat", "latitude", "xlat", "xlat_m", "nav_lat", "lat_0", "gridlat_0", "lats"],
    "lon": ["lon", "longitude", "xlong", "xlong_m", "nav_lon", "lon_0", "gridlon_0", "lons"],
    "time": ["time", "valid_time", "xtime", "times", "date", "t"],
}
ST_TIME_IN_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})[_ T](\d{2})[:_\-](\d{2})[:_\-](\d{2})")


def st_haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * ST_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def st_wrap180(lon):
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def st_to_plane(lat, lon, lat0, lon0):
    """local tangent plane (km east, km north) around (lat0, lon0); dateline safe"""
    x = ST_R_KM * np.radians(st_wrap180(lon - lon0)) * np.cos(np.radians(lat0))
    y = ST_R_KM * np.radians(np.asarray(lat) - lat0)
    return x, y


def st_from_plane(x, y, lat0, lon0):
    lat = lat0 + np.degrees(y / ST_R_KM)
    lon = lon0 + np.degrees(x / (ST_R_KM * np.cos(np.radians(lat0))))
    return float(lat), float(st_wrap180(lon))


def st_find(ds, key, required=False):
    want = ST_NAMES[key]
    for v in list(ds.data_vars) + list(ds.coords):
        if v.lower() in want or str(ds[v].attrs.get("standard_name", "")).lower() in want:
            return v
    if required:
        raise KeyError(f"no {key} variable found; looked for {want}. Use --var / rename.")
    return None


def st_open(path, grib_filter=None):
    low = path.lower()
    if low.endswith((".grb", ".grib", ".grb2", ".grib2")) or "pgrb2" in low:
        kw = {"indexpath": ""}
        tries = [grib_filter] if grib_filter else [{"typeOfLevel": "meanSea"}, {"typeOfLevel": "surface"}]
        last = None
        for flt in tries:
            try:
                return xr.open_dataset(path, engine="cfgrib", backend_kwargs={**kw, "filter_by_keys": flt})
            except Exception as e:
                last = e
        raise RuntimeError(f"cannot read GRIB {path}: {last}")
    if low.endswith(".zarr") or low.endswith(".zarr/"):
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def st_time_values(ds, tdim, path):
    """one pandas Timestamp per index of tdim (NaT if unknown)"""
    n = ds.sizes.get(tdim, 1) if tdim else 1
    if "Times" in ds.variables:
        raw = ds["Times"].values
        out = []
        for r in np.atleast_1d(raw):
            if hasattr(r, "dtype") and r.dtype.kind in "SU" and r.ndim:
                s = r.tobytes().decode("utf-8", errors="replace").replace("\x00", "").strip()
            else:
                s = str(r).strip("b'\"")
            out.append(pd.Timestamp(s.replace("_", " ")))
        if len(out) == n:
            return out
    for key in ("valid_time", "time", "xtime", "Time", "XTIME"):
        if key in ds.variables and np.issubdtype(ds[key].dtype, np.datetime64):
            v = np.atleast_1d(ds[key].values).ravel()
            if v.size == n:
                return [pd.Timestamp(x) for x in v]
            if v.size == 1:
                return [pd.Timestamp(v[0])] * n
    m = ST_TIME_IN_NAME.search(path)
    if m and n == 1:
        return [pd.Timestamp(*map(int, m.groups()))]
    return [pd.NaT] * n


@dataclass
class StormFrame:
    time: pd.Timestamp
    lat: np.ndarray
    lon: np.ndarray
    p: np.ndarray
    wspd: np.ndarray | None
    source: str


def st_frames(paths, var=None, mask_terrain=200.0, grib_filter=None):
    """return Frame objects from any list of files, sorted by time"""
    out = []
    for path in paths:
        ds = st_open(path, grib_filter)
        latv, lonv = st_find(ds, "lat", True), st_find(ds, "lon", True)
        pv = var or st_find(ds, "mslp")
        reduce = False
        if pv is None:
            pv = st_find(ds, "psfc", True)
            reduce = True
        hv, tv = st_find(ds, "hgt"), st_find(ds, "t2")
        uv, vv = st_find(ds, "u10"), st_find(ds, "v10")
        spatial = ds[pv].dims[-2:]
        extra = [d for d in ds[pv].dims if d not in spatial]
        tdim = extra[0] if extra else None
        times = st_time_values(ds, tdim, path)

        def at(name, k):
            """2-D slice of any variable at frame k: time dim -> k, any other non-spatial dim -> 0"""
            a = ds[name]
            if a.ndim == 1:
                return np.asarray(a.values, float)
            sel = {d: (k if d == tdim else 0) for d in a.dims if d not in spatial}
            return np.asarray(a.isel(sel).values, float).squeeze()

        for k in range(len(times)):
            lat, lon = at(latv, k), at(lonv, k)
            p = at(pv, k)
            if lat.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
            if np.nanmedian(p) > 2000:
                p = p / 100.0
            if reduce:
                if hv is None:
                    warnings.warn("only surface pressure and no terrain height: tracking raw surface pressure")
                else:
                    z = at(hv, k)
                    t = at(tv, k) if tv else 288.15 - 0.0065 * z
                    p = p * np.exp(ST_G * z / (ST_RD * (t + 0.0065 * z / 2)))
                    p = np.where(z > mask_terrain, np.nan, p)
            w = np.hypot(at(uv, k), at(vv, k)) if (uv and vv) else None
            out.append(StormFrame(times[k], lat, st_wrap180(lon), p, w, path))
        ds.close()
    out.sort(key=lambda f: (pd.Timestamp.max if pd.isna(f.time) else f.time))
    return out


def st_is_global(lon):
    """Global 2-D grid, without mistaking a local dateline-crossing nest for global."""
    if lon.ndim != 2:
        return False
    lon0 = float(np.asarray(lon).flat[0])
    continuous = lon0 + st_wrap180(lon - lon0)
    return float(np.nanmax(continuous) - np.nanmin(continuous)) > 350


def st_smooth(p, lat, lon, km):
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        warnings.warn("scipy missing: --smooth-km ignored")
        return p
    dy = np.nanmedian(st_haversine_km(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :]))
    sig = km / max(dy, 1e-6)
    ok = np.isfinite(p)
    num = gaussian_filter(np.where(ok, p, 0.0), sig)
    den = gaussian_filter(ok.astype(float), sig)
    return np.where(ok, num / np.maximum(den, 1e-9), np.nan)


def st_track(frs, first_guess=None, box=None, refine_km=150.0, max_speed=30.0, max_dev_speed=15.0,
          min_depth=1.0, smooth_km=0.0, vmax_km=200.0, max_misses=2, edge=5):
    """follow the storm through the frames; returns one row per frame (see README for columns)"""
    rows, misses = [], 0
    for f in frs:
        p = st_smooth(f.p, f.lat, f.lon, smooth_km) if smooth_km > 0 else f.p
        valid = np.isfinite(p)
        if edge > 0 and min(p.shape) > 2 * edge:
            border = np.ones(p.shape, bool)
            border[edge:-edge, edge:-edge] = False
            if not st_is_global(f.lon):
                valid &= ~border
        good = [r for r in rows if r["flag"] != "lost"]
        if not good:
            if first_guess is not None:
                c0, radius = first_guess, 500.0
            else:
                c0, radius = None, np.inf
                if box is not None:
                    la0, la1, lo0, lo1 = box
                    lo = f.lon
                    inside = (f.lat >= la0) & (f.lat <= la1) & (st_wrap180(lo - lo0) >= 0) & (st_wrap180(lo1 - lo) >= 0)
                    valid &= inside
        else:
            last = good[-1]
            dt = (f.time - last["time"]).total_seconds() if pd.notna(f.time) and pd.notna(last["time"]) else 3600.0
            if len(good) >= 2 and pd.notna(good[-2]["time"]):
                x, y = st_to_plane(last["lat"], last["lon"], good[-2]["lat"], good[-2]["lon"])
                ddt = (last["time"] - good[-2]["time"]).total_seconds() or 1.0
                c0 = st_from_plane(x / ddt * dt, y / ddt * dt, last["lat"], last["lon"])
                radius = max_dev_speed * dt / 1000 + 50
            else:
                c0, radius = (last["lat"], last["lon"]), max_speed * dt / 1000 + 50
            radius = float(np.clip(radius, 100, 600))
        if c0 is not None:
            valid &= st_haversine_km(c0[0], c0[1], f.lat, f.lon) <= radius
        if not valid.any():
            rows.append(dict(time=f.time, lat=np.nan, lon=np.nan, grid_lat=np.nan, grid_lon=np.nan,
                             pmin_hpa=np.nan, depth_hpa=np.nan, vmax_ms=np.nan, flag="lost", source=f.source))
            misses += 1
            if misses > max_misses:
                break
            continue
        misses = 0
        j, i = np.unravel_index(np.argmin(np.where(valid, p, np.inf)), p.shape)
        la0, lo0 = f.lat[j, i], f.lon[j, i]
        disc = np.isfinite(p) & (st_haversine_km(la0, lo0, f.lat, f.lon) <= refine_km)
        penv = np.nanpercentile(p[disc], 90)
        w = np.where(disc, np.clip(penv - p, 0, None), 0.0)
        if w.sum() > 0:
            x, y = st_to_plane(f.lat, f.lon, la0, lo0)
            clat, clon = st_from_plane((w * x).sum() / w.sum(), (w * y).sum() / w.sum(), la0, lo0)
        else:
            clat, clon = float(la0), float(lo0)
        depth = float(penv - p[j, i])
        vmax = np.nan
        if f.wspd is not None:
            near = st_haversine_km(clat, clon, f.lat, f.lon) <= vmax_km
            vmax = float(np.nanmax(np.where(near, f.wspd, np.nan)))
        rows.append(dict(time=f.time, lat=clat, lon=clon, grid_lat=float(la0), grid_lon=float(lo0),
                         pmin_hpa=float(f.p[j, i]), depth_hpa=depth, vmax_ms=vmax,
                         flag="ok" if depth >= min_depth else "weak", source=f.source))
    return st_motion(pd.DataFrame(rows))


def st_motion(df, window_h=3.0):
    """forward speed (m/s), heading (deg from north), east/north components from a local line fit"""
    df = df.copy()
    for c in ("speed_ms", "speed_kt", "heading_deg", "u_ms", "v_ms"):
        df[c] = np.nan
    ok = df[(df.flag != "lost") & df.time.notna()]
    for idx, r in ok.iterrows():
        win = ok[(ok.time - r.time).abs() <= pd.Timedelta(hours=window_h)]
        if len(win) < 2:
            continue
        s = (win.time - r.time).dt.total_seconds().values
        if np.ptp(s) == 0:
            continue
        x, y = st_to_plane(win.lat.values, win.lon.values, r.lat, r.lon)
        u = np.polyfit(s, x, 1)[0] * 1000
        v = np.polyfit(s, y, 1)[0] * 1000
        spd = np.hypot(u, v)
        df.loc[idx, ["u_ms", "v_ms", "speed_ms", "speed_kt", "heading_deg"]] = [
            u, v, spd, spd / 0.514444, np.degrees(np.arctan2(u, v)) % 360]
    return df


def st_plot(df, path, every=1, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.flag != "lost"].reset_index(drop=True)
    lon0 = float(d.lon.iloc[0])
    lon = pd.Series(lon0 + st_wrap180(d.lon - lon0), index=d.index)
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        pc = ccrs.PlateCarree(central_longitude=float(np.round(lon.mean())))
        fig, ax = plt.subplots(figsize=(9, 7.5), subplot_kw={"projection": pc})
        tr = ccrs.PlateCarree()
        ax.add_feature(cfeature.LAND, facecolor="#e6e3dc")
        ax.add_feature(cfeature.COASTLINE, lw=0.7)
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray")
        gl.top_labels = gl.right_labels = False
        kw = {"transform": tr}
    except ImportError:
        fig, ax = plt.subplots(figsize=(9, 7.5))
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        ax.set_aspect(1 / np.cos(np.radians(d.lat.mean())))
        kw = {}
    ax.plot(lon, d.lat, "-", color="k", lw=1.5, **kw)
    sc = ax.scatter(lon, d.lat, c=d.pmin_hpa, cmap="viridis_r", s=40, edgecolors="k", zorder=3, **kw)
    a = d.iloc[::max(every, 1)].dropna(subset=["u_ms"])
    q = ax.quiver(lon[a.index].values, a.lat.values, a.u_ms.values, a.v_ms.values, color="#d6604d",
                  angles="uv", scale_units="inches", scale=8, width=0.005, zorder=4, **kw)
    ax.quiverkey(q, 0.85, 0.05, 8, "motion 8 m/s", labelpos="E", coordinates="axes")
    pad = 1.5
    try:
        ax.set_extent([lon.min() - pad, lon.max() + pad, d.lat.min() - pad, d.lat.max() + pad], crs=kw["transform"])
    except (KeyError, AttributeError):
        pass
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.03)
    cb.locator = MaxNLocator(5)
    cb.formatter = FormatStrFormatter("%.1f" if np.ptp(d.pmin_hpa) < 5 else "%.0f")
    cb.update_ticks()
    cb.set_label("Minimum pressure (hPa)")
    lat_b = float(d.lat.min()) - 0.8 * pad
    width_km = max(float(st_haversine_km(lat_b, lon.min() - pad, lat_b, lon.max() + pad)), 1.0)
    bar = min([50, 100, 200, 300, 500, 1000, 2000], key=lambda b: abs(b - width_km / 5))
    lon_b = float(lon.min()) - 0.8 * pad
    dlon = bar / (111.32 * np.cos(np.radians(lat_b)))
    ax.plot([lon_b, lon_b + dlon], [lat_b, lat_b], color="k", lw=3, zorder=5, **kw)
    ax.text(lon_b + dlon / 2, lat_b + 0.08 * pad, f"{bar} km", ha="center", va="bottom", zorder=5, **kw)
    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.87), xycoords="axes fraction", ha="center",
                va="center", fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=1.5))
    t0, t1 = d.time.iloc[0], d.time.iloc[-1]
    ax.set_title(title or f"Track {t0:%d %b %HZ} – {t1:%d %b %HZ}", loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@dataclass
class StormOverlayStyle:
    """Every visual choice of overlay(). Edit the defaults here, or pass OverlayStyle(...) / keywords."""
    track_color: str = "k"
    track_lw: float = 2.0
    future_ls: str | None = "--"
    centre_marker: str = "o"
    centre_size: float = 12.0
    centre_face: str = "white"
    arrow_color: str = "#d6604d"
    arrow_every: int = 3
    arrow_ms_per_inch: object = 8.0
    arrow_width: float = 0.005
    past_arrow_frac: float = 0.5
    halo: bool = True
    label: bool = True
    units: str = "kt"
    key: bool = True
    key_xy: tuple = (0.80, 0.05)
    fontsize: float = 12.0
    zorder: float = 20.0


ST_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def st_read_track(path):
    """load a CSV written by stormtrack (times parsed)"""
    return pd.read_csv(path, parse_dates=["time"])


def st_track_at(df, t):
    """centre and motion interpolated to time t (None if t is outside the track)"""
    d = df[df.flag != "lost"].set_index("time").sort_index()
    t = pd.Timestamp(t)
    if d.empty or t < d.index.min() or t > d.index.max():
        return None
    x = d[["lat", "u_ms", "v_ms", "pmin_hpa"]].copy()
    x["lon"] = float(d.lon.iloc[0]) + st_wrap180(d.lon - float(d.lon.iloc[0]))
    r = x.reindex(x.index.union([t])).interpolate(method="time").loc[t]
    spd = float(np.hypot(r.u_ms, r.v_ms))
    return dict(time=t, lat=float(r.lat), lon=float(st_wrap180(r.lon)), u_ms=float(r.u_ms), v_ms=float(r.v_ms),
                speed_ms=spd, heading_deg=float(np.degrees(np.arctan2(r.u_ms, r.v_ms)) % 360),
                pmin_hpa=float(r.pmin_hpa))


def st_overlay(ax, df, valid_time=None, style=None, **kw):
    """Draw the tracked storm on an EXISTING map: past track, centre at valid_time, motion arrows, label.

    ax          any matplotlib Axes: a cartopy GeoAxes (e.g. a wrf-python chart, any projection) or a
                plain lon/lat Axes (0-360 or -180..180 longitudes are both handled)
    df          DataFrame from track() / read_track()
    valid_time  time of the chart being drawn; None = whole track, no current centre
    style       OverlayStyle; extra keywords override single fields: overlay(ax, df, t, arrow_color="m")
    returns     dict of the matplotlib artists, so anything can still be restyled afterwards
    """
    import matplotlib.patheffects as pe
    sty = style or StormOverlayStyle()
    for k, v in kw.items():
        if not hasattr(sty, k):
            raise TypeError(f"unknown style field {k!r}")
        setattr(sty, k, v)
    d = df[df.flag != "lost"].sort_values("time").reset_index(drop=True)
    geo = hasattr(ax, "projection")
    if geo:
        import cartopy.crs as ccrs
        pc = ccrs.PlateCarree()
        tkw = {"transform": pc}
        text_xy = pc._as_mpl_transform(ax)
    else:
        tkw, text_xy = {}, "data"
    lon0 = float(d.lon.iloc[0])
    lon = pd.Series(lon0 + st_wrap180(d.lon - lon0), index=d.index)
    use360 = (not geo) and ax.get_xlim()[1] > 180
    if use360:
        lon = lon % 360
    halo = [pe.withStroke(linewidth=sty.track_lw + 2.5, foreground="white")] if sty.halo else None
    halo_q = [pe.Stroke(linewidth=1.2, foreground="white"), pe.Normal()] if sty.halo else None
    z = sty.zorder
    art = {}
    if valid_time is None:
        past, fut = d.index, d.index[:0]
    else:
        vt = pd.Timestamp(valid_time)
        past, fut = d.index[d.time <= vt], d.index[d.time >= vt]
    art["track"] = ax.plot(lon[past], d.lat[past], "-", color=sty.track_color, lw=sty.track_lw,
                           path_effects=halo, zorder=z, **tkw)
    if sty.future_ls and len(fut) > 1:
        art["future"] = ax.plot(lon[fut], d.lat[fut], sty.future_ls, color=sty.track_color,
                                lw=sty.track_lw * 0.7, path_effects=halo, zorder=z, **tkw)
    if sty.arrow_ms_per_inch == "auto":
        w_in = ax.get_window_extent().width / ax.figure.dpi
        scale = max(float(np.nanmax(d.speed_ms)), 1.0) / (w_in / 8)
    else:
        scale = float(sty.arrow_ms_per_inch)
    qkw = dict(color=sty.arrow_color, angles="uv", scale_units="inches", scale=scale, **tkw)
    q = None
    if sty.arrow_every and len(past):
        a = d.loc[past].iloc[::sty.arrow_every].dropna(subset=["u_ms"])
        if len(a):
            pkw = dict(qkw, scale=scale / max(sty.past_arrow_frac, 1e-3))
            qp = ax.quiver(lon[a.index].values, a.lat.values, a.u_ms.values, a.v_ms.values,
                           width=sty.arrow_width * 0.8, zorder=z + 1, **pkw)
            if halo_q:
                qp.set_path_effects(halo_q)
            art["arrows"] = qp
    if valid_time is not None:
        c = st_track_at(d, valid_time)
        if c is not None:
            clon = c["lon"] % 360 if use360 else c["lon"]
            art["centre"] = ax.plot(clon, c["lat"], sty.centre_marker, mfc=sty.centre_face, mec=sty.track_color,
                                    mew=2.5, ms=sty.centre_size, zorder=z + 2, **tkw)
            qc = ax.quiver(np.array([clon]), np.array([c["lat"]]), np.array([c["u_ms"]]), np.array([c["v_ms"]]),
                           width=sty.arrow_width * 1.6, zorder=z + 3, **qkw)
            if halo_q:
                qc.set_path_effects(halo_q)
            art["centre_arrow"] = q = qc
            if sty.label:
                spd = c["speed_ms"] / 0.514444 if sty.units == "kt" else c["speed_ms"]
                txt = f"{ST_COMPASS[int((c['heading_deg'] + 11.25) // 22.5) % 16]} {spd:.0f} {sty.units}"
                h = np.radians(c["heading_deg"])
                off = (22 * np.cos(h), -22 * np.sin(h))
                art["label"] = ax.annotate(
                    txt, xy=(clon, c["lat"]), xycoords=text_xy, xytext=off, textcoords="offset points",
                    ha="right" if off[0] < 0 else "left", va="top" if off[1] < 0 else "bottom",
                    fontsize=sty.fontsize, fontweight="bold", zorder=z + 4,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
    if sty.key and q is None and "arrows" in art:
        q = art["arrows"]
    if sty.key and q is not None:
        ref = 10.0 if sty.units == "kt" else 5.0
        ref_ms = ref * 0.514444 if sty.units == "kt" else ref
        art["key"] = ax.quiverkey(q, *sty.key_xy, ref_ms, f"motion {ref:.0f} {sty.units}", labelpos="E",
                                  coordinates="axes", fontproperties={"size": sty.fontsize})
    return art


###############################################################################
# Main
###############################################################################
def main() -> None:
    args = normalize_cli_arguments(build_arg_parser().parse_args())

    if args.init_lat is not None and args.init_lon is None:
        raise SystemExit("ERROR: --init-lat requires --init-lon.")

    if args.init_lon is not None and args.init_lat is None:
        raise SystemExit("ERROR: --init-lon requires --init-lat.")

    if args.domain == "generic":
        df = track_generic_files(args)
    else:
        if wrf is None or Dataset is None:
            raise SystemExit(
                "ERROR: Native WRF mode needs wrf-python and netCDF4. "
                "Install these in your wrf-python environment, or use 'generic --files'."
            )
        print("Parent-domain fallback:", "ENABLED (--parent-fallback)" if args.parent_fallback
              else "OFF (independent-domain default; moving nests supported)")
        chain, by_domain = build_domain_frames(
            args.domain, args.wrf_dir, args.file_glob,
            use_parent_fallback=args.parent_fallback,
        )
        if not by_domain.get(args.domain):
            raise SystemExit(f"ERROR: No WRF frames found for requested domain {args.domain}.")
        df = track_vortex(chain, by_domain, args)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(f"{'stormtrack_hurricane_track' if args.domain == 'generic' else 'wrf_hurricane_track'}_{safe_tag(args.domain)}")
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    use_smoothed_output = not args.no_smoothed_output

    stem = make_output_stem(args.domain, df)

    csv_path = out_dir / f"{stem}.csv"
    atcf_path = out_dir / f"{stem}_atcf.dat"
    map_path = out_dir / f"{stem}_map.png"
    intensity_path = out_dir / f"{stem}_intensity.png"
    comparison_path = out_dir / f"{stem}_best_track_comparison.csv"

    df = add_map_quality_columns(df, max_speed_kt=args.map_jump_speed_kt,
                                 use_smoothed_output=use_smoothed_output)
    df.to_csv(csv_path, index=False)
    write_generic_atcf(df, atcf_path, use_smoothed_output=use_smoothed_output)

    best_df = None

    if args.best_track:
        comparison = compare_to_best_track(
            df,
            args.best_track,
            use_smoothed_output=use_smoothed_output,
        )

        if not comparison.empty:
            comparison.to_csv(comparison_path, index=False)
            best_df = read_best_track(args.best_track)
        else:
            print(
                "WARNING: Best-track file was read, but no matching times were found.",
                file=sys.stderr,
            )

    plot_track_map(
        df,
        map_path,
        best_df,
        dpi=args.dpi,
        use_smoothed_output=use_smoothed_output,
        label_interval_hours=args.label_interval_hours,
        arrow_every=args.arrow_every,
        show_cities=args.cities,
        map_jump_speed_kt=args.map_jump_speed_kt,
        map_padding_deg=args.map_padding_deg,
    )

    plot_intensity(
        df,
        intensity_path,
        best_df,
        dpi=args.dpi,
        use_smoothed_output=use_smoothed_output,
    )

    print("")
    count = int(df["map_suspect_jump"].sum())
    if count:
        print(f"MAP REVIEW: {count} suspect center-to-center jumps marked with dashed lines.")
        print("  Check map_suspect_jump and map_segment_motion_kt in the CSV.")
    print("Outputs written:")
    print(f"  CSV:       {csv_path}")
    print(f"  ATCF-like: {atcf_path}")
    print(f"  Map:       {map_path}")
    print(f"  Intensity: {intensity_path}")

    if args.best_track and comparison_path.exists():
        print(f"  Compare:   {comparison_path}")


if __name__ == "__main__":
    main()
