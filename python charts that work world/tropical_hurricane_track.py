#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WRF_Hurricane_Track.py

Operational-style WRF hurricane/vortex tracker for static or moving nests.

Required:
    domain

Typical command:
    python3 WRF_Hurricane_Track.py d02 --wrf-dir /path/to/WRF/run

Optional first guess:
    python3 WRF_Hurricane_Track.py d02 --wrf-dir /path/to/WRF/run --init-lat 24.5 --init-lon -94.0

Default operational smoothing:
    --max-jump-km 125
    --track-smooth-window 3
    --intensity-smooth-window 3
    --label-interval-hours 6

Outputs:
    * CSV with raw and smoothed centers
    * ATCF-style output
    * Track map
    * Intensity time series
    * Optional best-track comparison

Notes:
    This is WRF-only. No storm name, basin, storm number, or model ID is needed.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cartopy.crs as crs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import wrf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
from wrf import ALL_TIMES, to_np

###############################################################################
# Warning suppression
###############################################################################
warnings.filterwarnings("ignore")

###############################################################################
# Constants
###############################################################################
EARTH_RADIUS_KM = 6371.0
KNOTS_PER_MPS = 1.9438444924406046

GENERIC_ATCF_BASIN = "XX"
GENERIC_ATCF_NUMBER = "00"
GENERIC_ATCF_MODEL = "WRF"


###############################################################################
# Canonical helper function block from WRF plotting scripts
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
# Natural Earth features
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
# Cities
###############################################################################
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)


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


def parent_domain_chain(domain: str, use_parent_fallback: bool = True) -> list[str]:
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
    use_parent_fallback: bool,
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

    if dims and dims[0] == "Time":
        return np.asarray(var[time_index, ...])

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

    for candidate, weight in candidates:
        if (
            candidate.valid
            and np.isfinite(candidate.lat)
            and np.isfinite(candidate.lon)
            and weight > 0
        ):
            total += weight
            lat_sum += candidate.lat * weight
            lon_sum += candidate.lon * weight

    if total <= 0:
        return PointCandidate(np.nan, np.nan, np.nan, "weighted", valid=False)

    return PointCandidate(
        lat_sum / total,
        lon_sum / total,
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

    if args.center_method == "pressure":
        final_center = pressure_center
    elif args.center_method == "vorticity850":
        final_center = vort850_center if vort850_center.valid else pressure_center
    else:
        final_center = weighted_center(
            [
                (pressure_center, args.weight_pressure),
                (vort850_center, args.weight_vort850),
                (vort700_center, args.weight_vort700),
                (wind_center, args.weight_wind),
            ]
        )

        if not final_center.valid:
            final_center = pressure_center

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
        "avg_dx_km": fields.avg_dx_km,
        "avg_dy_km": fields.avg_dy_km,
        "center_method": args.center_method,
        "final_center_lat": final_center.lat,
        "final_center_lon": final_center.lon,
        "surface_pressure_center_lat": pressure_center.lat,
        "surface_pressure_center_lon": pressure_center.lon,
        "min_slp_hpa": float(pressure_center.value),
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
    }


def candidate_jump(candidate: dict[str, object]) -> float:
    value = candidate.get("jump_from_previous_km", np.nan)

    try:
        return float(value)
    except Exception:
        return np.nan


def track_vortex(
    chain: list[str],
    by_domain: dict[str, dict[datetime, FrameRef]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    requested_frames = sorted(
        by_domain.get(args.domain, {}).values(),
        key=lambda f: f.time_utc,
    )

    if not requested_frames:
        raise RuntimeError(f"No frames were found for requested domain {args.domain}.")

    if args.max_times and args.max_times > 0:
        requested_frames = requested_frames[: args.max_times]

    print(
        f"Tracking {len(requested_frames)} time(s) from "
        f"{requested_frames[0].time_utc} to {requested_frames[-1].time_utc}"
    )

    print("Domain search order:", " -> ".join(chain))

    guess_lat = args.init_lat
    guess_lon = args.init_lon
    rows: list[dict[str, object]] = []

    for frame_number, primary_frame in enumerate(requested_frames):
        time_utc = primary_frame.time_utc

        unrestricted_first_search = (
            frame_number == 0 and (guess_lat is None or guess_lon is None)
        )

        accepted: dict[str, object] | None = None
        rejected_by_jump: list[dict[str, object]] = []

        for domain in chain:
            frame = by_domain.get(domain, {}).get(time_utc)

            if frame is None:
                continue

            try:
                fields = load_fields(frame)

                candidate = analyze_fields(
                    fields,
                    guess_lat,
                    guess_lon,
                    args,
                    unrestricted_first_search=unrestricted_first_search,
                )

            except Exception as exc:
                print(f"WARNING: {domain} failed at {time_utc}: {exc}", file=sys.stderr)
                candidate = None

            if candidate is None:
                continue

            jump_km = candidate_jump(candidate)

            if (
                frame_number > 0
                and args.max_jump_km is not None
                and args.max_jump_km > 0
                and np.isfinite(jump_km)
                and jump_km > args.max_jump_km
            ):
                candidate["max_jump_exceeded"] = True
                rejected_by_jump.append(candidate)
                continue

            accepted = candidate
            break

        if accepted is None and rejected_by_jump:
            rejected_by_jump.sort(
                key=lambda row: candidate_jump(row)
                if np.isfinite(candidate_jump(row))
                else 1.0e9
            )
            accepted = rejected_by_jump[0]
            accepted["used_as_fallback_after_jump_limit"] = True

            print(
                "WARNING: all candidate centers exceeded --max-jump-km at "
                f"{time_utc.isoformat().replace('+00:00', 'Z')}; "
                "using closest candidate.",
                file=sys.stderr,
            )

        if accepted is None:
            print(f"WARNING: No valid center found at {time_utc}.", file=sys.stderr)
            continue

        rows.append(accepted)

        guess_lat = float(accepted["final_center_lat"])
        guess_lon = float(accepted["final_center_lon"])

        print(
            f"{time_utc.isoformat().replace('+00:00', 'Z')} "
            f"{accepted['domain_used']} "
            f"lat={guess_lat:.3f} lon={guess_lon:.3f} "
            f"mslp={float(accepted['min_slp_hpa']):.1f} hPa "
            f"vmax={float(accepted['vmax10_kt']):.0f} kt "
            f"jump={candidate_jump(accepted):.1f} km"
        )

    if not rows:
        raise RuntimeError("No vortex centers were found.")

    df = pd.DataFrame(rows)
    df = add_motion_columns(df, lat_col="final_center_lat", lon_col="final_center_lon", prefix="")
    df = add_smoothed_track_columns(df, window=args.track_smooth_window)
    df = add_smoothed_intensity_columns(df, window=args.intensity_smooth_window)

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

    out["smoothed_center_lat"] = centered_rolling_median(
        pd.to_numeric(out["final_center_lat"], errors="coerce"),
        window,
    )

    out["smoothed_center_lon"] = centered_rolling_median(
        pd.to_numeric(out["final_center_lon"], errors="coerce"),
        window,
    )

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
        time_utc = pd.to_datetime(row["time_utc"], utc=True)
        tau = int(round((time_utc - start_time).total_seconds() / 3600.0))
        ymdh = time_utc.strftime("%Y%m%d%H")

        lat_text = format_atcf_lat(float(row[lat_col]))
        lon_text = format_atcf_lon(float(row[lon_col]))

        vmax = int(round(float(row[vmax_col]))) if np.isfinite(row[vmax_col]) else 0
        mslp = int(round(float(row[slp_col]))) if np.isfinite(row[slp_col]) else 0

        line = (
            f"{GENERIC_ATCF_BASIN}, {GENERIC_ATCF_NUMBER}, {ymdh}, 03, "
            f"{GENERIC_ATCF_MODEL}, {tau:03d}, {lat_text}, {lon_text}, "
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
def map_extent_arrays(df: pd.DataFrame, lat_col: str, lon_col: str):
    all_lats = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
    all_lons = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()

    lat_min = np.nanmin(all_lats)
    lat_max = np.nanmax(all_lats)
    lon_min = np.nanmin(all_lons)
    lon_max = np.nanmax(all_lons)

    pad_lat = max(2.0, (lat_max - lat_min) * 0.20)
    pad_lon = max(2.0, (lon_max - lon_min) * 0.20)

    extent = [
        lon_min - pad_lon,
        lon_max + pad_lon,
        lat_min - pad_lat,
        lat_max + pad_lat,
    ]

    lons_np = np.array(
        [
            [extent[0], extent[1]],
            [extent[0], extent[1]],
        ]
    )
    lats_np = np.array(
        [
            [extent[2], extent[2]],
            [extent[3], extent[3]],
        ]
    )

    return extent, lats_np, lons_np


def plot_track_map(
    df: pd.DataFrame,
    path: Path,
    best_df: pd.DataFrame | None,
    dpi: int,
    use_smoothed_output: bool,
    label_interval_hours: int,
) -> None:
    if df.empty:
        return

    lat_col, lon_col = output_track_columns(df, use_smoothed_output)

    extent, lats_np, lons_np = map_extent_arrays(df, lat_col, lon_col)

    fig = plt.figure(figsize=(10, 13), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree())

    ax.set_extent(extent, crs=crs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor=cfeature.COLORS["land"])
    for feature in features:
        add_feature(ax, *feature)

    plot_cities(ax, lons_np, lats_np, 10.0, 10.0)
    add_latlon_gridlines(ax)

    ax.plot(
        df[lon_col],
        df[lat_col],
        marker="x",
        color="black",
        markersize=4,
        markerfacecolor="black",
        markeredgecolor="black",
        linewidth=1,
        transform=crs.PlateCarree(),
        label="WRF center",
    )

    if best_df is not None and not best_df.empty:
        ax.plot(
            best_df["best_lon"],
            best_df["best_lat"],
            marker="x",
            linestyle="--",
            color="black",
            markersize=4,
            markerfacecolor="black",
            markeredgecolor="black",
            linewidth=1,
            transform=crs.PlateCarree(),
            label="Best track",
        )

    times = pd.to_datetime(df["time_utc"], utc=True)

    if label_interval_hours and label_interval_hours > 0:
        start_time = times.iloc[0]

        for idx, row in df.iterrows():
            current_time = times.loc[idx]
            hours_since_start = (current_time - start_time).total_seconds() / 3600.0

            if abs(hours_since_start % label_interval_hours) < 0.01:
                label = current_time.strftime("%H:%M:%SZ %Y-%m-%d")

                ax.text(
                    float(row[lon_col]),
                    float(row[lat_col]),
                    label,
                    fontsize=8,
                    transform=crs.PlateCarree(),
                    bbox=dict(
                        boxstyle="round,pad=0.08",
                        facecolor="white",
                        alpha=0.4,
                    ),
                    clip_on=True,
                )

    title_suffix = "Smoothed Center Track" if use_smoothed_output else "Raw Center Track"
    ax.set_title(
        "Weather Research and Forecasting Model\n"
        f"{grid_spacing_title_text(df)}\n"
        "WRF Hurricane/Vortex Track\n"
        f"{title_suffix}",
        loc="left",
        fontsize=13,
    )
    ax.set_title(
        valid_title_text(df),
        loc="right",
        fontsize=13,
    )
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_intensity(
    df: pd.DataFrame,
    path: Path,
    best_df: pd.DataFrame | None,
    dpi: int,
    use_smoothed_output: bool,
) -> None:
    """
    Plot hurricane/vortex intensity as one PNG with two stacked panels:

        top:    minimum sea-level pressure (hPa)
        bottom: maximum 10 m wind (kt)

    This avoids the dual-axis clutter from plotting pressure and wind on the
    same axis while keeping the WRF-style left/right title format.
    """
    if df.empty:
        return

    slp_col, vmax_col = output_intensity_columns(df, use_smoothed_output)
    times = pd.to_datetime(df["time_utc"], utc=True)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        dpi=dpi,
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 1],
            "hspace": 0.12,
        },
    )

    # -------------------------------------------------------------------------
    # Top panel: minimum sea-level pressure
    # -------------------------------------------------------------------------
    ax1.plot(
        times,
        df[slp_col],
        marker="o",
        linewidth=1.5,
        color="black",
        markersize=4,
        markerfacecolor="black",
        markeredgecolor="black",
        
        label="WRF min SLP",
    )
    ax1.set_ylabel("Minimum sea-level pressure (hPa)")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")

    if best_df is not None and not best_df.empty:
        bt_times = pd.to_datetime(best_df["time_utc"], utc=True)

        if "best_mslp_hpa" in best_df.columns:
            ax1.plot(
                bt_times,
                best_df["best_mslp_hpa"],
                marker="x",
                linestyle="--",
                linewidth=1.5,
                label="Best-track MSLP",
            )
            ax1.legend(loc="best")

    # -------------------------------------------------------------------------
    # Bottom panel: maximum 10 m wind
    # -------------------------------------------------------------------------
    ax2.plot(
        times,
        df[vmax_col],
        marker="s",
        linewidth=1.5,
        markersize=4,
        label="WRF max 10 m wind",
    )
    ax2.set_ylabel("Maximum 10 m wind (kt)")
    ax2.set_xlabel("Valid time (UTC)")
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="best")

    if best_df is not None and not best_df.empty:
        bt_times = pd.to_datetime(best_df["time_utc"], utc=True)

        if "best_vmax_kt" in best_df.columns:
            ax2.plot(
                bt_times,
                best_df["best_vmax_kt"],
                marker="x",
                linestyle="--",
                linewidth=1.5,
                label="Best-track wind",
            )
            ax2.legend(loc="best")

    # -------------------------------------------------------------------------
    # WRF-style titles
    # -------------------------------------------------------------------------
    title_suffix = "Smoothed Intensity" if use_smoothed_output else "Raw Intensity"
    ax1.set_title(
        "Weather Research and Forecasting Model\n"
        f"{grid_spacing_title_text(df)}\n"
        "WRF Hurricane/Vortex Intensity Time Series\n"
        "Minimum Sea-Level Pressure (hPa)\n"
        "Maximum 10 m Wind (kt)\n"
        f"{title_suffix}",
        loc="left",
        fontsize=13,
        pad=10,
    )
    ax1.set_title(
        valid_title_text(df),
        loc="right",
        fontsize=13,
        pad=10,
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


###############################################################################
# Command line
###############################################################################
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track a WRF hurricane/vortex center from wrfout files. "
            "Only the WRF domain is required."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "domain",
        help=(
            "WRF domain to track first, such as d01, d02, or d03. "
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
        help="Output directory. Default is wrf_hurricane_track_<domain>.",
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
        choices=["weighted", "pressure", "vorticity850"],
        default="weighted",
    )

    parser.add_argument(
        "--hemisphere",
        choices=["auto", "north", "south"],
        default="auto",
    )

    parser.add_argument("--slp-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--vort-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--wind-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--edge-buffer-grid-points", type=int, default=5)
    parser.add_argument("--min-search-points", type=int, default=25)

    parser.add_argument(
        "--no-parent-fallback",
        action="store_true",
        help="Disable d03 -> d02 -> d01 style fallback.",
    )

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
        default=125.0,
        help=(
            "Maximum allowed one-step center jump before a candidate is rejected. "
            "If all candidates exceed this value, the closest candidate is still used "
            "and flagged in the CSV. Use 0 to disable."
        ),
    )

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
        default=6,
        help="Track map label interval in hours. Use 0 to disable labels.",
    )

    return parser


def make_output_stem(domain: str, df: pd.DataFrame) -> str:
    start = pd.to_datetime(df["time_utc"].iloc[0], utc=True).strftime("%Y%m%d%H")
    end = pd.to_datetime(df["time_utc"].iloc[-1], utc=True).strftime("%Y%m%d%H")

    return f"wrf_hurricane_track_{safe_tag(domain)}_{start}_to_{end}"


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

    if not is_wrf_domain(args.domain):
        raise SystemExit(
            "ERROR: Domain must look like d01, d02, or d03. "
            f"Received: {args.domain!r}"
        )

    return args


###############################################################################
# Main
###############################################################################
def main() -> None:
    args = normalize_cli_arguments(build_arg_parser().parse_args())

    if args.init_lat is not None and args.init_lon is None:
        raise SystemExit("ERROR: --init-lat requires --init-lon.")

    if args.init_lon is not None and args.init_lat is None:
        raise SystemExit("ERROR: --init-lon requires --init-lat.")

    chain, by_domain = build_domain_frames(
        args.domain,
        args.wrf_dir,
        args.file_glob,
        not args.no_parent_fallback,
    )

    if not by_domain.get(args.domain):
        raise SystemExit(f"ERROR: No WRF frames found for requested domain {args.domain}.")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(f"wrf_hurricane_track_{safe_tag(args.domain)}")
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    df = track_vortex(chain, by_domain, args)

    use_smoothed_output = not args.no_smoothed_output

    stem = make_output_stem(args.domain, df)

    csv_path = out_dir / f"{stem}.csv"
    atcf_path = out_dir / f"{stem}_atcf.dat"
    map_path = out_dir / f"{stem}_map.png"
    intensity_path = out_dir / f"{stem}_intensity.png"
    comparison_path = out_dir / f"{stem}_best_track_comparison.csv"

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
    )

    plot_intensity(
        df,
        intensity_path,
        best_df,
        dpi=args.dpi,
        use_smoothed_output=use_smoothed_output,
    )

    print("")
    print("Outputs written:")
    print(f"  CSV:       {csv_path}")
    print(f"  ATCF-like: {atcf_path}")
    print(f"  Map:       {map_path}")
    print(f"  Intensity: {intensity_path}")

    if args.best_track and comparison_path.exists():
        print(f"  Compare:   {comparison_path}")


if __name__ == "__main__":
    main()
