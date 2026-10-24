#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WRF tropical cyclone radial structure: standard radii with separate quadrant strip.

This standalone script generates WRF 10-m radial-mean wind and SLP charts,
WRF-derived RMW, and quadrant/mean R34, R50, and R64 wind radii.
The 3-row quadrant strip separates radius dots from plotted wind speed,
and a compact numerical key lists NE/SE/SW/NW radii in kilometers.
It does NOT calculate Tao–Chavas analytical wind curves, CKK pressure curves,
or potential-intensity / GPIv research indices.

Usage, without flags:
    python3 wrf_tropical_cyclone_radial_structure.py /path/to/WRF/run d02
    python3 wrf_tropical_cyclone_radial_structure.py /path/to/WRF/run d03
    python3 wrf_tropical_cyclone_radial_structure.py /path/to/WRF/run d03 frames
    python3 wrf_tropical_cyclone_radial_structure.py /path/to/WRF/run d03 gif

Scientific notes:
* WRF native instantaneous 10-m grid-cell winds are not NHC standardized
  one-minute sustained winds; model-derived radii are not NHC best-track.
* Outer R34/R50/R64 are detected in quadrant-bin maxima of WRF 10-m speed,
  not in the plotted azimuthal mean curves. If all four quadrants are resolved,
  the mean of their radii is reported as a *WRF-proxy mean radius*.
* No 0.85 empirical radius reduction, parametric tail fitting, temporal wind
  smoothing, or assumed storm-size cutoff is used. Unresolved radii stay blank.
* Parent-domain winds are preferred to extrapolation. Missing stationary-grid
  parent output can be interpolated only between closely timed bracketing
  records. The selected moving nest itself is never time-interpolated.
"""
from __future__ import annotations

import csv
import glob
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

KNOT_MS = 0.5144444
MPS_TO_MPH = 2.2369362920544
EARTH_RADIUS_KM = 6371.0088
QUADRANTS = ('NE', 'SE', 'SW', 'NW')
WIND_THRESHOLDS_KT = (64, 50, 34)

def _outer_34kt_crossing(r_km, cyclonic_tan_ms):
    """Outermost bracketed 34-kt crossing of radial-mean tangential wind."""
    threshold = 34.0 * KNOT_MS
    radius = np.asarray(r_km, dtype=float)
    winds = np.asarray(cyclonic_tan_ms, dtype=float)
    crossing = float('nan')
    for i in range(1, len(radius)):
        a, b = winds[i - 1], winds[i]
        if np.isfinite(a) and np.isfinite(b) and a >= threshold > b:
            crossing = float(radius[i - 1] +
                             (threshold - a) / (b - a) * (radius[i] - radius[i - 1]))
    return crossing


def valid_time_from_filename(path: str):
    """Fallback ONLY for a single record or the first time index."""
    match = re.search(r'wrfout_.*?_(\d{4}-\d{2}-\d{2})_(\d{2})[:_](\d{2})[:_](\d{2})',
                      Path(path).name)
    if not match:
        raise ValueError(f'No UTC timestamp can be determined for {path}')
    return datetime.strptime('_'.join(match.groups()), '%Y-%m-%d_%H_%M_%S')


def _utc_naive(value):
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError('Expected exactly one WRF Time value')
        value = value.reshape(-1)[0]
    if isinstance(value, np.datetime64):
        value = datetime.fromisoformat(np.datetime_as_string(value, unit='us'))
    if not isinstance(value, datetime):
        raise ValueError(f'Unsupported WRF time: {value!r}')
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def get_valid_time(nc, path, timeidx, wrf_module):
    try:
        return _utc_naive(wrf_module.extract_times(nc, timeidx=timeidx))
    except Exception:
        pass
    if 'Times' in nc.variables:
        raw = np.asarray(nc.variables['Times'][timeidx]).ravel()
        val = ''.join(
            x.decode('ascii') if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in raw
        ).strip().replace(' ', '_')
        return datetime.strptime(val, '%Y-%m-%d_%H:%M:%S')
    if timeidx == 0:
        return valid_time_from_filename(path)
    raise RuntimeError('Cannot infer valid time for additional records without WRF metadata')


def get_latlon(nc, timeidx):
    out = []
    for name in ('XLAT', 'XLONG'):
        if name not in nc.variables:
            raise KeyError(f'Missing required WRF coordinate: {name}')
        var = nc.variables[name]
        a = np.asarray(var[timeidx] if len(var.shape) == 3 else var[:], dtype=float)
        if a.ndim != 2:
            raise ValueError(f'{name} is not a 2-D horizontal field')
        out.append(a)
    if out[0].shape != out[1].shape:
        raise ValueError('XLAT and XLONG shapes do not match')
    return out


def distance_bearing_km(lat2, lon2, lat1, lon1):
    """Great-circle radius and clockwise-from-north bearing, including dateline."""
    a1 = np.deg2rad(lat1)
    a2 = np.deg2rad(lat2)
    dl = np.deg2rad((np.asarray(lon2) - lon1 + 180) % 360 - 180)
    dp = a2 - a1
    h = np.sin(dp / 2)**2 + np.cos(a1) * np.cos(a2) * np.sin(dl / 2)**2
    radii = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0., 1.)))
    east = np.sin(dl) * np.cos(a2)
    north = np.cos(a1) * np.sin(a2) - np.sin(a1) * np.cos(a2) * np.cos(dl)
    bearing = np.arctan2(east, north)
    return radii, bearing


def locate_center(slp, lats, lons, guess=None, search_km=100):
    """Select lowest WRF SLP, optionally inside a great-circle guess window."""
    search = np.isfinite(slp) & np.isfinite(lats) & np.isfinite(lons)
    if guess is not None:
        r, _ = distance_bearing_km(lats, lons, guess[0], guess[1])
        search &= r <= search_km
    if not np.any(search):
        raise ValueError('No valid WRF pressure cells inside the center search region')
    ix = np.unravel_index(np.argmin(np.where(search, slp, np.inf)), slp.shape)
    return float(lats[ix]), float(lons[ix]), float(slp[ix])


def _ring_mean_rmw(r_km, cyclonic_tan_ms):
    """Radius of peak cyclonic, annulus-mean 10-m wind, if identifiable."""
    valid = np.isfinite(cyclonic_tan_ms)
    if not valid.any() or float(np.nanmax(cyclonic_tan_ms)) <= 0:
        return float('nan')
    return float(r_km[np.nanargmax(cyclonic_tan_ms)])


def safe_ring_max_radius(lats, lons, center_lat, center_lon, limit_km=400.):
    """Keep entire sampling circle inside the rectangular WRF grid footprint.

    Conservative approximation for regular WRF grids, including moving nests.
    A 5% buffer reduces incomplete edge annuli, but geometry and coverage
    should still be checked for highly skewed domains.
    """
    edges = ((lats[0], lons[0]), (lats[-1], lons[-1]),
             (lats[:, 0], lons[:, 0]), (lats[:, -1], lons[:, -1]))
    finite_distances = []
    for y, x in edges:
        r, _ = distance_bearing_km(y, x, center_lat, center_lon)
        finite_distances.extend(np.asarray(r)[np.isfinite(r)].tolist())
    if not finite_distances:
        return float('nan')
    return min(float(limit_km), 0.95 * min(finite_distances))


def grid_bin_width_km(lats, lons):
    """Choose radial spacing appropriate for WRF resolution without flags."""
    if min(lats.shape) < 2:
        raise ValueError('WRF lat/lon grid must be at least 2 x 2')
    dx, _ = distance_bearing_km(lats[:, 1:], lons[:, 1:],
                                lats[:, :-1], lons[:, :-1])
    dy, _ = distance_bearing_km(lats[1:, :], lons[1:, :],
                                lats[:-1, :], lons[:-1, :])
    spacing = np.nanmedian(np.r_[np.asarray(dx).ravel(), np.asarray(dy).ravel()])
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Cannot determine horizontal WRF grid resolution')
    return float(np.clip(max(5.0, spacing * 1.5), 5., 40.))


def wind_axis_conversions(mph):
    """Return equivalent (m/s, knots) for any value expressed in mph."""
    mps = np.asarray(mph, dtype=float) / MPS_TO_MPH
    return mps, mps / KNOT_MS


def _wind_ticks_mph(maximum_mph):
    """At most about 7 shared, readable mph / m/s / kt tick positions."""
    if not np.isfinite(maximum_mph) or maximum_mph <= 0:
        maximum_mph = 60.0
    candidate_steps = (5., 10., 15., 20., 25., 30., 40., 50., 75., 100., 150., 200.)
    step = next((v for v in candidate_steps if v >= maximum_mph / 6.), 250.)
    ceiling = max(step * 2., np.ceil(maximum_mph * 1.06 / step) * step)
    return np.arange(0., ceiling + .25 * step, step), float(ceiling)


def _domain_bands(profile):
    """Broad approximate domain-reach bands, not per-bin vertical stripes.

    Every real radial bin retains its exact source_domain string in the CSV.
    The chart only marks each nested grid's outermost contributed radius,
    with a broad, transparent color block per domain.
    """
    radii = np.asarray(profile['r_km'], dtype=float)
    sources = list(profile.get('source_domain', []))
    if len(sources) != len(radii) or not len(radii):
        return []
    valid_wind = np.isfinite(profile['wind_speed_ms'])
    bounds = {}
    for dom in ('d03', 'd02', 'd01'):
        candidates = [i for i, src in enumerate(sources)
                      if valid_wind[i] and dom in str(src).split('+')]
        if candidates:
            bounds[dom] = float(radii[max(candidates)])
    if not bounds:
        return []
    regions = []
    cursor = 0.
    for dom in ('d03', 'd02', 'd01'):
        if dom not in bounds:
            continue
        bound = max(cursor, bounds[dom])
        if bound > cursor + 1e-6:
            regions.append((cursor, bound, dom))
        cursor = bound
    return regions


def _wrf_curve_value_at_radius(r_km, values, target_km):
    """Interpolate only between adjacent valid WRF samples, never through gaps.

    All annotations rely on native composite-WRF curves, not parametric values.
    An out-of-domain radius returns NaN and receives NO falsely placed dot.
    """
    radii = np.asarray(r_km, dtype=float)
    data = np.asarray(values, dtype=float)
    target = float(target_km)
    if not np.isfinite(target) or radii.ndim != 1 or len(radii) < 2:
        return float('nan')
    pos = int(np.searchsorted(radii, target))
    if pos < len(radii) and np.isclose(radii[pos], target, atol=1e-8):
        return float(data[pos]) if np.isfinite(data[pos]) else float('nan')
    if pos <= 0 or pos >= len(radii):
        return float('nan')
    a, b = pos - 1, pos
    if not (np.isfinite(data[a]) and np.isfinite(data[b])):
        return float('nan')
    spacing = np.diff(radii)
    representative = float(np.median(spacing[np.isfinite(spacing) & (spacing > 0)]))
    if radii[b] - radii[a] > 2.5 * representative:
        return float('nan')
    return float(data[a] + (data[b]-data[a]) * (target-radii[a])/(radii[b]-radii[a]))


def _peak_wrf_radial_mean(profile):
    """Return (peak wind m/s, bin-center radius km) from the plotted blue curve.

    This is a maximum of azimuthal MEAN values, not the maximum individual
    U10/V10 grid-cell wind. The two diagnostics must remain separate.
    Missing or non-finite bins are never treated as zero winds.
    """
    radii = np.asarray(profile['r_km'], dtype=float)
    wind = np.asarray(profile['wind_speed_ms'], dtype=float)
    if radii.ndim != 1 or wind.shape != radii.shape:
        raise ValueError('WRF radial radius and wind arrays do not align')
    valid = np.isfinite(radii) & np.isfinite(wind)
    if not np.any(valid):
        return float('nan'), float('nan')
    indices = np.flatnonzero(valid)
    index = indices[int(np.argmax(wind[valid]))]
    return float(wind[index]), float(radii[index])


def _wind_triplet(value_ms):
    """Return the consistently rounded mph, m/s, and kt wind label."""
    speed = float(value_ms)
    if not np.isfinite(speed):
        return 'unavailable'
    return f'{speed * MPS_TO_MPH:.1f} mph ({speed:.1f} m/s; {speed / KNOT_MS:.1f} kt)'


def make_gif(png_paths, path, duration=600):
    from PIL import Image
    if not png_paths:
        return
    first = Image.open(png_paths[0]).convert('RGB')
    # Palette-mode conversion controls GIF size for high-res frames.
    with first.quantize(colors=128) as initial:
        images = []
        try:
            for p in png_paths[1:]:
                with Image.open(p) as im:
                    images.append(im.convert('RGB').quantize(colors=128))
            initial.save(path, save_all=True, append_images=images,
                         duration=duration, loop=0, optimize=False)
        finally:
            for im in images:
                im.close()
    first.close()


def get_wrf_fields(nc, idx, wrf_module):
    lats, lons = get_latlon(nc, idx)
    slp = np.asarray(wrf_module.getvar(nc, 'slp', timeidx=idx, meta=False), dtype=float)
    uv = np.asarray(wrf_module.getvar(nc, 'uvmet10', timeidx=idx, meta=False), dtype=float)
    if uv.shape[0] != 2 or slp.shape != lats.shape or uv.shape[1:] != lats.shape:
        raise ValueError('WRF SLP/uvmet10 output dimensions are inconsistent')
    return lats, lons, slp, uv[0], uv[1]


def discover_frames(paths, Dataset):
    result = []
    for path in paths:
        with Dataset(path) as nc:
            if 'Time' in nc.dimensions:
                count = len(nc.dimensions['Time'])
            elif 'Times' in nc.variables:
                count = int(nc.variables['Times'].shape[0])
            else:
                count = 1
        result += [(path, idx) for idx in range(count)]
    return result


def _plane_km(lats, lons, center_lat, center_lon):
    """Local equirectangular x/y only for nest footprint containment."""
    x = ((np.asarray(lons) - center_lon + 180.) % 360. - 180.) * (111.195 * np.cos(np.deg2rad(center_lat)))
    y = (np.asarray(lats) - center_lat) * 111.195
    return np.asarray(x), np.asarray(y)


def _grid_footprint(lats, lons, center_lat, center_lon):
    """A WRF grid's actual four exterior edges, not its bounding rectangle."""
    from matplotlib.path import Path as MplPath
    def edge(ar):
        return np.r_[ar[0, :], ar[1:-1, -1], ar[-1, ::-1], ar[-2:0:-1, 0]]
    latb, lonb = edge(lats), edge(lons)
    xx, yy = _plane_km(latb, lonb, center_lat, center_lon)
    return MplPath(np.vstack((np.c_[xx, yy], [xx[0], yy[0]])), closed=True)


def _valid_in_grid(lats, lons, target_lat, target_lon):
    fp = _grid_footprint(lats, lons, target_lat, target_lon)
    return bool(fp.contains_point((0., 0.)))


def _read_all_domain_inventory(wrf_dir, requested_domain, Dataset, wrf_module):
    """Read timestamps from requested domain and only its ancestor domains.

    Never substitute a parent at the wrong valid time. If a parent has multiple
    frames in one file, its exact record index is preserved.
    """
    domain_num = int(requested_domain[1:])
    by_domain = {}
    for n in range(1, domain_num + 1):
        dom = f'd{n:02d}'
        paths = sorted(glob.glob(str(Path(wrf_dir).expanduser() / f'wrfout_{dom}*')))
        if n == domain_num and not paths:
            raise FileNotFoundError(f'No wrfout_{dom}* files found under {wrf_dir}')
        records = {}
        for path, idx in discover_frames(paths, Dataset):
            try:
                with Dataset(path) as nc:
                    valid = get_valid_time(nc, path, idx, wrf_module)
                if valid not in records:
                    records[valid] = (path, idx)
            except Exception as exc:
                print(f'[WARN] Cannot index {dom} {Path(path).name}[{idx}]: {exc}', file=sys.stderr)
        by_domain[dom] = records
        print(f'[INDEX] {dom}: {len(records)} unique UTC frames')
    return by_domain


def _interpolate_parent_fields(valid, matches, domain, Dataset, wrf_module, max_span_hours=3.):
    """Linear interpolation in time for stationary-grid parent output only.

    The parent must bracket the requested nest time with two real WRF records.
    Never extrapolate time, never interpolate moving-grid array indices across
    different geographical locations. When the grid is moving, return None.
    """
    from bisect import bisect_left
    times=sorted(matches)
    j=bisect_left(times,valid)
    if j==0 or j==len(times):return None
    left,right=times[j-1],times[j]
    duration=(right-left).total_seconds()
    if duration<=0 or duration>max_span_hours*3600.:return None
    frac=(valid-left).total_seconds()/duration
    if not 0.<frac<1.:return None
    frames=[]
    for stamp in (left,right):
        path,idx=matches[stamp]
        with Dataset(path) as nc:
            la,lo,p,u,v=get_wrf_fields(nc,idx,wrf_module)
        frames.append((la,lo,p,u,v))
    a,b=frames
    if a[0].shape!=b[0].shape or not np.allclose(a[0],b[0],atol=1e-4,rtol=0.):return None
    if not np.allclose(((a[1]-b[1]+180.)%360.)-180.,0.,atol=1e-4,rtol=0.):return None
    la,lo=a[:2]
    p=(1-frac)*a[2]+frac*b[2]
    u=(1-frac)*a[3]+frac*b[3]
    v=(1-frac)*a[4]+frac*b[4]
    return dict(domain=domain,lat=la,lon=lo,slp=p,u=u,v=v,interpolated=True,
                interval_utc=f'{left:%H:%M}Z-{right:%H:%M}Z',time_weight=frac,
                source_path='temporal_interpolation',timeidx=None)


def _read_matched_domains(valid, requested_domain, inventories, Dataset, wrf_module):
    """Per-timestep moving-nest fields; absent parents skipped, not fabricated."""
    grids = {}
    for n in range(1, int(requested_domain[1:]) + 1):
        dom = f'd{n:02d}'
        match = inventories[dom].get(valid)
        if match is None:
            if dom == requested_domain:
                continue
            try:
                interpolated=_interpolate_parent_fields(valid,inventories[dom],dom,Dataset,wrf_module)
                if interpolated is not None:
                    grids[dom]=interpolated
                    print(f'[INTERPOLATED] {dom} at {valid:%H:%M}Z from {interpolated["interval_utc"]}')
                else:
                    print(f'[WARN] {dom} lacks a matching or safely interpolatable time at {valid:%H:%M}Z')
            except Exception as exc:
                print(f'[WARN] Could not interpolate {dom}: {exc}',file=sys.stderr)
            continue
        path, idx = match
        try:
            with Dataset(path) as nc:
                lat, lon, p, u, v = get_wrf_fields(nc, idx, wrf_module)
            grids[dom] = dict(domain=dom, lat=lat, lon=lon, slp=p,
                              u=u, v=v, source_path=path, timeidx=idx,interpolated=False)
        except Exception as exc:
            print(f'[WARN] {dom} data failed at {valid:%Y-%m-%d %H:%M}Z: {exc}', file=sys.stderr)
    if requested_domain not in grids:
        raise RuntimeError(f'{requested_domain} unavailable at requested valid time')
    return grids


def _choose_center(grids, requested, guess=None):
    """Track cyclone via domain nesting, using a parent for a displaced nest.

    Start on largest available domain, then refine within 150 km through
    progressively finer domains. On later steps the previous center constrains
    parent selection, avoiding sudden switches to distant weather systems.
    """
    order = sorted(grids)
    if guess is None:
        origin = grids[order[0]]
        center = locate_center(origin['slp'], origin['lat'], origin['lon'])[:2]
        base = order[0]
    else:
        center = guess
        base = order[0]
    best = None
    for dom in order:
        g = grids[dom]
        if not _valid_in_grid(g['lat'], g['lon'], *center):
            continue
        try:
            lat, lon, pmin = locate_center(g['slp'], g['lat'], g['lon'], center, 150.)
            # Reject spurious boundary points that may be a truncated vortex.
            best = (lat, lon, pmin, dom)
            center = lat, lon
            base = dom
        except ValueError:
            pass
    if best is None:
        # The tracked cyclone may have left a moving nest. Use parent data.
        parent = grids[order[0]]
        lat, lon, p = locate_center(parent['slp'], parent['lat'], parent['lon'], guess,
                                     200.) if guess else locate_center(parent['slp'], parent['lat'], parent['lon'])
        return lat, lon, p, order[0]
    return best


def _stack_priority_cells(grids, center_lat, center_lon):
    """Keep high-resolution grid cells, masking overlapping parent centers.

    Each coarse cell represents area roughly dx*dy; area weighting avoids
    thousands of fine cells overwhelming the outer-ring parent contribution.
    The mask is applied at parent cell centers; partial boundary cells are an
    approximation, and bin coverage is reported explicitly.
    """
    from matplotlib.path import Path as MplPath
    ordered = sorted(grids, reverse=True)  # d03, d02, d01
    finer_footprints = []
    out = []
    spacing = {}
    for dom in ordered:
        g = grids[dom]
        lat, lon = g['lat'], g['lon']
        xx, yy = _plane_km(lat, lon, center_lat, center_lon)
        valid = (np.isfinite(lat) & np.isfinite(lon) & np.isfinite(g['slp']) &
                 np.isfinite(g['u']) & np.isfinite(g['v']))
        for polygon in finer_footprints:
            valid &= ~polygon.contains_points(np.c_[xx.ravel(), yy.ravel()], radius=0.).reshape(lat.shape)
        dy = np.asarray(distance_bearing_km(lat[1:,:],lon[1:,:], lat[:-1,:],lon[:-1,:])[0],float)
        dx = np.asarray(distance_bearing_km(lat[:,1:],lon[:,1:], lat[:,:-1],lon[:,:-1])[0],float)
        typical_x, typical_y = float(np.nanmedian(dx)), float(np.nanmedian(dy))
        spacing[dom] = float(np.sqrt(typical_x * typical_y))
        # Each cell has a geometric-area proxy; exact projected cell areas are
        # not required for regular structured WRF grids in a small TC region.
        area = np.full(lat.shape, typical_x * typical_y)
        if np.any(valid):
            r, theta = distance_bearing_km(lat[valid],lon[valid],center_lat,center_lon)
            u,v = g['u'][valid],g['v'][valid]
            sign = 1 if center_lat >= 0 else -1
            tan = sign*(-u*np.cos(theta) + v*np.sin(theta))
            out.append(dict(domain=dom, r=np.asarray(r), bearing=np.asarray(theta),
                            speed=np.hypot(u,v), tan=tan,
                            pressure=g['slp'][valid], area=area[valid]))
        # Mask coarser cells wherever the finer grid footprint geometrically lies.
        finer_footprints.append(_grid_footprint(lat,lon,center_lat,center_lon))
    if not out:
        raise ValueError('No valid stitched WRF cells')
    return out, spacing


def _full_circle_limit(grids, center_lat, center_lon):
    """Full-circle reach follows actual parent-grid geometry, not storm presets.

    The previous fixed 1600-km cap could discard valid parent-grid structure for
    broad cyclones. The radius is now bounded by the available domain footprint
    and its conservative geometric buffer, independently at each valid time.
    """
    largest = grids[sorted(grids)[0]]
    return safe_ring_max_radius(largest['lat'], largest['lon'],
                                center_lat, center_lon, limit_km=float('inf'))


def _adaptive_edges(max_radius, inner_spacing, outer_spacing):
    """Fine bins for inner core, larger bins to sample 20–25 km parent cells."""
    small = max(5., min(30., 1.5 * inner_spacing))
    medium = max(10., min(40., 1.5 * inner_spacing))
    large = max(20., min(60., 1.65 * outer_spacing))
    edges = [0.]
    while edges[-1] + small < min(150., max_radius): edges.append(edges[-1] + small)
    while edges[-1] + medium < min(300., max_radius): edges.append(edges[-1] + medium)
    while edges[-1] + large < max_radius: edges.append(edges[-1] + large)
    if max_radius - edges[-1] >= min(0.5 * large, 5.): edges.append(float(max_radius))
    return np.asarray(edges, dtype=float)


def _stitched_rings(cells, edges, source_order):
    """Equal-angle ring means from nested WRF cells with provenance and QC.

    Grid density can vary by 25:1 across a nested boundary. First compute a
    cell-area-weighted mean in each of 8 equal 45-degree sectors, then average
    those sector means with equal azimuthal weight. This avoids treating a
    densely sampled quadrant as representative of the entire storm.

    Crucially, no temporal smoothing, fixed storm radius, or assumed storm
    symmetry is imposed. A ring with fewer than six populated sectors remains
    missing; 6-7 sector rings are preserved but explicitly flagged PARTIAL.
    """
    mid = .5 * (edges[1:] + edges[:-1])
    n = len(mid)
    fields = ('wind_speed_ms', 'cyclonic_tangential_ms', 'slp_hpa')
    result = {'r_km': mid, 'gridcell_count': np.zeros(n, dtype=int)}
    out = {name: np.full(n, np.nan, dtype=float) for name in fields}
    sector_coverage = np.zeros(n, dtype=int)
    origin = []

    r = np.concatenate([c['r'] for c in cells])
    bearing = np.concatenate([c['bearing'] for c in cells])
    area = np.concatenate([c['area'] for c in cells])
    speed = np.concatenate([c['speed'] for c in cells])
    tan = np.concatenate([c['tan'] for c in cells])
    slp = np.concatenate([c['pressure'] for c in cells])
    which = np.concatenate([np.full(len(c['r']), c['domain']) for c in cells])

    # The red X and header MUST refer to the same physical grid cell.
    # Limit the search to the domain-stitched, full-circle radial analysis.
    valid_for_peak = (np.isfinite(speed) & np.isfinite(r) &
                      (r >= 0) & (r < edges[-1]))
    if np.any(valid_for_peak):
        cell_indices = np.flatnonzero(valid_for_peak)
        peak_i = int(cell_indices[np.argmax(speed[cell_indices])])
        max_speed = float(speed[peak_i])
        peak_radius = float(r[peak_i])
        peak_domain = str(which[peak_i])
        peak_bearing = float(bearing[peak_i])
    else:
        max_speed = float('nan')
        peak_radius = float('nan')
        peak_domain = ''
        peak_bearing = float('nan')
    bins = np.searchsorted(edges, r, side='right') - 1
    good = (np.isfinite(r) & (r >= 0) & (r < edges[-1]) &
            (bins >= 0) & (bins < n) & np.isfinite(area) & (area > 0) &
            np.isfinite(bearing))
    bins, bearing, area = bins[good], bearing[good], area[good]
    speed, tan, slp, which = speed[good], tan[good], slp[good], which[good]
    result['gridcell_count'] = np.bincount(bins, minlength=n)

    sectors = (np.floor(((bearing + 2 * np.pi) % (2 * np.pi)) /
                        (np.pi / 4)).astype(int)) % 8
    keys = bins * 8 + sectors
    for name, values in (('wind_speed_ms', speed),
                         ('cyclonic_tangential_ms', tan),
                         ('slp_hpa', slp)):
        finite = np.isfinite(values)
        weight = np.bincount(keys[finite], weights=area[finite], minlength=n * 8).reshape(n, 8)
        sums = np.bincount(keys[finite], weights=area[finite] * values[finite],
                           minlength=n * 8).reshape(n, 8)
        sector_values = np.divide(sums, weight, out=np.full((n, 8), np.nan),
                                  where=weight > 0)
        sufficient = np.count_nonzero(np.isfinite(sector_values), axis=1) >= 6
        with np.errstate(invalid='ignore'):
            # Numpy nanmean emits a warning for a fully uncovered bin; avoid it.
            numbers = np.nansum(sector_values, axis=1)
            counts = np.count_nonzero(np.isfinite(sector_values), axis=1)
            out[name][sufficient] = numbers[sufficient] / counts[sufficient]

    # Sector coverage for all three diagnostic fields, not merely geometry.
    for i in range(n):
        members = [d for d in source_order if np.any(which[bins == i] == d)]
        origin.append('+'.join(members) if members else 'unavailable')
        sector_ids = np.unique(sectors[(bins == i) & np.isfinite(speed)])
        sector_coverage[i] = len(sector_ids)

    # Count the source-domain changes in radius, but preserve the raw values.
    quality = []
    seam = np.zeros(n, dtype=int)
    for i, coverage in enumerate(sector_coverage):
        if i and origin[i] != origin[i-1] and origin[i] != 'unavailable':
            seam[i] = 1
        if coverage < 6 or result['gridcell_count'][i] < 3:
            for name in fields:
                out[name][i] = np.nan
            quality.append('INSUFFICIENT_COVERAGE')
        elif coverage < 8:
            quality.append('PARTIAL_AZIMUTH')
        elif '+' in origin[i]:
            quality.append('MIXED_DOMAIN')
        else:
            quality.append('FULL_AZIMUTH')

    result.update(out)
    result.update(source_domain=origin, angular_sectors_8=sector_coverage,
                  radial_coverage_fraction=sector_coverage / 8.,
                  radial_quality=quality, source_transition=seam,
                  max_10m_speed_ms=max_speed,
                  max_10m_radius_km=peak_radius,
                  max_10m_source_domain=peak_domain,
                  max_10m_bearing_deg=float(np.degrees(peak_bearing)))
    result['ring_mean_rmw_km'] = _ring_mean_rmw(mid, out['cyclonic_tangential_ms'])
    result['ring_mean_r34_km'] = _outer_34kt_crossing(mid, out['cyclonic_tangential_ms'])
    return result


def _source_domain_summary(record):
    profile=record['profile']; coverage=profile.get('source_domain',[])
    return ','.join(sorted(set(d for text in coverage for d in text.split('+') if d.startswith('d'))))


def quadrant_wind_radii(cells, edges, threshold_kt):
    """Detect an outer threshold crossing in each geographic quadrant.

    For each radial ring/quadrant, take the maximum *total* 10-m wind speed.
    A radius is WRF_RESOLVED only when the last above-threshold bin is followed
    by three consecutive sampled bins below the threshold. This is a research
    proxy for quadrant wind radii, not the NHC sustained-wind definition.

    Neither absent crossings nor missing parent-domain coverage is fabricated.
    """
    threshold = float(threshold_kt) * KNOT_MS
    radius = np.concatenate([c['r'] for c in cells])
    bearing = np.concatenate([c['bearing'] for c in cells])
    wind = np.concatenate([c['speed'] for c in cells])
    domains = np.concatenate([np.full(len(c['r']), c['domain']) for c in cells])
    qid = np.floor(((bearing + 2 * np.pi) % (2 * np.pi)) / (np.pi / 2)).astype(int) % 4
    mids = 0.5 * (edges[:-1] + edges[1:])
    n = len(mids)
    results, statuses, details = {}, {}, {}
    for quadrant_idx, quadrant in enumerate(QUADRANTS):
        maxima = np.full(n, np.nan)
        src = [''] * n
        selected = ((qid == quadrant_idx) & np.isfinite(radius) &
                    np.isfinite(wind) & (radius >= 0) & (radius < edges[-1]))
        bins = np.searchsorted(edges, radius[selected], side='right') - 1
        speeds = wind[selected]
        sources = domains[selected]
        for i in np.unique(bins[(bins >= 0) & (bins < n)]):
            mask = bins == i
            if np.count_nonzero(mask) >= 2:
                maxima[i] = float(np.max(speeds[mask]))
                src[i] = ','.join(sorted(set(sources[mask])))
        above = np.flatnonzero(np.isfinite(maxima) & (maxima >= threshold))
        if not len(above):
            statuses[quadrant] = 'NOT_DETECTED'
            details[quadrant] = f'No sampled {threshold_kt}-kt wind; cannot determine a crossing'
            continue
        last = int(above[-1])
        below = last + 1
        # Don't silently bridge missing outer rings or stop at a domain edge.
        if (below + 2 < n and np.all(np.isfinite(maxima[below:below + 3]))
                and np.all(maxima[below:below + 3] < threshold)):
            a, b = float(maxima[last]), float(maxima[below])
            if not (a > b):
                statuses[quadrant] = 'UNRESOLVED'
                details[quadrant] = 'Invalid threshold bracket'
                continue
            frac = (threshold - a) / (b - a)
            result = float(mids[last] + frac * (mids[below] - mids[last]))
            if np.isfinite(result) and result > 0:
                results[quadrant] = result
                statuses[quadrant] = 'WRF_RESOLVED'
                bands = 1 + int(np.count_nonzero(np.diff(above) > 1))
                details[quadrant] = (
                    f'{src[last]} -> {src[below]}; '
                    f'outermost of {bands} sampled above-threshold band(s)'
                )
                continue
        statuses[quadrant] = 'UNRESOLVED'
        details[quadrant] = 'Wind radius not bracketed by three sampled outer bins'
    return results, statuses, details


def calculate_standard_radii(cells, edges):
    """R34/R50/R64 quadrant radii; mean only with four resolved quadrants."""
    result, statuses, details, means = {}, {}, {}, {}
    for threshold in WIND_THRESHOLDS_KT:
        label = f'R{threshold}'
        result[label], statuses[label], details[label] = quadrant_wind_radii(
            cells, edges, threshold
        )
        means[label] = (float(np.mean(list(result[label].values())))
                        if len(result[label]) == 4 else float('nan'))
    qc = []
    for quadrant in QUADRANTS:
        radii = [result[f'R{kt}'].get(quadrant) for kt in WIND_THRESHOLDS_KT]
        if all(r is not None for r in radii):
            r64, r50, r34 = radii
            if not (r64 <= r50 <= r34):
                qc.append(f'{quadrant}: NONMONOTONIC_RADIUS_ORDER')
    return result, statuses, details, means, qc


def derive_stitched_frame(grids, valid, requested_domain, guess=None):
    """Use requested nest for core and valid parent-domain data for outer rings."""
    lat, lon, pmin, center_dom = _choose_center(grids, requested_domain, guess)
    cells, spacing = _stack_priority_cells(grids, lat, lon)
    limit = _full_circle_limit(grids, lat, lon)
    if not np.isfinite(limit) or limit < max(40., 4 * min(spacing.values())):
        raise ValueError('No sufficiently large, complete radial reach')
    edges = _adaptive_edges(limit, min(spacing.values()), max(spacing.values()))
    if len(edges) < 6:
        raise ValueError('Insufficient radial bins for a cyclone profile')
    profile = _stitched_rings(cells, edges, sorted(grids, reverse=True))
    radii, status, details, means, qc = calculate_standard_radii(cells, edges)
    return dict(
        valid=valid, domain=requested_domain,
        center_lat=lat, center_lon=lon, center_domain=center_dom,
        min_slp=pmin, profile=profile,
        quadrant_radii_km=radii, wind_radius_status=status,
        wind_radius_details=details, mean_radii_km=means, radii_qc=qc,
        source_domains=tuple(sorted(grids)),
        temporally_interpolated_domains=tuple(
            d for d in sorted(grids) if grids[d].get('interpolated', False)),
        available_radius_km=limit
    )


def _radius_text(radius):
    return f'{radius:.0f} km' if np.isfinite(radius) else 'unavailable'


def _peak_position(profile):
    """Actual radial distance and wind of the maximum model grid cell."""
    radius = float(profile.get('max_10m_radius_km', float('nan')))
    speed = float(profile.get('max_10m_speed_ms', float('nan')))
    if not (np.isfinite(radius) and np.isfinite(speed)):
        return None
    return radius, speed * MPS_TO_MPH


def plot_standard_radii(record, out_path):
    """Draw compact, publication-style WRF radial profiles and wind radii.

    Wind and pressure curves are exclusively WRF-derived. The quadrant dots
    occupy a categorical *radius* strip, never the wind speed axes. The radius
    summary at the right lists actual NE/SE/SW/NW crossing radii in kilometers.
    Missing wind radii are displayed as dashes, never guessed or filled.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

    profile = record['profile']
    radius = np.asarray(profile['r_km'], dtype=float)
    speed_ms = np.asarray(profile['wind_speed_ms'], dtype=float)
    tan_ms = np.asarray(profile['cyclonic_tangential_ms'], dtype=float)
    pressure = np.asarray(profile['slp_hpa'], dtype=float)
    if (radius.ndim != 1 or radius.size == 0 or
            any(arr.shape != radius.shape for arr in (speed_ms, tan_ms, pressure)) or
            not np.any(np.isfinite(speed_ms)) or not np.any(np.isfinite(pressure))):
        raise ValueError('No valid WRF radial wind and pressure samples')

    wind = speed_ms * MPS_TO_MPH
    tangential = tan_ms * MPS_TO_MPH
    blue, green = '#176aa8', '#137d5b'
    colors = {'RMW': '#9a2ca2', 'R64': '#cb7414',
              'R50': '#278644', 'R34': '#2966b8'}
    levels = [('R64', 64, 2), ('R50', 50, 1), ('R34', 34, 0)]
    shapes = {'NE': 'o', 'SE': 's', 'SW': 'D', 'NW': '^'}
    offsets = {'NE': -.135, 'SE': -.045, 'SW': .045, 'NW': .135}
    means = record['mean_radii_km']

    # A fixed layout across frames prevents the animation from jumping.
    # The middle panel is deliberately large enough for labeled radii rows.
    fig = plt.figure(figsize=(15.2, 9.7), facecolor='white')
    spec = fig.add_gridspec(
        3, 1, height_ratios=(1.75, .60, 1.05),
        left=.087, right=.805, top=.826, bottom=.157, hspace=.30
    )
    aw = fig.add_subplot(spec[0])
    aq = fig.add_subplot(spec[1], sharex=aw)
    ap = fig.add_subplot(spec[2], sharex=aw)
    for ax in (aw, aq, ap):
        ax.set_facecolor('white')
        ax.set_axisbelow(True)
        ax.minorticks_off()
        ax.tick_params(axis='both', labelsize=10, colors='#253544', pad=5)
        for spine in ax.spines.values():
            spine.set_color('#98a4b1')
            spine.set_linewidth(.8)
    for ax in (aw, ap):
        ax.grid(True, color='#aebac7', linewidth=.7, alpha=.18)
    aq.grid(axis='x', color='#aebac7', linewidth=.7, alpha=.16)

    # Large, extremely pale regions convey the nested-domain reach without
    # per-bin striping or plotting separate curves for overlapping domains.
    domain_tints = {'d03': '#dcefe1', 'd02': '#e2eff9', 'd01': '#eee7fa'}
    for left, right, domain in _domain_bands(profile):
        aw.axvspan(left, right, facecolor=domain_tints[domain],
                   alpha=.16, lw=0, zorder=0)

    aw.plot(radius, wind, color=blue, lw=2.8, zorder=4,
            label='WRF radial-mean 10-m speed')
    aw.plot(radius, tangential, color=green, lw=2.3, zorder=4,
            ls=(0, (6, 3)), label='WRF cyclonic tangential mean')
    ap.plot(radius, pressure, color=blue, lw=2.65,
            label='WRF radial-mean SLP', zorder=4)

    # The right scale is the *same physical axis* converted from mph.
    # Include the grid-cell maximum when selecting the scale so its X marker
    # never gets clipped by a curve-mean-derived upper limit.
    grid_peak = _peak_position(profile)
    wind_top_candidate = float(np.nanmax(wind))
    if np.any(np.isfinite(tangential)):
        wind_top_candidate = max(wind_top_candidate, float(np.nanmax(tangential)))
    if grid_peak is not None:
        wind_top_candidate = max(wind_top_candidate, grid_peak[1])
    tick_mph, top_mph = _wind_ticks_mph(wind_top_candidate)
    aw.set_ylim(0, top_mph)
    aw.yaxis.set_major_locator(FixedLocator(tick_mph))
    aw.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}'))
    aw.set_ylabel('10-m wind speed (mph)', fontsize=11.0, labelpad=9)
    aw.set_title('Radial 10-m wind profile', fontsize=12.8, weight='semibold', pad=8)
    equivalent = aw.twinx()
    equivalent.set_ylim(aw.get_ylim())
    equivalent.yaxis.set_major_locator(FixedLocator(tick_mph))
    equivalent.set_yticklabels(
        [f'{float(wind_axis_conversions(t)[0]):.1f} m/s  |  '
         f'{float(wind_axis_conversions(t)[1]):.1f} kt'
         for t in tick_mph], fontsize=8.8
    )
    equivalent.set_ylabel('Equivalent wind speed: m/s  |  knots',
                          fontsize=9.4, color='#455466', labelpad=13)
    equivalent.tick_params(axis='y', length=3, pad=7, colors='#455466')
    equivalent.grid(False)
    for side in ('top', 'bottom', 'left'):
        equivalent.spines[side].set_visible(False)
    equivalent.spines['right'].set_color('#98a4b1')

    # The requested cyclone determines the x limits; no fixed radius imposed.
    x_max = max(30., 1.02 * float(np.nanmax(radius)))
    aw.set_xlim(0, x_max)
    ap.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4))
    aw.tick_params(axis='x', labelbottom=False)
    aq.tick_params(axis='x', labelbottom=False)

    # Standard metrics only. Each reference is drawn when directly resolved.
    # Stagger labels only when radii are close compared with plot width.
    reference_radii = [
        ('RMW', float(profile.get('ring_mean_rmw_km', np.nan))),
        ('R64', float(means.get('R64', np.nan))),
        ('R50', float(means.get('R50', np.nan))),
        ('R34', float(means.get('R34', np.nan)))
    ]
    last_radius, last_level = float('-inf'), 0
    for tag, r_ref in sorted(reference_radii, key=lambda item: item[1]):
        if not (np.isfinite(r_ref) and 0 < r_ref <= x_max):
            continue
        color = colors[tag]
        style = '-.' if tag == 'RMW' else '--'
        aw.axvline(r_ref, color=color, ls=style, lw=1.05, alpha=.78, zorder=2)
        ap.axvline(r_ref, color=color, ls=style, lw=.85, alpha=.28, zorder=2)
        if r_ref - last_radius < .053 * x_max:
            level = 1 - last_level
        else:
            level = 0
        last_radius, last_level = r_ref, level
        aw.text(r_ref, .986 - .082 * level, tag,
                transform=aw.get_xaxis_transform(), color=color,
                ha='center', va='top', fontsize=9.3, weight='bold', zorder=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.91, pad=1.2))
        if tag == 'RMW':
            y = _wrf_curve_value_at_radius(radius, tan_ms, r_ref)
            if np.isfinite(y):
                aw.scatter([r_ref], [y * MPS_TO_MPH], s=40, c=green,
                           edgecolors='white', linewidths=.8, zorder=9)

    # Draw the maximum individual cell at its *own* radius and speed. Do not
    # relocate this marker to the radial-mean curve maximum.
    if grid_peak is not None:
        peak_radius, peak_mph = grid_peak
        aw.scatter([peak_radius], [peak_mph], marker='x', s=155,
                   c='#d3262c', lw=2.9, zorder=12)
        to_left = peak_radius > .72 * x_max
        aw.annotate('Max grid-cell wind', xy=(peak_radius, peak_mph),
                    xytext=(-22 if to_left else 22, -26),
                    textcoords='offset points',
                    ha='right' if to_left else 'left', va='top',
                    fontsize=9.8, weight='bold', color='#c7212b',
                    arrowprops=dict(arrowstyle='-', color='#c7212b',
                                    lw=1.0, shrinkA=3, shrinkB=5),
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.92, pad=1.2),
                    zorder=12)

    # Quadrant-radius strip. The horizontal coordinate is kilometers, while
    # categorical rows are 64, 50, and 34 knots. It is NOT a wind-speed plot.
    for tag, _, row in levels:
        color = colors[tag]
        aq.axhspan(row - .35, row + .35, color=color, alpha=.045, zorder=0, lw=0)
        quadrants = record['quadrant_radii_km'].get(tag, {})
        resolved = []
        for quadrant in QUADRANTS:
            value = quadrants.get(quadrant)
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue
            if not (np.isfinite(value) and value >= 0):
                continue
            resolved.append(value)
            aq.scatter([value], [row + offsets[quadrant]],
                       s=62, c=color, marker=shapes[quadrant],
                       edgecolors='white', linewidths=.85, zorder=6)
        if len(resolved) >= 2:
            aq.hlines(row, min(resolved), max(resolved),
                      color=color, linewidth=2.3, alpha=.38, zorder=2)
        mean = float(means.get(tag, np.nan))
        if len(resolved) == 4 and np.isfinite(mean):
            aq.vlines(mean, row - .28, row + .28, lw=2.45,
                      color='#293340', zorder=7)
        elif not resolved:
            aq.text(.988, row, 'not resolved', fontsize=8.7,
                    color='#687484', ha='right', va='center',
                    transform=aq.get_yaxis_transform())

    aq.set_ylim(-.49, 2.49)
    aq.set_yticks([2, 1, 0], labels=['R64', 'R50', 'R34'])
    aq.tick_params(axis='y', length=0, pad=10)
    for lab, key in zip(aq.get_yticklabels(), ('R64', 'R50', 'R34')):
        lab.set_color(colors[key])
        lab.set_fontweight('bold')
        lab.set_fontsize(10.5)
    aq.set_title('Quadrant wind radii  |  symbols = NE/SE/SW/NW  |  dark tick = 4-quadrant mean',
                 fontsize=10.7, color='#344455', pad=7, weight='medium')

    # A compact numerical key, separate from the data strip. Its figures are
    # the same values exported to the summary CSV. Dashes mean unavailable.
    def km_cell(tag, quadrant):
        value = record['quadrant_radii_km'].get(tag, {}).get(quadrant)
        try:
            return f'{float(value):.0f}' if np.isfinite(float(value)) else '—'
        except (ValueError, TypeError):
            return '—'

    legend_handles, legend_labels = [], []
    for quadrant in QUADRANTS:
        legend_handles.append(Line2D([], [], ls='None', marker=shapes[quadrant],
                                     markersize=7.7, color='#58687a',
                                     markerfacecolor='#58687a', markeredgecolor='white'))
        legend_labels.append(
            f'{quadrant:<2}  {km_cell("R64", quadrant):>4}  '
            f'{km_cell("R50", quadrant):>4}  {km_cell("R34", quadrant):>4}'
        )
    key_title = 'Quadrant radii (km)\n        R64   R50   R34'
    legend = aq.legend(legend_handles, legend_labels, title=key_title,
                       loc='center left', bbox_to_anchor=(1.013, .5),
                       borderaxespad=0., handlelength=1.0, handletextpad=.45,
                       labelspacing=.44, fontsize=8.6,
                       prop=FontProperties(family='DejaVu Sans Mono', size=8.6),
                       frameon=True, facecolor='white', edgecolor='#d8dee5',
                       framealpha=.96)
    legend.get_title().set_fontfamily('DejaVu Sans Mono')
    legend.get_title().set_fontsize(8.2)

    aw.legend(loc='upper center', bbox_to_anchor=(.62, 1.015),
              fontsize=8.9, ncol=2, framealpha=.97, facecolor='white',
              edgecolor='#cbd3dc')
    ap.set_title('Radial sea-level pressure profile', fontsize=12.8,
                 weight='semibold', pad=8)
    ap.set_ylabel('Sea-level pressure (hPa)', fontsize=10.8, labelpad=9)
    ap.set_xlabel('Radius from tropical cyclone center (km)', fontsize=11.1)
    ap.legend(loc='upper left', fontsize=8.9, framealpha=.97,
              facecolor='white', edgecolor='#cbd3dc')

    # Tightly grouped metadata. Every number is generated for this valid time.
    fig.suptitle(
        f"WRF {record['domain'].upper()} | Tropical Cyclone Radial Structure | "
        f"Valid: {record['valid']:%H:%M:%SZ %Y-%m-%d}",
        fontsize=15.3, weight='bold', y=.989
    )
    radial_peak_ms, radial_peak_km = _peak_wrf_radial_mean(profile)
    grid_peak_ms = float(profile.get('max_10m_speed_ms', np.nan))
    fig.text(.087, .953,
             f"Center {record['center_lat']:+.3f}°, {record['center_lon']:+.3f}° | "
             f"Minimum WRF SLP {record['min_slp']:.1f} hPa | "
             f"Max grid-cell 10-m wind {_wind_triplet(grid_peak_ms)}",
             fontsize=9.5)
    fig.text(.087, .932,
             f"Peak radial-mean 10-m wind {_wind_triplet(radial_peak_ms)} "
             f"at {radial_peak_km:.0f} km | Tangential RMW "
             f"{_radius_text(float(profile['ring_mean_rmw_km']))}",
             fontsize=9.25)
    fig.text(.087, .911,
             'Four-quadrant means: ' + ' | '.join(
                 f'{key} {_radius_text(float(means.get(key, np.nan)))}'
                 for key in ('R64', 'R50', 'R34')),
             fontsize=9.25)
    parent_interp = ','.join(record.get('temporally_interpolated_domains', ())) or 'none'
    fig.text(.087, .890,
             f'WRF sources: {_source_domain_summary(record)} | '
             f'Parent-time interpolation: {parent_interp} | '
             f'Center source: {record["center_domain"]}',
             fontsize=8.8, color='#526172')

    # Footnotes are deliberately short; QC stays in CSV and console.
    fig.text(.087, .069,
             'Red X: strongest individual WRF 10-m wind, at its own radius. '
             'The blue curve is the radial mean.',
             fontsize=8.7, color='#526172')
    fig.text(.087, .046,
             'Quadrant symbols are 34-, 50-, and 64-kt radial crossings, not '
             'wind speeds. Instantaneous WRF proxies, not NHC sustained-wind radii.',
             fontsize=8.25, color='#526172')
    if record.get('radii_qc'):
        fig.text(.087, .095,
                 'QC review: ' + '; '.join(record['radii_qc']),
                 fontsize=8.3, color='#995c2a')
    fig.savefig(out_path, dpi=180, facecolor='white')
    plt.close(fig)


def write_radial_csv(path, profile):
    """Native WRF radial means, area/sector coverage, and data provenance."""
    fields = [
        'radius_km', 'wrf_10m_radial_mean_ms', 'wrf_10m_radial_mean_mph',
        'wrf_10m_radial_mean_kt', 'wrf_cyclonic_tangential_ms',
        'wrf_cyclonic_tangential_mph', 'wrf_cyclonic_tangential_kt',
        'wrf_slp_hpa', 'gridcell_count', 'angular_sectors_out_of_8',
        'angular_coverage_fraction', 'wrf_source_domains',
        'radial_quality', 'source_transition'
    ]
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, radius in enumerate(profile['r_km']):
            wind = float(profile['wind_speed_ms'][i])
            tang = float(profile['cyclonic_tangential_ms'][i])
            writer.writerow(dict(
                radius_km=float(radius),
                wrf_10m_radial_mean_ms=wind,
                wrf_10m_radial_mean_mph=wind * MPS_TO_MPH,
                wrf_10m_radial_mean_kt=wind / KNOT_MS,
                wrf_cyclonic_tangential_ms=tang,
                wrf_cyclonic_tangential_mph=tang * MPS_TO_MPH,
                wrf_cyclonic_tangential_kt=tang / KNOT_MS,
                wrf_slp_hpa=float(profile['slp_hpa'][i]),
                gridcell_count=int(profile['gridcell_count'][i]),
                angular_sectors_out_of_8=int(profile['angular_sectors_8'][i]),
                angular_coverage_fraction=float(profile['radial_coverage_fraction'][i]),
                wrf_source_domains=profile['source_domain'][i],
                radial_quality=profile['radial_quality'][i],
                source_transition=int(profile['source_transition'][i])
            ))


def diagnostic_row(record):
    """One summary row per WRF valid time, including all 12 quadrant radii."""
    profile = record['profile']
    peak, peak_rad = _peak_wrf_radial_mean(profile)
    row = dict(
        valid_time_utc=record['valid'].strftime('%Y-%m-%dT%H:%M:%SZ'),
        domain=record['domain'],
        center_lat=record['center_lat'], center_lon=record['center_lon'],
        center_source_domain=record['center_domain'],
        wrf_source_domains=_source_domain_summary(record),
        parent_time_interpolation=','.join(record.get('temporally_interpolated_domains', ())),
        full_circle_wrf_radius_km=record['available_radius_km'],
        minimum_slp_hpa=record['min_slp'],
        maximum_gridcell_wind_ms=profile['max_10m_speed_ms'],
        maximum_gridcell_wind_mph=profile['max_10m_speed_ms'] * MPS_TO_MPH,
        maximum_gridcell_wind_kt=profile['max_10m_speed_ms'] / KNOT_MS,
        maximum_gridcell_radius_km=profile['max_10m_radius_km'],
        maximum_gridcell_source_domain=profile['max_10m_source_domain'],
        maximum_gridcell_bearing_deg=profile['max_10m_bearing_deg'],
        peak_radial_mean_wind_ms=peak,
        peak_radial_mean_wind_mph=peak * MPS_TO_MPH,
        peak_radial_mean_wind_kt=peak / KNOT_MS,
        peak_radial_mean_radius_km=peak_rad,
        wrf_tangential_rmw_km=profile['ring_mean_rmw_km'],
        partial_azimuth_bins=int(np.count_nonzero(
            np.asarray(profile['radial_quality']) == 'PARTIAL_AZIMUTH')),
        mixed_domain_bins=int(np.count_nonzero(
            np.asarray(profile['radial_quality']) == 'MIXED_DOMAIN')),
        source_transition_bins=int(np.count_nonzero(profile['source_transition'])),
        wind_radius_qc='; '.join(record.get('radii_qc', ()))
    )
    for kt in WIND_THRESHOLDS_KT:
        key = f'R{kt}'
        mean = float(record['mean_radii_km'][key])
        row[f'r{kt}_quadrant_mean_km'] = mean if np.isfinite(mean) else ''
        for quadrant in QUADRANTS:
            low = quadrant.lower()
            row[f'r{kt}_{low}_km'] = record['quadrant_radii_km'][key].get(quadrant, '')
            row[f'r{kt}_{low}_status'] = record['wind_radius_status'][key][quadrant]
            row[f'r{kt}_{low}_provenance'] = record['wind_radius_details'][key][quadrant]
    return row


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in (2, 3) or not re.fullmatch(r'd0[1-9]', args[1]):
        print('Usage: python3 wrf_tropical_cyclone_radial_structure.py WRF_RUN_DIR d02 [all|frames|gif]',
              file=sys.stderr)
        return 2
    wrf_dir, domain = args[:2]
    mode = args[2] if len(args) == 3 else 'all'
    if mode not in ('all', 'frames', 'gif'):
        print('Optional third positional value: all, frames, or gif', file=sys.stderr)
        return 2
    root = Path('WRF_TC_Standard_Radii_Output')
    images, data, animations = root / 'Images', root / 'CSV', root / 'Animation'
    for folder in (images, data, animations):
        folder.mkdir(parents=True, exist_ok=True)
    if mode == 'gif':
        paths = sorted(images.glob(f'wrf_{domain}_TC_Standard_Radii_*.png'))
        if not paths:
            raise FileNotFoundError('No existing radial-profile PNGs to animate')
        outfile = animations / f'wrf_{domain}_TC_Standard_Radii.gif'
        make_gif(paths, outfile)
        print(f'[PASS] GIF {outfile}')
        return 0
    try:
        from netCDF4 import Dataset
        import wrf
    except ImportError as exc:
        raise RuntimeError('Install or activate an environment with netCDF4 and wrf-python') from exc

    inventory = _read_all_domain_inventory(wrf_dir, domain, Dataset, wrf)
    times = sorted(inventory[domain])
    if not times:
        raise RuntimeError(f'No readable WRF frames available for {domain}')
    previous = None
    summary, pngs = [], []
    for valid in times:
        try:
            grids = _read_matched_domains(valid, domain, inventory, Dataset, wrf)
            guess = None
            if previous is not None:
                elapsed_h = (valid - previous['valid']).total_seconds() / 3600.
                if 0 < elapsed_h <= 6:
                    guess = (previous['center_lat'], previous['center_lon'])
            record = derive_stitched_frame(grids, valid, domain, guess)
            stamp = valid.strftime('%Y%m%d%H%M%S')
            basename = f'wrf_{domain}_TC_Standard_Radii_{stamp}'
            png = images / (basename + '.png')
            csv_path = data / (basename + '.csv')
            plot_standard_radii(record, png)
            write_radial_csv(csv_path, record['profile'])
            pngs.append(png)
            summary.append(diagnostic_row(record))
            previous = record
            if record.get('radii_qc'):
                print(f'[QC] {valid:%Y-%m-%d %H:%M}Z: {record["radii_qc"]}')
            radii_text = ', '.join(
                f'R{kt}={_radius_text(record["mean_radii_km"][f"R{kt}"])}'
                for kt in WIND_THRESHOLDS_KT)
            print(f'[PASS] {valid:%Y-%m-%d %H:%M}Z: {radii_text}; '
                  f'sources={_source_domain_summary(record)} -> {png}')
        except Exception as exc:
            print(f'[SKIP] {valid:%Y-%m-%d %H:%M}Z: {exc}', file=sys.stderr)
    if not summary:
        raise RuntimeError('No valid WRF frames processed. Check NetCDF files and parent coverage.')
    summary_file = data / f'wrf_{domain}_TC_Standard_Radii_Summary.csv'
    with summary_file.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'[PASS] Summary {summary_file}')
    if mode == 'all':
        gif = animations / f'wrf_{domain}_TC_Standard_Radii.gif'
        try:
            make_gif(pngs, gif)
            print(f'[PASS] GIF {gif}')
        except ImportError:
            print('[WARN] Pillow not installed; PNG and CSV outputs are complete')
    print(f'Completed {len(summary)} valid WRF time(s) using available parent domains.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
