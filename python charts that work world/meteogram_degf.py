#!/usr/bin/env python3
"""Standalone WRF five-panel meteogram, imperial units (°F, mph, in).

USAGE:
    python3 meteogram_degf_enhanced.py /path/to/WRF/run d02 'Salt Lake City, UT' 40.7608 -111.891

Plot panels: (1) sea-level pressure; (2) 2m temperature/dew point with
light-green 2m RH bars BEHIND the lines; (3) 10m wind and direction;
(4) exact trailing-one-hour total-precipitation and snowfall water-equivalent rates;
(5) surface downwelling shortwave radiation. No day/night shading,
no summary banner, no separate RH panel, and no summary boxes on data.

Moving-nest point selection is preserved: wrf.ll_to_xy(..., timeidx=...) is
called for every timestep, then x/y bounds are checked. No cross-domain
fallback is attempted. Outputs a PNG and full-precision CSV.

Scientific caveats:
* RAINC+RAINNC+(optional RAINSH) are total model precipitation accumulators,
  including frozen precipitation water equivalent, not exclusively rain.
* ACSNOW is used for snow accumulation where present; SNOWNC, when used,
  captures grid-scale snowfall only. SNOW is an instantaneous snowpack field
  and is never used as a snowfall accumulation counter.
* A trailing-hour rate needs an exact WRF output 60 minutes earlier;
  otherwise the plotted bar is missing, not fabricated zero.
* A fixed geographic site is sampled at the nearest moving-grid cell.
  Model accumulation counters may not be geographically continuous during
  moving-nest repositioning/restarts; invalid negative differences are masked,
  and these results should be verified for quantitative applications.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
import glob
import os
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Keep scientific routines separately testable without a local WRF installation.
try:
    from netCDF4 import Dataset
except ImportError:
    Dataset = None
try:
    import wrf
    from wrf import ll_to_xy
except ImportError:
    wrf = None
    ll_to_xy = None

UNIT_SYSTEM = 'imperial'
METRIC = UNIT_SYSTEM == 'metric'
MM_TO_IN = 1.0 / 25.4


# -----------------------------------------------------------------------------
# Existing moving-nest point-selection behavior, retained intentionally.
# -----------------------------------------------------------------------------
def get_domain_shape(ncfile):
    """Return unstaggered WRF horizontal shape as (ny, nx)."""
    ny = len(ncfile.dimensions['south_north'])
    nx = len(ncfile.dimensions['west_east'])
    return ny, nx


def moving_nest_ll_to_xy(ncfile, latitude: float, longitude: float, time_index: int):
    """Choose x/y on the specified WRF Time: essential for vortex-following nests."""
    xy_loc = ll_to_xy(
        ncfile,
        latitude,
        longitude,
        timeidx=time_index,
        as_int=True,
        meta=False,
    )
    x = int(xy_loc[0]); y = int(xy_loc[1])
    ny, nx = get_domain_shape(ncfile)
    if x < 0 or x >= nx or y < 0 or y >= ny:
        return None
    return x, y


# -----------------------------------------------------------------------------
# WRF time, quantities, and quality control.
# -----------------------------------------------------------------------------
def filename_valid_time(path):
    """WRF filename time is valid ONLY as a fallback for time_index=0."""
    match = re.search(r'wrfout_.*?_(\d{4}-\d{2}-\d{2})_(\d{2})[:_](\d{2})[:_](\d{2})',
                      Path(path).name)
    if not match:
        raise ValueError(f'Cannot determine UTC WRF time from {path}')
    return datetime.strptime('_'.join([match[1], match[2], match[3], match[4]]),
                             '%Y-%m-%d_%H_%M_%S')


def _datetime_from_raw(value):
    """Normalize wrf-python/netCDF timestamps as naive UTC datetime."""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError('A single valid time was expected')
        value = value.reshape(())[()]  # preserve numpy datetime64 precision
    if isinstance(value, np.datetime64):
        value = datetime.fromisoformat(np.datetime_as_string(value, unit='us'))
    if not isinstance(value, datetime):
        raise TypeError(f'Unexpected WRF time: {value!r}')
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def get_valid_time(ncfile, path, time_index):
    """Prefer WRF metadata, fall back to the specific 'Times' record."""
    try:
        return _datetime_from_raw(wrf.extract_times(ncfile, timeidx=time_index))
    except Exception:
        pass
    if 'Times' in ncfile.variables:
        raw = ncfile.variables['Times'][time_index]
        try:
            txt = b''.join(bytes(x) for x in raw).decode('ascii').strip()
        except Exception:
            txt = ''.join(x.decode('ascii') if isinstance(x, bytes) else str(x) for x in raw).strip()
        return datetime.strptime(txt.replace(' ', '_'), '%Y-%m-%d_%H:%M:%S')
    if time_index == 0:
        return filename_valid_time(path)
    raise RuntimeError(f'Missing valid time metadata for {path} time index={time_index}')


def raw_point(nc, name, idx, y, x, *, optional=False):
    """Read a model variable at one timestep and x/y; optional means missing."""
    if name not in nc.variables:
        if optional:
            return float('nan')
        raise KeyError(f'Missing required WRF variable {name}')
    v = nc.variables[name]
    dims = tuple(getattr(v, 'dimensions', ()))
    try:
        if dims and dims[0] == 'Time':
            return float(v[idx, y, x])
        return float(v[y, x])
    except Exception as exc:
        raise RuntimeError(f'Cannot sample {name}[time={idx}, j={y}, i={x}]: {exc}') from exc


def derived_point(nc, name, idx, y, x):
    try:
        return float(wrf.getvar(nc, name, timeidx=idx)[y, x])
    except Exception as exc:
        raise RuntimeError(f'Cannot obtain wrf-python {name} at time={idx}, j={y}, i={x}: {exc}') from exc


def true_wind_components(nc, idx, y, x):
    """Earth-relative components; never mislabel raw grid-relative direction."""
    try:
        uv = wrf.getvar(nc, 'uvmet10', timeidx=idx, meta=False)
        return float(uv[0, y, x]), float(uv[1, y, x])
    except Exception:
        u = raw_point(nc, 'U10', idx, y, x)
        v = raw_point(nc, 'V10', idx, y, x)
        c = raw_point(nc, 'COSALPHA', idx, y, x, optional=True)
        s = raw_point(nc, 'SINALPHA', idx, y, x, optional=True)
        if np.isfinite(c) and np.isfinite(s):
            return u * c - v * s, v * c + u * s
        # Speed magnitude is rotation invariant, but direction is not.
        return float(np.hypot(u, v)), float('nan')


def wind_direction_from_uv(u, v):
    if not np.isfinite(u) or not np.isfinite(v):
        return float('nan')
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def cardinal(degrees):
    if not np.isfinite(degrees):
        return 'N/A'
    points = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
    return points[int((degrees % 360.0 + 22.5) // 45.0) % 8]


def one_hour_accumulation(current, previous, *, tolerance=1e-7):
    """Return exact trailing-one-hour accumulation; NaN on gaps/resets."""
    if previous is None:
        return float('nan')
    if current['valid_time'] - previous['valid_time'] != timedelta(hours=1):
        return float('nan')
    val = current['precip_cumulative_mm'] - previous['precip_cumulative_mm']
    if not np.isfinite(val) or val < -tolerance:
        return float('nan')
    return max(0.0, float(val))


def one_hour_snow(current, previous, *, tolerance=1e-7):
    if previous is None:
        return float('nan')
    if current['valid_time'] - previous['valid_time'] != timedelta(hours=1):
        return float('nan')
    if current['snow_source'] is None or current['snow_source'] != previous['snow_source']:
        return float('nan')
    val = current['snow_cumulative_swe_mm'] - previous['snow_cumulative_swe_mm']
    if not np.isfinite(val) or val < -tolerance:
        return float('nan')
    return max(0.0, float(val))


def compute_one_hour_rates(samples):
    """Do not treat absent first-hour data as zero; match by valid UTC time."""
    table = {r['valid_time']: r for r in samples}
    for r in samples:
        prev = table.get(r['valid_time'] - timedelta(hours=1))
        # Comparison at the fixed-site nearest cell across WRF output times.
        # A moving-nest counter jump may remain unrecognized when positive.
        r['precip_hour_mm'] = one_hour_accumulation(r, prev)
        r['snow_hour_swe_mm'] = one_hour_snow(r, prev)
        r['snow_hour_10to1_mm'] = (10.0 * r['snow_hour_swe_mm']
                                   if np.isfinite(r['snow_hour_swe_mm']) else float('nan'))
    return samples


def discover_frames(paths):
    frames = []
    for path in paths:
        with Dataset(path) as nc:
            if 'Time' in nc.dimensions:
                n = len(nc.dimensions['Time'])
            elif 'Times' in nc.variables:
                n = nc.variables['Times'].shape[0]
            else:
                n = 1
        frames.extend((path, i) for i in range(n))
    return frames


def process_frame(args):
    path, idx, latitude, longitude = args
    with Dataset(path) as nc:
        dt = get_valid_time(nc, path, idx)
        # Existing per-time moving-nest mapping retained verbatim in behavior.
        try:
            xy = moving_nest_ll_to_xy(nc, latitude, longitude, idx)
        except Exception as exc:
            print(f'[SKIP] {dt:%Y-%m-%d %H:%MZ} location lookup: {exc}', flush=True)
            return None
        if xy is None:
            print(f'[SKIP] {dt:%Y-%m-%d %H:%MZ} point outside {Path(path).name}', flush=True)
            return None
        x, y = xy
        try:
            temp_c = raw_point(nc, 'T2', idx, y, x) - 273.15
            dew_c = derived_point(nc, 'td2', idx, y, x)
            rh = derived_point(nc, 'rh2', idx, y, x)
            pressure = derived_point(nc, 'slp', idx, y, x)
            solar = raw_point(nc, 'SWDOWN', idx, y, x)
            # RAINSH is not written by every WRF configuration.
            rainc = raw_point(nc, 'RAINC', idx, y, x)
            rainnc = raw_point(nc, 'RAINNC', idx, y, x)
            rainsh = raw_point(nc, 'RAINSH', idx, y, x, optional=True)
            precip = rainc + rainnc + (rainsh if np.isfinite(rainsh) else 0.0)
            if 'ACSNOW' in nc.variables:
                snow_source = 'ACSNOW'
            elif 'SNOWNC' in nc.variables:
                snow_source = 'SNOWNC'
            else:
                snow_source = None
            snow = raw_point(nc, snow_source, idx, y, x) if snow_source else float('nan')
            u, v = true_wind_components(nc, idx, y, x)
            if np.isfinite(u) and np.isfinite(v):
                speed = np.hypot(u, v)
                direction = wind_direction_from_uv(u, v)
            else:
                # Only magnitude is trustworthy when map rotation is unavailable.
                speed = abs(u)
                direction = float('nan')
        except Exception as exc:
            print(f'[SKIP] {dt:%Y-%m-%d %H:%MZ}: {exc}', flush=True)
            return None
        return dict(valid_time=dt, temp_c=float(temp_c), dew_c=float(dew_c),
                    rh_pct=float(rh), slp_hpa=float(pressure), solar_wm2=float(solar),
                    wind_ms=float(speed), wind_dir_deg=float(direction),
                    precip_cumulative_mm=float(precip),
                    snow_cumulative_swe_mm=float(snow), snow_source=snow_source,
                    grid_i=int(x), grid_j=int(y), wrf_file=Path(path).name,
                    time_index=int(idx))


# -----------------------------------------------------------------------------
# Plotting: white background, five panels, light-green RH behind temperature.
# -----------------------------------------------------------------------------
def variable_width_days(time_points):
    if len(time_points) < 2:
        return 0.6 / 24.0
    delta = np.diff(np.sort(mdates.date2num(time_points)))
    delta = delta[delta > 0]
    if len(delta) == 0:
        return 0.6 / 24.0
    return float(np.clip(np.median(delta) * 0.75, 2.0/1440.0, 1.5/24.0))


def _dates_across_period(first, last):
    dates = []
    day = first.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= last:
        dates.append(day)
        day += timedelta(days=1)
    return dates


def _setup_day_labels(ax, first, last):
    """Date headings appear only for multi-day UTC forecasts, outside data."""
    if first.date() == last.date():
        return []  # Same date already appears in the validity subtitle.
    days = _dates_across_period(first, last)
    locations = []
    labels = []
    for day in days:
        lo = max(first, day)
        hi = min(last, day + timedelta(days=1))
        if hi <= lo:
            continue
        locations.append(mdates.date2num(lo + (hi-lo)/2))
        labels.append(day.strftime('%a %d %b'))
    if locations:
        header = ax.secondary_xaxis('top')
        header.set_xticks(locations)
        header.set_xticklabels(labels, fontsize=10, weight='bold')
        header.tick_params(axis='x', length=0, pad=6)
    return days

def _forecast_lines(axes, dates):
    for ax in axes:
        ax.set_facecolor('white')
        ax.grid(True, which='major', axis='both', color='#cbd5e1', alpha=.72, lw=.65)
        ax.set_axisbelow(True)
        for day in dates:
            if day.hour == 0:
                ax.axvline(day, color='#64748b', linewidth=1.05, alpha=.8, zorder=1)
        ax.tick_params(axis='both', labelsize=9)


def _precip_label_indices(values, *, threshold, maximum=12):
    """Select meaningful positive bars; thin labels for crowded long forecasts.

    The labels are a presentation feature only. No meteorological values are
    modified. Missing values and exact zero accumulations are never annotated.
    """
    series = np.asarray(values, dtype=float)
    candidates = [i for i, val in enumerate(series)
                  if np.isfinite(val) and val >= threshold]
    if len(candidates) <= maximum:
        return candidates
    # For dense multi-day charts suppress annotations of tiny repeated bars.
    # Keep substantial peaks relative to the largest visible accumulation.
    maximum_value = max(series[i] for i in candidates)
    significance = max(threshold * 2.5, maximum_value * 0.12)
    substantial = [i for i in candidates if series[i] >= significance]
    if not substantial:
        substantial = [max(candidates, key=lambda i: series[i])]
    peaks = []
    for i in substantial:
        left = series[i-1] if i > 0 else -np.inf
        right = series[i+1] if i+1 < series.size else -np.inf
        if series[i] > left and series[i] >= right:
            peaks.append(i)
    if not peaks:
        peaks = [max(substantial, key=lambda i: series[i])]
    ranked = sorted(peaks, key=lambda i: (-series[i], i))
    selected = []
    for i in ranked:
        if all(abs(i-j) >= 2 for j in selected):
            selected.append(i)
        if len(selected) >= maximum:
            break
    return sorted(selected)


def _label_precip_bars(ax, bars, values, *, threshold, unit_system, color):
    """Write small value labels above significant bars without obscuring data."""
    fmt = '.2f' if unit_system == 'imperial' else '.1f'
    for i in _precip_label_indices(values, threshold=threshold):
        bar = bars.patches[i]
        value = float(values[i])
        if not np.isfinite(value):
            continue
        ax.annotate(f'{value:{fmt}}',
                    (bar.get_x()+bar.get_width()/2, value),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7.5,
                    fontweight='semibold', color=color,
                    clip_on=True, zorder=7)


def _format_forecast_validity(times, duration_hours, cadence):
    """A single UTC date is printed once; multi-day forecasts show both dates."""
    first, last = times[0], times[-1]
    if first.date() == last.date():
        period = (f'Valid: {first:%a %d %b %Y}  |  '
                  f'{first:%H:%MZ}–{last:%H:%MZ}')
    else:
        period = (f'Valid: {first:%d %b %Y %H:%MZ}  →  '
                  f'{last:%d %b %Y %H:%MZ}')
    return f'{period}  ({duration_hours:g} hours)  |  {cadence}'


def make_plot(samples, *, city, latitude, longitude, domain, output_path, dpi=200):
    if not samples:
        raise ValueError('No samples available for plotting')
    samples = sorted(samples, key=lambda r: r['valid_time'])
    times = [r['valid_time'] for r in samples]
    take = lambda name: np.asarray([r[name] for r in samples], dtype=float)
    temp_c = take('temp_c'); dew_c = take('dew_c'); rh = take('rh_pct')
    slp = take('slp_hpa'); wind = take('wind_ms'); wd = take('wind_dir_deg')
    rain = take('precip_hour_mm'); snow = take('snow_hour_swe_mm')
    solar = take('solar_wm2')
    if METRIC:
        temp, dew, tunit = temp_c, dew_c, '°C'
        wind_values, wind_unit = wind, 'm/s'
        rain_values, snow_values, punit = rain, snow * 10.0, 'mm'
        suffix = 'metric'
    else:
        temp, dew, tunit = 32.0 + 1.8*temp_c, 32.0 + 1.8*dew_c, '°F'
        wind_values, wind_unit = wind*2.2369362921, 'mph'
        rain_values, snow_values, punit = rain*MM_TO_IN, snow*10.0*MM_TO_IN, 'in'
        suffix = 'imperial'

    count = len(times)
    # The chart grows modestly with duration to keep multi-day timelines readable.
    duration_h = max(1.0, (times[-1]-times[0]).total_seconds()/3600.0)
    fig_width = max(16.0, min(27.0, 15.5 + 0.055*duration_h))
    fig, axs = plt.subplots(5, 1, figsize=(fig_width, 14.6), dpi=dpi,
                            sharex=True, gridspec_kw={'height_ratios':[1.05,1.25,1.05,1.05,1.05]})
    fig.patch.set_facecolor('white')
    # Single-day charts need no extra date-heading row; multi-day charts do.
    # Keep title/subtitle outside data while reclaiming the old blank top space.
    same_utc_day = times[0].date() == times[-1].date()
    fig.subplots_adjust(left=.075, right=.925, bottom=.095,
                        top=.914 if same_utc_day else .895, hspace=.32)
    width = variable_width_days(times)
    dates = _dates_across_period(times[0], times[-1])
    _forecast_lines(axs, dates)

    # Panel 1: Sea-level pressure. No overlay callout boxes.
    axs[0].plot(times, slp, color='#111827', marker='o', ms=3.0, lw=1.6, label='Sea Level Pressure')
    axs[0].set_ylabel('SLP (hPa)')
    axs[0].set_title('Sea Level Pressure', fontsize=11, weight='bold', pad=7)
    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=6))
    _setup_day_labels(axs[0], times[0], times[-1])

    # Panel 2: Temperature/dew point lines OVER faint RH columns, right RH scale.
    t_ax = axs[1]
    rh_ax = t_ax.twinx()
    rh_ax.set_zorder(1)
    t_ax.set_zorder(2)
    t_ax.patch.set_visible(False)
    rh_ax.set_ylim(0, 100)
    rh_ax.bar(times, np.clip(rh, 0, 100), width=width,
              color='#99dfa5', alpha=.34, edgecolor='none',
              align='center', label='2m RH (%)', zorder=1)
    rh_ax.set_ylabel('Relative Humidity (%)', color='#16783d', fontsize=9)
    rh_ax.tick_params(axis='y', colors='#16783d', labelsize=9)
    rh_ax.grid(False)
    t_ax.plot(times, temp, color='#ef2323', marker='o', ms=3, lw=1.7,
              label=f'2m Temperature ({tunit})', zorder=5)
    t_ax.plot(times, dew, color='#138a35', marker='o', ms=3, lw=1.7,
              label=f'2m Dew Point ({tunit})', zorder=5)
    t_ax.set_ylabel(f'Temperature ({tunit})')
    t_ax.set_title('2m Temperature, Dew Point, and Relative Humidity', fontsize=11, weight='bold', pad=7)
    t_ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    freeze = 0.0 if METRIC else 32.0
    if min(np.nanmin(temp), np.nanmin(dew)) <= freeze + 8.0:
        t_ax.axhline(freeze, ls='--', lw=1, color='#3385ff', alpha=.8, label=f'Freezing ({tunit})', zorder=3)

    # Panel 3: Wind. Actual earth-relative direction where available.
    w_ax = axs[2]
    w_ax.bar(times, wind_values, width=width, color='#b6b7bc', edgecolor='#76787d',
             linewidth=.4, label=f'Wind Speed ({wind_unit})', zorder=3)
    w_ax.set_ylabel(f'Wind Speed ({wind_unit})')
    w_ax.set_ylim(bottom=0)
    w_ax.set_title('10m Wind Speed and Direction', fontsize=11, weight='bold', pad=7)
    wd_ax = w_ax.twinx()
    wd_ax.set_ylim(0, 360)
    wd_ax.set_yticks([0,90,180,270,360])
    wd_ax.set_yticklabels(['0° N','90° E','180° S','270° W','360° N'])
    wd_ax.scatter(times, wd, color='#ec6509', s=18, label='Wind Direction (°)', zorder=5)
    wd_ax.set_ylabel('Wind Direction (°)', fontsize=9)
    wd_ax.grid(False)

    # Panel 4: Both accumulation components in the same (water-equivalent) units.
    p_ax = axs[3]
    x = mdates.date2num(times)
    rain_bars = p_ax.bar(x - width*.27, rain_values, width=width*.5,
             color='#008923', edgecolor='none', label=f'Total Precipitation ({punit})', zorder=3)
    snow_bars = None
    if np.any(np.isfinite(snow_values)):
        name = 'Estimated Snowfall (10:1 ratio)' if any(r['snow_source']=='ACSNOW' for r in samples) else 'Estimated Grid-scale Snowfall (10:1 ratio)'
        snow_bars = p_ax.bar(x + width*.27, snow_values, width=width*.5, color='#14b9ef',
                 edgecolor='none', label=f'{name} ({punit})', zorder=3)
    # Headroom leaves value annotations clear of the top axis frame.
    valid_precip = np.concatenate((rain_values[np.isfinite(rain_values)],
                                   snow_values[np.isfinite(snow_values)]))
    largest = max(0.0, float(np.max(valid_precip))) if valid_precip.size else 0.0
    p_ax.set_ylim(0, largest*1.24 if largest > 0 else (0.10 if not METRIC else 2.0))
    threshold = 0.01 if not METRIC else 0.25
    _label_precip_bars(p_ax, rain_bars, rain_values,
                       threshold=threshold, unit_system=UNIT_SYSTEM, color='#006c1e')
    if snow_bars is not None:
        _label_precip_bars(p_ax, snow_bars, snow_values,
                           threshold=threshold, unit_system=UNIT_SYSTEM, color='#00799d')
    p_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    p_ax.set_ylabel(f'Total Precipitation ({punit})')
    p_ax.set_title('Trailing 1-Hour Totals' +
                   (' (snowfall unavailable)' if not np.any(np.isfinite(snow_values)) else ''),
                   fontsize=11, weight='bold', pad=7)
    snow_ax = p_ax.twinx()
    snow_ax.set_ylim(p_ax.get_ylim())
    snow_ax.set_ylabel(f'Estimated Snowfall (10:1 ratio) ({punit})', fontsize=9)
    snow_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    snow_ax.grid(False)

    # Panel 5: Solar; zero remains zero, and the axis is never negative.
    s_ax = axs[4]
    s_ax.plot(times, solar, color='#f59e0b', marker='o', ms=3, lw=1.5)
    s_ax.set_ylabel('SWDOWN (W/m²)')
    s_ax.set_ylim(bottom=0)
    if not np.any(np.isfinite(solar) & (solar > 0)):
        s_ax.set_ylim(0, 25)
    s_ax.set_title('Surface Downwelling Shortwave Radiation', fontsize=11, weight='bold', pad=7)

    # No day/night background, no summary banner, no RH-only panel.
    interval_min = np.median(np.diff(mdates.date2num(times)))*1440 if count>1 else np.nan
    dtstr = f'{interval_min:g}-minute output' if np.isfinite(interval_min) else 'single frame'
    title = f'WRF Meteogram for {city} ({latitude:.4f}° N, {abs(longitude):.4f}° {"W" if longitude<0 else "E"})  |  Domain: {domain}'
    fig.suptitle(title, y=.989, fontsize=16, weight='bold')
    fig.text(.5, .957, _format_forecast_validity(times, duration_h, dtstr),
             ha='center', fontsize=11)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=13)
    axs[-1].xaxis.set_major_locator(locator)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%MZ'))
    axs[-1].tick_params(axis='x', labelrotation=0)
    axs[-1].set_xlabel('UTC Time (Z)' if not same_utc_day else
                        f'UTC Time (Z) — {times[0]:%d %b %Y}')
    for ax in axs:
        ax.set_xlim(times[0] - timedelta(minutes=max(10, interval_min*.5 if np.isfinite(interval_min) else 20)),
                    times[-1] + timedelta(minutes=max(10, interval_min*.5 if np.isfinite(interval_min) else 20)))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return output_path


def write_csv(samples, path):
    keys = ['valid_time','temp_c','dew_c','rh_pct','slp_hpa','solar_wm2',
            'wind_ms','wind_dir_deg','precip_cumulative_mm','precip_hour_mm',
            'snow_cumulative_swe_mm','snow_hour_swe_mm','snow_hour_10to1_mm','snow_source',
            'grid_i','grid_j','wrf_file','time_index']
    if not METRIC:
        keys += ['temp_f','dew_f','wind_mph','precip_hour_in','snow_hour_swe_in','snow_hour_10to1_in']
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        for r in samples:
            o={k:r.get(k) for k in keys}
            o['valid_time']=r['valid_time'].strftime('%Y-%m-%dT%H:%M:%SZ')
            if not METRIC:
                o.update(temp_f=32+1.8*r['temp_c'], dew_f=32+1.8*r['dew_c'],
                         wind_mph=2.2369362921*r['wind_ms'],
                         precip_hour_in=r['precip_hour_mm']*MM_TO_IN,
                         snow_hour_swe_in=r['snow_hour_swe_mm']*MM_TO_IN,
                         snow_hour_10to1_in=(r['snow_hour_10to1_mm']*MM_TO_IN if np.isfinite(r.get('snow_hour_10to1_mm', float('nan'))) else float('nan')))
            w.writerow(o)


def main(argv=None):
    if Dataset is None or wrf is None:
        raise SystemExit('Missing runtime requirements: install netCDF4 and wrf-python in your WRF Python environment.')
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('wrf_dir',help='Directory containing wrfout_dXX files')
    parser.add_argument('domain',help='WRF domain such as d01 or d02')
    parser.add_argument('city',help='City/station label; quote spaces')
    parser.add_argument('latitude',type=float)
    parser.add_argument('longitude',type=float)
    parser.add_argument('--workers',type=int,default=4,help='Parallel readers (default: 4)')
    parser.add_argument('--dpi',type=int,default=200,help='PNG resolution (default: 200)')
    parser.add_argument('--output-dir',default=None,help='Override output directory')
    parser.add_argument('--no-csv',action='store_true',help='Do not export numeric CSV')
    args=parser.parse_args(argv)
    files=sorted(glob.glob(os.path.join(args.wrf_dir,f'wrfout_{args.domain}*')))
    if not files:
        raise SystemExit(f'No wrfout_{args.domain} files found in {args.wrf_dir}')
    frames=discover_frames(files)
    jobs=[(path,idx,args.latitude,args.longitude) for path,idx in frames]
    if not jobs:
        raise SystemExit('No valid WRF time records found')
    workers=max(1,min(args.workers,len(jobs)))
    if workers==1:
        data=[process_frame(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            data=list(executor.map(process_frame,jobs))
    samples=[x for x in data if x is not None]
    if not samples:
        raise SystemExit('No forecast samples inside this WRF domain for the specified location')
    samples.sort(key=lambda x: x['valid_time'])
    dedup={}
    for r in samples:
        if r['valid_time'] in dedup:
            print(f'[WARN] Duplicate valid time {r["valid_time"]}; keeping first frame, ignoring {r["wrf_file"]}')
            continue
        dedup[r['valid_time']]=r
    samples=list(dedup.values())
    compute_one_hour_rates(samples)
    safe_city = re.sub(r'[^\w .,-]+','_',args.city).strip()
    out_dir=Path(args.output_dir) if args.output_dir else Path(f'meteogram_{safe_city}_{UNIT_SYSTEM}')
    png=out_dir/f'meteogram_{safe_city}_{UNIT_SYSTEM}.png'
    make_plot(samples,city=args.city,latitude=args.latitude,longitude=args.longitude,
              domain=args.domain, output_path=png,dpi=args.dpi)
    print(f'[OK] Meteogram PNG: {png}')
    if not args.no_csv:
        csv_path=out_dir/f'meteogram_{safe_city}_{UNIT_SYSTEM}.csv'
        write_csv(samples,csv_path)
        print(f'[OK] Full-precision CSV: {csv_path}')
    print(f'[INFO] Valid samples: {len(samples)}, skipped: {len(jobs)-len(samples)}')
    gaps=sum(not np.isfinite(x['precip_hour_mm']) for x in samples)
    print(f'[INFO] Missing exact one-hour precipitation comparisons: {gaps}')
    return 0

if __name__=='__main__':
    sys.exit(main())
