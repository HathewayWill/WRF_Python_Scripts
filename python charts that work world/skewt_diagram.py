#!/usr/bin/env python3
"""
WRF Skew-T / Hodograph / Severe Parameters (v3 playbook style)

Features:
    * Supports multiple wrfout_<domain>* files, each with one or more timesteps.
    * Supports a single wrfout file containing many timesteps.
    * One frame = one (file, time_index) pair → one Skew-T/Hodo PNG.
    * Uses netCDF4 + wrf-python only (no xarray).
    * Physics/diagnostics: identical to original script (MetPy parcel,
      CAPE/CIN, SRH, shear, STP, SCP, indices, etc.).
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

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import wrf
from metpy.plots import Hodograph, SkewT
from metpy.units import units
from netCDF4 import Dataset
from PIL import Image
from wrf import ALL_TIMES, to_np  # ALL_TIMES not used but kept for parity

###############################################################################
# Warning suppression
###############################################################################
warnings.filterwarnings("ignore")


###############################################################################
# Canonical helper function block (time truth + utilities) – contiguous
###############################################################################
def _is_finite_scalar(val):
    """
    Test for a finite scalar value (works with Pint Quantities, numpy, etc.).
    """
    try:
        arr = np.asarray(getattr(val, "m", val))
    except Exception:
        arr = np.asarray(val)
    return np.isfinite(arr).all()


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


###############################################################################
# Per-frame processing function (Skew-T + Hodograph + severe params)
###############################################################################
def process_frame(args):
    """
    Worker function that builds a Skew-T / hodograph plot and saves a PNG
    for a single (WRF file, time_index) pair.

    args = (
        ncfile_path,
        time_index,
        domain,
        city,
        lat,
        lon,
        path_figures,
    )
    """
    (
        ncfile_path,
        time_index,
        domain,
        city,
        lat,
        lon,
        path_figures,
    ) = args

    image_folder = os.path.join(path_figures, "Images")

    # Open WRF file for this frame (no shared Dataset across processes)
    with Dataset(ncfile_path) as ncfile:

        # Valid time from metadata (fallback to filename)
        valid_dt = get_valid_time(ncfile, ncfile_path, time_index)
        print(f"Plotting data: {valid_dt:%Y/%m/%d %H:%M:%S} UTC")

        # -------------------------------------------------------------------------
        # Extract column at requested lat/lon
        # -------------------------------------------------------------------------
        lat_lon = [lat, lon]
        x_y = wrf.ll_to_xy(ncfile, lat_lon[0], lat_lon[1])

        # WRF variables (physics unchanged)
        p1 = wrf.getvar(ncfile, "pres", timeidx=time_index, units="hPa")
        T1 = wrf.getvar(ncfile, "temp", timeidx=time_index, units="degC")
        Td1 = wrf.getvar(ncfile, "td", timeidx=time_index, units="degC")
        u1 = wrf.getvar(ncfile, "ua", timeidx=time_index, units="kt")
        v1 = wrf.getvar(ncfile, "va", timeidx=time_index, units="kt")
        z1 = wrf.getvar(ncfile, "height_agl", timeidx=time_index, units="m")

        # Keep column extraction semantics; ensure Pint units exist robustly
        p_col = to_np(p1[:, x_y[1], x_y[0]])
        T_col = to_np(T1[:, x_y[1], x_y[0]])
        Td_col = to_np(Td1[:, x_y[1], x_y[0]])
        u_col = to_np(u1[:, x_y[1], x_y[0]])
        v_col = to_np(v1[:, x_y[1], x_y[0]])
        z_col = to_np(z1[:, x_y[1], x_y[0]])

        p = units.Quantity(p_col, units.hectopascal)
        T = units.Quantity(T_col, units.degC)
        Td = units.Quantity(Td_col, units.degC)
        u = units.Quantity(u_col, units.knots)
        v = units.Quantity(v_col, units.knots)
        z = units.Quantity(z_col, units.meter)

        # -------------------------------------------------------------------------
        # Figure & Skew-T setup (structure only; physics unchanged)
        # -------------------------------------------------------------------------
        dpi = plt.rcParams.get("figure.dpi", 400)
        fig = plt.figure(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi)
        skew = SkewT(fig, rotation=45, rect=(0.05, 0.05, 0.50, 0.90))

        skew.ax.set_ylim(1070, 100)
        skew.ax.set_xlim(-50, 50)

        skew.ax.set_xlabel("Temperature ($^\\circ$C)")
        skew.ax.set_ylabel("Pressure (hPa)")

        fig.set_facecolor("whitesmoke")
        skew.ax.set_facecolor("whitesmoke")

        # Simple shaded isotherm pattern (unchanged)
        x1 = np.linspace(-100, 40, 8)
        x2 = np.linspace(-90, 50, 8)
        y = [1100, 50]
        for i in range(0, 8):
            skew.shade_area(y=y, x1=x1[i], x2=x2[i], color="gray", alpha=0.02, zorder=1)

        # Plot T/Td
        skew.plot(p, T, "r", lw=2, label="TEMPERATURE")
        skew.plot(p, Td, "g", lw=2, label="DEWPOINT")

        # -------------------------------------------------------------------------
        # Wind barbs (resampled)
        # -------------------------------------------------------------------------
        interval = np.logspace(2.113, 3, 40) * units.hPa

        p_vals = getattr(p, "magnitude", p)
        p_hpa = np.ma.asarray(p_vals).astype(float)
        interval_hpa = interval.to("hPa").magnitude

        idx = mpcalc.resample_nn_1d(p_hpa, interval_hpa)

        blank_len = len(u[idx])
        blank = np.zeros(blank_len)
        skew.plot_barbs(
            pressure=p[idx],
            u=blank,
            v=blank,
            xloc=0.955,
            fill_empty=True,
            sizes=dict(emptybarb=0.075, width=0.18, height=0.4),
        )
        skew.plot_barbs(
            pressure=p[idx],
            u=u[idx],
            v=v[idx],
            xloc=0.955,
            fill_empty=True,
            sizes=dict(emptybarb=0.075, width=0.18, height=0.4),
            length=7,
        )

        # Skew-T background lines (unchanged)
        skew.ax.axvline(0 * units.degC, linestyle="--", color="blue", alpha=0.5)
        skew.plot_dry_adiabats(lw=1, alpha=0.4)
        skew.plot_moist_adiabats(lw=1, alpha=0.4)
        skew.plot_mixing_lines(lw=1, alpha=0.4)

        # -------------------------------------------------------------------------
        # Parcel levels: LCL, LFC, EL (unchanged physics)
        # -------------------------------------------------------------------------
        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
        if _is_finite_scalar(lcl_pressure) and _is_finite_scalar(lcl_temperature):
            skew.plot(lcl_pressure, lcl_temperature, "ko", markerfacecolor="black")

        lfc_pressure, lfc_temperature = mpcalc.lfc(p, T, Td)
        if _is_finite_scalar(lfc_pressure) and _is_finite_scalar(lfc_temperature):
            skew.plot(lfc_pressure, lfc_temperature, "ko", markerfacecolor="red")

        el_pressure, el_temperature = mpcalc.el(p, T, Td)
        if _is_finite_scalar(el_pressure) and _is_finite_scalar(el_temperature):
            skew.plot(el_pressure, el_temperature, "ko", markerfacecolor="green")

        # -------------------------------------------------------------------------
        # Parcel profiles & shading (unchanged physics)
        # -------------------------------------------------------------------------
        ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=50 * units.hPa)
        ml_p, _, _ = mpcalc.mixed_parcel(p, T, Td, depth=50 * units.hPa)
        mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, Td, depth=50 * units.hPa)

        mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(
            p, T, Td, depth=50 * units.hPa
        )
        mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=50 * units.hPa)

        sbprof = mpcalc.parcel_profile(p[:], T[0], Td[0]).to("degC")
        skew.plot(
            p, sbprof, "darkorange", linewidth=1.5, ls="--", label="SB PARCEL PATH"
        )

        muprof = mpcalc.parcel_profile(p[:], mu_t, mu_td).to("degC")
        skew.plot(p, muprof, "red", linewidth=1.5, ls="--", label="MU PARCEL PATH")

        mlprof = mpcalc.parcel_profile(p[:], ml_t, ml_td).to("degC")
        skew.plot(p, mlprof, "gold", linewidth=1.5, ls="--", label="ML PARCEL PATH")

        skew.shade_cin(p, T, sbprof, Td, alpha=0.4, label="SBCIN")
        skew.shade_cape(p, T, sbprof, alpha=0.4, label="SBCAPE")
        skew.shade_cin(p, T, muprof, Td, alpha=0.3, label="MUCIN")
        skew.shade_cape(p, T, muprof, alpha=0.3, label="MUCAPE")
        skew.shade_cin(p, T, mlprof, Td, alpha=0.2, label="MLCIN")
        skew.shade_cape(p, T, mlprof, alpha=0.2, label="MLCAPE")

        # -------------------------------------------------------------------------
        # Text labels for LCL/LFC/EL/FRZ level (unchanged logic)
        # -------------------------------------------------------------------------
        if _is_finite_scalar(lcl_pressure):
            plt.text(
                0.82,
                lcl_pressure,
                "←LCL",
                weight="bold",
                color="gray",
                alpha=0.9,
                fontsize=11,
                transform=skew.ax.get_yaxis_transform(),
            )

        if _is_finite_scalar(lfc_pressure):
            plt.text(
                0.82,
                lfc_pressure,
                "←LFC",
                weight="bold",
                color="gray",
                alpha=0.9,
                fontsize=11,
                transform=skew.ax.get_yaxis_transform(),
            )

        if _is_finite_scalar(el_pressure):
            plt.text(
                0.82,
                el_pressure,
                "←EL",
                weight="bold",
                color="gray",
                alpha=0.9,
                fontsize=11,
                transform=skew.ax.get_yaxis_transform(),
            )

        T_np = T.magnitude
        p_np = p.magnitude
        zero_deg_index = np.argmin(np.abs(T_np + 0))
        frz_pt_p = p_np[zero_deg_index]

        if _is_finite_scalar(frz_pt_p):
            plt.text(
                0.82,
                frz_pt_p,
                "←FRZ",
                weight="bold",
                color="blue",
                alpha=0.3,
                fontsize=11,
                transform=skew.ax.get_yaxis_transform(),
            )

        # -------------------------------------------------------------------------
        # Hodograph inset (unchanged physics)
        # -------------------------------------------------------------------------
        hodo_ax = plt.axes((0.48, 0.45, 0.5, 0.5))
        h = Hodograph(hodo_ax, component_range=80.0)

        h.add_grid(increment=20, ls="-", lw=1.5, alpha=0.5)
        h.add_grid(increment=10, ls="--", lw=1, alpha=0.2)

        h.ax.set_box_aspect(1)
        h.ax.set_yticklabels([])
        h.ax.set_xticklabels([])
        h.ax.set_xticks([])
        h.ax.set_yticks([])
        h.ax.set_xlabel(" ")
        h.ax.set_ylabel(" ")

        plt.xticks(np.arange(0, 0, 1))
        plt.yticks(np.arange(0, 0, 1))
        for i in range(0, 120, 20):
            h.ax.annotate(
                str(i),
                (i, 0),
                xytext=(0, 2),
                textcoords="offset pixels",
                clip_on=True,
                fontsize=10,
                weight="bold",
                alpha=0.3,
                zorder=0,
            )
        for i in range(0, 120, 20):
            h.ax.annotate(
                str(i),
                (0, i),
                xytext=(0, 2),
                textcoords="offset pixels",
                clip_on=True,
                fontsize=10,
                weight="bold",
                alpha=0.3,
                zorder=0,
            )

        h.plot_colormapped(
            u[z.magnitude <= 12000],
            v[z.magnitude <= 12000],
            c=z[z.magnitude <= 12000],
            linewidth=6,
            label="0-12km WIND",
        )

        RM, LM, MW = mpcalc.bunkers_storm_motion(p, u, v, z)
        if _is_finite_scalar(RM[0]) and _is_finite_scalar(RM[1]):
            h.ax.text(
                RM[0].m + 0.5,
                RM[1].m - 0.5,
                "RM",
                weight="bold",
                ha="left",
                fontsize=13,
                alpha=0.6,
            )
        if _is_finite_scalar(LM[0]) and _is_finite_scalar(LM[1]):
            h.ax.text(
                LM[0].m + 0.5,
                LM[1].m - 0.5,
                "LM",
                weight="bold",
                ha="left",
                fontsize=13,
                alpha=0.6,
            )
        if _is_finite_scalar(MW[0]) and _is_finite_scalar(MW[1]):
            h.ax.text(
                MW[0].m + 0.5,
                MW[1].m - 0.5,
                "MW",
                weight="bold",
                ha="left",
                fontsize=13,
                alpha=0.6,
            )
        if _is_finite_scalar(RM[0]) and _is_finite_scalar(RM[1]):
            h.ax.arrow(
                0,
                0,
                RM[0].m - 0.3,
                RM[1].m - 0.3,
                linewidth=2,
                color="black",
                alpha=0.2,
                label="Bunkers Vector",
                length_includes_head=True,
                head_width=2,
            )

        # -------------------------------------------------------------------------
        # Severe parameters box (unchanged physics)
        # -------------------------------------------------------------------------
        fig.patches.extend(
            [
                plt.Rectangle(
                    (0.563, 0.05),
                    0.334,
                    0.37,
                    edgecolor="black",
                    facecolor="white",
                    linewidth=1,
                    alpha=1,
                    transform=fig.transFigure,
                    figure=fig,
                )
            ]
        )

        # Stability indices
        kindex = mpcalc.k_index(p, T, Td)
        total_totals = mpcalc.total_totals_index(p, T, Td)

        # -------------------------------------------------------------------------
        # FIX: thickness_hydrostatic units error (MetPy 1.x)
        # Ensure new_p/new_t are true pint.Quantity (not masked arrays without units)
        # -------------------------------------------------------------------------
        p_mag = np.ma.filled(np.ma.asarray(p.magnitude).astype(float), np.nan)
        T_mag = np.ma.filled(np.ma.asarray(T.magnitude).astype(float), np.nan)

        if _is_finite_scalar(lcl_pressure) and _is_finite_scalar(lcl_temperature):
            mask_sel = p_mag > float(lcl_pressure.magnitude)
            new_p_vals = np.append(p_mag[mask_sel], float(lcl_pressure.magnitude))
            new_t_vals = np.append(T_mag[mask_sel], float(lcl_temperature.magnitude))
        else:
            new_p_vals = p_mag.copy()
            new_t_vals = T_mag.copy()

        new_p = units.Quantity(new_p_vals, units.hectopascal)
        new_t = units.Quantity(new_t_vals, units.degC)

        lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)

        # Surface-based CAPE
        sbcape, sbcin = mpcalc.surface_based_cape_cin(p, T, Td)

        # Lifted Index
        LI = mpcalc.lifted_index(p, T, sbprof)

        # Showalter Index
        if p[0] < 850 * units.mbar:
            SW = np.nan
        else:
            SW = mpcalc.showalter_index(p, T, Td)

        # Storm-relative helicity
        (u_storm, v_storm), *_ = mpcalc.bunkers_storm_motion(p, u, v, z)
        *_, total_helicity1 = mpcalc.storm_relative_helicity(
            z, u, v, depth=1 * units.km, storm_u=u_storm, storm_v=v_storm
        )
        *_, total_helicity3 = mpcalc.storm_relative_helicity(
            z, u, v, depth=3 * units.km, storm_u=u_storm, storm_v=v_storm
        )
        *_, total_helicity6 = mpcalc.storm_relative_helicity(
            z, u, v, depth=6 * units.km, storm_u=u_storm, storm_v=v_storm
        )

        # Bulk shear
        ubshr1, vbshr1 = mpcalc.bulk_shear(p, u, v, height=z, depth=1 * units.km)
        bshear1 = mpcalc.wind_speed(ubshr1, vbshr1)
        ubshr3, vbshr3 = mpcalc.bulk_shear(p, u, v, height=z, depth=3 * units.km)
        bshear3 = mpcalc.wind_speed(ubshr3, vbshr3)
        ubshr6, vbshr6 = mpcalc.bulk_shear(p, u, v, height=z, depth=6 * units.km)
        bshear6 = mpcalc.wind_speed(ubshr6, vbshr6)

        # Significant Tornado parameter
        sig_tor = mpcalc.significant_tornado(
            sbcape, lcl_height, total_helicity3, bshear3
        ).to_base_units()

        # Supercell composite
        super_comp = mpcalc.supercell_composite(mucape, total_helicity3, bshear3)

        # -------------------------------------------------------------------------
        # Text: thermodynamic and kinematic parameters (unchanged)
        # -------------------------------------------------------------------------
        plt.figtext(
            0.58, 0.37, "SBCAPE: ", weight="bold", fontsize=15, color="black", ha="left"
        )
        plt.figtext(
            0.71,
            0.37,
            f"{sbcape:.0f~P}",
            weight="bold",
            fontsize=15,
            color="orangered",
            ha="right",
        )
        plt.figtext(
            0.58, 0.34, "SBCIN: ", weight="bold", fontsize=15, color="black", ha="left"
        )
        plt.figtext(
            0.71,
            0.34,
            f"{sbcin:.0f~P}",
            weight="bold",
            fontsize=15,
            color="lightblue",
            ha="right",
        )
        plt.figtext(
            0.58, 0.29, "MLCAPE: ", weight="bold", fontsize=15, color="black", ha="left"
        )
        plt.figtext(
            0.71,
            0.29,
            f"{mlcape:.0f~P}",
            weight="bold",
            fontsize=15,
            color="orangered",
            ha="right",
        )
        plt.figtext(
            0.58, 0.26, "MLCIN: ", weight="bold", fontsize=15, color="black", ha="left"
        )
        plt.figtext(
            0.71,
            0.26,
            f"{mlcin:.0f~P}",
            weight="bold",
            fontsize=15,
            color="lightblue",
            ha="right",
        )
        plt.figtext(
            0.58, 0.21, "MUCAPE: ", weight="bold", fontsize=15, color="black", ha="left"
        )
        plt.figtext(
            0.71,
            0.21,
            f"{mucape:.0f~P}",
            weight="bold",
            fontsize=15,
            color="orangered",
            ha="right",
        )
        plt.figtext(
            0.58, 0.18, "MUCIN: ", weight="bold", fontsize=15, color="black", ha="left"
        )
        plt.figtext(
            0.71,
            0.18,
            f"{mucin:.0f~P}",
            weight="bold",
            fontsize=15,
            color="lightblue",
            ha="right",
        )
        plt.figtext(
            0.58,
            0.13,
            "TT-INDEX: ",
            weight="bold",
            fontsize=15,
            color="black",
            ha="left",
        )
        plt.figtext(
            0.71,
            0.13,
            f"{total_totals:.0f~P}",
            weight="bold",
            fontsize=15,
            color="orangered",
            ha="right",
        )
        plt.figtext(
            0.58,
            0.10,
            "K-INDEX: ",
            weight="bold",
            fontsize=15,
            color="black",
            ha="left",
        )
        plt.figtext(
            0.71,
            0.10,
            f"{kindex:.0f~P}",
            weight="bold",
            fontsize=15,
            color="orangered",
            ha="right",
        )
        plt.figtext(
            0.58,
            0.07,
            "LIFTED-INDEX: ",
            weight="bold",
            fontsize=15,
            color="black",
            ha="left",
        )
        plt.figtext(
            0.71,
            0.07,
            f"{LI[0]:.0f~P}",
            weight="bold",
            fontsize=15,
            color="orangered",
            ha="right",
        )

        plt.figtext(
            0.73,
            0.37,
            "0-1km SRH: ",
            weight="bold",
            fontsize=15,
            color="black",
            ha="left",
        )
        plt.figtext(
            0.88,
            0.37,
            f"{total_helicity1:.0f~P}",
            weight="bold",
            fontsize=15,
            color="navy",
            ha="right",
        )
        plt.figtext(
            0.73,
            0.34,
            "0-1km SHEAR: ",
            weight="bold",
            fontsize=15,
            color="black",
            ha="left",
        )
        plt.figtext(
            0.88,
            0.34,
            f"{bshear1:.0f~P}",
            weight="bold",
            fontsize=15,
            color="blue",
            ha="right",
        )
        plt.figtext(
            0.73,
            0.29,
            "0-3km SRH: ",
            weight="bold",
            fontsize=15,
            color="black",
            ha="left",
        )
        plt.figtext(
            0.88,
            0.29,
            f"{total_helicity3:.0f~P}",
            weight="bold",
            fontsize=15,
            color="navy",
            ha="right",
        )
        plt.figtext(
            0.73,
            0.26,
            "0-3km SHEAR: ",
            weight="bold",
            fontsize=15,
            color="black",
            ha="left",
        )
        plt.figtext(
            0.88,
            0.26,
            f"{bshear3:.0f~P}",
            weight="bold",
            fontsize=15,
            color="blue",
            ha="right",
        )
        plt.figtext(
            0.73,
            0.21,
            "0-6km SRH: ",
            weight="bold",
            fontsize=15,
            color="black",
            ha="left",
        )
        plt.figtext(
            0.88,
            0.21,
            f"{total_helicity6:.0f~P}",
            weight="bold",
            fontsize=15,
            color="navy",
            ha="right",
        )
        plt.figtext(
            0.73,
            0.18,
            "0-6km SHEAR: ",
            weight="bold",
            fontsize=15,
            color="black",
            ha="left",
        )
        plt.figtext(
            0.88,
            0.18,
            f"{bshear6:.0f~P}",
            weight="bold",
            fontsize=15,
            color="blue",
            ha="right",
        )
        plt.figtext(
            0.73,
            0.13,
            "SIG TORNADO: ",
            weight="bold",
            fontsize=15,
            color="black",
            ha="left",
        )
        plt.figtext(
            0.88,
            0.13,
            f"{sig_tor[0]:.0f~P}",
            weight="bold",
            fontsize=15,
            color="orangered",
            ha="right",
        )
        plt.figtext(
            0.73,
            0.10,
            "SUPERCELL COMP: ",
            weight="bold",
            fontsize=15,
            color="black",
            ha="left",
        )
        plt.figtext(
            0.88,
            0.10,
            f"{super_comp[0]:.0f~P}",
            weight="bold",
            fontsize=15,
            color="orangered",
            ha="right",
        )
        if not np.isnan(SW):
            plt.figtext(
                0.73,
                0.07,
                "SW-INDEX: ",
                weight="bold",
                fontsize=15,
                color="black",
                ha="left",
            )
            plt.figtext(
                0.88,
                0.07,
                f"{SW[0]:.0f~P}",
                weight="bold",
                fontsize=15,
                color="orangered",
                ha="right",
            )

        # Legends
        skewleg = skew.ax.legend(loc="upper left")
        hodoleg = h.ax.legend(loc="upper left")

        # -------------------------------------------------------------------------
        # Titles & saving
        # -------------------------------------------------------------------------
        fig.text(
            0.0,
            0.97,
            "Weather Research and Forecasting Model\n"
            f"Model Skew-T Plot at {city} Lat: {lat}, Lon: {lon}\n"
            "Hodograph\nSevere Weather Parameters",
            fontsize=13,
            ha="left",
            va="top",
        )

        fig.text(
            1.0,
            0.97,
            f"Valid: {valid_dt:%H:%M:%SZ %Y-%m-%d}",
            fontsize=13,
            ha="right",
            va="top",
        )

        fname_time = valid_dt.strftime("%Y%m%d%H%M%S")
        file_out = f"wrf_{domain}_SkewT_LogP_{fname_time}.png"

        plt.savefig(
            os.path.join(image_folder, file_out),
            bbox_inches="tight",
            dpi=100,
        )

        plt.close(fig)

        return os.path.join(image_folder, file_out)


###############################################################################
# Frame discovery (multi-file + multi-time)
###############################################################################
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


###############################################################################
# GIF creation utilities (v3 canonical pattern)
###############################################################################
def create_gif(path_figures, image_folder, domain):
    animation_folder = os.path.join(path_figures, "Animation")
    if not os.path.isdir(animation_folder):
        os.mkdir(animation_folder)

    png_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    if not png_files:
        print("No PNG files found for GIF generation. Skipping GIF step.")
        return

    png_files_sorted = sorted(png_files)
    print("Creating .gif file from sorted .png files")

    images = [
        Image.open(os.path.join(image_folder, filename))
        for filename in png_files_sorted
    ]

    if not images:
        print("No images loaded for GIF creation. Skipping GIF step.")
        return

    gif_file_out = f"wrf_{domain}_SkewT_LogP.gif"
    gif_path = os.path.join(animation_folder, gif_file_out)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0,
    )

    print(f"GIF generation complete: {gif_path}")


###############################################################################
# Main CLI / controller (v3 style)
###############################################################################
def main():
    # Arguments: path_wrf, domain, city, [lat lon optional]
    if len(sys.argv) not in [4, 6]:
        print(
            "\nEnter the required arguments: path_wrf, domain, city, "
            "[latitude and longitude (optional)]\n"
            "For example:\n"
            "    script_name.py /home/WRF/test/em_real d01 Paris [48.8566 2.3522]\n"
        )
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]
    city = sys.argv[3]

    # Lat/Lon handling
    if len(sys.argv) == 6:
        try:
            lat = float(sys.argv[4])
            lon = float(sys.argv[5])
        except ValueError:
            print("Invalid latitude or longitude. Please enter valid decimal numbers.")
            sys.exit(1)
    else:
        while True:
            try:
                lat = float(
                    input("Enter latitude in decimal format (e.g., 48.8566): ").strip()
                )
                break
            except ValueError:
                print(
                    "Invalid input. Please enter latitude in decimal format (e.g., 48.8566)."
                )

        while True:
            try:
                lon = float(
                    input("Enter longitude in decimal format (e.g., 2.3522): ").strip()
                )
                break
            except ValueError:
                print(
                    "Invalid input. Please enter longitude in decimal format (e.g., 2.3522)."
                )

    print(f"Latitude: {lat}, Longitude: {lon}")

    # Output directories
    path_figures = f"wrf_SkewT_LogP_{city}_Lat_{lat}_Long_{lon}"
    image_folder = os.path.join(path_figures, "Images")
    animation_folder = os.path.join(path_figures, "Animation")

    for folder in (path_figures, image_folder, animation_folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    # Find WRF files
    ncfile_paths = sorted(glob.glob(os.path.join(path_wrf, f"wrfout_{domain}*")))
    if not ncfile_paths:
        print("No WRF output files found matching pattern.")
        return

    # Discover frames (multi-file + multi-time)
    frames = discover_frames(ncfile_paths)
    if not frames:
        print("No timesteps found in provided WRF files.")
        return

    # Build task list for ProcessPoolExecutor
    args_list = [
        (ncfile_path, time_index, domain, city, lat, lon, path_figures)
        for (ncfile_path, time_index) in frames
    ]

    max_workers = min(4, len(args_list)) if args_list else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in executor.map(process_frame, args_list):
            pass

    print("Skew-T / Hodograph plot generation complete.")
    create_gif(path_figures, image_folder, domain)


###############################################################################
# Script entry point
###############################################################################
if __name__ == "__main__":
    main()
