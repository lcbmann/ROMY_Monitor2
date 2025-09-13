#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – Beat Frequency Drift Analysis (full, single file)
=======================================================

Generates the "beat drift" overview plot for all four ROMY rings and saves:
    <repo>/new_figures/html_beatdrift.png

Key changes vs. original:
  • Self-contained script (no wrapper that execs another file).
  • Output is written to the local repo's 'new_figures' directory.
  • Robust y-limits using central 98% coverage (1st–99th percentiles) so most
    of the data is visible while suppressing outliers.
  • Falls back to fixed nominal band per ring if percentile limits are invalid.

It tries to read beat (LJ*) data from:
    $ROMY_MOUNT/temp_archive  → then  $ROMY_MOUNT/romy_archive
with $ROMY_MOUNT defaulting to /import/freenas-ffb-01-data

Optional overlays (maintenance LXX and MLTI) keep the original imports.
If those helper modules are unavailable, overlays are skipped gracefully.
"""

from __future__ import annotations
import os, gc, warnings
from pathlib import Path
from datetime import date
from typing import Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obspy import UTCDateTime, Stream

# Optional helpers (overlays); if not present, we continue without them.
try:
    from functions.get_mlti_intervals import __get_mlti_intervals
    from functions.mlti_intervals_to_NaN import __mlti_intervals_to_NaN  # unused, kept for reference
    from functions.load_mlti import __load_mlti
    from functions.smoothing import __smooth
    from functions.interpolate_nan import __interpolate_nan              # unused, kept for reference
    from functions.get_mlti_statistics import __get_mlti_statistics
    from functions.get_lxx_intervals import __get_lxx_intervals
    from functions.load_lxx import __load_lxx
    HAVE_OVERLAY_FUNCS = True
except Exception:
    HAVE_OVERLAY_FUNCS = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["agg.path.chunksize"] = 10_000

# ─────────── Paths / repo layout ─────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
FIG_DIR    = REPO_ROOT / "new_figures"
FIG_DIR.mkdir(exist_ok=True)

MOUNT = Path(os.getenv("ROMY_MOUNT", "/import/freenas-ffb-01-data")).expanduser()

# Prefer temp_archive if it exists, else fallback to romy_archive
SDS_ROOT_CANDIDATES = [
    MOUNT / "temp_archive",
    MOUNT / "romy_archive",
]

# ─────────── Config ─────────────────────────────────────────────────────
config = {}

# days shown
config["time_interval"] = int(os.getenv("ROMY_BEATDRIFT_DAYS", "14"))

# time span (UTC)
config["tend"] = UTCDateTime.now()
config["tbeg"] = config["tend"] - config["time_interval"] * 86400
config["tbeg"] = config["tbeg"].replace(hour=0, minute=0, second=0)

# nominal frequency windows (fallback clamps)
Zlower, Zupper = 553.20, 553.80
Ulower, Uupper = 302.45, 302.60
Vlower, Vupper = 447.65, 447.90
Wlower, Wupper = 447.65, 447.90

# Plot colors
config["colors"] = {"Z": "tab:orange", "U": "deeppink", "V": "tab:blue", "W": "darkblue"}

# ─────────── SDS reading ─────────────────────────────────────────────────
def read_sds_any(root_candidates, seed: str, tbeg: UTCDateTime, tend: UTCDateTime) -> Stream:
    """
    Try multiple SDS roots and return the first non-empty Stream.
    """
    from obspy.clients.filesystem.sds import Client
    net, sta, loc, cha = seed.split(".")
    for root in root_candidates:
        if not root.is_dir():
            continue
        try:
            cli = Client(str(root), sds_type="D", format="MSEED")
            st = cli.get_waveforms(net, sta, loc, cha, tbeg - 10, tend + 10, merge=-1)
            st = st.trim(tbeg, tend, nearest_sample=False)
            if st and len(st[0].data) > 0:
                return st
        except Exception:
            pass
    return Stream()

# ─────────── Robust y-limit helper (98% coverage) ────────────────────────
def robust_ylim(data: np.ndarray,
                nominal_low: float,
                nominal_high: float,
                coverage: float = 98.0) -> Tuple[float, float]:
    """
    Compute central `coverage` percentile limits (default 98% → 1st..99th)
    and clamp to the provided nominal [low, high] band.
    """
    if data is None or data.size == 0:
        return nominal_low, nominal_high

    a = np.asarray(data, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return nominal_low, nominal_high

    low_p = (100.0 - coverage) / 2.0
    high_p = 100.0 - low_p
    try:
        lo = float(np.nanpercentile(a, low_p))
        hi = float(np.nanpercentile(a, high_p))
    except Exception:
        return nominal_low, nominal_high

    # Clamp into nominal band
    lo_c = max(lo, nominal_low)
    hi_c = min(hi, nominal_high)

    # If degenerate or inverted, fallback to nominal
    if not np.isfinite(lo_c) or not np.isfinite(hi_c) or lo_c >= hi_c:
        return nominal_low, nominal_high

    return lo_c, hi_c

# ─────────── Load data ───────────────────────────────────────────────────
def load_all_beats(tbeg: UTCDateTime, tend: UTCDateTime):
    # Beat seeds (LJ*)
    seeds = {
        "Z": "BW.ROMY.XX.LJZ",
        "U": "BW.ROMY.XX.LJU",
        "V": "BW.ROMY.XX.LJV",
        "W": "BW.ROMY.XX.LJW",
    }
    beats = {}
    for ring, seed in seeds.items():
        try:
            st = read_sds_any(SDS_ROOT_CANDIDATES, seed, tbeg, tend)
        except Exception:
            st = Stream()
        beats[ring] = st
    return beats

def try_load_overlays(tbeg: UTCDateTime, tend: UTCDateTime):
    """
    Load LXX (maintenance) and MLTI (per ring) if helper modules are present.
    Returns dictionaries; missing entries are handled by the caller.
    """
    overlays = dict(
        lxx_t1=None, lxx_t2=None,
        mlti_t1=dict(), mlti_t2=dict(),
        mlti_stats=dict(),
    )
    if not HAVE_OVERLAY_FUNCS:
        return overlays

    # Maintenance LXX
    try:
        lxx = __load_lxx(tbeg, tend, str(MOUNT) + "/")
        lxx_t1, lxx_t2 = __get_lxx_intervals(lxx.datetime)
        overlays["lxx_t1"], overlays["lxx_t2"] = lxx_t1, lxx_t2
    except Exception:
        pass

    # MLTI per ring
    for ring in ["Z", "U", "V", "W"]:
        try:
            mlti = __load_mlti(tbeg, tend, ring, str(MOUNT) + "/")
            t1, t2 = __get_mlti_intervals(mlti.time_utc)
            overlays["mlti_t1"][ring] = t1
            overlays["mlti_t2"][ring] = t2

            stats = __get_mlti_statistics(mlti, tbeg, tend, intervals=True, plot=False, ylog=False)
            stats["mlti_series_avg"] = __smooth(stats["mlti_series"] * 30, 86400, win="boxcar") * 100
            overlays["mlti_stats"][ring] = stats
        except Exception:
            pass

    return overlays

# ─────────── Plotting ────────────────────────────────────────────────────
def make_plot(config, beats: dict, overlays: dict):
    """
    Create the 5-row figure:
      RZ, RU, RV, RW beat frequency time series + MLTI density panel.
    """
    Nrow, Ncol = 5, 1
    font = 12
    ref_date = config["tbeg"]

    fig, ax = plt.subplots(Nrow, Ncol, figsize=(10, 7), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    # Convenience
    def _plot_ring(row_idx: int, ring: str, nominal: Tuple[float, float], label: str):
        st = beats.get(ring, Stream())
        if st and len(st) > 0 and len(st[0].data) > 0:
            data = st[0].data
            times = st[0].times(reftime=ref_date)
            ax[row_idx].plot(times, data, color=config["colors"][ring], alpha=0.9, label=label)
            lo, hi = robust_ylim(data, nominal[0], nominal[1], coverage=98.0)
            ax[row_idx].set_ylim(lo, hi)
        else:
            ax[row_idx].text(0.5, 0.5, "No Data",
                             transform=ax[row_idx].transAxes,
                             ha="center", va="center",
                             fontsize=14, color="gray")
            ax[row_idx].set_ylim(nominal[0], nominal[1])

        ax[row_idx].ticklabel_format(useOffset=False)
        ax[row_idx].set_ylabel(label, fontsize=font)

    _plot_ring(0, "Z", (Zlower, Zupper), "Horizontal\nring (Hz)")
    _plot_ring(1, "U", (Ulower, Uupper), "Northern\nring (Hz)")
    _plot_ring(2, "V", (Vlower, Vupper), "Western\nring (Hz)")
    _plot_ring(3, "W", (Wlower, Wupper), "Eastern\nring (Hz)")

    # MLTI density panel (bottom)
    ax[4].set_ylabel("MLTI\nDensity (%)", fontsize=font)
    if overlays.get("mlti_stats"):
        for ring, stats in overlays["mlti_stats"].items():
            try:
                n_total = int(stats["cumsum"][-1])
                ax[4].plot(stats["tsec"], stats["mlti_series_avg"],
                           color=config["colors"][ring], label=f"R{ring} (N={n_total})")
                ax[4].fill_between(stats["tsec"], 0, stats["mlti_series_avg"],
                                   color=config["colors"][ring], alpha=0.3)
            except Exception:
                pass
        ax[4].legend(loc=1, ncol=3)
    ax[4].set_ylim(bottom=0)

    # X limits and grids + overlays (maintenance + MLTI intervals)
    for i in range(Nrow):
        ax[i].grid(ls=":", zorder=0)
        ax[i].set_xlim(0, (config["tend"] - config["tbeg"]))
        # Shadings use absolute seconds since ref_date along x
        _, _, ylo, yhi = ax[i].axis()

        # Maintenance shading (yellow)
        lxx_t1, lxx_t2 = overlays.get("lxx_t1"), overlays.get("lxx_t2")
        if lxx_t1 is not None and lxx_t2 is not None:
            for t1, t2 in zip(lxx_t1, lxx_t2):
                x1 = (t1 - UTCDateTime(ref_date))
                x2 = (t2 - UTCDateTime(ref_date))
                ax[i].fill_betweenx([ylo, yhi], x1, x2, color="yellow", alpha=0.3)

        # MLTI (red) shading per ring track
        ring_for_row = {0: "Z", 1: "U", 2: "V", 3: "W"}.get(i)
        if ring_for_row is not None:
            t1_list = overlays.get("mlti_t1", {}).get(ring_for_row)
            t2_list = overlays.get("mlti_t2", {}).get(ring_for_row)
            if t1_list is not None and t2_list is not None:
                for t1, t2 in zip(t1_list, t2_list):
                    x1 = (t1 - UTCDateTime(ref_date))
                    x2 = (t2 - UTCDateTime(ref_date))
                    ax[i].fill_betweenx([ylo, yhi], x1, x2, color="red", alpha=0.2)

    # Date ticks on the bottom axis
    days_to_show = config["time_interval"]
    day_sec = 86400
    if days_to_show <= 7:
        tick_spacing = 1
    elif days_to_show <= 14:
        tick_spacing = 2
    elif days_to_show <= 31:
        tick_spacing = 3
    else:
        tick_spacing = 7

    tick_positions, tick_labels = [], []
    for d in range(0, days_to_show + 1, tick_spacing):
        pos = d * day_sec
        tick_positions.append(pos)
        dts = (config["tbeg"] + d * day_sec).datetime
        tick_labels.append(dts.strftime("%Y-%m-%d\n%H:%M:%S"))

    ax[-1].set_xticks(tick_positions)
    ax[-1].set_xticklabels(tick_labels, rotation=0, ha="center")

    # minor ticks for other midnights
    minor_positions = [d * day_sec for d in range(days_to_show + 1) if d % tick_spacing != 0]
    ax[-1].set_xticks(minor_positions, minor=True)

    for i in range(Nrow):
        ax[i].minorticks_on()
        ax[i].grid(which="both", lw=0.4, color="grey", zorder=0, ls=":")

    return fig

# ─────────── Main ────────────────────────────────────────────────────────
def main():
    tbeg, tend = config["tbeg"], config["tend"]

    # Load data
    beats = load_all_beats(tbeg, tend)
    overlays = try_load_overlays(tbeg, tend) if HAVE_OVERLAY_FUNCS else dict()

    # Build and save figure
    fig = make_plot(config, beats, overlays)
    out = FIG_DIR / "html_beatdrift.png"
    fig.savefig(out, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ beatdrift → {out}")

if __name__ == "__main__":
    main()
