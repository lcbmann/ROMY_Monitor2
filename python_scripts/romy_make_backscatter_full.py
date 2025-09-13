#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – Backscatter (robust replication)
=======================================

Generates html_backscatter.png in <repo>/new_figures using the same visual
layout as Andreas' makeplot_backscatter.py, but with robust handling when
no/partial data exists (e.g., missing time1/time2).

Usage:
    python romy_make_backscatter_full.py [Z|U|V|W]
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_cache")

import sys
from pathlib import Path
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from pandas import DataFrame

# --- repo paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
OUT_DIR = REPO / "new_figures"
OUT_DIR.mkdir(exist_ok=True)

# --- imports from the original function module
# (these must exist under python_scripts/functions/)
from functions.load_backscatter_data import __load_backscatter_data
from functions.backscatter_correction import __backscatter_correction

def _node_paths():
    """Replicate node-based path selection."""
    node = os.uname().nodename
    if node == 'lighthouse':
        archive_path = '/home/andbro/freenas/'
    elif node == 'kilauea':
        archive_path = '/import/freenas-ffb-01-data/'
    elif node == 'teide':
        archive_path = '/freenas-ffb-01/'
    elif node in ['lin-ffb-01', 'ambrym', 'hochfelln']:
        archive_path = '/import/freenas-ffb-01-data/'
    else:
        # sensible default for unknown hosts
        archive_path = '/import/freenas-ffb-01-data/'
    return dict(archive_path=archive_path)

def _placeholder(msg: str, sub: str = ""):
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.64, "Backscatter", ha="center", va="center",
                fontsize=16, weight="bold")
        if msg:
            ax.text(0.5, 0.42, msg, ha="center", va="center",
                    fontsize=11, color="crimson", wrap=True)
        if sub:
            ax.text(0.5, 0.24, sub, ha="center", va="center",
                    fontsize=9, color="0.4", wrap=True)
        ax.axis("off")
        out = OUT_DIR / "html_backscatter.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"ℹ️  Wrote placeholder → {out}")
    except Exception:
        print("💥 Failed to write placeholder image.")

def _find_max_min(lst, pp=99, perc=0, add_percent=None):
    """Percentile-based y-limit helper (same logic as original)."""
    from numpy import nanpercentile
    maxs, mins = [], []
    for arr in lst:
        maxs.append(nanpercentile(arr, pp))
        mins.append(nanpercentile(arr, 100-pp))
    if perc == 0:
        out_min, out_max = min(mins), max(maxs)
    else:
        _min = min(mins)
        _max = max(maxs)
        xx = _max*(1+perc) - _max
        out_min, out_max = _min-xx, _max+xx
    if add_percent is None:
        return out_min, out_max
    else:
        return out_min - out_min*add_percent, out_max + out_max*add_percent

def _makeplot(df, tbeg):
    """Visual replication of Andreas' __makeplot2 (with minor safety tweaks)."""
    from matplotlib.ticker import MultipleLocator

    Nrow, Ncol = 5, 1
    font = 12

    # Time axis (hours since tbeg)
    tscale, tunit = 1/3600, "hours"
    t_axis = df['time_sec'] * tscale

    fig, ax = plt.subplots(Nrow, Ncol, figsize=(15, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    # Row 0: f1_fs, f2_fs + twin fj_fs, w_s
    ax[0].plot(t_axis, df['f1_fs'], zorder=2, label="f1", color="tab:orange")
    ax[0].plot(t_axis, df['f2_fs'], zorder=2, label="f2", color="tab:red")
    ax[0].set_ylabel("$\\delta$f (Hz)", fontsize=font)
    ax[0].ticklabel_format(useOffset=False)
    ax[0].set_ylim(_find_max_min([df['f1_fs'], df['f2_fs']], pp=98))
    ax[0].legend(loc=1, ncol=2)

    ax00 = ax[0].twinx()
    ax00.plot(t_axis, df['fj_fs'], zorder=2, color="tab:blue", label="fj")
    # prefer 'w_s' if present, else fall back to our computed 'fj_bs' (if we added it)
    if 'w_s' in df.columns:
        ax00.plot(t_axis, df['w_s'], zorder=2, ls="--", color="k", label="backscatter removed")
    elif 'fj_bs' in df.columns:
        ax00.plot(t_axis, df['fj_bs'], zorder=2, ls="--", color="k", label="backscatter removed")
    ax00.set_ylabel("$\\delta$f (Hz)", fontsize=font)
    ax00.spines['right'].set_color('tab:blue')
    ax00.yaxis.label.set_color('tab:blue')
    ax00.tick_params(axis='y', colors='tab:blue')
    ax00.ticklabel_format(useOffset=False)
    ax00.set_ylim(_find_max_min([df['fj_fs']], pp=98))
    ax00.legend(loc=4, ncol=2)

    # Row 1: AC (f1,f2) + twin fj_ac
    ax[1].plot(t_axis, df['f1_ac']*1e3, zorder=2, label="f1", color="tab:orange")
    ax[1].plot(t_axis, df['f2_ac']*1e3, zorder=2, label="f2", color="tab:red")
    ax[1].set_ylabel("AC (mV)", fontsize=font)
    ax[1].set_ylim(_find_max_min([df['f1_ac']*1e3, df['f2_ac']*1e3], pp=99.5, add_percent=0.05))

    ax11 = ax[1].twinx()
    ax11.plot(t_axis, df['fj_ac']*1e3, zorder=2, label="fj")
    ax11.set_ylabel("AC (mV)", fontsize=font)
    ax11.spines['right'].set_color('tab:blue')
    ax11.yaxis.label.set_color('tab:blue')
    ax11.tick_params(axis='y', colors='tab:blue')
    ax11.set_ylim(_find_max_min([df['fj_ac']*1e3], pp=99.5, add_percent=0.05))

    # Row 2: DC (f1,f2) + twin fj_dc
    ax[2].plot(t_axis, df['f1_dc']*1e3, zorder=2, label="f1", color="tab:orange")
    ax[2].plot(t_axis, df['f2_dc']*1e3, zorder=2, label="f2", color="tab:red")
    ax[2].set_ylabel("DC (mV)", fontsize=font)
    ax[2].set_ylim(_find_max_min([df['f1_dc']*1e3, df['f2_dc']*1e3], pp=99.5, add_percent=0.05))

    ax21 = ax[2].twinx()
    ax21.plot(t_axis, df['fj_dc']*1e3, zorder=2, label="fj")
    ax21.set_ylabel("DC (mV)", fontsize=font)
    ax21.spines['right'].set_color('tab:blue')
    ax21.yaxis.label.set_color('tab:blue')
    ax21.tick_params(axis='y', colors='tab:blue')
    ax21.set_ylim(_find_max_min([df['fj_dc']*1e3], pp=99.5, add_percent=0.05))

    # Row 3: AC/DC + twin fj (AC/DC)
    ax[3].plot(t_axis, df['f1_ac']/df['f1_dc']*1e3, zorder=2, label="f1", color="tab:orange")
    ax[3].plot(t_axis, df['f2_ac']/df['f2_dc']*1e3, zorder=2, label="f2", color="tab:red")
    ax[3].set_ylabel("AC/DC", fontsize=font)
    ax[3].set_ylim(_find_max_min([df['f1_ac']/df['f1_dc']*1e3, df['f2_ac']/df['f2_dc']*1e3], pp=99.9))

    ax31 = ax[3].twinx()
    ax31.plot(t_axis, df['fj_ac']/df['fj_dc']*1e3, zorder=2, label="fj")
    ax31.set_ylabel("AC/DC", fontsize=font)
    ax31.spines['right'].set_color('tab:blue')
    ax31.yaxis.label.set_color('tab:blue')
    ax31.tick_params(axis='y', colors='tab:blue')
    ax31.set_ylim(_find_max_min([df['fj_ac']/df['fj_dc']*1e3], pp=99.5, add_percent=0.05))

    # Row 4: phase diff
    ax[4].plot(t_axis, df['f1_ph']-df['f2_ph'], color="tab:orange", zorder=2, label="f1-f2")
    ax[4].plot(t_axis, df['f1_ph']-df['f2_ph'], color="tab:red", zorder=2, ls="--")
    ax[4].set_ylabel("$\\Delta$ Phase (rad)", fontsize=font)
    ax[4].set_ylim(_find_max_min([df['f1_ph']-df['f2_ph']], pp=99.5, add_percent=0.05))

    # axes cosmetics
    for i in range(Nrow):
        ax[i].set_xlim(0, 24)
        ax[i].xaxis.set_major_locator(MultipleLocator(4))
        ax[i].xaxis.set_minor_locator(MultipleLocator(1))
        ax[i].grid(ls=":", zorder=0, which="both")
        ax[i].legend(loc=1, ncol=3)
    ax[4].set_xlabel(f"Time ({tunit}) since {tbeg.date} {str(tbeg.time)[:8]} UTC", fontsize=font)

    fig.tight_layout()
    return fig

def main():
    # --- CLI ring override
    ring = "Z"
    if len(sys.argv) > 1 and sys.argv[1].strip().upper() in {"Z","U","V","W"}:
        ring = sys.argv[1].strip().upper()

    # --- config (mirrors original, but output always new_figures/)
    paths = _node_paths()
    config = {}
    config['ring'] = ring
    config['time_interval'] = 1  # days
    config['last_reset'] = UTCDateTime("2024-10-01 14:00")
    config['tend'] = UTCDateTime(UTCDateTime().now().date)
    if abs(config['tend'] - config['last_reset']) > config['time_interval']*86400:
        config['tbeg'] = config['tend'] - config['time_interval']*86400
    else:
        config['tbeg'] = config['last_reset']
    config['ring_sagnac'] = {"U":303.05, "V":447.5, "W":447.5, "Z":553.2}
    config['nominal_sagnac'] = config['ring_sagnac'][config['ring']]
    config['path_to_data'] = paths['archive_path'] + "romy_autodata/backscatter/"
    config['path_to_figs'] = OUT_DIR.as_posix() + "/"

    # --- load data
    try:
        bs = __load_backscatter_data(config['tbeg'], config['tend'], config['ring'], config['path_to_data'])
    except Exception as e:
        _placeholder("Backscatter loader failed", str(e))
        return 0

    # --- validate required columns
    required = {'time1','time2','f1_ac','f1_dc','f2_ac','f2_dc','f1_ph','f2_ph','fj_fs','fj_ac','fj_dc'}
    has_required = isinstance(bs, DataFrame) and (not bs.empty) and required.issubset(set(bs.columns))
    if not has_required:
        _placeholder("No backscatter data available for selected interval",
                     f"Ring R{ring}   {config['tbeg']} – {config['tend']}")
        return 0

    # --- compute time axis seconds
    try:
        bs['time_sec'] = bs.time2 - bs.time1 + (bs.time1 - bs.time1.loc[0])
    except Exception:
        _placeholder("Backscatter data missing usable time columns",
                     f"Ring R{ring}   {config['tbeg']} – {config['tend']}")
        return 0

    # --- compute backscatter corrected signal if not present
    # original script computes fj_bs but plots 'w_s' (usually provided by the pickles).
    # If 'w_s' missing, we add fj_bs and plot that instead.
    try:
        if 'w_s' not in bs.columns:
            bs['fj_bs'] = __backscatter_correction(
                bs.f1_ac/bs.f1_dc,
                bs.f2_ac/bs.f2_dc,
                np.unwrap(bs.f1_ph) - np.unwrap(bs.f2_ph),
                bs.fj_fs,
                np.nanmedian(bs.fj_fs),
                cm_filter_factor=1.033,
            )
    except Exception as e:
        # Not fatal—still plot the rest; placeholder only if we can't even draw axes
        print(f"⚠️  backscatter_correction failed: {e}")

    # --- make plot
    try:
        fig = _makeplot(bs, config['tbeg'])
        out = OUT_DIR / "html_backscatter.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✔ backscatter → {out}")
    except Exception as e:
        _placeholder("Plotting failed", str(e))

    return 0

if __name__ == "__main__":
    sys.exit(main())
