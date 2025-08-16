#!/usr/bin/env python3
"""Generate backscatter diagnostic plot (modern style)

Outputs: new_figures/backscatter_R<ring>.png

Refactored from makeplot_backscatter.py with:
 - CLI args: ring (Z,U,V,W) and optional end date (YYYY-MM-DD)
 - Uses ROMY_MOUNT env var else default server path
 - Headless matplotlib Agg
 - Simplified configuration + error guards
"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from pandas import DataFrame
warnings.filterwarnings("ignore", category=RuntimeWarning)

MOUNT = Path(os.getenv("ROMY_MOUNT", "/import/freenas-ffb-01-data")).expanduser()
REPO  = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2 or sys.argv[1].upper() not in "ZUVW":
    sys.exit("Usage: python romy_make_backscatter.py <Z|U|V|W> [YYYY-MM-DD]")
RING = sys.argv[1].upper()
TEND = (sys.argv[2] if len(sys.argv) > 2 else str(UTCDateTime().now().date))
TEND_UTC = UTCDateTime(TEND)
TIME_INTERVAL_DAYS = 1
TBEG_UTC = TEND_UTC - TIME_INTERVAL_DAYS*86400

# nominal sagnac frequencies (Hz)
RING_F = dict(U=303.05, V=447.5, W=447.5, Z=553.2)[RING]

DATA_PATH = MOUNT/"romy_autodata"/"backscatter"

from functions.load_backscatter_data import __load_backscatter_data  # type: ignore
from functions.backscatter_correction import __backscatter_correction  # type: ignore


def find_min_max(series_list, pp=98):
    mx = []
    mn = []
    for s in series_list:
        arr = np.asarray(s)
        if arr.size == 0: continue
        mx.append(np.nanpercentile(arr, pp))
        mn.append(np.nanpercentile(arr, 100-pp))
    if not mx: return 0,1
    return min(mn), max(mx)


def load_bs():
    try:
        bs = __load_backscatter_data(TBEG_UTC, TEND_UTC, RING, str(DATA_PATH)+"/")
        bs['time_sec'] = bs.time2 - bs.time1 + (bs.time1 - bs.time1.loc[0])
        return bs
    except Exception as e:
        print("✖ backscatter load failed:", e)
        return DataFrame()


def compute_correction(bs):
    try:
        bs['fj_bs'] = __backscatter_correction(bs.f1_ac/bs.f1_dc,
                                               bs.f2_ac/bs.f2_dc,
                                               np.unwrap(bs.f1_phw) - np.unwrap(bs.f2_phw),
                                               bs.fj_fs,
                                               np.nanmedian(bs.fj_fs),
                                               cm_filter_factor=1.033)
    except Exception as e:
        print("✖ backscatter correction failed:", e)
    return bs


def make_figure(bs):
    if bs.empty:
        fig, ax = plt.subplots(figsize=(8,3))
        ax.text(.5,.5,"No data", ha='center', va='center'); ax.axis('off')
        return fig
    from matplotlib.ticker import MultipleLocator
    fig, ax = plt.subplots(5,1, figsize=(12,8), sharex=True)
    plt.subplots_adjust(hspace=0.08)
    t = bs.time_sec/3600.0
    # f1/f2
    ax[0].plot(t, bs.f1_fs, label='f1', color='tab:orange', lw=0.8)
    ax[0].plot(t, bs.f2_fs, label='f2', color='tab:red', lw=0.8)
    ax[0].legend(ncol=2, fontsize=8)
    ymin,ymax = find_min_max([bs.f1_fs, bs.f2_fs])
    ax[0].set_ylabel('Δf (Hz)')
    ax[0].set_ylim(ymin, ymax)
    # fj + corrected
    try:
        ax0b = ax[0].twinx()
        ax0b.plot(t, bs.fj_fs, color='tab:blue', lw=0.6, label='fj')
        if 'fj_bs' in bs:
            ax0b.plot(t, bs.fj_bs, color='k', lw=0.6, ls='--', label='fj_bs')
        ax0b.set_ylabel('Δf (Hz)', color='tab:blue')
    except Exception:
        pass

    # AC
    ax[1].plot(t, bs.f1_ac*1e3, color='tab:orange', lw=0.6, label='f1_ac')
    ax[1].plot(t, bs.f2_ac*1e3, color='tab:red', lw=0.6, label='f2_ac')
    ax[1].set_ylabel('AC (mV)')
    ymin,ymax = find_min_max([bs.f1_ac*1e3, bs.f2_ac*1e3], pp=99.5)
    ax[1].set_ylim(ymin, ymax)

    # DC
    ax[2].plot(t, bs.f1_dc*1e3, color='tab:orange', lw=0.6, label='f1_dc')
    ax[2].plot(t, bs.f2_dc*1e3, color='tab:red', lw=0.6, label='f2_dc')
    ax[2].set_ylabel('DC (mV)')
    ymin,ymax = find_min_max([bs.f1_dc*1e3, bs.f2_dc*1e3], pp=99.5)
    ax[2].set_ylim(ymin, ymax)

    # AC/DC
    ax[3].plot(t, bs.f1_ac/bs.f1_dc*1e3, color='tab:orange', lw=0.6)
    ax[3].plot(t, bs.f2_ac/bs.f2_dc*1e3, color='tab:red', lw=0.6)
    ax[3].set_ylabel('AC/DC')

    # Phase diff
    try:
        ax[4].plot(t, bs.f1_ph - bs.f2_ph, color='tab:purple', lw=0.5)
        ax[4].set_ylabel('ΔPhase (rad)')
    except Exception:
        pass

    for a in ax:
        a.grid(ls=':', alpha=.6)
        a.set_xlim(0,24)
        a.xaxis.set_major_locator(MultipleLocator(4))
        a.xaxis.set_minor_locator(MultipleLocator(1))
    ax[-1].set_xlabel(f'Time (h) since {TBegDate()} UTC')
    fig.suptitle(f'Backscatter R{RING} – {TEND}', fontsize=14)
    return fig


def TBegDate():
    return f"{TBEG_UTC.date} {str(TBEG_UTC.time)[:8]}"


def main():
    bs = load_bs()
    if not bs.empty:
        bs = compute_correction(bs)
    fig = make_figure(bs)
    out = OUT_DIR / "html_backscatter.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('✔ saved', out)

if __name__ == '__main__':
    main()
