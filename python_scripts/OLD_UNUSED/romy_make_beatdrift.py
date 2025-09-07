#!/usr/bin/env python3
"""Generate Sagnac beat drift multi-ring figure (modern style)

Outputs: new_figures/beatdrift.png
Refactored from makeplot_sagnacdrift_v2.py
"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Stream
warnings.filterwarnings('ignore')

MOUNT = Path(os.getenv('ROMY_MOUNT', '/import/freenas-ffb-01-data')).expanduser()
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO/'docs'/'figures'; OUT_DIR.mkdir(parents=True, exist_ok=True)
TIME_DAYS = 14
TEND = UTCDateTime().now()
TBEG = TEND - TIME_DAYS*86400

SDS = MOUNT/'temp_archive'

from functions.read_sds import __read_sds  # type: ignore
from functions.find_max_min import __find_max_min  # type: ignore
from functions.load_lxx import __load_lxx  # type: ignore
from functions.get_lxx_intervals import __get_lxx_intervals  # type: ignore

BOUNDS = dict(Z=(553.20,553.80), U=(302.45,302.60), V=(447.65,447.90), W=(447.65,447.90))
COLORS = dict(Z='tab:orange', U='deeppink', V='tab:blue', W='darkblue')
SEEDS = dict(Z='BW.ROMY.XX.LJZ', U='BW.ROMY.XX.LJU', V='BW.ROMY.XX.LJV', W='BW.ROMY.XX.LJW')


def load(seed):
    try:
        return __read_sds(str(SDS), seed, TBEG, TEND)
    except Exception:
        return Stream()


def load_lxx():
    try:
        lxx = __load_lxx(TBEG, TEND, str(MOUNT))
        return __get_lxx_intervals(lxx.datetime)
    except Exception:
        return [],[]


def make_figure(traces, lxx_t1, lxx_t2):
    fig, ax = plt.subplots(4,1, figsize=(10,7), sharex=True); plt.subplots_adjust(hspace=.08)
    ref = TBEG; scale=1
    for i, ring in enumerate(['Z','U','V','W']):
        tr = traces.get(ring)
        a = ax[i]
        if tr:
            a.plot(tr[0].times(reftime=ref)*scale, tr[0].data, color=COLORS[ring], lw=.8)
            lo, hi = BOUNDS[ring]
            try:
                dmin,dmax = __find_max_min([tr[0].data], 99)
                a.set_ylim(max(lo,dmin), min(hi,dmax))
            except Exception:
                a.set_ylim(lo,hi)
        else:
            a.text(.5,.5,'No data', ha='center', va='center')
        a.set_ylabel(f'R{ring}\n(Hz)')
        a.grid(ls=':', alpha=.5)
        # maintenance shading
        for t1,t2 in zip(lxx_t1,lxx_t2):
            a.fill_betweenx(a.get_ylim(), (t1-ref), (t2-ref), color='yellow', alpha=.3)
    # ticks: daily
    total = (TEND-TBEG)
    day = 86400
    days = int(total/day)+1
    positions = [d*day for d in range(days)]
    labels = [(ref + d*day).strftime('%Y-%m-%d\n%H:%M:%S') for d in range(days)]
    ax[-1].set_xticks(positions); ax[-1].set_xticklabels(labels, rotation=0)
    fig.suptitle(f'Sagnac Beat Drift last {TIME_DAYS} days', fontsize=14)
    return fig


def main():
    traces = {r:load(SEEDS[r]) for r in 'ZUVW'}
    l1,l2 = load_lxx()
    fig = make_figure(traces, l1,l2)
    out = OUT_DIR/'html_beatdrift.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('✔ saved', out)

if __name__=='__main__':
    main()
