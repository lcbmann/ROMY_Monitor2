#!/usr/bin/env python3
"""Generate raw Sagnac signal panel for all rings.

Output: docs/figures/new/sagnac_signals.png
Shows 24h (yesterday) FJ* raw counts (converted to Hz approx via derivative if needed?)
Simplified: just plots raw beat frequency channels if present.
"""
import os, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Stream
warnings.filterwarnings('ignore')

MOUNT = Path(os.getenv('ROMY_MOUNT','/import/freenas-ffb-01-data')).expanduser()
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO/'docs'/'figures'/'new'; OUT_DIR.mkdir(parents=True, exist_ok=True)

YDAY = str((UTCDateTime().now() - 86400).date)
T1 = UTCDateTime(f"{YDAY} 00:00:00")
T2 = T1 + 24*3600

from functions.read_sds import __read_sds  # type: ignore

RINGS = 'ZUVW'
SEEDS = {r:f'BW.DROMY..FJ{r}' for r in RINGS}
SDS = MOUNT/'romy_archive'


def load(seed):
    try:
        return __read_sds(str(SDS), seed, T1, T2)
    except Exception:
        return Stream()


def make_figure(traces):
    fig, ax = plt.subplots(len(RINGS),1, figsize=(12,8), sharex=True); plt.subplots_adjust(hspace=.05)
    for i,r in enumerate(RINGS):
        tr = traces.get(r)
        a = ax[i]
        if tr:
            t = tr[0].times(reftime=T1)/3600
            a.plot(t, tr[0].data, lw=.5, color='k')
            a.set_ylabel(f'R{r}\n(counts)')
        else:
            a.text(.5,.5,'No data', ha='center', va='center')
        a.grid(ls=':', alpha=.4)
    ax[-1].set_xlabel('Time (h) UTC')
    fig.suptitle(f'Raw Sagnac Beat Signals – {YDAY}', fontsize=14)
    return fig


def main():
    traces = {r:load(SEEDS[r]) for r in RINGS}
    fig = make_figure(traces)
    out = OUT_DIR/'sagnac_signals.png'
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print('✔ saved', out)

if __name__=='__main__':
    main()
