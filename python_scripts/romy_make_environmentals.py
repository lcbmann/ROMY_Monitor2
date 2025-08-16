#!/usr/bin/env python3
"""Generate condensed environmental overview (modern style)

Outputs: new_figures/environmentals.png
Refactors: makeplot_environmentals.py (reduced scope: key panels only)
"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Stream
warnings.filterwarnings('ignore')

MOUNT = Path(os.getenv('ROMY_MOUNT','/import/freenas-ffb-01-data')).expanduser()
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO/'docs'/'figures'; OUT_DIR.mkdir(parents=True, exist_ok=True)

# Interval config
RING='Z'
DAYS=10
TEND = UTCDateTime().now()
TBEG = TEND - DAYS*86400

# Imports from functions
from functions.load_backscatter_data import __load_backscatter_data  # type: ignore
from functions.backscatter_correction import __backscatter_correction  # type: ignore
from functions.read_sds import __read_sds  # type: ignore
from functions.load_beam_wander_data_v2 import __load_beam_wander_data  # type: ignore
from functions.find_max_min import __find_max_min  # type: ignore

# Paths
BACK_PATH = MOUNT/'romy_autodata'/'backscatter'
SDS_TEMP  = MOUNT/'temp_archive'
IDS_BASE  = MOUNT/'ids'

CAMS = ['00','03','07']
CONV = {"00":5.3e-3, "01":5.3e-3, "03":5.3e-3, "05":5.3e-3, "07":5.3e-3}
COLS = {"00":"tab:blue","03":"tab:red","07":"tab:purple"}


def load_backscatter():
    try:
        bs = __load_backscatter_data(TBEG, TEND, RING, str(BACK_PATH)+"/")
        bs['time_sec'] = bs.time2 - bs.time1 + (bs.time1 - bs.time1.loc[0])
        bs['fj_bs'] = __backscatter_correction(bs.f1_ac/bs.f1_dc,
                                               bs.f2_ac/bs.f2_dc,
                                               np.unwrap(bs.f1_ph) - np.unwrap(bs.f2_ph),
                                               bs.fj_fs,
                                               np.nanmedian(bs.fj_fs),
                                               cm_filter_factor=1.033)
        return bs
    except Exception as e:
        print('backscatter fail', e)
        import pandas as pd
        return pd.DataFrame()


def load_beam(cam):
    try:
        bw = __load_beam_wander_data(TBEG.date, TEND.date, str(IDS_BASE/f'data{cam}/'), cam, verbose=False)
        bw = bw[(bw.time > TBEG) & (bw.time < TEND)]
        bw['x_um'] = bw.x*CONV[cam]*1e6
        bw['y_um'] = bw.y*CONV[cam]*1e6
        bw = bw[(bw.amp>20)&(bw.amp<255)]
        bw['tsec'] = bw.time - TBEG
        return bw
    except Exception:
        import pandas as pd
        return pd.DataFrame()


def load_pressure():
    try:
        st = Stream()
        st += __read_sds(str(SDS_TEMP), 'BW.FFBI.30.LDF', TBEG, TEND)
        st += __read_sds(str(SDS_TEMP), 'BW.FFBI.30.LDO', TBEG, TEND)
        return st
    except Exception:
        return Stream()


def make_figure(bs, beams, press):
    fig, ax = plt.subplots(5,1, figsize=(11,10), sharex=False); plt.subplots_adjust(hspace=.15)
    ref = TBEG
    # Panel 0: Sagnac raw & corrected
    if not bs.empty:
        t = bs.time_sec/86400
        ax[0].plot(t, bs.fj_fs, color='grey', alpha=.4, lw=.5,label='raw')
        ax[0].plot(t, bs.fj_bs, color='k', lw=.7,label='BS corr')
        fmin,fmax = __find_max_min([bs.fj_bs],99)
        ax[0].set_ylim(fmin-0.001,fmax+0.001)
        ax[0].set_ylabel('δf (Hz)')
        ax[0].legend(fontsize=8)
        ax[0].grid(ls=':',alpha=.5)
    else:
        ax[0].text(.5,.5,'No Sagnac data', ha='center', va='center')

    # Panel 1: Beam wander X
    for cam in CAMS:
        bw = beams.get(cam)
        if bw is None or bw.empty: continue
        ax[1].plot(bw.tsec/86400, bw.x_um, lw=.5, color=COLS[cam], label=f'{cam} X')
    ax[1].set_ylabel('Beam X (µm)'); ax[1].grid(ls=':',alpha=.5)
    ax[1].legend(ncol=3, fontsize=7)

    # Panel 2: Beam wander Y
    for cam in CAMS:
        bw = beams.get(cam)
        if bw is None or bw.empty: continue
        ax[2].plot(bw.tsec/86400, bw.y_um, lw=.5, color=COLS[cam], label=f'{cam} Y')
    ax[2].set_ylabel('Beam Y (µm)'); ax[2].grid(ls=':',alpha=.5)

    # Panel 3: Differential pressure & abs
    try:
        bdf = press.select(channel='*F')[0]
        bdo = press.select(channel='*O')[0]
        ax[3].plot(bdf.times(reftime=ref)/86400, bdf.data, color='tab:red', lw=.6, label='ΔP')
        ax3b = ax[3].twinx(); ax3b.plot(bdo.times(reftime=ref)/86400, bdo.data, color='tab:orange', lw=.5, label='P')
        ax[3].set_ylabel('ΔP (Pa)'); ax3b.set_ylabel('P (hPa)')
        ax[3].legend(fontsize=7)
        ax[3].grid(ls=':',alpha=.5)
    except Exception:
        ax[3].text(.5,.5,'No pressure', ha='center', va='center')

    # Panel 4: Beam scatter cloud (latest cam 00)
    try:
        bw0 = beams.get('00')
        if bw0 is not None and not bw0.empty:
            sc = ax[4].scatter(bw0.x_um, bw0.y_um, c=bw0.tsec, s=3, cmap='viridis')
            ax[4].set_xlabel('X (µm)'); ax[4].set_ylabel('Y (µm)')
            ax[4].set_title('Beam positions cam 00')
    except Exception:
        ax[4].text(.5,.5,'No beam data', ha='center', va='center')
    for a in ax: a.tick_params(labelsize=8)
    fig.suptitle(f'Environmental Overview R{RING} last {DAYS} days', fontsize=14)
    return fig


def main():
    bs = load_backscatter()
    beams = {cam:load_beam(cam) for cam in CAMS}
    press = load_pressure()
    fig = make_figure(bs, beams, press)
    out = OUT_DIR/'html_environmentals.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('✔ saved', out)

if __name__=='__main__':
    main()
