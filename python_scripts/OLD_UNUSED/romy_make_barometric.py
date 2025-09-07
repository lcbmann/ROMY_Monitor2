#!/usr/bin/env python3
"""Generate barometric & rotation correlation plot (modern style).

Output: docs/figures/new/barometric.png
Simplified from makeplot_baro.py focusing on pressure + rotation & regression residuals.
"""
import os, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Stream, read_inventory
warnings.filterwarnings('ignore')

MOUNT = Path(os.getenv('ROMY_MOUNT','/import/freenas-ffb-01-data')).expanduser()
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO/'docs'/'figures'; OUT_DIR.mkdir(parents=True, exist_ok=True)

# Time window: last 3h ending yesterday (to allow data completeness)
T2 = UTCDateTime().now() - 86400
TOFF = 3*3600
T1 = T2 - TOFF
TBEG = T1 - TOFF
TEND = T2 + TOFF

SDS_TEMP = MOUNT/'temp_archive'
STXML = MOUNT/'stationxml_ringlaser'

from functions.read_sds import __read_sds  # type: ignore
from functions.regressions import regressions  # type: ignore

COUNTS_REFTEK = 6.28099e5


def load_stream():
    st = Stream()
    # rotation (BJZ, BJN, BJE) optional
    for ch in ['BJZ','BJN','BJE']:
        try:
            st += __read_sds(str(SDS_TEMP), f'BW.ROMY.00.{ch}', TBEG, TEND)
        except Exception:
            pass
    # pressure FFBI
    try:
        inv = read_inventory(str(STXML/'station_BW_FFBI.xml'))
        st += __read_sds(str(SDS_TEMP), 'BW.FFBI..BDO', TBEG, TEND)
        st += __read_sds(str(SDS_TEMP), 'BW.FFBI..BDF', TBEG, TEND)
        st.merge()
        for tr in st.select(station='FFBI'):
            if tr.stats.channel.endswith('DO'):
                tr.data = tr.data/1.0/COUNTS_REFTEK/1e-5  # counts->Pa (approx)
        # Hilbert of BDO
        try:
            bdo = st.select(channel='BDO')[0]
            from scipy.signal import hilbert
            h = np.imag(hilbert(bdo.data))
            from obspy import Trace
            trh = Trace(data=h, header=bdo.stats)
            trh.stats.channel = 'BDH'
            st += trh
        except Exception:
            pass
    except Exception:
        pass
    st.merge()
    # detrend/filter
    for tr in st:
        try:
            tr.detrend('demean').detrend('linear').taper(0.05)
        except Exception:
            pass
    return st.trim(T1, T2)


def regress_all(st):
    out = {}
    try:
        bdo = st.select(channel='BDO')[0]
        bdh = st.select(channel='BDH')[0]
    except Exception:
        return out
    time = bdo.times()
    for tr in st.select(channel='BJ*'):
        import pandas as pd
        df = pd.DataFrame({'pressure':bdo.data, 'hilbert':bdh.data, 'rot':tr.data})
        try:
            reg = regressions(df, features=['pressure','hilbert'], target='rot', reg='ransac', verbose=False)
            model = np.array(reg['dp'])
            resid = tr.data - model
            vr = (1 - np.var(resid)/np.var(tr.data))*100
            out[tr.stats.channel] = dict(time=time, data=tr.data, model=model, resid=resid, vr=vr)
        except Exception:
            continue
    return out


def make_figure(st, regs):
    fig, ax = plt.subplots(9,1, figsize=(14,10), sharex=True); plt.subplots_adjust(hspace=.08)
    tscale = 1/3600
    try:
        bdx = st.select(channel='BDO')[0]
        ax[0].plot(bdx.times()*tscale, bdx.data, color='k', lw=.8, label='BDO (Pa)')
        ax[0].legend(fontsize=8); ax[0].set_ylabel('P (Pa)')
    except Exception:
        ax[0].text(.5,.5,'No BDO', ha='center', va='center')
    try:
        bdh = st.select(channel='BDH')[0]
        ax[1].plot(bdh.times()*tscale, bdh.data, color='k', lw=.8, label='Hilbert(P)')
        ax[1].legend(fontsize=8); ax[1].set_ylabel('Hilb P')
    except Exception:
        ax[1].text(.5,.5,'No BDH', ha='center', va='center')
    comps = ['BJZ','BJN','BJE']
    for i,ch in enumerate(comps):
        try:
            r = regs[ch]
            ax[2+i].plot(r['time']*tscale, r['data']*1e9, color='k', lw=.7, label=f'{ch}')
            ax[2+i].plot(r['time']*tscale, r['resid']*1e9, color='red', ls='--', lw=.6, label=f'Residual (VR={r["vr"]:.1f}%)')
            ax[2+i].set_ylabel('ω̇ (nrad)')
            ax[2+i].legend(fontsize=7)
        except Exception:
            ax[2+i].text(.5,.5,f'No {ch}', ha='center', va='center')
    # placeholders for tilt slots to keep layout similar
    for j in range(3):
        ax[5+j].axis('off')
    ax[-1].set_xlabel(f'Time (h) from {str(TBEG).split(".")[0]} UTC')
    for a in ax: a.grid(ls=':', alpha=.4)
    fig.suptitle('Barometric – 3h window (pressure vs rotation)', fontsize=14)
    return fig


def main():
    st = load_stream()
    regs = regress_all(st)
    fig = make_figure(st, regs)
    out = OUT_DIR/'html_romy_baro.png'
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print('✔ saved', out)

if __name__=='__main__':
    main()
