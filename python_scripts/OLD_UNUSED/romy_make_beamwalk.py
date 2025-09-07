#!/usr/bin/env python3
"""Generate beam walk composite figure (modern style)

Outputs: new_figures/beamwalk.png
Refactors makeplot_beamwalk.py (reduced to essentials; keeps scatter & latest images)
"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy import UTCDateTime
warnings.filterwarnings('ignore')

MOUNT = Path(os.getenv('ROMY_MOUNT', '/import/freenas-ffb-01-data')).expanduser()
REPO  = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'docs' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config
CAMS = ['00','03','07','05']
TIME_INTERVAL_DAYS = 14
LAST_RESET = UTCDateTime('2025-07-03 00:00')

archive_path = MOUNT
IDS_PATH = archive_path/'ids'

from functions.load_beam_wander_data_v2 import __load_beam_wander_data  # type: ignore
from functions.find_max_min import __find_max_min  # type: ignore

CONV = {"00":5.3e-3, "01":5.3e-3, "03":5.3e-3, "05":5.3e-3, "07":5.3e-3}
COLORS = {"00":"tab:blue","01":"tab:orange","03":"tab:red","05":"tab:green","07":"tab:purple"}

TEND = UTCDateTime().now()
TBEG = TEND - TIME_INTERVAL_DAYS*86400 if (TEND-LAST_RESET) > TIME_INTERVAL_DAYS*86400 else LAST_RESET

def latest_png(path_pattern):
    from glob import glob
    try:
        files = glob(path_pattern)
        if not files: return None, None
        latest = max(files, key=os.path.getmtime)
        base = os.path.basename(latest).split('_')
        dt = f"{base[0][:4]}-{base[0][4:6]}-{base[0][6:8]} {base[1][:2]}:{base[1][2:4]}:{base[1][4:6]} UTC"
        return latest, dt
    except Exception:
        return None, None


def load_cam(cam):
    try:
        bw = __load_beam_wander_data(TBEG.date, TEND.date, str(IDS_PATH/f'data{cam}/'), cam)
        bw = bw[(bw.time > TBEG) & (bw.time < TEND)]
        # conversion
        bw['x_mm'] = bw.x*CONV[cam]
        bw['y_mm'] = bw.y*CONV[cam]
        bw = bw[bw.amp > 20]
        bw = bw[bw.amp < 255]
        bw['time_utc'] = bw.time
        bw['time_sec'] = bw.time_utc - TBEG
        return bw
    except Exception as e:
        print('cam', cam, 'load fail', e)
        import pandas as pd
        return pd.DataFrame()


def make_figure(data):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16,9))
    gs = GridSpec(8,8,figure=fig,hspace=.3)
    ax0 = fig.add_subplot(gs[0:4,0:4])
    ax1 = fig.add_subplot(gs[4:8,0:4])
    scatter_axes = [fig.add_subplot(gs[i*2:(i+1)*2,4:6]) for i in range(4)]
    image_axes   = [fig.add_subplot(gs[i*2:(i+1)*2,6:8]) for i in range(4)]

    # Y positions
    ydata=[]
    for cam in CAMS:
        df = data.get(cam)
        if df is None or df.empty: continue
        mask_ok = df.amp < 255
        ax0.scatter(df.loc[mask_ok,'time_sec']/86400, df.loc[mask_ok,'y_mm']*1e3, s=2, color=COLORS[cam], label=f'IDS-{cam}')
        ydata.append(df.y_mm.values*1e3)
    if ydata:
        ymin,ymax = __find_max_min(ydata, pp=99)
        ax0.set_ylim(ymin,ymax)
    ax0.set_ylabel('rel Y (µm)'); ax0.grid(ls=':', alpha=.5); ax0.legend(ncol=4, markerscale=3, fontsize=8)

    # X positions
    xdata=[]
    for cam in CAMS:
        df = data.get(cam)
        if df is None or df.empty: continue
        mask_ok = df.amp < 255
        ax1.scatter(df.loc[mask_ok,'time_sec']/86400, df.loc[mask_ok,'x_mm']*1e3, s=2, color=COLORS[cam], label=f'IDS-{cam}')
        xdata.append(df.x_mm.values*1e3)
    if xdata:
        xmin,xmax = __find_max_min(xdata, pp=99)
        ax1.set_ylim(xmin,xmax)
    ax1.set_ylabel('rel X (µm)'); ax1.set_xlabel(f'Time (days) since {TBEG.date}'); ax1.grid(ls=':', alpha=.5)

    # Scatter clouds + latest images
    for idx, cam in enumerate(CAMS):
        if idx >=4: break
        df = data.get(cam)
        axS = scatter_axes[idx]; axI = image_axes[idx]
        if df is not None and not df.empty:
            sc = axS.scatter(df.x_mm*1e3, df.y_mm*1e3, c=df.time_sec, s=2, cmap='viridis')
            axS.set_title(f'IDS-{cam} path')
            axS.set_xticks([]); axS.set_yticks([]); axS.grid(ls=':', alpha=.3)
        img, dt = latest_png(str(IDS_PATH/f"data{cam}/outfigs/*.png"))
        if img:
            try:
                axI.imshow(plt.imread(img)); axI.set_title(f'IDS-{cam} {dt}', fontsize=8)
            except Exception:
                pass
        axI.axis('off')

    fig.suptitle(f'Beam Walk (reset {LAST_RESET.date} UTC)', fontsize=14)
    return fig


def main():
    data={cam:load_cam(cam) for cam in CAMS}
    fig=make_figure(data)
    out = OUT_DIR/ 'html_beamwalk.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('✔ saved', out)

if __name__=='__main__':
    main()
