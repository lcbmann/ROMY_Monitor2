#!/usr/bin/env python3
"""Generate daily raw Sagnac signal plot per ring

Legacy filenames: docs/figures/html_sagnacsignal_R<ring>.png
Assumes seed BW.DROMY..FJ<RING> one-day data; shows 24h time series (frequency proxy counts to Hz not converted here if unknown scaling).
"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Stream
from datetime import date, timedelta
warnings.filterwarnings('ignore')

MOUNT = Path(os.getenv('ROMY_MOUNT','/import/freenas-ffb-01-data')).expanduser()
REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO/'new_figures'; FIG_DIR.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2 or sys.argv[1].upper() not in 'ZUVW':
    sys.exit('Usage: python romy_make_sagnacsignal.py <Z|U|V|W> [YYYY-MM-DD]')
RING = sys.argv[1].upper()
if len(sys.argv) > 2:
    RUN_DATE = sys.argv[2]
else:
    yesterday = date.today() - timedelta(days=1)
    RUN_DATE = yesterday.isoformat()

from obspy.clients.filesystem.sds import Client

def read_day(seed, day):
    n,s,l,c = seed.split('.')
    cli = Client(str(MOUNT/'romy_archive'))
    t1 = UTCDateTime(day)
    t2 = t1 + 86400
    try:
        return cli.get_waveforms(n,s,l,c,t1,t2,merge=-1)
    except Exception:
        return Stream()

seed = f'BW.DROMY..FJ{RING}'
st = read_day(seed, RUN_DATE)
if not st:
    print('✖ no data – writing placeholder image')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15,4))
    ax.text(0.5,0.55,f"R{RING} Sagnac signal\n{RUN_DATE}",ha='center',va='center',fontsize=16,weight='bold')
    ax.text(0.5,0.25,'No data available',ha='center',va='center',fontsize=12,color='crimson')
    ax.axis('off')
    out = FIG_DIR/f'html_sagnacsignal_R{RING}.png'
    fig.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig)
    sys.exit(0)
tr = st[0]; tr.trim(UTCDateTime(RUN_DATE), UTCDateTime(RUN_DATE)+86400)

# Downsample for plotting if large
if tr.stats.sampling_rate > 5:
    dec = int(tr.stats.sampling_rate // 5) or 1
    if dec>1:
        tr.decimate(dec, no_filter=True)

t = np.linspace(0,24,len(tr.data))
fig, ax = plt.subplots(figsize=(15,4))

# Use a more visible color scheme based on the ring
ring_colors = {
    'Z': 'tab:orange',
    'U': 'deeppink',
    'V': 'tab:blue', 
    'W': 'darkblue'
}
line_color = ring_colors.get(RING, 'tab:blue')  # Default to blue if ring not in dict

ax.plot(t, tr.data, lw=0.6, color=line_color)
ax.set_xlim(0,24)
ax.set_xlabel('Hours UTC', fontsize=11)
ax.set_ylabel('Counts', fontsize=11)
ax.set_title(f'Sagnac signal R{RING} – {RUN_DATE}', fontsize=13)
ax.grid(ls=':',alpha=.6)
ax.tick_params(labelsize=10)

# Add background to improve readability
ax.set_facecolor('#f8f9fa')
fig.tight_layout()
out = FIG_DIR/f'html_sagnacsignal_R{RING}.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print('✔ saved', out)
