#!/usr/bin/env python3
"""Generate combined Rotation Spectra + Helicorder figure per ring

Output filename (legacy‑style for web inclusion):
    html_rotationcombo_R<ring>.png  (placed in new_figures/)

Invocation:
    python romy_make_rotation_combo.py <Z|U|V|W> [YYYY-MM-DD]

If the required *spectra* helper cannot be imported or no data is found, a
placeholder image is produced instead so downstream pages never break.
"""
from __future__ import annotations
import sys, os, warnings
from pathlib import Path
from datetime import date, timedelta
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obspy import UTCDateTime, Stream

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR   = REPO_ROOT / 'new_figures'; FIG_DIR.mkdir(exist_ok=True)
MOUNT     = Path(os.getenv('ROMY_MOUNT', '/import/freenas-ffb-01-data')).expanduser()
SDS_ARCH  = MOUNT / 'romy_archive'

if len(sys.argv) < 2 or sys.argv[1].upper() not in 'ZUVW':
    sys.exit('Usage: python romy_make_rotation_combo.py <Z|U|V|W> [YYYY-MM-DD]')
RING = sys.argv[1].upper()
RUN_DATE = (sys.argv[2] if len(sys.argv) > 2 else str(date.today() - timedelta(days=1)))

sys.path.append(str(REPO_ROOT / 'develop' / 'spectra'))
try:
    from spectra import spectra  # noqa
except Exception as e:  # fallback placeholder
    print(f"✖ Could not import spectra module: {e}")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.text(0.5,0.55,f"R{RING} Rotation+Helicorder",ha='center',va='center',fontsize=16,weight='bold')
    ax.text(0.5,0.30, RUN_DATE, ha='center', va='center', fontsize=12)
    ax.text(0.5,0.12,'Dependency missing',ha='center',va='center',fontsize=11,color='crimson')
    ax.axis('off')
    out = FIG_DIR / f'html_rotationcombo_R{RING}.png'
    fig.savefig(out,dpi=140,bbox_inches='tight'); plt.close(fig)
    sys.exit(0)

def read_sds(archive: Path, seed: str, t1, t2) -> Stream:
    from obspy.clients.filesystem.sds import Client
    if not archive.is_dir():
        return Stream()
    n,s,l,c = seed.split('.')
    try:
        cli = Client(str(archive), sds_type='D')
        return cli.get_waveforms(n,s,l,c, UTCDateTime(t1), UTCDateTime(t2), merge=-1)
    except Exception:
        return Stream()

def spec_cfg(ring: str, run_date: str):
    loc = dict(Z='10', U='', V='', W='')[ring]
    seed = f"BW.ROMY.{loc}.BJ{ring}"
    return {
        'tbeg': f"{run_date} 00:00:00",
        'tend': f"{run_date} 23:59:59",
        'seeds': [seed],
        'path_to_sds': str(SDS_ARCH),
        'path_to_data_out':   str(REPO_ROOT / 'local_output' / 'rotation_data'),
        'path_to_figures_out': str(REPO_ROOT / 'local_output' / 'rotation_plots'),
        'tinterval': 3600,
        'toverlap': 0,
        'method': 'welch',
        'fmin': 0.001,
        'fmax': 5.0,
        'apply_average': True,
        'fraction_of_octave': 12,
        'averaging': 'median',
        'data_unit': 'rad/s',
        'quality_filter': 'good',
        'threshold': 1e-15,
        'zero_seq_limit': 20,
        'high_seq_limit': 20,
        'flat_seq_limit': 20,
        'verbose': False,
        'remove_response': False,
    }

def main():
    print(f"Generating rotation+helicorder combo for R{RING} {RUN_DATE}")
    cfg = spec_cfg(RING, RUN_DATE)
    sp = spectra(cfg)
    seed = cfg['seeds'][0]
    big_stream = Stream()
    for hr in range(24):
        t1 = UTCDateTime(RUN_DATE) + hr*3600
        t2 = t1 + 3600
        st = read_sds(SDS_ARCH, seed, t1-30, t2+30)
        if st:
            big_stream += st
    if not big_stream:
        print('✖ No data found – writing placeholder')
        fig, ax = plt.subplots(figsize=(10,5))
        ax.text(0.5,0.55,f"R{RING} Rotation+Helicorder",ha='center',va='center',fontsize=16,weight='bold')
        ax.text(0.5,0.30, RUN_DATE, ha='center', va='center', fontsize=12)
        ax.text(0.5,0.12,'No data available',ha='center',va='center',fontsize=11,color='crimson')
        ax.axis('off')
        out = FIG_DIR / f'html_rotationcombo_R{RING}.png'
        fig.savefig(out,dpi=140,bbox_inches='tight'); plt.close(fig)
        return
    big_stream.merge(method=1, fill_value='interpolate')
    sp.add_trace(big_stream[0])
    sp.get_collection(cfg['tinterval'], cfg['toverlap'], cfg['method'])
    if cfg['apply_average']:
        sp.get_fband_average(cfg['fraction_of_octave'], cfg['averaging'])
    if cfg['quality_filter']:
        sp.classify_collection_quality(threshold=cfg['threshold'],
                                       zero_seq_limit=cfg['zero_seq_limit'],
                                       high_seq_limit=cfg['high_seq_limit'],
                                       flat_seq_limit=cfg['flat_seq_limit'])
    out_png = FIG_DIR / f'html_rotationcombo_R{RING}.png'
    try:
        # Use rainbow (same as legacy make_spectra config) and disable quality filtering
        # so intervals retain colour even if classified 'bad' (grey only for true gaps).
        sp.plot_spectra_and_helicorder(fmin=cfg['fmin'], fmax=cfg['fmax'],
                                       cmap='rainbow', alpha=0.9,
                                       data_unit=cfg['data_unit'],
                                       quality_filter=None,
                                       savefig=str(out_png), show=False, out=False)
        print('✔ saved', out_png)
    except Exception as e:
        print(f"✖ Plot failed: {e}")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.text(0.5,0.55,f"R{RING} Rotation+Helicorder",ha='center',va='center',fontsize=16,weight='bold')
        ax.text(0.5,0.30, RUN_DATE, ha='center', va='center', fontsize=12)
        ax.text(0.5,0.12,'Plot error',ha='center',va='center',fontsize=11,color='crimson')
        ax.axis('off')
        fig.savefig(out_png,dpi=140,bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    main()
