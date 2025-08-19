#!/usr/bin/env python3
"""Generate combined sagnac signals plots for all rings

This script creates a single combined plot showing all rings' sagnac signals,
ensuring consistent ordering (Z,U,V,W) and color coding.
"""

import sys, warnings, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Stream
from datetime import date, timedelta

# Silence warnings
warnings.filterwarnings('ignore')

# Paths
MOUNT = Path(os.getenv('ROMY_MOUNT','/import/freenas-ffb-01-data')).expanduser()
REPO = Path(__file__).resolve().parents[1] 
FIG_DIR = REPO/'new_figures'; FIG_DIR.mkdir(parents=True, exist_ok=True)

# Get date parameter
if len(sys.argv) > 1:
    RUN_DATE = sys.argv[1]
else:
    yesterday = date.today() - timedelta(days=1)
    RUN_DATE = yesterday.isoformat()

# Define rings in consistent order (Z,U,V,W)
RINGS = ['Z', 'U', 'V', 'W']

# Ring colors (consistent with other scripts)
RING_COLORS = {
    'Z': 'tab:orange',
    'U': 'deeppink',
    'V': 'tab:blue', 
    'W': 'darkblue'
}

def read_sds(seed, day):
    """Read data from SDS archive for a specific day"""
    from obspy.clients.filesystem.sds import Client
    n,s,l,c = seed.split('.')
    cli = Client(str(MOUNT/'romy_archive'))
    t1 = UTCDateTime(day)
    t2 = t1 + 86400
    try:
        return cli.get_waveforms(n,s,l,c,t1,t2,merge=-1)
    except Exception:
        return Stream()

def main():
    """Main execution function"""
    print(f"Generating combined Sagnac signals plot for {RUN_DATE}")
    
    # Create figure with subplots for each ring
    fig, axes = plt.subplots(len(RINGS), 1, figsize=(15, 10), sharex=True)
    plt.subplots_adjust(hspace=0.2)
    
    # For each ring (in consistent order)
    for i, ring in enumerate(RINGS):
        ax = axes[i]
        
        seed = f'BW.DROMY..FJ{ring}'
        st = read_sds(seed, RUN_DATE)
        
        # Get color for this ring
        ring_color = RING_COLORS.get(ring)
        
        if st:
            tr = st[0]
            tr.trim(UTCDateTime(RUN_DATE), UTCDateTime(RUN_DATE)+86400)
            
            # Downsample for plotting if large
            if tr.stats.sampling_rate > 5:
                dec = int(tr.stats.sampling_rate // 5) or 1
                if dec > 1:
                    tr.decimate(dec, no_filter=True)
            
            t = np.linspace(0, 24, len(tr.data))
            ax.plot(t, tr.data, lw=0.8, color=ring_color)
            ax.set_ylabel(f'R{ring}\nCounts')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, color='crimson')
        
        # Formatting
        ax.grid(ls=':', alpha=0.6)
        ax.set_title(f'Ring {ring}', fontweight='bold', loc='left')
        ax.set_facecolor('#f8f9fa')  # Light gray background for readability
        
        # Only bottom plot gets x-axis label
        if i == len(RINGS) - 1:
            ax.set_xlabel('Hours UTC')
    
    # Overall title
    fig.suptitle(f'Sagnac Signals - {RUN_DATE}', fontsize=16)
    
    # Save figure
    out = FIG_DIR/'html_sagnac_signals_combined.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('✓ saved', out)

if __name__ == "__main__":
    main()
