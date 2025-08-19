#!/usr/bin/env python3
"""Generate helicorder images specifically for the beam wander analysis page

This script ensures that the beam wander analysis page has the necessary helicorder/rotation PSD images
by creating them if they're missing from the standard rotation and helicorder generation process.
"""

import sys, os, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "new_figures"; FIG_DIR.mkdir(exist_ok=True)
DOCS_FIGS = REPO_ROOT / "docs" / "figures"; DOCS_FIGS.mkdir(exist_ok=True)
DOCS_FIGS_NEW = DOCS_FIGS / "new"; DOCS_FIGS_NEW.mkdir(exist_ok=True)

def ensure_beam_wander_images():
    """Ensure all necessary beam wander page images are present in docs/figures"""
    
    # Static images expected by beam-wander.html
    static_images = [
        'BW_ROMY_10_BJZ.png',
        'BW_ROMY__BJU.png',
        'BW_ROMY__BJV.png',
        'BW_ROMY__BJW.png'
    ]
    
    # Create copies in docs/figures/new if they don't exist
    for img_name in static_images:
        target_file = DOCS_FIGS_NEW / img_name
        if not target_file.exists():
            source_file = DOCS_FIGS / img_name
            if not source_file.exists():
                source_file = REPO_ROOT / "figures" / img_name
            
            if source_file.exists():
                # Copy the file using Python to avoid shell command issues
                try:
                    with open(source_file, 'rb') as src, open(target_file, 'wb') as dst:
                        dst.write(src.read())
                    print(f"✓ Copied {img_name} to docs/figures/new")
                except Exception as e:
                    print(f"✖ Failed to copy {img_name}: {e}")
            else:
                # Create a placeholder
                try:
                    fig, ax = plt.subplots(figsize=(8,6))
                    ax.text(0.5, 0.6, f"Beam Position {img_name.split('__')[-1].split('.')[0]}", 
                           ha='center', va='center', fontsize=14, weight='bold')
                    ax.text(0.5, 0.4, "(Source image not available)", 
                           ha='center', va='center', fontsize=12, color='crimson')
                    ax.axis('off')
                    fig.savefig(target_file, dpi=110, bbox_inches='tight')
                    plt.close(fig)
                    print(f"✓ Created placeholder for {img_name}")
                except Exception as e:
                    print(f"✖ Failed to create placeholder for {img_name}: {e}")

if __name__ == "__main__":
    print("Ensuring beam wander analysis images are available...")
    ensure_beam_wander_images()
    print("Done!")
