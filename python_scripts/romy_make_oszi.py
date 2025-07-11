#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – Oscilloscope Screenshot Capture
======================================

Captures screenshots from RIGOL DS1054Z Oscilloscope and produces

    <repo>/figures/html_oszi.png
"""

# ─────────── std-lib / 3-party ───────────────────────────────────────────
import os, sys, warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # head-less rendering
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

# Oscilloscope library
try:
    from ds1054z import DS1054Z
except ImportError:
    print("Error: ds1054z library not found. Install with: pip install ds1054z")
    sys.exit(1)

warnings.filterwarnings("ignore")
plt.rcParams["agg.path.chunksize"] = 10_000

# ─────────── User-specific paths ─────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR   = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ─────────── Configuration ───────────────────────────────────────────────
CFG = dict(
    scope_ip     = "10.153.82.107",     # RIGOL DS1054Z IP address
    probe_ratio  = 1,                   # probe ratio setting
    dpi          = 150,                 # output image DPI
    temp_file    = "tmp_oszi.bmp",      # temporary file name
    timeout      = 10,                  # connection timeout in seconds
)

# ─────────── Helper: capture oscilloscope screenshot ────────────────────
def capture_scope_screenshot():
    """Capture screenshot from RIGOL DS1054Z oscilloscope."""
    temp_file = FIG_DIR / CFG["temp_file"]
    
    try:
        # Initialize scope connection
        scope = DS1054Z(CFG["scope_ip"])
        print(f"✔ Connected to oscilloscope at {CFG['scope_ip']}")
    except Exception as e:
        print(f"✖ Failed to connect to oscilloscope: {e}")
        return None
    
    try:
        # Stop scope for screenshot
        scope.stop()
        scope.set_probe_ratio(1, CFG["probe_ratio"])
        
        # Capture screenshot
        bmap_scope = scope.display_data
        
        # Save temporary BMP file
        with open(temp_file, "wb") as f:
            f.write(bmap_scope)
        
        # Restart scope
        scope.run()
        print("✔ Screenshot captured successfully")
        return temp_file
        
    except Exception as e:
        print(f"✖ Failed to capture screenshot: {e}")
        try:
            scope.run()  # Try to restart scope
        except:
            pass
        return None

# ─────────── Helper: create figure with timestamp ───────────────────────
def make_figure(image_path):
    """Create matplotlib figure with oscilloscope image and timestamp."""
    try:
        # Load image
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(image_path).convert('RGBA')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        dt = datetime.utcnow()
        
        ax.imshow(img)
        ax.set_title(f"ROMY Oscilloscope – {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC", 
                     fontsize=14, pad=20)
        ax.axis("off")
        
        fig.tight_layout()
        return fig
        
    except Exception as e:
        print(f"✖ Failed to create figure: {e}")
        return None

# ═══════════════════════════ main routine ════════════════════════════════
def main():
    """Main execution function."""
    print(f"Starting oscilloscope capture from {CFG['scope_ip']}")
    
    # Capture screenshot
    temp_file = capture_scope_screenshot()
    if not temp_file:
        return
    
    # Create figure
    fig = make_figure(temp_file)
    if not fig:
        return
    
    # Save figure
    try:
        png_name = FIG_DIR / "html_oszi.png"
        fig.savefig(png_name, dpi=CFG["dpi"], bbox_inches="tight", 
                   facecolor='white', edgecolor='none')
        print(f"✔ saved figure → {png_name}")
    except PermissionError:
        png_name = REPO_ROOT / "local_output" / "html_oszi.png"
        png_name.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_name, dpi=CFG["dpi"], bbox_inches="tight",
                   facecolor='white', edgecolor='none')
        print(f"✔ saved figure → {png_name} (fallback)")
    except Exception as e:
        print(f"✖ Failed to save figure: {e}")
    
    plt.close(fig)
    
    # Clean up temporary file
    try:
        temp_file.unlink()
        print("✔ Cleaned up temporary files")
    except:
        pass

# ─────────── entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()
