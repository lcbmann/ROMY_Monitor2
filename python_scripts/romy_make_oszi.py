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
import numpy as np
from PIL import Image, ImageFile

# Oscilloscope library
try:
    from ds1054z import DS1054Z
except ImportError:
    print("⚠ ds1054z library not found – creating placeholder oscilloscope image")
    import matplotlib.pyplot as plt
    REPO_ROOT = Path(__file__).resolve().parents[1]
    FIG_DIR   = REPO_ROOT / "new_figures"; FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.text(0.5,0.55,"Oscilloscope Capture",ha='center',va='center',fontsize=18,weight='bold')
    ax.text(0.5,0.35,'Dependency ds1054z missing',ha='center',va='center',fontsize=12,color='crimson')
    ax.text(0.5,0.2,datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),ha='center',va='center',fontsize=10,color='0.3')
    ax.axis('off')
    out = FIG_DIR / 'html_oszi.png'
    fig.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig)
    sys.exit(0)

warnings.filterwarnings("ignore")
plt.rcParams["agg.path.chunksize"] = 10_000

# ─────────── User-specific paths ─────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
# Use unified build artifact directory new_figures
FIG_DIR   = REPO_ROOT / "new_figures"
FIG_DIR.mkdir(exist_ok=True)

# ─────────── Configuration ───────────────────────────────────────────────
CFG = dict(
    scope_ip     = "10.153.82.107",     # RIGOL DS1054Z IP address
    probe_ratio  = 1,                   # probe ratio setting
    dpi          = 150,                 # output image DPI
    temp_file    = "tmp_oszi.bmp",      # temporary file name
    timeout      = 10,                  # connection timeout in seconds
    # Fractional crop margins to trim UI from screenshot (0.0-0.4 typical)
    # Positive values remove that fraction from each edge respectively
    crop_top     = 0.07,
    crop_bottom  = 0,
    crop_left    = 0.07,
    crop_right   = 0.13,
    bottom_strip_frac  = 0.05,  # fraction of image height to shift
    bottom_strip_shift = 0.08,  # fraction of width to offset slice to the right
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

        def _apply_crop(pil_img):
            """Crop a bit from the top and sides to remove UI buttons."""
            w, h = pil_img.size
            # Clamp crop fractions to [0, 0.4] to avoid excessive cropping
            ct = max(0.0, min(0.4, float(CFG.get("crop_top", 0.12))))
            cb = max(0.0, min(0.4, float(CFG.get("crop_bottom", 0.02))))
            cl = max(0.0, min(0.4, float(CFG.get("crop_left", 0.06))))
            cr = max(0.0, min(0.4, float(CFG.get("crop_right", 0.06))))
            left   = int(round(w * cl))
            right  = int(round(w * (1.0 - cr)))
            top    = int(round(h * ct))
            bottom = int(round(h * (1.0 - cb)))
            # Ensure valid box
            left = max(0, min(left, w - 2))
            right = max(left + 1, min(right, w))
            top = max(0, min(top, h - 2))
            bottom = max(top + 1, min(bottom, h))
            try:
                return pil_img.crop((left, top, right, bottom))
            except Exception:
                return pil_img

        def _shift_bottom_strip(pil_img):
            """Shift the bottom strip rightward before cropping to retain labels."""
            try:
                arr = np.array(pil_img)
                if arr.ndim == 3 and arr.shape[2] == 4:
                    h, w, _ = arr.shape
                    strip_frac = float(CFG.get("bottom_strip_frac", 0.05))
                    shift_frac = float(CFG.get("bottom_strip_shift", CFG.get("crop_left", 0.07)))
                    strip_h = max(1, int(round(h * np.clip(strip_frac, 0.0, 0.2))))
                    shift_px = int(round(w * np.clip(shift_frac, 0.0, 0.5)))
                    if strip_h < h and 0 < shift_px < w:
                        strip = arr[-strip_h:, :].copy()
                        shifted = np.zeros_like(strip)
                        shifted[:, shift_px:, :] = strip[:, : w - shift_px, :]
                        if strip_h < h - 1 and shift_px:
                            filler_row = arr[-strip_h-1, :shift_px, :].copy()[np.newaxis, ...]
                        else:
                            filler_row = strip[0:1, :shift_px, :]
                        if filler_row.size:
                            filler = np.broadcast_to(
                                filler_row,
                                (strip_h, shift_px, strip.shape[2]),
                            )
                            shifted[:, :shift_px, :] = filler
                        arr[-strip_h:, :, :] = shifted
                        print(
                            "ℹ Bottom strip shifted "
                            f"{shift_px}px over {strip_h}px height (left filled from row above)"
                        )
                        return Image.fromarray(arr)
                    print(
                        "ℹ Skipped bottom strip shift (strip outside bounds): "
                        f"strip_h={strip_h}, shift_px={shift_px}, image={w}x{h}"
                    )
                else:
                    print("ℹ Skipped bottom strip shift (unexpected image mode)")
            except Exception as exc:
                print(f"✖ Bottom strip shift failed: {exc}")
            return pil_img

        img = _shift_bottom_strip(img)
        img = _apply_crop(img)


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
        # write placeholder
        fig, ax = plt.subplots(figsize=(10,6))
        ax.text(0.5,0.55,"Oscilloscope Capture",ha='center',va='center',fontsize=18,weight='bold')
        ax.text(0.5,0.35,'Connection failed',ha='center',va='center',fontsize=12,color='crimson')
        ax.text(0.5,0.2,datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),ha='center',va='center',fontsize=10,color='0.3')
        ax.axis('off')
        out = FIG_DIR / 'html_oszi.png'
        fig.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig)
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
