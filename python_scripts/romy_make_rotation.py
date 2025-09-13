#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – Rotation Spectra (old-style, PSD-only)
=============================================

Generates the daily rotation PSD figure **using the same code path**
as the old scripts, but outputs ONLY the colored spectra panel
(no helicorder).

Key points (to match old behavior):
  - Data read via spectra.read_from_sds(..., merge=True)
  - Per-hour Welch PSD collection via spectra.get_collection(...)
  - Optional octave-band averaging via spectra.get_fband_average(...)
  - Quality classification via spectra.classify_collection_quality(...)
  - Plotting reproduces the left-panel axes/labels/style from
    spectra.plot_spectra_and_helicorder(), but without the helicorder.

Usage:
  python romy_make_rotation.py                 # all rings for yesterday
  python romy_make_rotation.py Z               # ring Z for yesterday
  python romy_make_rotation.py V 2025-09-10    # specific ring & date

Output →  <repo>/new_figures/rotation_spectrum_R<ring>.png
"""

# ───────────────── std-lib / third-party ────────────────────────────────
import sys
import os
import warnings
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import get_cmap
from obspy import UTCDateTime

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ───────────────── repo paths & SDS root ────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR   = REPO_ROOT / "new_figures"
FIG_DIR.mkdir(exist_ok=True)

# SDS archive root (override via env if needed)
MOUNT    = Path(os.getenv("ROMY_MOUNT", "/import/freenas-ffb-01-data")).expanduser()
SDS_ARCH = Path(os.getenv("ROMY_SDS", str(MOUNT / "romy_archive"))).expanduser()

# ───────────────── spectra helper import (old pipeline) ─────────────────
sys.path.append(str(REPO_ROOT / "develop" / "spectra"))
try:
    from spectra import spectra  # old helper class, required
except Exception as e:
    # Create a simple placeholder image per requested ring and exit
    print(f"✖ Could not import spectra module: {e}")
    rings = list("ZUVW")
    if len(sys.argv) >= 2 and sys.argv[1].upper() in "ZUVW":
        rings = [sys.argv[1].upper()]
    for r in rings:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.55, f"R{r} Rotation Spectrum", ha="center", va="center",
                fontsize=16, weight="bold")
        ax.text(0.5, 0.35, "Dependency missing (spectra.py)", ha="center", va="center",
                fontsize=12, color="crimson")
        ax.axis("off")
        out = FIG_DIR / f"rotation_spectrum_R{r}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  → wrote placeholder: {out}")
    sys.exit(0)

# ───────────────── CLI args ─────────────────────────────────────────────
if len(sys.argv) < 2:
    RINGS = list("ZUVW")
    RUN_DATE = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
elif sys.argv[1].upper() in "ZUVW":
    RINGS = [sys.argv[1].upper()]
    RUN_DATE = sys.argv[2] if len(sys.argv) > 2 else (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
else:
    sys.exit("Usage: python romy_make_rotation.py [Z|U|V|W] [YYYY-MM-DD]")

# ───────────────── ROMY ring → seed mapping ─────────────────────────────
# ROMY rotation channels are BJZ/BJU/BJV/BJW; only Z has location code "10".
LOC_BY_RING = dict(Z="10", U="", V="", W="")

def romy_seed_for(ring: str) -> str:
    loc = LOC_BY_RING[ring]
    return f"BW.ROMY.{loc}.BJ{ring}"

# ───────────────── Default processing config (old style) ────────────────
# Mirrors the old YAML/JSON flow used by make_spectra.py; tuned for rotation.
def build_cfg(ring: str, day_str: str) -> dict:
    tbeg = f"{day_str} 00:00:00"
    tend = f"{day_str} 23:59:59"
    seed = romy_seed_for(ring)

    return {
        # time range and seed
        "tbeg": tbeg,
        "tend": tend,
        "seeds": [seed],

        # SDS + bookkeeping paths (spectra.set_config validates/creates)
        "path_to_sds": str(SDS_ARCH),
        "path_to_data_out":   str(REPO_ROOT / "local_output" / "rotation_data"),
        "path_to_figures_out": str(REPO_ROOT / "local_output" / "rotation_plots"),

        # per-interval PSD parameters (identical structure as old)
        "tinterval": 3600,          # 1h windows
        "toverlap": 0,
        "method": "welch",

        # plotting limits for PSD
        "fmin": 0.001,
        "fmax": 5.0,

        # averaging in octave bands (kept on to match typical old setup)
        "apply_average": True,
        "fraction_of_octave": 6,    # common old config
        "averaging": "mean",

        # units shown on PSD axis (rotation rates)
        "data_unit": r"rad/s",

        # quality filtering (same knobs as spectra.py)
        "quality_filter": "good",
        "threshold": 1e-15,
        "zero_seq_limit": 20,
        "high_seq_limit": 20,
        "flat_seq_limit": 20,

        # plotting cosmetics to match old figures (colorful)
        "cmap": "rainbow",
        "alpha": 0.8,

        # response removal is typically NOT applied for BJ* rotation; keep off
        "remove_response": False,
        "inventory_file": "",

        "verbose": False,
    }

# ───────────────── Plotting (PSD-only, left-panel replica) ──────────────
def plot_colored_psd_only(sp: spectra, cfg: dict, out_png: Path):
    """
    Reproduce the left PSD panel from spectra.plot_spectra_and_helicorder(),
    including discrete per-interval colors and greyed-out gaps/filtered traces.
    """
    if not hasattr(sp, "collection"):
        raise AttributeError("No collection available. Run get_collection() first.")

    # Build discrete colormap with one color per interval
    n_intervals = len(sp.collection.get("time", []))
    base_cmap = get_cmap(cfg.get("cmap", "rainbow"))
    colors = base_cmap(np.linspace(0, 1, max(n_intervals, 2)))
    discrete_cmap = ListedColormap(list(colors))
    color_norm = Normalize(vmin=-0.5, vmax=n_intervals - 0.5)  # for parity with old code (even if no colorbar)

    # Build quality mask like the old function
    quality_filter = cfg.get("quality_filter", None)
    if quality_filter is not None and "quality" in sp.collection:
        quality_mask = [q == quality_filter for q in sp.collection["quality"]]
    else:
        # fall back to accepting all intervals
        quality_mask = [True] * n_intervals

    # Gap detection (same logic as in plot_spectra_and_helicorder)
    gap_intervals = []
    for i, ((t1, t2), is_good_quality) in enumerate(zip(sp.time_intervals, quality_mask)):
        tr_slice = sp.tr.slice(t1, t2)
        data = tr_slice.data
        has_gaps = False

        if len(data) > 0:
            # Method 1: zero sequences
            if np.any(data == 0.0):
                zero_mask = data == 0
                zero_starts = np.where(np.diff(np.concatenate(([False], zero_mask))))[0]
                zero_ends   = np.where(np.diff(np.concatenate((zero_mask, [False]))))[0]
                min_len = min(len(zero_starts), len(zero_ends))
                if min_len > 0:
                    zero_lengths = zero_ends[:min_len] - zero_starts[:min_len]
                    if np.any(zero_lengths > 10):
                        has_gaps = True

            # Method 2: flat-line segments
            if not has_gaps:
                n_size = 20
                for j in range(0, len(data) - n_size, n_size):
                    if np.all(data[j:j + n_size] == data[j]):
                        has_gaps = True
                        break

        # honor quality filter by greying out non-matching spectra
        if not has_gaps and not is_good_quality:
            has_gaps = True

        if has_gaps:
            gap_intervals.append(i)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each interval spectrum with color or grey
    alpha = float(cfg.get("alpha", 0.8))
    for i, (f, s) in enumerate(zip(sp.collection.get("freq", []), sp.collection.get("spec", []))):
        if i in gap_intervals:
            color = "grey"
        else:
            color = discrete_cmap(i / max(n_intervals - 1, 1))
        ax.plot(f, s, color=color, alpha=alpha, lw=1)

    # Axes setup (match old left-panel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)

    fmin = cfg.get("fmin", None)
    fmax = cfg.get("fmax", None)
    if fmin is not None:
        ax.set_xlim(left=fmin)
    if fmax is not None:
        ax.set_xlim(right=fmax)

    ax.set_xlabel("Frequency (Hz)")
    if getattr(sp, "method", "welch") in ["welch", "multitaper"]:
        data_unit = cfg.get("data_unit", None)
        if data_unit:
            hz = r"$^2$/Hz"
            ax.set_ylabel(f"Power Spectral Density (({data_unit}){hz})")
        else:
            ax.set_ylabel("Power Spectral Density")
    else:
        data_unit = cfg.get("data_unit", None)
        if data_unit:
            hz = r"$\sqrt{Hz}$"
            ax.set_ylabel(f"Amplitude Spectrum ({data_unit}/{hz})")
        else:
            ax.set_ylabel("Amplitude Spectrum")

    # Title formatting like old function's suptitle
    title = f"{sp.tr_id}"
    if hasattr(sp, "tbeg"):
        try:
            title += f" | {sp.tbeg.strftime('%Y-%m-%d')}"
        except Exception:
            pass
    if hasattr(sp, "method"):
        title += f" | {str(sp.method).upper()}"
    if quality_filter is not None:
        n_good = sum(quality_mask) if quality_mask else 0
        title += f" | {quality_filter} quality ({n_good}/{len(quality_mask)})"
    plt.suptitle(title)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ───────────────── One-day, one-ring processor (old path) ───────────────
def process_ring(ring: str, run_date: str) -> bool:
    print(f"→ R{ring} | {run_date}")

    cfg = build_cfg(ring, run_date)
    sp = spectra(cfg)

    # Day bounds (inclusive end with +1s like old)
    t1 = UTCDateTime(cfg["tbeg"])
    t2 = UTCDateTime(cfg["tend"]) + 1

    seed = cfg["seeds"][0]

    # Read via the SAME function the old pipeline uses (merge=True, trim, detrend)
    st = sp.read_from_sds(
        path_to_archive=cfg["path_to_sds"],
        seed=seed,
        tbeg=t1,
        tend=t2,
        data_format="MSEED",
        merge=True,
    )

    if not st:
        print("   ✖ No data for the day – writing placeholder")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.55, f"R{ring} Rotation Spectrum\n{run_date}",
                ha="center", va="center", fontsize=18, weight="bold")
        ax.text(0.5, 0.35, "No data available", ha="center", va="center",
                fontsize=12, color="crimson")
        ax.axis("off")
        out_png = FIG_DIR / f"rotation_spectrum_R{ring}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"   → saved: {out_png}")
        return False

    # (Optional) response removal – off by default for rotation BJ*
    if cfg.get("remove_response"):
        try:
            st = sp.remove_response(
                st,
                cfg["inventory_file"],
                output=cfg.get("output_type", "VEL"),
            )
        except Exception as e:
            print(f"   ! response removal failed, continuing without: {e}")

    # Hook into spectra flow exactly like old code
    sp.add_trace(st[0])

    sp.get_collection(
        tinterval=cfg["tinterval"],
        toverlap=cfg["toverlap"],
        method=cfg["method"],
    )

    if cfg.get("apply_average"):
        sp.get_fband_average(
            fraction_of_octave=cfg.get("fraction_of_octave", 6),
            average=cfg.get("averaging", "mean"),
        )

    if cfg.get("quality_filter"):
        sp.classify_collection_quality(
            threshold=cfg.get("threshold", 1e-15),
            zero_seq_limit=cfg.get("zero_seq_limit", 20),
            high_seq_limit=cfg.get("high_seq_limit", 20),
            flat_seq_limit=cfg.get("flat_seq_limit", 20),
        )

    # Produce ONLY the colored PSD (left panel replica)
    out_png = FIG_DIR / f"rotation_spectrum_R{ring}.png"
    plot_colored_psd_only(sp, cfg, out_png)
    print(f"   ✔ saved: {out_png}")
    return True

# ───────────────────────────────── Main ──────────────────────────────────
if __name__ == "__main__":
    print("ROMY Rotation Spectra (old-style, PSD-only)")
    print(f"SDS : {SDS_ARCH}")
    print(f"Date: {RUN_DATE}")
    ok = 0
    for r in RINGS:
        try:
            ok += 1 if process_ring(r, RUN_DATE) else 0
        except Exception as e:
            print(f"   ! R{r} failed: {e}")
    print(f"✔ Completed {ok}/{len(RINGS)} rings")
