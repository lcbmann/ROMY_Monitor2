#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – Sagnac Spectrum Only (helicorder-matched colors)
======================================================

Reads one day of beat-note (FJ*) miniSEED from the local SDS archive
and produces **only** the Sagnac spectrum plot with:

  • the **same hour color code** as the helicorders (CFG['cmap'] used identically)
  • **fixed y-axis limits** for inter-comparability: 1e-12 … 1e1
  • a **legend** mapping each hour to its color (as in the original daily plots)
  • **round brackets** for units in axis labels

Notes on draw order & transparency (to avoid “red covering”):
  • Lines are drawn from hour 23 down to 00 so late-day reds are plotted first.
  • Per-line transparency alpha=0.3 (matches the original scripts).

Output:
    <repo>/new_figures/sagnac_spectrum_R<ring>.png

Usage:
    python romy_make_sagnac_spectrum.py <Z|U|V|W> [YYYY-MM-DD]
"""

# ─────────── std-lib / 3-party ──────────────────────────────────────────
import sys, os, gc, warnings
from datetime import date, timedelta
from pathlib  import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # head-less rendering
import matplotlib.pyplot as plt
from obspy        import UTCDateTime, Stream
from scipy.signal import welch, get_window
from tqdm.auto    import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["agg.path.chunksize"] = 10_000

# ─────────── User-specific paths ─────────────────────────────────────────
MOUNT     = Path(os.getenv("ROMY_MOUNT", "/import/freenas-ffb-01-data")).expanduser()
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR   = REPO_ROOT / "new_figures"
FIG_DIR.mkdir(exist_ok=True)

# ─────────── CLI arguments ───────────────────────────────────────────────
if len(sys.argv) < 2 or sys.argv[1].upper() not in "ZUVW":
    sys.exit("Usage:  python romy_make_sagnac_spectrum.py  <Z|U|V|W>  [YYYY-MM-DD]")

RING     = sys.argv[1].upper()
RUN_DATE = (sys.argv[2] if len(sys.argv) > 2
            else str((date.today() - timedelta(days=1))))   # yesterday (local date)

EXPECTED_F = dict(Z=553, U=302, V=448, W=448)[RING]
LOC_CODE   = dict(Z="10", U="", V="", W="")[RING]           # not used for FJ*, kept for completeness

CFG = dict(
    band        = 30,           # ± Hz around EXPECTED_F for plotting
    chunk_sec   = 3600,         # seconds per PSD window (1 h)
    seg_sec     = 600,          # Welch segment length (s)
    counts2V    = 0.59604645e-6,# counts → volts
    dpi         = 150,          # image DPI
    cmap        = "rainbow",    # <<< match helicorder colormap
    alpha       = 0.3,          # <<< match original spectra transparency
    ylim_fixed  = (1e-12, 1e1), # <<< fixed y-axis for inter comparability
)

# ─────────── SDS root ───────────────────────────────────────────────────
SDS_ARCH = MOUNT / "romy_archive"  # SDS miniSEED root

# ─────────── Helpers ────────────────────────────────────────────────────
def read_sds(archive: Path, seed: str, t1, t2) -> Stream:
    """Return an ObsPy Stream (empty if files missing)."""
    from obspy.clients.filesystem.sds import Client
    if not archive.is_dir():
        return Stream()
    n, s, l, c = seed.split(".")
    cli = Client(str(archive), sds_type="D", format="MSEED")
    try:
        st = cli.get_waveforms(n, s, l, c, UTCDateTime(t1)-10, UTCDateTime(t2)+10, merge=-1)
    except Exception:
        return Stream()
    # For beat-note PSDs we keep processing minimal (no detrend, keep volts)
    st = st.trim(UTCDateTime(t1), UTCDateTime(t2), nearest_sample=False)
    return st

def welch_psd(arr: np.ndarray, fs: float, win_sec: int):
    """Welch PSD (density) with Hann window and 50% overlap."""
    f, p = welch(
        arr,
        fs=fs,
        window=get_window("hann", int(fs * win_sec)),
        nperseg=int(fs * win_sec),
        noverlap=int(fs * win_sec * 0.5),
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return f, p

def hour_colors_helicorder(n_intervals: int, cmap_name: str):
    """
    Build the same discrete color ramp as the helicorder:
      colors = get_cmap(CFG['cmap'])(linspace(0,1,n_intervals))
    Hour h uses colors[h] (no reversing). We will draw in reverse order
    so late-day reds are placed *under* early-day blues.
    """
    base = plt.get_cmap(cmap_name)
    return base(np.linspace(0, 1, n_intervals))

# ─────────── Sagnac figure (PSD only, helicorder-matched colors) ─────────
def make_sagnac_figure(freq: np.ndarray,
                       hour_psds: dict[int, np.ndarray],
                       run_date: str) -> plt.Figure:
    """
    Plot per-hour Sagnac PSD around EXPECTED_F using the *same* hour→color
    mapping as the helicorder. Show a legend (hours → colors), apply fixed
    y-limits (1e-12 .. 1e1), and use round brackets in unit labels.
    """
    hours_present = sorted(hour_psds.keys())
    n_intervals   = 24  # fixed day structure for color indexing
    colors_all    = hour_colors_helicorder(n_intervals, CFG["cmap"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Draw chronologically but ensure later hours appear on top via z-order
    draw_order = sorted(hours_present)

    handles = []
    labels  = []
    for h in draw_order:
        z_base = 10 + h  # keep later hours visually on top
        line, = ax.plot(
            freq,
            hour_psds[h],
            color=colors_all[h],
            alpha=CFG["alpha"],
            lw=1.0,
            label=f"{h:02d}:00",
            zorder=z_base,
        )
        handles.append(line)
        labels.append(f"{h:02d}:00")

    # median PSD over available hours (robust to NaNs) — draw on top
    try:
        stack = np.vstack([hour_psds[h] for h in hours_present])
        med   = np.nanmedian(stack, axis=0)
        ax.plot(freq, med, color="k", lw=1.4, label="median", zorder=2)
    except Exception:
        pass

    # axes cosmetics
    ax.set_xlim(EXPECTED_F - CFG["band"], EXPECTED_F + CFG["band"])
    ax.set_yscale("log")
    ax.set_ylim(*CFG["ylim_fixed"])  # fixed inter-comparability limits
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V^2/Hz)")    # round brackets for units
    ax.set_title(f"Sagnac Spectrum R{RING} – {run_date}")
    ax.grid(ls="--", alpha=0.4)

    # Legend mapping hour→color (compact, readable) for the hours actually plotted
    if handles:
        leg = ax.legend(ncol=2, fontsize=9, frameon=True)
        # Thicken legend lines like in the original plots
        for line in leg.get_lines():
            line.set_linewidth(3.0)

    fig.tight_layout()
    return fig

# ═══════════════════════════ main routine ════════════════════════════════
def main():
    print(f"Processing Sagnac spectrum for R{RING} on {RUN_DATE}")

    seed_raw = f"BW.DROMY..FJ{RING}"

    hour_psds: dict[int, np.ndarray] = {}  # hour → PSD (band-limited)
    fig_freq = None

    for hour in tqdm(range(24), desc=f"Day {RUN_DATE}", ncols=74):
        t1 = UTCDateTime(RUN_DATE) + hour * 3600
        t2 = t1 + CFG["chunk_sec"]

        st_raw = read_sds(SDS_ARCH, seed_raw, t1, t2)
        if not st_raw or len(st_raw[0].data) == 0:
            continue

        tr = st_raw[0].copy()
        tr.data = tr.data.astype(float) * CFG["counts2V"]  # counts → volts

        # Welch PSD for this hour
        f_all, p_all = welch_psd(tr.data, tr.stats.sampling_rate, CFG["seg_sec"])

        # limit to band around EXPECTED_F for plotting
        mask = (f_all >= EXPECTED_F - CFG["band"]) & (f_all <= EXPECTED_F + CFG["band"])
        if not np.any(mask):
            continue

        if fig_freq is None:
            fig_freq = f_all[mask]

        hour_psds[hour] = p_all[mask]

        # free memory
        del st_raw, tr
        gc.collect()

    if not hour_psds:
        print(f"✖ No usable data found for {RUN_DATE} – writing placeholder image")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.55, f"R{RING} Sagnac Spectrum\n{RUN_DATE}",
                ha='center', va='center', fontsize=16, weight='bold')
        ax.text(0.5, 0.25, 'No data available',
                ha='center', va='center', fontsize=12, color='crimson')
        ax.axis('off')
        placeholder = FIG_DIR / f"sagnac_spectrum_R{RING}.png"
        fig.savefig(placeholder, dpi=CFG['dpi'], bbox_inches='tight'); plt.close(fig)
        return

    # Build the figure (PSD only, helicorder-matched colors & transparency)
    fig = make_sagnac_figure(fig_freq, hour_psds, RUN_DATE)

    out_png = FIG_DIR / f"sagnac_spectrum_R{RING}.png"
    fig.savefig(out_png, dpi=CFG["dpi"], bbox_inches="tight",
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✔ saved figure → {out_png}")

# ─────────── entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()
