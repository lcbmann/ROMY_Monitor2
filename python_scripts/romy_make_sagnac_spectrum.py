#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – Sagnac Spectrum Only
===========================

Reads one day of beat-note (FJ*) miniSEED from the local
**file system** and produces only the Sagnac spectrum plot

    <repo>/figures/sagnac_spectrum_R<ring>.png
"""

# ─────────── std-lib / 3-party ───────────────────────────────────────────
import sys, gc, warnings, os
from datetime      import date, timedelta
from pathlib       import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # head-less rendering
import matplotlib.pyplot as plt
from obspy        import UTCDateTime, Stream
from scipy.signal import welch, get_window
from pandas       import DataFrame
from tqdm.auto    import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["agg.path.chunksize"] = 10_000

# ─────────── User-specific paths ─────────────────────────────────────────
# • By default we try the server path.  Override with $ROMY_MOUNT if needed.
MOUNT = Path(os.getenv("ROMY_MOUNT", "/import/freenas-ffb-01-data")).expanduser()

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR   = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ─────────── CLI arguments ───────────────────────────────────────────────
if len(sys.argv) < 2 or sys.argv[1].upper() not in "ZUVW":
    sys.exit("Usage:  python romy_make_sagnac_spectrum.py  <Z|U|V|W>  [YYYY-MM-DD]")

RING     = sys.argv[1].upper()
RUN_DATE = (sys.argv[2] if len(sys.argv) > 2
            else str((date.today() - timedelta(days=1))))   # yesterday (local date)

EXPECTED_F = dict(Z=553, U=302, V=448, W=448)[RING]
LOC_CODE   = dict(Z="10", U="", V="", W="")[RING]           # BJ* location

CFG = dict(
    band      = 30,          # ± Hz plotted
    chunk_sec = 3600,        # seconds per PSD window
    seg_sec   = 600,         # Welch segment length
    counts2V  = 0.59604645e-6,
    dpi       = 150,         # output image DPI
)

# ─────────── Derived data paths ──────────────────────────────────────────
SDS_ARCH   = MOUNT / "romy_archive"                       # SDS miniSEED root

# ─────────── Helper: read SDS miniSEED ───────────────────────────────────
def read_sds(archive: Path, seed: str, t1, t2) -> Stream:
    """Return an ObsPy Stream (empty if files missing)."""
    from obspy.clients.filesystem.sds import Client
    if not archive.is_dir():
        return Stream()
    n,s,l,c = seed.split(".")
    cli     = Client(str(archive), sds_type="D")
    try:
        return cli.get_waveforms(n, s, l, c,
                                 UTCDateTime(t1), UTCDateTime(t2),
                                 merge=-1)
    except Exception:
        return Stream()

# ─────────── Helper: Welch PSD ───────────────────────────────────────────
def welch_psd(arr: np.ndarray, fs: float, win_sec: int):
    f, p = welch(arr,
                 fs=fs,
                 window=get_window("hann", int(fs*win_sec)),
                 nperseg=int(fs*win_sec),
                 noverlap=int(fs*win_sec*0.5),
                 detrend="constant",
                 scaling="density")
    return f, p

# ─────────── Helper: create Sagnac spectrum figure ──────────────────────
def make_sagnac_figure(freq, psds):
    """Create Sagnac spectrum figure only."""
    colours = plt.cm.jet_r(np.linspace(0, 1, len(psds)))
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # PSD panel
    for i, p in enumerate(psds[::-1]):
        ax.plot(freq, p, color=colours[i], alpha=.4, lw=.8)
    ax.plot(freq, np.nanmedian(psds, axis=0), color="k", lw=1.3, label="median")
    
    ax.set(xlim=(EXPECTED_F-CFG["band"], EXPECTED_F+CFG["band"]),
           yscale="log",
           xlabel="Frequency [Hz]",
           ylabel="PSD [V²/Hz]",
           title=f"Sagnac Spectrum R{RING} – {RUN_DATE}")
    ax.grid(ls=":")
    ax.legend()

    fig.tight_layout()
    return fig

# ═══════════════════════════ main routine ════════════════════════════════
def main():
    """Main execution function."""
    print(f"Processing Sagnac spectrum for R{RING} on {RUN_DATE}")

    seed_raw = f"BW.DROMY..FJ{RING}"
    psds, full_freq = [], None

    for hour in tqdm(range(24), desc=f"Day {RUN_DATE}", ncols=74):
        t1 = UTCDateTime(RUN_DATE) + hour*3600
        t2 = t1 + CFG["chunk_sec"]

        st_raw = read_sds(SDS_ARCH, seed_raw, t1-30, t2+30)
        if not st_raw:
            continue                      # skip empty hour

        tr_raw = st_raw[0]
        tr_raw.trim(t1, t2)
        tr_raw.data = tr_raw.data.astype(float) * CFG["counts2V"]

        full_freq, psd = welch_psd(tr_raw.data, tr_raw.stats.sampling_rate,
                                   CFG["seg_sec"])
        psds.append(psd)

    if not psds:
        print(f"✖ No usable data found for {RUN_DATE}")
        return

    # ---------- figure ----------------------------------------------------
    mask     = (full_freq >= EXPECTED_F-CFG["band"]) & (full_freq <= EXPECTED_F+CFG["band"])
    fig_freq = full_freq[mask]
    fig_psds = np.array(psds)[:, mask]

    fig = make_sagnac_figure(fig_freq, fig_psds)
    
    try:
        png_name = FIG_DIR / f"sagnac_spectrum_R{RING}.png"
        fig.savefig(png_name, dpi=CFG["dpi"], bbox_inches="tight",
                   facecolor='white', edgecolor='none')
        print(f"✔ saved figure → {png_name}")
    except PermissionError:
        png_name = (REPO_ROOT / "local_output" /
                    f"sagnac_spectrum_R{RING}.png")
        png_name.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_name, dpi=CFG["dpi"], bbox_inches="tight",
                   facecolor='white', edgecolor='none')
        print(f"✔ saved figure → {png_name} (fallback)")
    
    plt.close(fig)
    print("✔ Sagnac spectrum generation completed")

# ─────────── entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()
