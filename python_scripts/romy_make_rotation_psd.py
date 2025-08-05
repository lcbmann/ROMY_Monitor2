#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – Rotation PSD Only
========================

Reads one day of rotation (BJ*) miniSEED from the local
**file system** and produces only the rotation power spectral density plot

    <repo>/figures/rotation_psd_R<ring>.png
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
    sys.exit("Usage:  python romy_make_rotation_psd.py  <Z|U|V|W>  [YYYY-MM-DD]")

RING     = sys.argv[1].upper()
RUN_DATE = (sys.argv[2] if len(sys.argv) > 2
            else str((date.today() - timedelta(days=1))))   # yesterday (local date)

LOC_CODE   = dict(Z="10", U="", V="", W="")[RING]           # BJ* location

CFG = dict(
    freq_band = (0.001, 10),  # Frequency range for rotation PSD (Hz)
    chunk_sec = 3600,         # seconds per PSD window
    seg_sec   = 600,          # Welch segment length
    dpi       = 150,          # output image DPI
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
    # Ensure consistent window sizing
    nperseg = int(fs * win_sec)
    
    # Make sure we have enough data for at least 2 segments
    if len(arr) < nperseg * 2:
        nperseg = len(arr) // 4  # Use 1/4 of available data as window
        nperseg = max(nperseg, int(fs * 10))  # At least 10 seconds
    
    f, p = welch(arr,
                 fs=fs,
                 window=get_window("hann", nperseg),
                 nperseg=nperseg,
                 noverlap=int(nperseg*0.5),
                 detrend="constant",
                 scaling="density")
    return f, p

# ─────────── Helper: create rotation PSD figure ─────────────────────────
def make_rotation_psd_figure(freq, psds):
    """Create rotation power spectral density figure only."""
    colours = plt.cm.jet_r(np.linspace(0, 1, len(psds)))
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # PSD panel
    for i, p in enumerate(psds[::-1]):
        ax.plot(freq, p, color=colours[i], alpha=.4, lw=.8)
    ax.plot(freq, np.nanmedian(psds, axis=0), color="k", lw=1.3, label="median")
    
    ax.set(xlim=CFG["freq_band"],
           xscale="log",
           yscale="log", 
           xlabel="Frequency [Hz]",
           ylabel="PSD [(rad/s)²/Hz]",
           title=f"Rotation PSD R{RING} – {RUN_DATE}")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    fig.tight_layout()
    return fig

# ═══════════════════════════ main routine ════════════════════════════════
def main():
    """Main execution function."""
    print(f"Processing rotation PSD for R{RING} on {RUN_DATE}")

    seed_rot = f"BW.ROMY.{LOC_CODE}.BJ{RING}"
    psds, full_freq = [], None
    reference_freq = None  # Store reference frequency array

    for hour in tqdm(range(24), desc=f"Day {RUN_DATE}", ncols=74):
        t1 = UTCDateTime(RUN_DATE) + hour*3600
        t2 = t1 + CFG["chunk_sec"]

        st_rot = read_sds(SDS_ARCH, seed_rot, t1-30, t2+30)
        if not st_rot:
            continue                      # skip empty hour

        tr_rot = st_rot[0]
        tr_rot.trim(t1, t2)
        tr_rot.data = tr_rot.data.astype(float)

        # Skip if data is too short
        if len(tr_rot.data) < 1000:  # Less than ~16 seconds at 60 Hz
            continue

        try:
            # Use fixed window size to ensure consistent frequency arrays
            win_sec = min(CFG["seg_sec"], len(tr_rot.data) // (4 * int(tr_rot.stats.sampling_rate)))
            win_sec = max(60, win_sec)  # At least 60 seconds
            
            freq, psd = welch_psd(tr_rot.data, tr_rot.stats.sampling_rate, win_sec)
            
            # Store the first valid frequency array as reference
            if reference_freq is None:
                reference_freq = freq
                full_freq = freq
            
            # Only keep PSDs that match the reference frequency array length
            if len(freq) == len(reference_freq):
                psds.append(psd)
            
        except Exception as e:
            print(f"  -> Error processing hour {hour}: {e}")
            continue

    if not psds:
        print(f"✖ No usable data found for {RUN_DATE}")
        return

    # ---------- figure ----------------------------------------------------
    mask     = (full_freq >= CFG["freq_band"][0]) & (full_freq <= CFG["freq_band"][1])
    fig_freq = full_freq[mask]
    fig_psds = np.array(psds)[:, mask]

    fig = make_rotation_psd_figure(fig_freq, fig_psds)
    
    try:
        png_name = FIG_DIR / f"rotation_psd_R{RING}.png"
        fig.savefig(png_name, dpi=CFG["dpi"], bbox_inches="tight",
                   facecolor='white', edgecolor='none')
        print(f"✔ saved figure → {png_name}")
    except PermissionError:
        png_name = (REPO_ROOT / "local_output" /
                    f"rotation_psd_R{RING}.png")
        png_name.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_name, dpi=CFG["dpi"], bbox_inches="tight",
                   facecolor='white', edgecolor='none')
        print(f"✔ saved figure → {png_name} (fallback)")
    
    plt.close(fig)
    print("✔ Rotation PSD generation completed")

# ─────────── entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()
