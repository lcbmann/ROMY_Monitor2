#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – daily Sagnac PSD  +  mini-helicorder
==========================================

Reads one day of beat-note (FJ*) + rotation (BJ*) miniSEED from the local
**file system** (no SSH-FS needed on the LMU servers) and produces

    romy_autodata/<year>/R<ring>/spectra/R<ring>_YYYYMMDD_spectra.pkl
    <repo>/figures/html_sagnacspectra_R<ring>.png
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
    sys.exit("Usage:  python romy_make_spectra.py  <Z|U|V|W>  [YYYY-MM-DD]")

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
)

# ─────────── Derived data paths ──────────────────────────────────────────
AUTOD_YEAR = MOUNT / "romy_autodata" / RUN_DATE[:4] / f"R{RING}"
PKL_DIR    = AUTOD_YEAR / "spectra"                       # server location
PKL_DIR.mkdir(parents=True, exist_ok=True)

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

# ─────────── Helper: final figure ────────────────────────────────────────
def make_figure(freq, psds, rot_traces):
    colours = plt.cm.jet_r(np.linspace(0, 1, len(psds)))
    fig, ax = plt.subplots(1, 2, figsize=(18, 8));  plt.subplots_adjust(wspace=.15)

    # PSD panel
    for i, p in enumerate(psds[::-1]):
        ax[0].plot(freq, p, color=colours[i], alpha=.4, lw=.8)
    ax[0].plot(freq, np.nanmedian(psds, axis=0), color="k", lw=1.3, label="median")
    ax[0].set(xlim=(EXPECTED_F-CFG["band"], EXPECTED_F+CFG["band"]),
              yscale="log",
              xlabel="Frequency [Hz]",
              ylabel="PSD [V²/Hz]",
              title=f"Sagnac spectra R{RING} – {RUN_DATE}")
    ax[0].grid(ls=":")
    ax[0].legend()

    # mini-helicorder
    t = np.linspace(0, 60, len(rot_traces[0]))
    for i, tr in enumerate(rot_traces[::-1]):          # oldest top
        norm = np.nanmax(np.abs(tr)) or 1
        ax[1].plot(t, 0.8*tr/norm + i, color=colours[i], lw=.5)
    ax[1].set(ylim=(-1,24),
              yticks=np.arange(0,24,3),
              yticklabels=[f"{h:02d}:00" for h in range(23,-1,-3)],
              xlabel="Time (min)",
              title="Helicorder (norm.)")
    ax[1].grid(ls=":")
    fig.tight_layout()
    return fig

# ═══════════════════════════ main routine ════════════════════════════════
def main():

    seed_raw = f"BW.DROMY..FJ{RING}"
    seed_rot = f"BW.ROMY.{LOC_CODE}.BJ{RING}"

    psds, rots, full_freq = [], [], None

    for hour in tqdm(range(24), desc=f"Day {RUN_DATE}", ncols=74):
        t1 = UTCDateTime(RUN_DATE) + hour*3600
        t2 = t1 + CFG["chunk_sec"]

        st_raw = read_sds(SDS_ARCH, seed_raw, t1-30, t2+30)
        st_rot = read_sds(SDS_ARCH, seed_rot, t1-30, t2+30)
        if not (st_raw and st_rot):
            continue                      # skip empty hour

        tr_raw, tr_rot = st_raw[0], st_rot[0]
        tr_raw.trim(t1, t2);  tr_rot.trim(t1, t2)

        tr_raw.data = tr_raw.data.astype(float) * CFG["counts2V"]

        full_freq, psd = welch_psd(tr_raw.data, tr_raw.stats.sampling_rate,
                                   CFG["seg_sec"])
        psds.append(psd)
        rots.append(tr_rot.data.astype(float))

    if not psds:
        print(f"✖ No usable data found for {RUN_DATE}")
        return

    # ---------- pickle ----------------------------------------------------
    pkl_name = PKL_DIR / f"R{RING}_{RUN_DATE.replace('-','')}_spectra.pkl"
    DataFrame({"frequencies":[full_freq], "psds":[np.array(psds)]}).to_pickle(pkl_name)
    print("✔ saved pickle  →", pkl_name)

    # ---------- figure ----------------------------------------------------
    mask     = (full_freq >= EXPECTED_F-CFG["band"]) & (full_freq <= EXPECTED_F+CFG["band"])
    fig_freq = full_freq[mask]
    fig_psds = np.array(psds)[:, mask]

    fig = make_figure(fig_freq, fig_psds, rots)
    png_name = FIG_DIR / f"html_sagnacspectra_R{RING}.png"
    fig.savefig(png_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✔ saved figure →", png_name)

    gc.collect()

# ─────────── entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()
