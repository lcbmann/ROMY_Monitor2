#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – Rotation Spectra (colour‑matched to Helicorder)
=====================================================

Creates daily colour‑matched rotation‑spectra PNGs that visually pair with the
helicorder drum‑plots.  The spectra are generated with Andreas M.‑style Welch
processing via the *spectra* helper class, but we now read the raw miniSEED
straight from the SDS archive with the **same** logic that the helicorder
script uses – this guarantees we find data where the helicorder does.

Output →  <repo>/figures/rotation_spectrum_R<ring>.png
"""

# ─────────── std‑lib / 3‑party ───────────────────────────────────────────
import sys, warnings, os
from datetime      import date, timedelta
from pathlib       import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # head‑less rendering
import matplotlib.pyplot as plt
from obspy        import UTCDateTime, Stream

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["agg.path.chunksize"] = 10_000

# ─────────── User‑specific paths ─────────────────────────────────────────
MOUNT     = Path(os.getenv("ROMY_MOUNT", "/import/freenas-ffb-01-data")).expanduser()
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR   = REPO_ROOT / "new_figures"; FIG_DIR.mkdir(exist_ok=True)

# Spectra helper (Andreas)
sys.path.append(str(REPO_ROOT / "develop" / "spectra"))
try:
    from spectra import spectra                # noqa: E402
except Exception as e:  # create placeholder and exit gracefully
    import matplotlib.pyplot as plt
    print(f"✖ Could not import spectra module: {e} – creating placeholder images")
    for ring in (['Z'] if 'RINGS' not in globals() else RINGS):
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5,0.55,f"R{ring} Rotation Spectrum\n{date.today()}",ha='center',va='center',fontsize=16,weight='bold')
        ax.text(0.5,0.25,'Dependency missing',ha='center',va='center',fontsize=12,color='crimson')
        ax.axis('off')
        out = FIG_DIR / f"rotation_spectrum_R{ring}.png"
        fig.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig)
    sys.exit(0)

# Progress bar (graceful fallback)
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, desc="", ncols=None):
        print(f"{desc}…")
        return x

# ─────────── CLI args ───────────────────────────────────────────────────
if len(sys.argv) < 2:
    RINGS = list("ZUVW")
    RUN_DATE = str(date.today() - timedelta(days=1))
elif sys.argv[1].upper() in "ZUVW":
    RINGS = [sys.argv[1].upper()]
    RUN_DATE = sys.argv[2] if len(sys.argv) > 2 else str(date.today() - timedelta(days=1))
else:
    sys.exit("Usage: python romy_make_rotation.py [Z|U|V|W] [YYYY-MM-DD]")

CFG = dict(
    chunk_sec = 3600,         # spectra window (1 h)
    freq_band = (0.001, 5.0), # Hz
    dpi       = 150,
)

SDS_ARCH = MOUNT / "romy_archive"           # SDS miniSEED root

# ─────────── SDS reader (identical to helicorder) ───────────────────────

def read_sds(archive: Path, seed: str, t1, t2) -> Stream:
    """Return an ObsPy *Stream* (empty if files missing)."""
    from obspy.clients.filesystem.sds import Client
    if not archive.is_dir():
        return Stream()
    n, s, l, c = seed.split(".")
    cli = Client(str(archive), sds_type="D")
    try:
        return cli.get_waveforms(n, s, l, c, UTCDateTime(t1), UTCDateTime(t2), merge=-1)
    except Exception:
        return Stream()

# ─────────── Spectra config helper ──────────────────────────────────────

def spec_cfg(ring: str, run_date: str):
    loc = dict(Z="10", U="", V="", W="")[ring]
    seed = f"BW.ROMY.{loc}.BJ{ring}"
    return {
        "tbeg": f"{run_date} 00:00:00",
        "tend": f"{run_date} 23:59:59",
        "seeds": [seed],
        "path_to_sds": str(SDS_ARCH),
        # required bookkeeping paths for spectra
        "path_to_data_out":   str(REPO_ROOT / "local_output" / "rotation_data"),
        "path_to_figures_out": str(REPO_ROOT / "local_output" / "rotation_plots"),
        "tinterval": 3600,
        "toverlap": 0,
        "method": "welch",
        "fmin": CFG["freq_band"][0],
        "fmax": CFG["freq_band"][1],
        "apply_average": True,
        "fraction_of_octave": 12,
        "averaging": "median",
        "data_unit": "rad/s",
        "quality_filter": "good",
        "threshold": 1e-15,
        "zero_seq_limit": 20,
        "high_seq_limit": 20,
        "flat_seq_limit": 20,
        "verbose": False,
    }

# ─────────── Plot helper (colour‑matched) ───────────────────────────────

def plot_spectra(sp, ring: str, run_date: str, out_png: Path):
    """Plot the hour‑wise spectra using helicorder colours.

    *Any* interval that contains only NaNs / zeros (i.e. genuinely empty) is
    rendered in grey; everything else gets its hour‑colour regardless of the
    internal quality flag.  This mirrors the behaviour of
    ``plot_spectra_and_helicorder`` in *make_spectra.py* which always shows
    useful data in colour.
    """
    if not hasattr(sp, "collection"):
        raise AttributeError("Call get_collection() first")

    n_int = len(sp.collection["time"])
    colours = plt.cm.jet_r(np.linspace(0, 1, n_int))[::-1]  # hour‑mapped colours

    fig, ax = plt.subplots(figsize=(10, 8))

    good_cnt = 0
    for i, (f, s) in enumerate(zip(sp.collection["freq"], sp.collection["spec"])):
        # Decide whether this interval really has usable data
        if s is None or len(s) == 0 or np.all(np.isnan(s)):
            colour = "grey"
        else:
            colour = colours[i]; good_cnt += 1
        ax.plot(f, s, lw=1, alpha=.8, color=colour)

    # Axis cosmetics
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(CFG["freq_band"])
    unit = sp.config.get("data_unit", "")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"PSD ({unit}$^2$/Hz)" if unit else "Power Spectral Density")
    ax.set_title(f"{sp.tr_id} | {run_date} | {good_cnt}/{n_int} coloured (data) intervals")
    ax.grid(True, which="both", ls="--", alpha=.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=CFG["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ─────────── Processing per ring ────────────────────────────────────────

def work(ring: str, run_date: str) -> bool:
    print(f"Processing R{ring} for {run_date}")
    cfg = spec_cfg(ring, run_date)
    sp  = spectra(cfg)

    # Build 24‑h trace by concatenating hourly reads (like helicorder)
    loc = dict(Z="10", U="", V="", W="")[ring]
    seed = f"BW.ROMY.{loc}.BJ{ring}"

    big_stream = Stream()
    for hr in range(24):
        t1 = UTCDateTime(run_date) + hr * 3600
        t2 = t1 + CFG["chunk_sec"]
        st = read_sds(SDS_ARCH, seed, t1 - 30, t2 + 30)
        if st:
            big_stream += st

    if not big_stream:
        print("  ✖ No data – writing placeholder")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5,0.55,f"R{ring} Rotation Spectrum\n{run_date}",ha='center',va='center',fontsize=16,weight='bold')
        ax.text(0.5,0.25,'No data available',ha='center',va='center',fontsize=12,color='crimson')
        ax.axis('off')
        out_png = FIG_DIR / f"rotation_spectrum_R{ring}.png"
        fig.savefig(out_png,dpi=CFG['dpi'],bbox_inches='tight'); plt.close(fig)
        return False

    big_stream.merge(method=1, fill_value="interpolate")
    sp.add_trace(big_stream[0])

    sp.get_collection(cfg["tinterval"], cfg["toverlap"], cfg["method"])
    if cfg["apply_average"]:
        sp.get_fband_average(cfg["fraction_of_octave"], cfg["averaging"])
    if cfg["quality_filter"]:
        sp.classify_collection_quality(threshold=cfg["threshold"],
                                       zero_seq_limit=cfg["zero_seq_limit"],
                                       high_seq_limit=cfg["high_seq_limit"],
                                       flat_seq_limit=cfg["flat_seq_limit"])

    out_png = FIG_DIR / f"rotation_spectrum_R{ring}.png"
    plot_spectra(sp, ring, run_date, out_png)
    print(f"  ✔ saved → {out_png}")
    return True

# ─────────── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ROMY Rotation‑Spectra Generator (colour‑matched)")
    print(f"Date : {RUN_DATE}")
    ok = sum(work(r, RUN_DATE) for r in RINGS)
    print(f"✔ Completed {ok}/{len(RINGS)} rings")
