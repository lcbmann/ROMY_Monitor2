#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – Helicorder Only
======================

Reads one day of rotation (BJ*) miniSEED from the local
**file system** and produces only the helicorder (drumplot) visualization

    <repo>/figures/helicorder_R<ring>.png
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
    sys.exit("Usage:  python romy_make_helicorder.py  <Z|U|V|W>  [YYYY-MM-DD]")

RING     = sys.argv[1].upper()
RUN_DATE = (sys.argv[2] if len(sys.argv) > 2
            else str((date.today() - timedelta(days=1))))   # yesterday (local date)

LOC_CODE   = dict(Z="10", U="", V="", W="")[RING]           # BJ* location

CFG = dict(
    chunk_sec = 3600,        # seconds per hour window
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

# ─────────── Helper: create helicorder figure ───────────────────────────
def make_helicorder_figure(rot_traces):
    """Create helicorder (drumplot) figure only."""
    colours = plt.cm.jet_r(np.linspace(0, 1, len(rot_traces)))
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # helicorder
    for i, tr in enumerate(rot_traces[::-1]):      # oldest on top
        if len(tr) == 0:
            continue
        t_vec = np.linspace(0, 60, len(tr))        # individual time axis per hour
        norm  = np.nanmax(np.abs(tr)) or 1.0
        ax.plot(t_vec, 0.8 * tr / norm + i,
                color=colours[i], lw=.5)

    ax.set(ylim=(-1, 24),
           yticks=np.arange(0, 24, 3),
           yticklabels=[f"{h:02d}:00" for h in range(23, -1, -3)],
           xlabel="Time (minutes)",
           ylabel="Hour (UTC)",
           title=f"Rotation Helicorder R{RING} – {RUN_DATE}")

    ax.grid(ls=":")
    fig.tight_layout()
    return fig

# ═══════════════════════════ main routine ════════════════════════════════
def main():
    """Main execution function."""
    print(f"Processing helicorder for R{RING} on {RUN_DATE}")

    seed_rot = f"BW.ROMY.{LOC_CODE}.BJ{RING}"
    rots = []

    for hour in tqdm(range(24), desc=f"Day {RUN_DATE}", ncols=74):
        t1 = UTCDateTime(RUN_DATE) + hour*3600
        t2 = t1 + CFG["chunk_sec"]

        st_rot = read_sds(SDS_ARCH, seed_rot, t1-30, t2+30)
        if not st_rot:
            rots.append(np.array([]))  # append empty array for missing hours
            continue

        tr_rot = st_rot[0]
        tr_rot.trim(t1, t2)
        rots.append(tr_rot.data.astype(float))

    if all(len(r) == 0 for r in rots):
        print(f"✖ No usable data found for {RUN_DATE}")
        return

    # ---------- figure ----------------------------------------------------
    fig = make_helicorder_figure(rots)
    
    try:
        png_name = FIG_DIR / f"helicorder_R{RING}.png"
        fig.savefig(png_name, dpi=CFG["dpi"], bbox_inches="tight",
                   facecolor='white', edgecolor='none')
        print(f"✔ saved figure → {png_name}")
    except PermissionError:
        png_name = (REPO_ROOT / "local_output" /
                    f"helicorder_R{RING}.png")
        png_name.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_name, dpi=CFG["dpi"], bbox_inches="tight",
                   facecolor='white', edgecolor='none')
        print(f"✔ saved figure → {png_name} (fallback)")
    
    plt.close(fig)
    print("✔ Helicorder generation completed")

# ─────────── entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()
