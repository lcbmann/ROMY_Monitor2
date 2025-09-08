#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – Helicorder Only
======================

Reads one day of rotation (BJ*) miniSEED from the local file system
and produces only the helicorder (drumplot) visualization:

    <repo>/new_figures/helicorder_R<ring>.png
"""

import sys, warnings, os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Stream

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["agg.path.chunksize"] = 10_000

# Paths
MOUNT = Path(os.getenv("ROMY_MOUNT", "/import/freenas-ffb-01-data")).expanduser()
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "new_figures"
FIG_DIR.mkdir(exist_ok=True)

# CLI
if len(sys.argv) < 2 or sys.argv[1].upper() not in "ZUVW":
    sys.exit("Usage:  python romy_make_helicorder.py  <Z|U|V|W>  [YYYY-MM-DD]")

RING = sys.argv[1].upper()
RUN_DATE = (sys.argv[2] if len(sys.argv) > 2
            else str((date.today() - timedelta(days=1))))  # yesterday (local date)

LOC_CODE = dict(Z="10", U="", V="", W="")[RING]

CFG = dict(
    tinterval=3600,       # 1-hour windows
    toverlap=0,
    dpi=150,
    cmap="rainbow",
    alpha=0.8,
)

SDS_ARCH = MOUNT / "romy_archive"  # SDS miniSEED root

# Helpers
def read_sds(archive: Path, seed: str, t1, t2) -> Stream:
    from obspy.clients.filesystem.sds import Client
    if not archive.is_dir():
        return Stream()
    net, sta, loc, cha = seed.split(".")
    cli = Client(str(archive), sds_type="D", format="MSEED")
    try:
        st = cli.get_waveforms(net, sta, loc, cha, UTCDateTime(t1)-10, UTCDateTime(t2)+10, merge=-1)
    except Exception:
        return Stream()
    st = st.detrend("linear").detrend("demean")
    st = st.merge(fill_value=0)
    st = st.trim(UTCDateTime(t1), UTCDateTime(t2), nearest_sample=False)
    return st

def time_intervals(tbeg: UTCDateTime, tend: UTCDateTime, t_interval: float, t_overlap: float):
    times = []
    t1, t2 = tbeg, tbeg + t_interval
    while t2 <= tend:
        times.append((t1, t2))
        t1 = t1 + t_interval - t_overlap
        t2 = t2 + t_interval - t_overlap
    return times

def has_gap(data: np.ndarray) -> bool:
    if data.size == 0:
        return True
    zero_mask = (data == 0.0)
    if zero_mask.any():
        starts = np.where(np.diff(np.concatenate(([False], zero_mask))))[0]
        ends   = np.where(np.diff(np.concatenate((zero_mask, [False]))))[0]
        m = min(len(starts), len(ends))
        if m > 0 and np.any((ends[:m] - starts[:m]) > 10):
            return True
    n_size = 20
    for j in range(0, max(0, len(data) - n_size), n_size):
        if np.all(data[j:j+n_size] == data[j]):
            return True
    return False

def make_helicorder(tr, tints, dt, title):
    n_intervals = len(tints)
    base = plt.get_cmap(CFG["cmap"])
    colors = base(np.linspace(0, 1, n_intervals))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    yticks, yticklabels = [], []

    for j, (t1, t2) in enumerate(tints):
        if t1 <= tr.stats.endtime and t2 >= tr.stats.starttime:
            tr_slice = tr.slice(max(t1, tr.stats.starttime), min(t2, tr.stats.endtime))
            data = np.asarray(tr_slice.data, dtype=float)
            if data.size > 0 and np.any(np.abs(data) > 0):
                # de-mean per hour → equal spacing
                m = np.nanmean(data)
                if np.isfinite(m):
                    data = data - m
                norm = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 1.0
                if norm == 0.0:
                    norm = 1.0
                start_min = float(tr_slice.stats.starttime - t1) / 60.0
                times = start_min + np.arange(data.size) * dt / 60.0
                color = 'grey' if has_gap(data) else colors[j]
                ax.plot(times, (data / norm) - j, color=color, alpha=CFG["alpha"], lw=1)

        yticks.append(-j)
        yticklabels.append(t1.strftime('%H:%M'))

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Hour of day (UTC)')
    ax.set_xlim(0, 60)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(False)
    ax.set_title(title)

    # >>> Tight y-limits so 00:00 and 23:00 fill the frame
    m = 0.8
    ax.set_ylim(m, -(n_intervals - 1) - m)

    fig.tight_layout()
    return fig

# main
def main():
    print(f"Processing helicorder for R{RING} on {RUN_DATE}")

    day_start = UTCDateTime(RUN_DATE)
    day_end   = day_start + 86400
    tints = time_intervals(day_start, day_end, CFG["tinterval"], CFG["toverlap"])

    seed_rot = f"BW.ROMY.{LOC_CODE}.BJ{RING}"
    st = read_sds(SDS_ARCH, seed_rot, day_start, day_end)

    if not st or len(st[0].data) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.55, f"R{RING} Helicorder\n{RUN_DATE}",
                ha='center', va='center', fontsize=16, weight='bold')
        ax.text(0.5, 0.25, 'No data available',
                ha='center', va='center', fontsize=12, color='crimson')
        ax.axis('off')
        out = FIG_DIR / f"helicorder_R{RING}.png"
        fig.savefig(out, dpi=CFG['dpi'], bbox_inches='tight'); plt.close(fig)
        print(f"✖ No usable data — wrote {out}")
        return

    tr = st[0]
    dt = tr.stats.delta

    title = f"Rotation Helicorder R{RING} – {RUN_DATE}"
    fig = make_helicorder(tr, tints, dt, title)

    out = FIG_DIR / f"helicorder_R{RING}.png"
    fig.savefig(out, dpi=CFG["dpi"], bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✔ saved figure → {out}")

if __name__ == "__main__":
    main()
