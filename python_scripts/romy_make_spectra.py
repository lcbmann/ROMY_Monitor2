#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – daily Sagnac PSD  +  mini-helicorder (side-by-side)
=========================================================

Reads one day of beat-note (FJ*) + rotation (BJ*) miniSEED from the local
file system and produces:

    romy_autodata/<year>/R<ring>/spectra/R<ring>_YYYYMMDD_spectra.pkl
    <repo>/new_figures/html_sagnacspectra_R<ring>.png

This combines the visual styles of:
  • romy_make_sagnac_spectrum.py  (fixed PSD y-limits, legend, units in brackets,
    helicorder-matched colors, alpha=0.3, reverse draw order to avoid red covering)
  • romy_make_helicorder.py       (hour slicing, de-mean & normalize per hour,
    same colormap, alpha=0.8, identical hour→color mapping)
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
MOUNT     = Path(os.getenv("ROMY_MOUNT", "/import/freenas-ffb-01-data")).expanduser()
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR   = REPO_ROOT / "new_figures"; FIG_DIR.mkdir(exist_ok=True)

# ─────────── CLI arguments ───────────────────────────────────────────────
if len(sys.argv) < 2 or sys.argv[1].upper() not in "ZUVW":
    sys.exit("Usage:  python romy_make_spectra.py  <Z|U|V|W>  [YYYY-MM-DD]")

RING     = sys.argv[1].upper()
RUN_DATE = (sys.argv[2] if len(sys.argv) > 2
            else str((date.today() - timedelta(days=1))))   # yesterday (local date)

EXPECTED_F = dict(Z=553, U=302, V=448, W=448)[RING]
LOC_CODE   = dict(Z="10", U="", V="", W="")[RING]           # BJ* location

CFG = dict(
    # Sagnac PSD settings (match romy_make_sagnac_spectrum.py)
    band        = 30,            # ± Hz plotted around EXPECTED_F
    chunk_sec   = 3600,          # seconds per PSD window
    seg_sec     = 600,           # Welch segment length
    counts2V    = 0.59604645e-6, # counts → volts
    psd_alpha   = 0.3,           # transparency per PSD line
    psd_ylim    = (1e-12, 1e1),  # fixed for inter-comparability
    # Helicorder settings (match romy_make_helicorder.py)
    tinterval   = 3600,          # 1-hour windows
    toverlap    = 0,
    heli_alpha  = 0.8,
    # Shared
    cmap        = "rainbow",
    dpi         = 150,
)

# ─────────── Derived data paths ──────────────────────────────────────────
AUTOD_YEAR = MOUNT / "romy_autodata" / RUN_DATE[:4] / f"R{RING}"
PKL_DIR    = AUTOD_YEAR / "spectra"; PKL_DIR.mkdir(parents=True, exist_ok=True)
SDS_ARCH   = MOUNT / "romy_archive"  # SDS miniSEED root

# ─────────── SDS readers ─────────────────────────────────────────────────
def read_sds_raw(archive: Path, seed: str, t1, t2) -> Stream:
    """Beat-note (FJ*) read: minimal processing, keep Volts conversion outside."""
    from obspy.clients.filesystem.sds import Client
    if not archive.is_dir():
        return Stream()
    n, s, l, c = seed.split(".")
    cli = Client(str(archive), sds_type="D", format="MSEED")
    try:
        st = cli.get_waveforms(n, s, l, c, UTCDateTime(t1)-10, UTCDateTime(t2)+10, merge=-1)
    except Exception:
        return Stream()
    st = st.trim(UTCDateTime(t1), UTCDateTime(t2), nearest_sample=False)
    return st

def read_sds_rot(archive: Path, seed: str, t1, t2) -> Stream:
    """Rotation (BJ*) read: detrend, merge fill=0, trim (helicorder style)."""
    from obspy.clients.filesystem.sds import Client
    if not archive.is_dir():
        return Stream()
    n, s, l, c = seed.split(".")
    cli = Client(str(archive), sds_type="D", format="MSEED")
    try:
        st = cli.get_waveforms(n, s, l, c, UTCDateTime(t1)-10, UTCDateTime(t2)+10, merge=-1)
    except Exception:
        return Stream()
    st = st.detrend("linear").detrend("demean")
    st = st.merge(fill_value=0)
    st = st.trim(UTCDateTime(t1), UTCDateTime(t2), nearest_sample=False)
    return st

# ─────────── Welch PSD ───────────────────────────────────────────────────
def welch_psd(arr: np.ndarray, fs: float, win_sec: int):
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

# ─────────── Time windows / gap detection (helicorder helpers) ──────────
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

# ─────────── Color ramp (shared by both panels) ──────────────────────────
def hour_colors(n_intervals: int, cmap_name: str):
    """Exact discrete ramp (like helicorder): colors[h] is hour h color."""
    base = plt.get_cmap(cmap_name)
    return base(np.linspace(0, 1, n_intervals))

# ─────────── Left panel: Sagnac PSD (as in romy_make_sagnac_spectrum.py) ─
def draw_sagnac_psd(ax, freq: np.ndarray, hour_psds: dict[int, np.ndarray], colors):
    hours_present = sorted(hour_psds.keys())
    # Draw from 23 → 00 so late reds are underneath
    for h in sorted(hours_present, reverse=True):
        ax.plot(
            freq,
            hour_psds[h],
            color=colors[h],
            alpha=CFG["psd_alpha"],
            lw=1.0,
            label=f"{h:02d}:00",
            zorder=1,
        )
    # Median on top
    try:
        stack = np.vstack([hour_psds[h] for h in hours_present])
        med   = np.nanmedian(stack, axis=0)
        ax.plot(freq, med, color="k", lw=1.4, label="median", zorder=2)
    except Exception:
        pass

    ax.set_xlim(EXPECTED_F - CFG["band"], EXPECTED_F + CFG["band"])
    ax.set_yscale("log")
    ax.set_ylim(*CFG["psd_ylim"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V^2/Hz)")
    ax.set_title(f"Sagnac Spectrum R{RING} – {RUN_DATE}")
    ax.grid(ls="--", alpha=0.4)

    # Legend (hours → colors)
    leg = ax.legend(ncol=2, fontsize=9, frameon=True)
    if leg is not None:
        for line in leg.get_lines():
            line.set_linewidth(3.0)

# ─────────── Right panel: mini-helicorder (as in romy_make_helicorder.py) ─
def draw_mini_helicorder(ax, tr, tints, dt, colors):
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
                ax.plot(times, (data / norm) - j, color=color, alpha=CFG["heli_alpha"], lw=1)

        yticks.append(-j)
        yticklabels.append(t1.strftime('%H:%M'))

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Hour of day (UTC)')
    ax.set_xlim(0, 60)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(False)
    ax.set_title(f"Rotation Helicorder R{RING} – {RUN_DATE}")

    # Tight y-limits so 00:00 and 23:00 fill the frame
    m = 0.8
    ax.set_ylim(m, -(len(tints) - 1) - m)

# ─────────── Main ────────────────────────────────────────────────────────
def main():

    # Seeds
    seed_raw = f"BW.DROMY..FJ{RING}"                  # beat-note
    seed_rot = f"BW.ROMY.{LOC_CODE}.BJ{RING}"         # rotation

    # Prepare colors for 24 hours (shared by both panels)
    colors24 = hour_colors(24, CFG["cmap"])

    # Build rotation stream for the whole day (helicorder panel)
    day_start = UTCDateTime(RUN_DATE)
    day_end   = day_start + 86400
    tints     = time_intervals(day_start, day_end, CFG["tinterval"], CFG["toverlap"])

    st_rot_day = read_sds_rot(SDS_ARCH, seed_rot, day_start, day_end)
    have_rot   = bool(st_rot_day) and len(st_rot_day[0].data) > 0
    if have_rot:
        tr_rot = st_rot_day[0]
        dt_rot = tr_rot.stats.delta

    # Compute Sagnac PSDs per hour
    hour_psds = {}
    fig_freq  = None

    for hour in tqdm(range(24), desc=f"Day {RUN_DATE}", ncols=74):
        t1 = day_start + hour * 3600
        t2 = t1 + CFG["chunk_sec"]

        st_raw = read_sds_raw(SDS_ARCH, seed_raw, t1, t2)
        if not st_raw or len(st_raw[0].data) == 0:
            continue

        tr_raw = st_raw[0].copy()
        tr_raw.data = tr_raw.data.astype(float) * CFG["counts2V"]  # counts → volts

        f_all, p_all = welch_psd(tr_raw.data, tr_raw.stats.sampling_rate, CFG["seg_sec"])

        mask = (f_all >= EXPECTED_F - CFG["band"]) & (f_all <= EXPECTED_F + CFG["band"])
        if not np.any(mask):
            continue

        if fig_freq is None:
            fig_freq = f_all[mask]
        hour_psds[hour] = p_all[mask]

        del st_raw, tr_raw
        gc.collect()

    # Write daily pickle of Sagnac spectra (full‐band frequencies & array of PSDs)
    if hour_psds:
        # Rebuild full_freq and psds stack for pickle
        # (save *untrimmed* frequency axis if available; else save trimmed)
        # Since we only kept band-limited values, store the used freq (fig_freq)
        pkl_name = PKL_DIR / f"R{RING}_{RUN_DATE.replace('-','')}_spectra.pkl"
        try:
            DataFrame({"frequencies": [fig_freq],
                       "psds":       [np.vstack([hour_psds[h] for h in sorted(hour_psds.keys())])]}).to_pickle(pkl_name)
            print("✔ saved pickle  →", pkl_name)
        except PermissionError:
            local_dir = (REPO_ROOT / "local_output" / RUN_DATE[:4] / f"R{RING}")
            local_dir.mkdir(parents=True, exist_ok=True)
            pkl_name = local_dir / f"R{RING}_{RUN_DATE.replace('-','')}_spectra.pkl"
            DataFrame({"frequencies": [fig_freq],
                       "psds":       [np.vstack([hour_psds[h] for h in sorted(hour_psds.keys())])]}).to_pickle(pkl_name)
            print("✔ saved pickle  →", pkl_name, "(fallback)")

    # ── Figure (side-by-side) ────────────────────────────────────────────
    fig, (ax_psd, ax_heli) = plt.subplots(1, 2, figsize=(18, 8))
    plt.subplots_adjust(wspace=.15)

    # Left: Sagnac PSD (or placeholder)
    if hour_psds:
        draw_sagnac_psd(ax_psd, fig_freq, hour_psds, colors24)
    else:
        ax_psd.text(0.5, 0.55, f"R{RING} Sagnac Spectrum\n{RUN_DATE}",
                    ha='center', va='center', fontsize=16, weight='bold')
        ax_psd.text(0.5, 0.25, 'No Sagnac data', ha='center', va='center',
                    fontsize=12, color='crimson')
        ax_psd.axis('off')

    # Right: mini-helicorder (or placeholder)
    if have_rot:
        draw_mini_helicorder(ax_heli, tr_rot, tints, dt_rot, colors24)
    else:
        ax_heli.text(0.5, 0.55, f"R{RING} Helicorder\n{RUN_DATE}",
                     ha='center', va='center', fontsize=16, weight='bold')
        ax_heli.text(0.5, 0.25, 'No rotation data', ha='center', va='center',
                     fontsize=12, color='crimson')
        ax_heli.axis('off')

    fig.tight_layout()

    # Save
    try:
        png_name = FIG_DIR / f"html_sagnacspectra_R{RING}.png"
        fig.savefig(png_name, dpi=CFG["dpi"], bbox_inches="tight",
                    facecolor='white', edgecolor='none')
        print("✔ saved figure →", png_name)
    except PermissionError:
        png_name = (REPO_ROOT / "local_output" / f"html_sagnacspectra_R{RING}.png")
        png_name.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_name, dpi=CFG["dpi"], bbox_inches="tight",
                    facecolor='white', edgecolor='none')
        print("✔ saved figure →", png_name, "(fallback)")

    plt.close(fig)

# ─────────── entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()
