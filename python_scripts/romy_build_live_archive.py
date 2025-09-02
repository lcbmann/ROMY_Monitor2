#!/usr/bin/env python3
"""Build (or incrementally update) a rolling archive of live helicorder &
rotation spectra images for the last N days (default 7).

Features:
  * Generates missing per‑day images (skips those already present).
  * Stores images under  docs/figures/live/<YYYY-MM-DD>/
        helicorder_live_R<ring>_<date>.png
        rotation_spectrum_live_R<ring>_<date>.png
  * Deletes any date directories older than retention window.
  * Writes a manifest JSON: docs/figures/live/manifest.json for the web UI.

Use:
    python python_scripts/romy_build_live_archive.py [--days 7]

Intended to be run via cron (e.g. once per day shortly after midnight UTC).
"""
from __future__ import annotations
import argparse, json, sys, shutil
from pathlib import Path
from datetime import date, timedelta, datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = REPO_ROOT / 'docs' / 'figures' / 'live'
LIVE_ROOT.mkdir(parents=True, exist_ok=True)

sys.path.append(str(REPO_ROOT / 'live'))
try:
    from live_generation import generate_helicorder, generate_rotation_spectra
except Exception as e:
    print('✖ Failed to import live_generation:', e)
    sys.exit(1)

RINGS = list('ZUVW')

def iso(d: date) -> str:
    return d.strftime('%Y-%m-%d')

def build_date_list(days: int) -> list[str]:
    today = date.today()
    return [iso(today - timedelta(days=i)) for i in range(days)]

def have_all_images(day_dir: Path, date_str: str) -> bool:
    for r in RINGS:
        h = day_dir / f'helicorder_live_R{r}_{date_str}.png'
        s = day_dir / f'rotation_spectrum_live_R{r}_{date_str}.png'
        if not (h.exists() and s.exists()):
            return False
    return True

def generate_for_date(date_str: str):
    day_dir = LIVE_ROOT / date_str
    day_dir.mkdir(exist_ok=True)
    # Skip if everything already there
    if have_all_images(day_dir, date_str):
        print(f'✔ {date_str} complete – skip')
        return
    print(f'… generating {date_str}')
    for r in RINGS:
        h_path = day_dir / f'helicorder_live_R{r}_{date_str}.png'
        if not h_path.exists():
            try:
                generate_helicorder(r, date_str, day_dir)
            except Exception as e:
                print(f'  ✖ helicorder R{r} {date_str}: {e}')
        s_path = day_dir / f'rotation_spectrum_live_R{r}_{date_str}.png'
        if not s_path.exists():
            try:
                generate_rotation_spectra(r, date_str, day_dir)
            except Exception as e:
                print(f'  ✖ rotation R{r} {date_str}: {e}')

def prune_old(retention_days: int):
    cutoff = datetime.utcnow().date() - timedelta(days=retention_days)
    for p in LIVE_ROOT.iterdir():
        if not p.is_dir():
            continue
        try:
            d = datetime.strptime(p.name, '%Y-%m-%d').date()
        except ValueError:
            continue
        if d < cutoff:
            print(f'… deleting old {p.name}')
            shutil.rmtree(p, ignore_errors=True)

def build_manifest(retention_days: int):
    manifest = {
        'generated_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'retention_days': retention_days,
        'dates': [],
        'images': {}
    }
    # Collect directories (sorted descending by date)
    dirs = []
    for p in LIVE_ROOT.iterdir():
        if p.is_dir():
            try:
                datetime.strptime(p.name, '%Y-%m-%d')
            except ValueError:
                continue
            dirs.append(p.name)
    dirs.sort(reverse=True)
    manifest['dates'] = dirs
    for d in dirs:
        day_dir = LIVE_ROOT / d
        manifest['images'][d] = {}
        for r in RINGS:
            entry = {}
            h = day_dir / f'helicorder_live_R{r}_{d}.png'
            s = day_dir / f'rotation_spectrum_live_R{r}_{d}.png'
            if h.exists():
                entry['helicorder'] = str(h.relative_to(REPO_ROOT / 'docs'))
            if s.exists():
                entry['rotation'] = str(s.relative_to(REPO_ROOT / 'docs'))
            if entry:
                manifest['images'][d][r] = entry
    out = LIVE_ROOT / 'manifest.json'
    out.write_text(json.dumps(manifest, indent=2))
    print(f'✔ wrote manifest with {len(dirs)} dates → {out}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=7, help='Retention window (days) & generation range (default 7)')
    args = ap.parse_args()
    date_list = build_date_list(args.days)
    for d in date_list:
        generate_for_date(d)
    prune_old(args.days)
    build_manifest(args.days)
    print('✔ archive build complete')

if __name__ == '__main__':
    main()
