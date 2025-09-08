#!/usr/bin/env python3
"""ROMY Monitor – Build Live Rolling Archive

Maintains last N days (default 7) of daily helicorder + rotation spectrum PNGs
and a manifest for the web UI.

Creates (per date under docs/figures/live/<YYYY-MM-DD>/):
    helicorder_live_R<ring>_<date>.png
    rotation_spectrum_live_R<ring>_<date>.png

Also:
    • Skips dates already complete
    • Prunes directories older than retention window
    • Writes docs/figures/live/manifest.json

CLI:
    python romy_build_live_archive.py [--days N]

Designed for daily cron shortly after UTC midnight.
Uses the same core scripts as the main monitoring system to ensure consistency.
"""
from __future__ import annotations
import argparse, json, sys, shutil, subprocess
from pathlib import Path
from datetime import date, timedelta, datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = REPO_ROOT / 'docs' / 'figures' / 'live'
LIVE_ROOT.mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = REPO_ROOT / "python_scripts"
PYTHON_CMD = sys.executable

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

def run_script(script_name, args=None, timeout=1800):
    """Run a Python script with proper error handling."""
    cmd = [PYTHON_CMD, str(SCRIPT_DIR / script_name)]
    if args:
        cmd.extend(args)
    
    cmd_str = " ".join(cmd)
    print(f"🚀 Running: {cmd_str}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True, None
        else:
            print(f"   Error: {result.stderr.strip()}")
            return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)

def generate_for_date(date_str: str):
    day_dir = LIVE_ROOT / date_str
    day_dir.mkdir(exist_ok=True)
    # Skip if everything already there
    if have_all_images(day_dir, date_str):
        print(f'✔ {date_str} complete – skip')
        return
    print(f'… generating {date_str}')
    for r in RINGS:
        # Run helicorder script
        h_path = day_dir / f'helicorder_live_R{r}_{date_str}.png'
        if not h_path.exists():
            try:
                # Run the helicorder script with output to our live dir
                success, error = run_script("romy_make_helicorder.py", [r, date_str])
                if success:
                    # Copy the result to our live directory with the appropriate name
                    src = REPO_ROOT / "new_figures" / f"helicorder_R{r}.png"
                    if src.exists():
                        shutil.copy2(src, h_path)
                        print(f"  ✓ helicorder R{r} {date_str}")
                    else:
                        print(f"  ✖ helicorder R{r} {date_str}: output file not found")
            except Exception as e:
                print(f"  ✖ helicorder R{r} {date_str}: {e}")
        
        # Run rotation spectrum script
        s_path = day_dir / f'rotation_spectrum_live_R{r}_{date_str}.png'
        if not s_path.exists():
            try:
                # Run the rotation spectrum script
                success, error = run_script("romy_make_rotation.py", [r, date_str])
                if success:
                    # Copy the result to our live directory with the appropriate name
                    src = REPO_ROOT / "new_figures" / f"rotation_spectrum_R{r}.png"
                    if src.exists():
                        shutil.copy2(src, s_path)
                        print(f"  ✓ rotation R{r} {date_str}")
                    else:
                        print(f"  ✖ rotation R{r} {date_str}: output file not found")
            except Exception as e:
                print(f"  ✖ rotation R{r} {date_str}: {e}")

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
