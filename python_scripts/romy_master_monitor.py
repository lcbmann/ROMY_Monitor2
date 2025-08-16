#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMY – Master Monitoring Script
===============================

Runs all ROMY monitoring scripts for complete data processing and visualization.
This script coordinates the execution of:
- Sagnac spectra generation for all rings
- Rotation PSD analysis for all rings  
- Helicorder generation for all rings
- Combined spectra plots for all rings
- Oscilloscope capture

Usage: python romy_master_monitor.py [YYYY-MM-DD]
"""

# ─────────── std-lib / 3-party ───────────────────────────────────────────
import sys, os, subprocess, time, argparse, shutil
from datetime import date, timedelta, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────── User-specific paths ─────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "python_scripts"

# Check if we're in a virtual environment
PYTHON_CMD = sys.executable

def _parse_cli():
    parser = argparse.ArgumentParser(
        description="Run ROMY monitoring scripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rings", nargs="*", default=["U","V","W","Z"],
                        help="Subset of rings to process (U V W Z). If omitted: all.")
    parser.add_argument("--date", dest="run_date", default=None,
                        help="Target date YYYY-MM-DD (defaults to yesterday)")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Parallel ring workers")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-script timeout (s)")
    args = parser.parse_args()
    # Validate rings
    rings = [r.upper() for r in args.rings if r.upper() in {"U","V","W","Z"}]
    if not rings:
        parser.error("No valid rings specified (choose from U V W Z)")
    # Resolve date
    if args.run_date is None:
        run_date = str(date.today() - timedelta(days=1))
    else:
        try:
            datetime.strptime(args.run_date, "%Y-%m-%d")
        except ValueError:
            parser.error("--date must be in YYYY-MM-DD format")
        run_date = args.run_date
    return dict(rings=rings, run_date=run_date, max_workers=args.max_workers, timeout=args.timeout)

# ─────────── Configuration (populated at runtime) ───────────────────────
CFG = {}

# ─────────── Script definitions ──────────────────────────────────────────
SCRIPTS = {
    "sagnac_spectrum": "romy_make_sagnac_spectrum.py",
    # rotation spectra script (colour matched) expects [RING [DATE]]
    "rotation_psd": "romy_make_rotation.py", 
    "helicorder": "romy_make_helicorder.py",
    "combined_spectra": "romy_make_spectra.py",
    "oscilloscope": "romy_make_oszi.py",
    "backscatter_full": "romy_make_backscatter_full.py",
    "beamwalk_full": "romy_make_beamwalk_full.py",
    "beatdrift_full": "romy_make_beatdrift_full.py",
    "environmentals_full": "romy_make_environmentals_full.py",
    "barometric_full": "romy_make_barometric_full.py",
    "sagnac_signal": "romy_make_sagnacsignal.py",
}

# ─────────── Helper: run script with error handling ─────────────────────
def run_script(script_name, script_file, args=None, timeout=None):
    """Run a Python script with proper error handling and logging."""
    if timeout is None:
        timeout = CFG.get("timeout", 1800)
    cmd = [PYTHON_CMD, str(SCRIPT_DIR / script_file)]
    if args:
        cmd.extend(args)
    
    cmd_str = " ".join(cmd)
    print(f"🚀 Starting: {script_name}")
    print(f"   Command: {cmd_str}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Completed: {script_name} ({duration:.1f}s)")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True, result.stdout, None
        else:
            print(f"❌ Failed: {script_name} ({duration:.1f}s)")
            print(f"   Error: {result.stderr.strip()}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {script_name} (>{timeout}s)")
        return False, "", "Script timeout"
    except Exception as e:
        print(f"💥 Exception: {script_name} - {str(e)}")
        return False, "", str(e)

# ─────────── Helper: run ring-specific scripts ──────────────────────────
def run_ring_scripts(ring):
    """Run all ring-specific scripts for a given ring."""
    ring_results = {}
    
    # Sequential execution for each ring (to avoid resource conflicts)
    for script_type in ["sagnac_spectrum", "rotation_psd", "helicorder", "combined_spectra", "sagnac_signal"]:
        script_file = SCRIPTS[script_type]
        script_name = f"{script_type}_{ring}"
        # Per‑script CLI conventions
        if script_type == "rotation_psd":
            # romy_make_rotation.py accepts optional ring (already per-ring here) and optional date
            args = [ring, CFG["run_date"]]
        elif script_type in {"sagnac_spectrum", "helicorder"}:
            args = [ring, CFG["run_date"]]
        elif script_type == "combined_spectra":
            # romy_make_spectra.py <RING> <DATE>
            args = [ring, CFG["run_date"]]
        elif script_type == "sagnac_signal":
            args = [ring, CFG["run_date"]]
        else:
            args = [ring, CFG["run_date"]]
        
        success, stdout, stderr = run_script(script_name, script_file, args)
        ring_results[script_type] = {
            "success": success,
            "stdout": stdout, 
            "stderr": stderr
        }
        
        # Small delay between scripts to reduce system load
        time.sleep(2)
    
    return ring, ring_results

# ═══════════════════════════ main routine ════════════════════════════════
def main():
    global CFG
    if not CFG:  # only parse once
        CFG = _parse_cli()
    """Main execution function."""
    print("🔥 ROMY Master Monitoring Script")
    print("=" * 50)
    print(f"📅 Processing date: {CFG['run_date']}")
    print(f"🎯 Target rings: {', '.join(CFG['rings'])}")
    print(f"🐍 Python executable: {PYTHON_CMD}")
    print(f"📂 Script directory: {SCRIPT_DIR}")
    print()
    
    start_time = time.time()
    results = {}
    
    # ─────────── Phase 1: Ring-specific scripts (parallel by ring) ────────
    print("📊 Phase 1: Processing ring-specific scripts...")
    
    with ThreadPoolExecutor(max_workers=CFG["max_workers"]) as executor:
        # Submit jobs for each ring
        future_to_ring = {
            executor.submit(run_ring_scripts, ring): ring 
            for ring in CFG["rings"]
        }
        
        # Collect results
        for future in as_completed(future_to_ring):
            ring = future_to_ring[future]
            try:
                ring, ring_results = future.result()
                results[ring] = ring_results
                print(f"🏁 Ring {ring} processing completed")
            except Exception as e:
                print(f"💥 Ring {ring} failed: {str(e)}")
                results[ring] = {"error": str(e)}
    
    print()
    
    # ─────────── Phase 2: Oscilloscope capture (independent) ──────────────
    print("📷 Phase 2: Oscilloscope capture...")
    
    success, stdout, stderr = run_script(
        "oscilloscope", 
        SCRIPTS["oscilloscope"]
    )
    results["oscilloscope"] = {
        "success": success,
        "stdout": stdout,
        "stderr": stderr
    }
    
    print()
    # ─────────── Phase 3: Full legacy replication scripts ────────────────
    print("🖼️  Phase 3: Full legacy plot replications...")
    legacy_scripts = [
        ("backscatter_full", []),
        ("beamwalk_full", []),
        ("beatdrift_full", []),
        ("environmentals_full", ["Z"]),  # environmentals primarily ring Z
        ("barometric_full", []),
    ]
    for key, args in legacy_scripts:
        success, stdout, stderr = run_script(key, SCRIPTS[key], args)
        results[key] = {"success": success, "stdout": stdout, "stderr": stderr}
        time.sleep(1)
    print()

    # ─────────── Ensure ring image completeness (placeholders) ───────────
    def ensure_full_ring_set(run_date: str):
        """Create placeholder images in new_figures for any missing ring outputs.

        Expected per ring: rotation_spectrum, sagnac_spectrum, helicorder,
        html_sagnacspectra, html_sagnacsignal.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        new_dir = REPO_ROOT / 'new_figures'
        new_dir.mkdir(exist_ok=True)
        expected_templates = [
            'rotation_spectrum_R{r}.png',
            'sagnac_spectrum_R{r}.png',
            'helicorder_R{r}.png',
            'html_sagnacspectra_R{r}.png',
            'html_sagnacsignal_R{r}.png',
        ]
        for r in ['U','V','W','Z']:
            for tmpl in expected_templates:
                fname = tmpl.format(r=r)
                fpath = new_dir / fname
                if not fpath.exists():
                    fig, ax = plt.subplots(figsize=(6,3.2))
                    ax.text(0.5,0.6,f"R{r} {fname.split('_R')[0]}",ha='center',va='center',fontsize=12,weight='bold')
                    ax.text(0.5,0.35, run_date, ha='center', va='center', fontsize=10, color='0.3')
                    ax.text(0.5,0.15,'(placeholder – no data)',ha='center',va='center',fontsize=9,color='crimson')
                    ax.axis('off')
                    fig.savefig(fpath, dpi=110, bbox_inches='tight'); plt.close(fig)
                    print(f"ℹ️  Placeholder created → {fpath.name}")
    ensure_full_ring_set(CFG['run_date'])

    # Sync replicated figures (and static assets) into docs/figures/new
    try:
        new_dir = REPO_ROOT / 'new_figures'
        docs_figs_new = REPO_ROOT / 'docs' / 'figures' / 'new'
        docs_figs_new.mkdir(parents=True, exist_ok=True)
        # Clean out old contents
        for old in docs_figs_new.glob('*.png'):
            try: old.unlink()
            except Exception: pass
        expected_patterns = [
            'rotation_spectrum_R*.png',
            'sagnac_spectrum_R*.png',
            'helicorder_R*.png',
            'html_sagnacspectra_R*.png',
            'html_sagnacsignal_R*.png',
            'html_oszi.png',
            'html_backscatter.png', 'html_beamwalk.png', 'html_beatdrift.png',
            'html_environmentals.png', 'html_romy_baro.png'
        ]
        copied = 0
        if new_dir.exists():
            # Copy everything present
            for png in new_dir.glob('*.png'):
                try:
                    shutil.copy2(png, docs_figs_new / png.name)
                    copied += 1
                except Exception as e:
                    print(f"⚠️  Copy failed for {png.name}: {e}")
            # Copy static beam‑wander images + oops fallback if referenced
            static_sources = [REPO_ROOT / 'docs' / 'figures', REPO_ROOT / 'figures']
            static_names = ['BW_ROMY_10_BJZ.png','BW_ROMY__BJU.png','BW_ROMY__BJV.png','BW_ROMY__BJW.png','oops.png']
            for name in static_names:
                if not (docs_figs_new / name).exists():
                    for src_dir in static_sources:
                        src_file = src_dir / name
                        if src_file.exists():
                            try:
                                shutil.copy2(src_file, docs_figs_new / name)
                                copied += 1
                            except Exception as e:
                                print(f"⚠️  Copy static failed for {name}: {e}")
                            break
            # Report missing expected patterns
            missing = []
            for pattern in expected_patterns:
                if not list(new_dir.glob(pattern)):
                    missing.append(pattern)
            if missing:
                print('⚠️  Missing expected outputs:', ', '.join(missing))
        print(f"🔄 Synced new_figures → docs/figures/new ({copied} files)")
    except Exception as e:
        print(f"⚠️  Figure sync skipped: {e}")
    
    # ─────────── Summary Report ────────────────────────────────────────────
    total_time = time.time() - start_time
    print("📋 EXECUTION SUMMARY")
    print("=" * 50)
    print(f"⏱️  Total execution time: {total_time:.1f}s")
    print()
    
    # Ring-by-ring summary
    for ring in CFG["rings"]:
        if ring in results and "error" not in results[ring]:
            ring_data = results[ring]
            successes = sum(1 for r in ring_data.values() if r.get("success", False))
            total_scripts = len(ring_data)
            print(f"🔍 Ring {ring}: {successes}/{total_scripts} scripts successful")
            
            for script_type, result in ring_data.items():
                status = "✅" if result.get("success", False) else "❌"
                print(f"    {status} {script_type}")
        else:
            print(f"💥 Ring {ring}: Failed to process")
    
    print()
    
    # Oscilloscope + legacy summary
    osc_result = results.get("oscilloscope", {})
    osc_status = "✅" if osc_result.get("success", False) else "❌"
    print(f"📷 Oscilloscope: {osc_status}")
    for key in ["backscatter_full","beamwalk_full","beatdrift_full","environmentals_full","barometric_full"]:
        if key in results:
            status = "✅" if results[key].get("success", False) else "❌"
            print(f"    {status} {key}")
    
    print()
    
    # Overall status
    total_successes = 0
    total_scripts = 0
    
    for ring in CFG["rings"]:
        if ring in results and "error" not in results[ring]:
            ring_data = results[ring]
            total_successes += sum(1 for r in ring_data.values() if r.get("success", False))
            total_scripts += len(ring_data)
    
    if results.get("oscilloscope", {}).get("success", False):
        total_successes += 1
    total_scripts += 1
    for key in ["backscatter_full","beamwalk_full","beatdrift_full","environmentals_full","barometric_full"]:
        if key in results:
            total_scripts += 1
            if results[key].get("success", False):
                total_successes += 1
    
    print(f"🏆 OVERALL: {total_successes}/{total_scripts} scripts completed successfully")
    
    if total_successes == total_scripts:
        print("🎉 All monitoring scripts completed successfully!")
        return 0
    else:
        print("⚠️  Some scripts failed. Check output above for details.")
        return 1

# ─────────── entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
