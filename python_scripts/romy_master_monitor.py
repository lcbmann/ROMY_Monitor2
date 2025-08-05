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
import sys, os, subprocess, time
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────── User-specific paths ─────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "python_scripts"

# Check if we're in a virtual environment
PYTHON_CMD = sys.executable

# ─────────── Configuration ───────────────────────────────────────────────
CFG = dict(
    rings = ["U", "V", "W", "Z"],           # All ROMY rings
    run_date = (sys.argv[1] if len(sys.argv) > 1 
                else str(date.today() - timedelta(days=1))),  # Default: yesterday
    max_workers = 2,                        # Parallel execution limit
    timeout = 1800,                         # 30 minutes timeout per script
)

# ─────────── Script definitions ──────────────────────────────────────────
SCRIPTS = {
    "sagnac_spectrum": "romy_make_sagnac_spectrum.py",
    "rotation_psd": "romy_make_rotation_psd.py", 
    "helicorder": "romy_make_helicorder.py",
    "combined_spectra": "romy_make_spectra.py",
    "oscilloscope": "romy_make_oszi.py",
}

# ─────────── Helper: run script with error handling ─────────────────────
def run_script(script_name, script_file, args=None, timeout=CFG["timeout"]):
    """Run a Python script with proper error handling and logging."""
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
    for script_type in ["sagnac_spectrum", "rotation_psd", "helicorder", "combined_spectra"]:
        script_file = SCRIPTS[script_type]
        script_name = f"{script_type}_{ring}"
        args = [ring]
        if CFG["run_date"]:
            args.append(CFG["run_date"])
        
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
    
    # Oscilloscope summary
    osc_result = results.get("oscilloscope", {})
    osc_status = "✅" if osc_result.get("success", False) else "❌"
    print(f"📷 Oscilloscope: {osc_status}")
    
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
