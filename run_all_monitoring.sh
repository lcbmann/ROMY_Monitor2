#!/bin/bash
# ROMY Master Monitoring Script
# =============================
# Runs all ROMY monitoring scripts for complete data processing

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)/python_scripts"
RINGS="U V W Z"
RUN_DATE="${1:-$(date -d 'yesterday' +%Y-%m-%d)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    printf "\033[0;34mℹ️  %s\033[0m\n" "$1"
}

log_success() {
    printf "\033[0;32m✅ %s\033[0m\n" "$1"
}

log_error() {
    printf "\033[0;31m❌ %s\033[0m\n" "$1"
}

log_warning() {
    printf "\033[1;33m⚠️  %s\033[0m\n" "$1"
}

# Script execution with error handling
run_script() {
    local script_name="$1"
    local script_file="$2"
    shift 2
    local args="$@"
    
    log_info "Starting: $script_name"
    
    if python "$SCRIPT_DIR/$script_file" $args; then
        log_success "Completed: $script_name"
        return 0
    else
        log_error "Failed: $script_name"
        return 1
    fi
}

# Main execution
main() {
    echo "🔥 ROMY Master Monitoring Script"
    echo "=================================="
    log_info "Processing date: $RUN_DATE"
    log_info "Target rings: $RINGS"
    log_info "Script directory: $SCRIPT_DIR"
    echo ""
    
    local start_time=$(date +%s)
    local total_scripts=0
    local successful_scripts=0
    
    # Phase 1: Ring-specific scripts
    log_info "Phase 1: Processing ring-specific scripts..."
    echo ""
    
    for ring in $RINGS; do
        log_info "Processing Ring $ring..."
        
        # Sagnac Spectrum
        if run_script "Sagnac Spectrum $ring" "romy_make_sagnac_spectrum.py" "$ring" "$RUN_DATE"; then
            successful_scripts=$((successful_scripts + 1))
        fi
        total_scripts=$((total_scripts + 1))
        
        # Rotation Spectrum
        if run_script "Rotation Spectrum $ring" "romy_make_rotation.py" "$ring" "$RUN_DATE"; then
            successful_scripts=$((successful_scripts + 1))
        fi
        total_scripts=$((total_scripts + 1))
        
        # Helicorder
        if run_script "Helicorder $ring" "romy_make_helicorder.py" "$ring" "$RUN_DATE"; then
            successful_scripts=$((successful_scripts + 1))
        fi
        total_scripts=$((total_scripts + 1))
        
        # Combined Spectra (original script)
        if run_script "Combined Spectra $ring" "romy_make_spectra.py" "$ring" "$RUN_DATE"; then
            successful_scripts=$((successful_scripts + 1))
        fi
        total_scripts=$((total_scripts + 1))
        
        echo ""
    done
    
    # Phase 2: Oscilloscope capture
    log_info "Phase 2: Oscilloscope capture..."
    if run_script "Oscilloscope" "romy_make_oszi.py"; then
        successful_scripts=$((successful_scripts + 1))
    fi
    total_scripts=$((total_scripts + 1))
    
    # Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    echo "📋 EXECUTION SUMMARY"
    echo "===================="
    log_info "Total execution time: ${duration}s"
    log_info "Scripts completed: $successful_scripts/$total_scripts"
    
    if [ $successful_scripts -eq $total_scripts ]; then
        log_success "All monitoring scripts completed successfully! 🎉"
        return 0
    else
        log_warning "Some scripts failed. Check output above for details."
        return 1
    fi
}

# Help function
show_help() {
    echo "ROMY Master Monitoring Script"
    echo ""
    echo "Usage: $0 [DATE]"
    echo ""
    echo "Arguments:"
    echo "  DATE    Date to process (YYYY-MM-DD format)"
    echo "          Default: yesterday"
    echo ""
    echo "Examples:"
    echo "  $0                    # Process yesterday's data"
    echo "  $0 2025-07-21         # Process specific date"
    echo ""
    echo "This script runs the following for all rings (U,V,W,Z):"
    echo "  - Sagnac spectrum generation"
    echo "  - Rotation spectrum analysis"  
    echo "  - Helicorder generation"
    echo "  - Combined spectra plots"
    echo "  - Oscilloscope capture"
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Run main function
main "$@"
