#!/bin/bash
# =============================================================================
# Run All Dynamic NeRF Training
# This script trains both Straightforward and Deformation networks 
# on all D-NeRF dataset scenes
# =============================================================================

# Exit on error
set -e

# Configuration
SCENES=("bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex")
METHODS=("straightforward" "deform")

# Parse arguments
RUN_STRAIGHTFORWARD=true
RUN_DEFORM=true
SELECTED_SCENES=()

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --straightforward    Run only straightforward method"
    echo "  --deform             Run only deformation method"
    echo "  --scene SCENE        Run only specified scene (can be repeated)"
    echo "  --help               Show this help message"
    echo ""
    echo "Available scenes: ${SCENES[*]}"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --straightforward)
            RUN_DEFORM=false
            shift
            ;;
        --deform)
            RUN_STRAIGHTFORWARD=false
            shift
            ;;
        --scene)
            SELECTED_SCENES+=("$2")
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# If specific scenes selected, use those; otherwise use all
if [ ${#SELECTED_SCENES[@]} -gt 0 ]; then
    SCENES=("${SELECTED_SCENES[@]}")
fi

# Print configuration
echo "============================================="
echo "Dynamic NeRF Training Configuration"
echo "============================================="
echo "Scenes: ${SCENES[*]}"
echo "Run Straightforward: $RUN_STRAIGHTFORWARD"
echo "Run Deformation: $RUN_DEFORM"
echo "============================================="
echo ""

# Create logs directory if not exists
mkdir -p logs

# Training loop
for scene in "${SCENES[@]}"; do
    echo ""
    echo "============================================="
    echo "Processing scene: $scene"
    echo "============================================="
    
    # Straightforward method
    if [ "$RUN_STRAIGHTFORWARD" = true ]; then
        config_file="configs/${scene}_straightforward.txt"
        if [ -f "$config_file" ]; then
            echo ""
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training Straightforward on $scene..."
            python train.py --config "$config_file"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Straightforward on $scene"
        else
            echo "Warning: Config file not found: $config_file"
        fi
    fi
    
    # Deformation method
    if [ "$RUN_DEFORM" = true ]; then
        config_file="configs/${scene}_deform.txt"
        if [ -f "$config_file" ]; then
            echo ""
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training Deformation on $scene..."
            python train.py --config "$config_file"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Deformation on $scene"
        else
            echo "Warning: Config file not found: $config_file"
        fi
    fi
done

echo ""
echo "============================================="
echo "All training completed!"
echo "============================================="
echo ""

# Summary
echo "Training Summary:"
for scene in "${SCENES[@]}"; do
    echo "  $scene:"
    if [ "$RUN_STRAIGHTFORWARD" = true ]; then
        log_dir="logs/dnerf_straightforward_${scene}"
        if [ -d "$log_dir" ]; then
            echo "    - Straightforward: $log_dir"
        fi
    fi
    if [ "$RUN_DEFORM" = true ]; then
        log_dir="logs/dnerf_deformation_${scene}"
        if [ -d "$log_dir" ]; then
            echo "    - Deformation: $log_dir"
        fi
    fi
done
