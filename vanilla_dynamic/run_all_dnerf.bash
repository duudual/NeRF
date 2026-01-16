#!/bin/bash
# =============================================================================
# Run All Dynamic NeRF Pipeline (Three-Stage)
# Stage 1: Train models on all scenes
# Stage 2: Render videos from trained models
# Stage 3: Evaluate models and generate metrics
# =============================================================================

# Exit on error
set -e

# =============================================================================
# Configuration
# =============================================================================
# All D-NeRF dataset scenes
SCENES=("bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex")

# =============================================================================
# Paths Configuration - Modify according to your environment
# =============================================================================
# Windows example:
# DATA_BASEDIR="D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data"
# MODEL_BASEDIR="D:/lecture/2.0_xk/CV/finalproject/NeRF/vanilla_dynamic"

# Linux example:
# DATA_BASEDIR="/media/fengwu/ZX1 1TB/code/cv_finalproject/data/D_NeRF_Dataset/data"
# MODEL_BASEDIR="/media/fengwu/ZX1 1TB/code/cv_finalproject/dynamic"

# Current configuration:
# DATA_BASEDIR="D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data"
# MODEL_BASEDIR="D:/lecture/2.0_xk/CV/finalproject/NeRF/vanilla_dynamic"

# =============================================================================
# Training Parameters
# =============================================================================
# Number of training iterations (50000-200000 for good results)
N_ITERS=10000

# Resolution: use --half_res for 400x400, remove for full 800x800
HALF_RES="--half_res"

# Learning rate (default: 5e-4, works well for both methods)
LRATE="5e-4"

# Learning rate decay (in 1000s of steps)
LRATE_DECAY=250

# Batch size (rays per gradient step)
N_RAND=1024

# Number of coarse/fine samples
N_SAMPLES=64
N_IMPORTANCE=128

# Logging frequency
I_PRINT=500
I_WEIGHTS=10000

# =============================================================================
# Video Rendering Parameters
# =============================================================================
VIDEO_N_FRAMES=120
VIDEO_FPS=30
VIDEO_TIME_MODES=("cycle")  # Options: "cycle", "linear", "fixed"

# =============================================================================
# Argument Parsing
# =============================================================================
RUN_STRAIGHTFORWARD=true
RUN_DEFORMATION=true
SELECTED_SCENES=()
RUN_TRAIN=true
RUN_RENDER=true
RUN_EVAL=true

# Methods to run (both by default)
METHODS=("straightforward" "deformation")

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Dynamic NeRF Training, Rendering, and Evaluation Pipeline"
    echo ""
    echo "This script trains both Straightforward (6D) and Deformation network"
    echo "approaches on D-NeRF dataset scenes, renders videos, and evaluates results."
    echo ""
    echo "Pipeline Control:"
    echo "  --train-only         Run only training stage"
    echo "  --render-only        Run only video rendering stage"
    echo "  --eval-only          Run only evaluation stage"
    echo "  --skip-train         Skip training stage"
    echo "  --skip-render        Skip video rendering stage"
    echo "  --skip-eval          Skip evaluation stage"
    echo ""
    echo "Method Selection:"
    echo "  --straightforward    Run only straightforward method (6D input)"
    echo "  --deformation        Run only deformation method (canonical + deform net)"
    echo ""
    echo "Scene Selection:"
    echo "  --scene SCENE        Run only specified scene (can be repeated)"
    echo ""
    echo "Training Parameters (modify in script header):"
    echo "  N_ITERS=${N_ITERS}       Training iterations"
    echo "  LRATE=${LRATE}           Learning rate"
    echo "  N_RAND=${N_RAND}         Batch size (rays per step)"
    echo "  N_SAMPLES=${N_SAMPLES}   Coarse samples per ray"
    echo "  N_IMPORTANCE=${N_IMPORTANCE}  Fine samples per ray"
    echo ""
    echo "Other:"
    echo "  --help               Show this help message"
    echo ""
    echo "Available scenes: ${SCENES[*]}"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run full pipeline for all scenes"
    echo "  $0 --scene lego --straightforward    # Run lego with straightforward only"
    echo "  $0 --skip-train                      # Skip training, only render and evaluate"
    echo "  $0 --train-only                      # Only train models"
    echo "  $0 --scene bouncingballs --scene lego  # Run two specific scenes"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --train-only)
            RUN_RENDER=false
            RUN_EVAL=false
            shift
            ;;
        --render-only)
            RUN_TRAIN=false
            RUN_EVAL=false
            shift
            ;;
        --eval-only)
            RUN_TRAIN=false
            RUN_RENDER=false
            shift
            ;;
        --skip-train)
            RUN_TRAIN=false
            shift
            ;;
        --skip-render)
            RUN_RENDER=false
            shift
            ;;
        --skip-eval)
            RUN_EVAL=false
            shift
            ;;
        --straightforward)
            RUN_DEFORMATION=false
            shift
            ;;
        --deformation)
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

# If specific scenes selected, use those; otherwise use all
if [ ${#SELECTED_SCENES[@]} -gt 0 ]; then
    SCENES=("${SELECTED_SCENES[@]}")
fi

# =============================================================================
# Print Configuration
# =============================================================================
echo "============================================="
echo "Dynamic NeRF Three-Stage Pipeline"
echo "============================================="
echo "Scenes: ${SCENES[*]}"
echo "Methods:"
[ "$RUN_STRAIGHTFORWARD" = true ] && echo "  - Straightforward"
[ "$RUN_DEFORMATION" = true ] && echo "  - Deformation"
echo ""
echo "Pipeline Stages:"
[ "$RUN_TRAIN" = true ] && echo "  ✓ Stage 1: Training"
[ "$RUN_RENDER" = true ] && echo "  ✓ Stage 2: Video Rendering"
[ "$RUN_EVAL" = true ] && echo "  ✓ Stage 3: Evaluation"
echo ""
echo "Paths:"
echo "  Data: ${DATA_BASEDIR}"
echo "  Models: ${MODEL_BASEDIR}"
echo "============================================="
echo ""

# Create necessary directories
mkdir -p "${MODEL_BASEDIR}"
mkdir -p "${MODEL_BASEDIR}/logs"
mkdir -p "${MODEL_BASEDIR}/results"

# =============================================================================
# STAGE 1: TRAINING
# =============================================================================
if [ "$RUN_TRAIN" = true ]; then
    echo ""
    echo "============================================="
    echo "STAGE 1: TRAINING MODELS"
    echo "============================================="
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    TOTAL_EXPERIMENTS=$((${#SCENES[@]} * (${RUN_STRAIGHTFORWARD} + ${RUN_DEFORMATION})))
    CURRENT_EXP=0
    
    for scene in "${SCENES[@]}"; do
        DATA_DIR="${DATA_BASEDIR}/${scene}"
        
        # Check if scene exists
        if [ ! -d "${DATA_DIR}" ]; then
            echo "WARNING: Scene not found, skipping: ${DATA_DIR}"
            continue
        fi
        
        echo ""
        echo "---------------------------------------------"
        echo "Training scene: ${scene}"
        echo "---------------------------------------------"
        
        # Straightforward method
        if [ "$RUN_STRAIGHTFORWARD" = true ]; then
            CURRENT_EXP=$((CURRENT_EXP + 1))
            NETWORK_TYPE="straightforward"
            EXP_NAME="dnerf_${NETWORK_TYPE}_${scene}"
            
            echo ""
            echo "[${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] Training: ${scene} (${NETWORK_TYPE})"
            echo "Experiment: ${EXP_NAME}"
            echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
            
            python train.py \
                --datadir "${DATA_DIR}" \
                --basedir "${MODEL_BASEDIR}" \
                --expname "${EXP_NAME}" \
                --network_type "${NETWORK_TYPE}" \
                --N_iters ${N_ITERS} \
                --N_rand ${N_RAND} \
                --N_samples ${N_SAMPLES} \
                --N_importance ${N_IMPORTANCE} \
                --lrate ${LRATE} \
                --lrate_decay ${LRATE_DECAY} \
                --use_viewdirs \
                --i_print ${I_PRINT} \
                --i_weights ${I_WEIGHTS} \
                --no_reload \
                ${HALF_RES} \
                2>&1 | tee "${MODEL_BASEDIR}/logs/${EXP_NAME}_train.log"
            
            echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
        fi
        
        # Deformation method
        if [ "$RUN_DEFORMATION" = true ]; then
            CURRENT_EXP=$((CURRENT_EXP + 1))
            NETWORK_TYPE="deformation"
            EXP_NAME="dnerf_${NETWORK_TYPE}_${scene}"
            
            echo ""
            echo "[${CURRENT_EXP}/${TOTAL_EXPERIMENTS}] Training: ${scene} (${NETWORK_TYPE})"
            echo "Experiment: ${EXP_NAME}"
            echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
            
            python train.py \
                --datadir "${DATA_DIR}" \
                --basedir "${MODEL_BASEDIR}" \
                --expname "${EXP_NAME}" \
                --network_type "${NETWORK_TYPE}" \
                --N_iters ${N_ITERS} \
                --N_rand ${N_RAND} \
                --N_samples ${N_SAMPLES} \
                --N_importance ${N_IMPORTANCE} \
                --lrate ${LRATE} \
                --lrate_decay ${LRATE_DECAY} \
                --use_viewdirs \
                --i_print ${I_PRINT} \
                --i_weights ${I_WEIGHTS} \
                --no_reload \
                ${HALF_RES} \
                2>&1 | tee "${MODEL_BASEDIR}/logs/${EXP_NAME}_train.log"
            
            echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
        fi
    done
    
    echo ""
    echo "============================================="
    echo "STAGE 1 COMPLETED: All models trained"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================="
fi

# =============================================================================
# STAGE 2: VIDEO RENDERING
# =============================================================================
if [ "$RUN_RENDER" = true ]; then
    echo ""
    echo "============================================="
    echo "STAGE 2: RENDERING VIDEOS"
    echo "============================================="
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    for scene in "${SCENES[@]}"; do
        DATA_DIR="${DATA_BASEDIR}/${scene}"
        
        if [ ! -d "${DATA_DIR}" ]; then
            echo "WARNING: Scene not found, skipping: ${DATA_DIR}"
            continue
        fi
        
        echo ""
        echo "---------------------------------------------"
        echo "Rendering scene: ${scene}"
        echo "---------------------------------------------"
        
        # Straightforward method
        if [ "$RUN_STRAIGHTFORWARD" = true ]; then
            NETWORK_TYPE="straightforward"
            EXP_NAME="dnerf_${NETWORK_TYPE}_${scene}"
            MODEL_DIR="${MODEL_BASEDIR}/${EXP_NAME}"
            
            # Check if best checkpoint exists
            BEST_CKPT="${MODEL_DIR}/best.tar"
            if [ ! -f "${BEST_CKPT}" ]; then
                echo "WARNING: Best checkpoint not found for ${EXP_NAME}, skipping"
                continue
            fi
            
            echo ""
            echo "Rendering: ${scene} (${NETWORK_TYPE})"
            
            # Render videos with different time modes
            for TIME_MODE in "${VIDEO_TIME_MODES[@]}"; do
                echo "  - Time mode: ${TIME_MODE}"
                
                python render_video.py \
                    --data_basedir "${DATA_BASEDIR}" \
                    --model_basedir "${MODEL_BASEDIR}" \
                    --scene "${scene}" \
                    --network_type "${NETWORK_TYPE}" \
                    --time_mode "${TIME_MODE}" \
                    --n_frames ${VIDEO_N_FRAMES} \
                    --fps ${VIDEO_FPS} \
                    ${HALF_RES} \
                    2>&1 | tee "${MODEL_BASEDIR}/logs/${EXP_NAME}_render_${TIME_MODE}.log"
            done
            
            echo "Completed rendering: ${scene} (${NETWORK_TYPE})"
        fi
        
        # Deformation method
        if [ "$RUN_DEFORMATION" = true ]; then
            NETWORK_TYPE="deformation"
            EXP_NAME="dnerf_${NETWORK_TYPE}_${scene}"
            MODEL_DIR="${MODEL_BASEDIR}/${EXP_NAME}"
            
            # Check if best checkpoint exists
            BEST_CKPT="${MODEL_DIR}/best.tar"
            if [ ! -f "${BEST_CKPT}" ]; then
                echo "WARNING: Best checkpoint not found for ${EXP_NAME}, skipping"
                continue
            fi
            
            echo ""
            echo "Rendering: ${scene} (${NETWORK_TYPE})"
            
            # Render videos with different time modes
            for TIME_MODE in "${VIDEO_TIME_MODES[@]}"; do
                echo "  - Time mode: ${TIME_MODE}"
                
                python render_video.py \
                    --data_basedir "${DATA_BASEDIR}" \
                    --model_basedir "${MODEL_BASEDIR}" \
                    --scene "${scene}" \
                    --network_type "${NETWORK_TYPE}" \
                    --time_mode "${TIME_MODE}" \
                    --n_frames ${VIDEO_N_FRAMES} \
                    --fps ${VIDEO_FPS} \
                    ${HALF_RES} \
                    2>&1 | tee "${MODEL_BASEDIR}/logs/${EXP_NAME}_render_${TIME_MODE}.log"
            done
            
            echo "Completed rendering: ${scene} (${NETWORK_TYPE})"
        fi
    done
    
    echo ""
    echo "============================================="
    echo "STAGE 2 COMPLETED: All videos rendered"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================="
fi

# =============================================================================
# STAGE 3: EVALUATION
# =============================================================================
if [ "$RUN_EVAL" = true ]; then
    echo ""
    echo "============================================="
    echo "STAGE 3: EVALUATION"
    echo "============================================="
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Create results directory
    RESULTS_DIR="${MODEL_BASEDIR}/results"
    mkdir -p "${RESULTS_DIR}"
    
    for scene in "${SCENES[@]}"; do
        DATA_DIR="${DATA_BASEDIR}/${scene}"
        
        if [ ! -d "${DATA_DIR}" ]; then
            echo "WARNING: Scene not found, skipping: ${DATA_DIR}"
            continue
        fi
        
        echo ""
        echo "---------------------------------------------"
        echo "Evaluating scene: ${scene}"
        echo "---------------------------------------------"
        
        # Evaluate Straightforward method
        if [ "$RUN_STRAIGHTFORWARD" = true ]; then
            NETWORK_TYPE="straightforward"
            EXP_NAME="dnerf_${NETWORK_TYPE}_${scene}"
            CKPT="${MODEL_BASEDIR}/${EXP_NAME}/best.tar"
            
            if [ -f "${CKPT}" ]; then
                echo ""
                echo "Evaluating: ${scene} (${NETWORK_TYPE})"
                
                python evaluate.py \
                    --data_basedir "${DATA_BASEDIR}" \
                    --model_basedir "${MODEL_BASEDIR}" \
                    --scene "${scene}" \
                    --network_type "${NETWORK_TYPE}" \
                    ${HALF_RES} \
                    2>&1 | tee "${MODEL_BASEDIR}/logs/${EXP_NAME}_eval.log"
                
                echo "Completed evaluation: ${scene} (${NETWORK_TYPE})"
            else
                echo "WARNING: Checkpoint not found: ${CKPT}"
            fi
        fi
        
        # Evaluate Deformation method
        if [ "$RUN_DEFORMATION" = true ]; then
            NETWORK_TYPE="deformation"
            EXP_NAME="dnerf_${NETWORK_TYPE}_${scene}"
            CKPT="${MODEL_BASEDIR}/${EXP_NAME}/best.tar"
            
            if [ -f "${CKPT}" ]; then
                echo ""
                echo "Evaluating: ${scene} (${NETWORK_TYPE})"
                
                python evaluate.py \
                    --data_basedir "${DATA_BASEDIR}" \
                    --model_basedir "${MODEL_BASEDIR}" \
                    --scene "${scene}" \
                    --network_type "${NETWORK_TYPE}" \
                    ${HALF_RES} \
                    2>&1 | tee "${MODEL_BASEDIR}/logs/${EXP_NAME}_eval.log"
                
                echo "Completed evaluation: ${scene} (${NETWORK_TYPE})"
            else
                echo "WARNING: Checkpoint not found: ${CKPT}"
            fi
        fi
    done
    
    echo ""
    echo "============================================="
    echo "STAGE 3 COMPLETED: All evaluations done"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================="
    
    # Generate summary report
    echo ""
    echo "---------------------------------------------"
    echo "Generating Summary Report"
    echo "---------------------------------------------"
    
    SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
    echo "Dynamic NeRF Evaluation Summary" > "${SUMMARY_FILE}"
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')" >> "${SUMMARY_FILE}"
    echo "========================================" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
    
    for scene in "${SCENES[@]}"; do
        echo "Scene: ${scene}" >> "${SUMMARY_FILE}"
        echo "----------------------------------------" >> "${SUMMARY_FILE}"
        
        # Straightforward results
        if [ "$RUN_STRAIGHTFORWARD" = true ]; then
            SF_METRICS="${MODEL_BASEDIR}/dnerf_straightforward_${scene}/evaluation/metrics.json"
            if [ -f "${SF_METRICS}" ]; then
                echo "  Straightforward:" >> "${SUMMARY_FILE}"
                python -c "import json; data=json.load(open('${SF_METRICS}')); m=data['metrics']; print(f\"    PSNR: {m['psnr']:.2f} ± {m['psnr_std']:.2f} dB\"); print(f\"    SSIM: {m['ssim']:.4f} ± {m['ssim_std']:.4f}\"); print(f\"    LPIPS: {m.get('lpips', 0):.4f} ± {m.get('lpips_std', 0):.4f}\" if 'lpips' in m else '')" >> "${SUMMARY_FILE}" 2>/dev/null || echo "    (metrics parsing failed)" >> "${SUMMARY_FILE}"
            fi
        fi
        
        # Deformation results
        if [ "$RUN_DEFORMATION" = true ]; then
            DF_METRICS="${MODEL_BASEDIR}/dnerf_deformation_${scene}/evaluation/metrics.json"
            if [ -f "${DF_METRICS}" ]; then
                echo "  Deformation:" >> "${SUMMARY_FILE}"
                python -c "import json; data=json.load(open('${DF_METRICS}')); m=data['metrics']; print(f\"    PSNR: {m['psnr']:.2f} ± {m['psnr_std']:.2f} dB\"); print(f\"    SSIM: {m['ssim']:.4f} ± {m['ssim_std']:.4f}\"); print(f\"    LPIPS: {m.get('lpips', 0):.4f} ± {m.get('lpips_std', 0):.4f}\" if 'lpips' in m else '')" >> "${SUMMARY_FILE}" 2>/dev/null || echo "    (metrics parsing failed)" >> "${SUMMARY_FILE}"
            fi
        fi
        
        echo "" >> "${SUMMARY_FILE}"
    done
    
    echo "Summary report saved to: ${SUMMARY_FILE}"
    echo ""
    cat "${SUMMARY_FILE}"
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "============================================="
echo "PIPELINE COMPLETED!"
echo "============================================="
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "Summary of Outputs:"
echo ""

for scene in "${SCENES[@]}"; do
    echo "Scene: ${scene}"
    echo "  Methods:"
    
    if [ "$RUN_STRAIGHTFORWARD" = true ]; then
        EXP_NAME="dnerf_straightforward_${scene}"
        MODEL_DIR="${MODEL_BASEDIR}/${EXP_NAME}"
        
        if [ -d "${MODEL_DIR}" ]; then
            echo "    ✓ Straightforward"
            [ -f "${MODEL_DIR}/best.tar" ] && echo "      - Model: ${MODEL_DIR}/best.tar"
            [ -d "${MODEL_DIR}/videos" ] && echo "      - Videos: ${MODEL_DIR}/videos/"
            [ -d "${MODEL_DIR}/evaluation" ] && echo "      - Evaluation: ${MODEL_DIR}/evaluation/"
        fi
    fi
    
    if [ "$RUN_DEFORMATION" = true ]; then
        EXP_NAME="dnerf_deformation_${scene}"
        MODEL_DIR="${MODEL_BASEDIR}/${EXP_NAME}"
        
        if [ -d "${MODEL_DIR}" ]; then
            echo "    ✓ Deformation"
            [ -f "${MODEL_DIR}/best.tar" ] && echo "      - Model: ${MODEL_DIR}/best.tar"
            [ -d "${MODEL_DIR}/videos" ] && echo "      - Videos: ${MODEL_DIR}/videos/"
            [ -d "${MODEL_DIR}/evaluation" ] && echo "      - Evaluation: ${MODEL_DIR}/evaluation/"
        fi
    fi
    
    echo ""
done

echo "============================================="
echo "All logs saved to: ${MODEL_BASEDIR}/logs/"
echo "All results saved to: ${RESULTS_DIR}/"
echo "============================================="
