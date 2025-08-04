#!/bin/bash

# CAZO Perturbation Sensitivity Analysis
# This script tests the effect of different perturbation numbers (2, 4, 6, ..., 30)

PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO

# Fixed parameters (control variables)
batch_size=64
lr=0.01
adapter_layer=3
reduction_factor=384
adapter_style="parallel"
fitness_lambda=0.4

# Hessian related parameters
epsilon=0.1
nu=0.8

# Optimizer related parameters
optimizer="sgd"
beta=0.9

# GPU setting
export CUDA_VISIBLE_DEVICES=3

# Create base output and log directories
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/cazo_pertub_sensitivity"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/cazo_pertub_sensitivity"

mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${BASE_LOG_DIR}

# Log file for summary
SUMMARY_LOG="${BASE_LOG_DIR}/cazo_pertub_sensitivity_summary.log"
echo "CAZO Perturbation Sensitivity Analysis Started at $(date)" > ${SUMMARY_LOG}
echo "Testing pertub values: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}

# Array to store results for final summary
declare -a RESULTS=()

# Loop through pertub values from 2 to 30 with step 2
for pertub in $(seq 2 2 30); do
    echo "=========================================="
    echo "Starting experiment with pertub=${pertub}"
    echo "=========================================="
    
    # Record start time
    start_time=$(date)
    echo "Experiment pertub=${pertub} started at: ${start_time}" >> ${SUMMARY_LOG}
    
    # Create specific output and log directories for this pertub value
    # OUTPUT_DIR="${BASE_OUTPUT_DIR}/cazo_pertub_${pertub}"
    # LOG_DIR="${BASE_LOG_DIR}/cazo_pertub_${pertub}"
    
    mkdir -p ${BASE_OUTPUT_DIR}
    mkdir -p ${BASE_LOG_DIR}
    
    # Generate tag for this experiment
    tag="_pertub_sensitivity_${pertub}_bs${batch_size}_lr${lr}_adapter${adapter_layer}_rf${reduction_factor}_${adapter_style}_eps${epsilon}_nu${nu}"
    
    # Run the experiment
    python ${PROJECT_ROOT}/main.py \
        --batch_size ${batch_size} \
        --workers 16 \
        --data /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet \
        --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
        --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
        --data_corruption /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-c/imagenet-c \
        --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
        --output ${BASE_OUTPUT_DIR} \
        --root_log_dir ${BASE_LOG_DIR} \
        --algorithm "cazo" \
        --lr ${lr} \
        --pertub ${pertub} \
        --adapter_layer ${adapter_layer} \
        --reduction_factor ${reduction_factor} \
        --adapter_style ${adapter_style} \
        --optimizer ${optimizer} \
        --beta ${beta} \
        --epsilon ${epsilon} \
        --nu ${nu} \
        --fitness_lambda ${fitness_lambda} \
        --tag "${tag}"
    
    # Check if experiment completed successfully
    if [ $? -eq 0 ]; then
        end_time=$(date)
        echo "Experiment pertub=${pertub} completed successfully at: ${end_time}" >> ${SUMMARY_LOG}
        
        # Try to extract accuracy results from log (adjust path as needed)
        LOG_FILE="${BASE_LOG_DIR}/cazo${tag}/"*"-log.txt"
        if ls ${LOG_FILE} 1> /dev/null 2>&1; then
            # Extract final accuracy results
            RESULT=$(tail -20 ${LOG_FILE} | grep -E "(mean acc|Top-1 Accuracy)" | tail -1)
            RESULTS+=("pertub=${pertub}: ${RESULT}")
            echo "Result - pertub=${pertub}: ${RESULT}" >> ${SUMMARY_LOG}
        else
            echo "Warning: Log file not found for pertub=${pertub}" >> ${SUMMARY_LOG}
        fi
    else
        echo "ERROR: Experiment pertub=${pertub} failed!" >> ${SUMMARY_LOG}
        echo "ERROR: Experiment with pertub=${pertub} failed!"
    fi
    
    echo "Experiment pertub=${pertub} finished. Moving to next..."
    echo "==========================================\n"
    
    # Optional: Add a small delay between experiments
    sleep 10
done

# Generate final summary
echo "" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}
echo "FINAL SUMMARY OF ALL EXPERIMENTS" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}

for result in "${RESULTS[@]}"; do
    echo "${result}" >> ${SUMMARY_LOG}
done

echo "" >> ${SUMMARY_LOG}
echo "All experiments completed at $(date)" >> ${SUMMARY_LOG}

# Display summary on console
echo "=========================================="
echo "ALL PERTURBATION SENSITIVITY EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Summary saved to: ${SUMMARY_LOG}"
echo ""
echo "Results preview:"
for result in "${RESULTS[@]}"; do
    echo "${result}"
done

echo ""
echo "Detailed logs can be found in: ${BASE_LOG_DIR}"
echo "Detailed outputs can be found in: ${BASE_OUTPUT_DIR}"