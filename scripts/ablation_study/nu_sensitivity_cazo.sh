#!/bin/bash

# CAZO Nu Parameter Sensitivity Analysis
# This script tests the effect of different nu values (0.1-0.9 with 0.1 step, plus 0.95, 0.98, 0.99)
# Total 12 experiments distributed across 7 GPUs

PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO

cd ${PROJECT_ROOT}

# Fixed parameters (control variables)
batch_size=64
lr=0.01
pertub=20
adapter_layer=3
reduction_factor=384
adapter_style="parallel"
fitness_lambda=0.4

# Other Hessian related parameters (fixed)
epsilon=0.1

# Optimizer related parameters (fixed)
optimizer="sgd"
beta=0.9

# Available GPUs (7 GPUs)
GPUS=(0 1 2 3 4 5 6)

# Nu values to test
NU_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.98 0.99)

# Create base output and log directories
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/cazo_nu_sensitivity"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/cazo_nu_sensitivity"

mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${BASE_LOG_DIR}

# Log file for summary
SUMMARY_LOG="${BASE_LOG_DIR}/cazo_nu_sensitivity_summary.log"
echo "CAZO Nu Parameter Sensitivity Analysis Started at $(date)" > ${SUMMARY_LOG}
echo "Testing nu values: ${NU_VALUES[*]}" >> ${SUMMARY_LOG}
echo "Using ${#GPUS[@]} GPUs: ${GPUS[*]}" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}

# Function to run experiment on specific GPU
run_experiment() {
    local nu=$1
    local gpu_id=$2
    local exp_id=$3
    
    echo "=========================================="
    echo "Starting experiment ${exp_id}: nu=${nu} on GPU ${gpu_id}"
    echo "=========================================="
    
    # Set GPU for this experiment
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    
    # Record start time
    start_time=$(date)
    echo "Experiment ${exp_id} (nu=${nu}, GPU=${gpu_id}) started at: ${start_time}" >> ${SUMMARY_LOG}
    
    # Generate tag for this experiment
    tag="_nu_sensitivity_${nu}_bs${batch_size}_lr${lr}_pertub${pertub}_adapter${adapter_layer}_rf${reduction_factor}_${adapter_style}_eps${epsilon}"
    
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
        echo "Experiment ${exp_id} (nu=${nu}, GPU=${gpu_id}) completed successfully at: ${end_time}" >> ${SUMMARY_LOG}
        
        # Try to extract accuracy results from log
        LOG_FILE="${BASE_LOG_DIR}/cazo${tag}/"*"-log.txt"
        if ls ${LOG_FILE} 1> /dev/null 2>&1; then
            # Extract final accuracy results
            RESULT=$(tail -20 ${LOG_FILE} | grep -E "(mean acc|Top-1 Accuracy)" | tail -1)
            echo "Result - Exp ${exp_id} (nu=${nu}, GPU=${gpu_id}): ${RESULT}" >> ${SUMMARY_LOG}
            echo "Experiment ${exp_id} (nu=${nu}) completed with result: ${RESULT}"
        else
            echo "Warning: Log file not found for Exp ${exp_id} (nu=${nu})" >> ${SUMMARY_LOG}
        fi
    else
        echo "ERROR: Experiment ${exp_id} (nu=${nu}, GPU=${gpu_id}) failed!" >> ${SUMMARY_LOG}
        echo "ERROR: Experiment ${exp_id} with nu=${nu} failed!"
    fi
    
    echo "Experiment ${exp_id} (nu=${nu}) on GPU ${gpu_id} finished."
    echo "==========================================\n"
}

# GPU assignment strategy for 12 experiments on 7 GPUs:
# GPU 0: experiments 1, 8    (nu=0.1, 0.8)
# GPU 1: experiments 2, 9    (nu=0.2, 0.9) 
# GPU 2: experiments 3, 10   (nu=0.3, 0.95)
# GPU 3: experiments 4, 11   (nu=0.4, 0.98)
# GPU 4: experiments 5, 12   (nu=0.5, 0.99)
# GPU 5: experiment 6        (nu=0.6)
# GPU 6: experiment 7        (nu=0.7)

echo "Starting parallel execution of nu sensitivity experiments..."
echo "GPU assignment:"
echo "GPU 0: nu=0.1, 0.8"
echo "GPU 1: nu=0.2, 0.9"
echo "GPU 2: nu=0.3, 0.95"
echo "GPU 3: nu=0.4, 0.98"
echo "GPU 4: nu=0.5, 0.99"
echo "GPU 5: nu=0.6"
echo "GPU 6: nu=0.7"
echo ""

# Array to store PIDs of background processes
PIDS=()

# First round: Start one experiment on each GPU
for i in {0..6}; do
    nu=${NU_VALUES[i]}
    gpu_id=${GPUS[i]}
    exp_id=$((i+1))
    
    echo "Launching Experiment ${exp_id} (nu=${nu}) on GPU ${gpu_id} in background..."
    run_experiment ${nu} ${gpu_id} ${exp_id} &
    PIDS+=($!)
    
    # Small delay to avoid race conditions
    sleep 5
done

# Wait for first 5 experiments to complete before starting second round
echo "Waiting for first 5 experiments to complete..."
for i in {0..4}; do
    wait ${PIDS[i]}
    echo "Experiment $((i+1)) completed"
done

# Second round: Start remaining experiments on first 5 GPUs
echo "Starting second round of experiments..."
for i in {0..4}; do
    nu=${NU_VALUES[$((i+7))]}  # nu values: 0.8, 0.9, 0.95, 0.98, 0.99
    gpu_id=${GPUS[i]}
    exp_id=$((i+8))
    
    echo "Launching Experiment ${exp_id} (nu=${nu}) on GPU ${gpu_id} in background..."
    run_experiment ${nu} ${gpu_id} ${exp_id} &
    
    # Small delay to avoid race conditions
    sleep 5
done

# Wait for remaining experiments to complete
echo "Waiting for all remaining experiments to complete..."
wait

# Generate final summary
echo "" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}
echo "FINAL SUMMARY OF ALL NU SENSITIVITY EXPERIMENTS" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}

# Extract all results from log
echo "Extracting all results..." >> ${SUMMARY_LOG}
grep "Result - Exp" ${SUMMARY_LOG} | sort -V >> ${SUMMARY_LOG}

echo "" >> ${SUMMARY_LOG}
echo "All nu sensitivity experiments completed at $(date)" >> ${SUMMARY_LOG}

# Display summary on console
echo "=========================================="
echo "ALL NU SENSITIVITY EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Summary saved to: ${SUMMARY_LOG}"
echo ""
echo "Results preview:"
grep "Result - Exp" ${SUMMARY_LOG} | sort -V

echo ""
echo "Detailed logs can be found in: ${BASE_LOG_DIR}"
echo "Detailed outputs can be found in: ${BASE_OUTPUT_DIR}"

echo ""
echo "Nu sensitivity analysis completed successfully!"
echo "Total experiments: ${#NU_VALUES[@]}"
echo "GPUs used: ${#GPUS[@]}" 