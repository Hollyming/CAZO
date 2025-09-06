#!/bin/bash

# Simplified Main Experiments Script for CAZO Project
# Running 9 algorithms with 5 different seeds each
# Configurable GPU count with parallel execution across GPUs

PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO

cd ${PROJECT_ROOT}

# Create main experiment directories
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/main_experiments"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/main_experiments"

mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${BASE_LOG_DIR}

# ======================== Configuration ========================
# Experiment parameters
batch_size=64
workers=16

# Seeds for experiments
seeds=(42 2020 2025 1234 888)

# Algorithms to test
# algorithms=(no_adapt lame t3a tent cotta sar foa zo_base cazo cazo_lit cozo)
algorithms=(cozo)

# GPU configuration - MODIFY THIS TO SET YOUR GPU COUNT
GPU_COUNT=7  # Change this to your desired GPU count
GPU_IDS=(0 1 2 3 4 5 6)  # Modify this array to match your available GPUs

# ================================================================

# Create summary log
MAIN_SUMMARY_LOG="${BASE_LOG_DIR}/main_experiments_summary.log"
echo "Main Experiments Started at $(date)" > ${MAIN_SUMMARY_LOG}
echo "Algorithms: ${algorithms[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Seeds: ${seeds[@]}" >> ${MAIN_SUMMARY_LOG}
echo "GPU Count: ${GPU_COUNT}" >> ${MAIN_SUMMARY_LOG}
echo "GPU IDs: ${GPU_IDS[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Total experiments: $((${#algorithms[@]} * ${#seeds[@]}))" >> ${MAIN_SUMMARY_LOG}
echo "============================================" >> ${MAIN_SUMMARY_LOG}

# Function to get algorithm-specific parameters
get_algorithm_params() {
    local algorithm=$1
    local params=""
    
    case ${algorithm} in
        "no_adapt")
            params=""
            ;;
        "lame")
            params=""
            ;;
        "t3a")
            params=""
            ;;
        "tent")
            params=""
            ;;
        "cotta")
            params=""
            ;;
        "sar")
            params="--margin_e0 0.4"
            ;;
        "foa")
            params="--num_prompts 3 --fitness_lambda 0.4"
            ;;
        "zo_base")
            params="--lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor 384 --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1"
            ;;
        "cazo")
            params="--lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor 384 --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1 --nu 0.8 --fitness_lambda 0.4"
            ;;
        "cazo_lit")
            params="--lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor 384 --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1 --nu 0.8 --fitness_lambda 0.4"
            ;;
        "cozo")
            params="--lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor 384 --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1 --fitness_lambda 0.4 --mode cov_only"
            ;;
    esac
    
    echo "${params}"
}

# Function to run experiments on a single GPU (serial execution)
run_gpu_experiments() {
    local gpu_id=$1
    local exp_list=("${@:2}")  # Get remaining arguments as experiment list
    
    echo "GPU ${gpu_id}: Starting ${#exp_list[@]} experiments"
    
    for exp_info in "${exp_list[@]}"; do
        # Parse experiment info: "algorithm:seed:exp_id"
        IFS=':' read -r algorithm seed exp_id <<< "${exp_info}"
        
        echo "GPU ${gpu_id}: Starting experiment ${exp_id} - ${algorithm} with seed ${seed}"
        
        # Create algorithm-specific directories
        local output_dir="${BASE_OUTPUT_DIR}/${algorithm}"
        local log_dir="${BASE_LOG_DIR}/${algorithm}"
        mkdir -p ${output_dir}
        mkdir -p ${log_dir}
        
        # Get algorithm parameters
        local algo_params=$(get_algorithm_params ${algorithm})
        local tag="_seed${seed}_bs${batch_size}"
        
        # Log experiment start
        local start_time=$(date)
        echo "Experiment ${exp_id} (${algorithm}, seed ${seed}) started at: ${start_time} on GPU ${gpu_id}" >> ${MAIN_SUMMARY_LOG}
        
        # Run experiment
        CUDA_VISIBLE_DEVICES=${gpu_id} python ${PROJECT_ROOT}/main.py \
            --batch_size ${batch_size} \
            --workers ${workers} \
            --seed ${seed} \
            --data /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet \
            --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
            --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
            --data_corruption /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-c/imagenet-c \
            --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
            --output ${output_dir} \
            --root_log_dir ${log_dir} \
            --algorithm ${algorithm} \
            ${algo_params} \
            --tag "${tag}"
        
        local exit_code=$?
        local end_time=$(date)
        
        # Log experiment completion
        if [ ${exit_code} -eq 0 ]; then
            echo "Experiment ${exp_id} (${algorithm}, seed ${seed}) completed successfully at: ${end_time}" >> ${MAIN_SUMMARY_LOG}
            echo "GPU ${gpu_id}: ✓ Experiment ${exp_id} completed - ${algorithm} seed ${seed}"
        else
            echo "ERROR: Experiment ${exp_id} (${algorithm}, seed ${seed}) failed at: ${end_time}" >> ${MAIN_SUMMARY_LOG}
            echo "GPU ${gpu_id}: ✗ Experiment ${exp_id} failed - ${algorithm} seed ${seed}"
        fi
    done
    
    echo "GPU ${gpu_id}: All experiments completed"
}

# Distribute experiments across GPUs
echo "Distributing experiments across ${GPU_COUNT} GPUs..."

# Create experiment list
experiment_list=()
experiment_id=0

for algorithm in "${algorithms[@]}"; do
    for seed in "${seeds[@]}"; do
        experiment_list+=("${algorithm}:${seed}:${experiment_id}")
        experiment_id=$((experiment_id + 1))
    done
done

echo "Total experiments to run: ${#experiment_list[@]}"

# Distribute experiments evenly across GPUs
declare -a gpu_experiment_lists
for ((i=0; i<GPU_COUNT; i++)); do
    gpu_experiment_lists[i]=""
done

# Round-robin assignment
for ((i=0; i<${#experiment_list[@]}; i++)); do
    gpu_idx=$((i % GPU_COUNT))
    if [ -z "${gpu_experiment_lists[gpu_idx]}" ]; then
        gpu_experiment_lists[gpu_idx]="${experiment_list[i]}"
    else
        gpu_experiment_lists[gpu_idx]="${gpu_experiment_lists[gpu_idx]} ${experiment_list[i]}"
    fi
done

# Print experiment distribution
echo ""
echo "Experiment distribution:"
for ((i=0; i<GPU_COUNT; i++)); do
    exp_count=$(echo ${gpu_experiment_lists[i]} | wc -w)
    echo "GPU ${GPU_IDS[i]}: ${exp_count} experiments"
done
echo ""

# Start experiments on each GPU in parallel
pids=()
for ((i=0; i<GPU_COUNT; i++)); do
    gpu_id=${GPU_IDS[i]}
    if [ -n "${gpu_experiment_lists[i]}" ]; then
        # Convert space-separated string to array
        IFS=' ' read -ra gpu_experiments <<< "${gpu_experiment_lists[i]}"
        
        echo "Starting experiments on GPU ${gpu_id}..."
        run_gpu_experiments ${gpu_id} "${gpu_experiments[@]}" &
        pids+=($!)
    fi
done

# Wait for all GPU processes to complete
echo ""
echo "Waiting for all GPU processes to complete..."
for pid in "${pids[@]}"; do
    wait ${pid}
done

# Final summary
echo ""
echo "=========================================="
echo "ALL MAIN EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Completed at: $(date)"
echo "Summary saved to: ${MAIN_SUMMARY_LOG}"
echo ""
echo "Results location: ${BASE_OUTPUT_DIR}"
echo "Logs location: ${BASE_LOG_DIR}"
echo ""
echo "To analyze results, run:"
echo "  cd logs_new/main_experiments"
echo "  python analyze_all_seeds.py"

# Final log entry
echo "" >> ${MAIN_SUMMARY_LOG}
echo "All experiments completed at $(date)" >> ${MAIN_SUMMARY_LOG}
echo "Results saved to: ${BASE_OUTPUT_DIR}" >> ${MAIN_SUMMARY_LOG}
echo "Logs saved to: ${BASE_LOG_DIR}" >> ${MAIN_SUMMARY_LOG} 