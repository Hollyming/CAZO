#!/bin/bash

# Simplified Main Experiments Script for CAZO Project
# Running 9 algorithms with 5 different seeds each
# Configurable GPU count with parallel execution across GPUs

PROJECT_ROOT=/home/zjm/Workspace/CAZO

cd ${PROJECT_ROOT}

DATASET_CHOICE="main_experiments"  # Options: "main_experiments" or "other_datasets"

if [ "${DATASET_CHOICE}" == "main_experiments" ]; then
    echo "Using dataset configuration for other datasets."
    DATASET_STYLE="imagenet_c"
elif [ "${DATASET_CHOICE}" == "other_datasets" ]; then
    echo "Using dataset configuration for other datasets."
    DATASET_STYLE="imagenet_r_s_v2"
else
    echo "Invalid DATASET_CHOICE: ${DATASET_CHOICE}. Exiting."
    exit 1
fi

# Create main experiment directories
# BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/main_experiments"
# BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/main_experiments"
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/${DATASET_CHOICE}"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/${DATASET_CHOICE}"

mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${BASE_LOG_DIR}

# ======================== Configuration ========================
# Experiment parameters
batch_size=64
workers=16

# Architectures to test
archs=(deit_base swin_tiny)  # Options: vit_base, deit_base, swin_tiny, resnet50

# Seeds for experiments
seeds=(42 2020 2025 1234 888)

# Algorithms to test
# algorithms=(no_adapt lame t3a tent cotta sar foa zo_base cazo cazo_lit cozo deyo rotta eta eata)
algorithms=(cazo)

# GPU configuration - MODIFY THIS TO SET YOUR GPU COUNT
GPU_COUNT=5  # Change this to your desired GPU count
GPU_IDS=(1 2 5 6 7)  # Modify this array to match your available GPUs

# ================================================================

# Create summary log
MAIN_SUMMARY_LOG="${BASE_LOG_DIR}/main_experiments_summary.log"
echo "Main Experiments Started at $(date)" > ${MAIN_SUMMARY_LOG}
echo "Architectures: ${archs[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Algorithms: ${algorithms[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Seeds: ${seeds[@]}" >> ${MAIN_SUMMARY_LOG}
echo "GPU Count: ${GPU_COUNT}" >> ${MAIN_SUMMARY_LOG}
echo "GPU IDs: ${GPU_IDS[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Total experiments: $((${#archs[@]} * ${#algorithms[@]} * ${#seeds[@]}))" >> ${MAIN_SUMMARY_LOG}
echo "============================================" >> ${MAIN_SUMMARY_LOG}

# Function to get algorithm-specific parameters
get_algorithm_params() {
    local algorithm=$1
    local arch=$2
    local params=""
    
    # Determine reduction_factor based on architecture
    local reduction_factor=384  # default for vit_base, deit_base
    if [ "${arch}" == "swin_tiny" ] || [ "${arch}" == "resnet50" ]; then
        reduction_factor=48
    fi
    
    case ${algorithm} in
        "no_adapt")
            params="--arch ${arch}"
            ;;
        "lame")
            params="--arch ${arch}"
            ;;
        "t3a")
            params="--arch ${arch}"
            ;;
        "tent")
            params="--arch ${arch}"
            ;;
        "cotta")
            params="--arch ${arch}"
            ;;
        "sar")
            params="--arch ${arch} --margin_e0 0.4"
            ;;
        "eta")
            params="--arch ${arch}"
            ;;
        "eata")
            params="--arch ${arch}"
            ;;
        "deyo")
            params="--arch ${arch}"
            ;;
        "rotta")
            params="--arch ${arch}"
            ;;
        "foa")
            params="--arch ${arch} --num_prompts 3 --fitness_lambda 0.4"
            ;;
        "zo_base")
            params="--arch ${arch} --lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor ${reduction_factor} --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1"
            ;;
        "cazo")
            params="--arch ${arch} --lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor ${reduction_factor} --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1 --nu 0.8 --fitness_lambda 0.4"
            ;;
        "cazo_lit")
            params="--arch ${arch} --lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor ${reduction_factor} --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1 --nu 0.8 --fitness_lambda 0.4"
            ;;
        "cozo")
            params="--arch ${arch} --lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor ${reduction_factor} --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1 --fitness_lambda 0.4 --mode cov_only"
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
        # Parse experiment info: "algorithm:seed:arch:exp_id"
        IFS=':' read -r algorithm seed arch exp_id <<< "${exp_info}"
        
        echo "GPU ${gpu_id}: Starting experiment ${exp_id} - ${algorithm} (${arch}) with seed ${seed}"
        
        # Create algorithm-specific directories with arch subdirectory
        local output_dir="${BASE_OUTPUT_DIR}/${arch}/${algorithm}"
        local log_dir="${BASE_LOG_DIR}/${arch}/${algorithm}"
        mkdir -p ${output_dir}
        mkdir -p ${log_dir}
        
        # Get algorithm parameters
        local algo_params=$(get_algorithm_params ${algorithm} ${arch})
        local tag="_seed${seed}_bs${batch_size}_${arch}"
        
        # Log experiment start
        local start_time=$(date)
        echo "Experiment ${exp_id} (${algorithm}, ${arch}, seed ${seed}) started at: ${start_time} on GPU ${gpu_id}" >> ${MAIN_SUMMARY_LOG}
        
        # Run experiment
        CUDA_VISIBLE_DEVICES=${gpu_id} python ${PROJECT_ROOT}/main.py \
            --batch_size ${batch_size} \
            --workers ${workers} \
            --seed ${seed} \
            --data /home/DATA/imagenet \
            --data_v2 /media/DATA/imagenetv2/ \
            --data_sketch /media/DATA/imagenet-sketch/sketch \
            --data_corruption /home/DATA/imagenet-c \
            --data_rendition /media/DATA/imagenet-r/imagenet-r \
            --dataset_style ${DATASET_STYLE} \
            --output ${output_dir} \
            --root_log_dir ${log_dir} \
            --algorithm ${algorithm} \
            ${algo_params} \
            --tag "${tag}"
        
        local exit_code=$?
        local end_time=$(date)
        
        # Log experiment completion
        if [ ${exit_code} -eq 0 ]; then
            echo "Experiment ${exp_id} (${algorithm}, ${arch}, seed ${seed}) completed successfully at: ${end_time}" >> ${MAIN_SUMMARY_LOG}
            echo "GPU ${gpu_id}: ✓ Experiment ${exp_id} completed - ${algorithm} (${arch}) seed ${seed}"
        else
            echo "ERROR: Experiment ${exp_id} (${algorithm}, ${arch}, seed ${seed}) failed at: ${end_time}" >> ${MAIN_SUMMARY_LOG}
            echo "GPU ${gpu_id}: ✗ Experiment ${exp_id} failed - ${algorithm} (${arch}) seed ${seed}"
        fi
    done
    
    echo "GPU ${gpu_id}: All experiments completed"
}

# Distribute experiments across GPUs
echo "Distributing experiments across ${GPU_COUNT} GPUs..."

# Create experiment list
experiment_list=()
experiment_id=0

for arch in "${archs[@]}"; do
    for algorithm in "${algorithms[@]}"; do
        for seed in "${seeds[@]}"; do
            experiment_list+=("${algorithm}:${seed}:${arch}:${experiment_id}")
            experiment_id=$((experiment_id + 1))
        done
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