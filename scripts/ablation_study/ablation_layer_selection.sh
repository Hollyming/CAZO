#!/bin/bash

# Ablation Study Script for Adapter Layer Selection
# Testing different adapter layer positions for CAZO algorithm
# Grid search over: archs × adapter_layers × seeds

PROJECT_ROOT=/home/zjm/Workspace/CAZO

cd ${PROJECT_ROOT}

DATASET_CHOICE="ablation"  # Options: "ablation" for layer ablation study
DATASET_STYLE="imagenet_c"

# Create ablation experiment directories
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

# Adapter layer positions to test (0-11 for 12-layer models)
adapter_layers=(0 1 2 3 4 5 6 7 8 9 10 11)

# Seeds for experiments
seeds=(42)  #default=(42 2020 2025 1234 888)

# Algorithm - focus on CAZO for ablation
algorithm="cazo"

# GPU configuration
GPU_COUNT=5  # Change this to your desired GPU count
GPU_IDS=(1 2 5 6 7)  # Modify this array to match your available GPUs

# ================================================================

# Create summary log
MAIN_SUMMARY_LOG="${BASE_LOG_DIR}/ablation_summary.log"
echo "Adapter Layer Ablation Study Started at $(date)" > ${MAIN_SUMMARY_LOG}
echo "Architectures: ${archs[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Algorithm: ${algorithm}" >> ${MAIN_SUMMARY_LOG}
echo "Adapter Layers: ${adapter_layers[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Seeds: ${seeds[@]}" >> ${MAIN_SUMMARY_LOG}
echo "GPU Count: ${GPU_COUNT}" >> ${MAIN_SUMMARY_LOG}
echo "GPU IDs: ${GPU_IDS[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Total experiments: $((${#archs[@]} * ${#adapter_layers[@]} * ${#seeds[@]}))" >> ${MAIN_SUMMARY_LOG}
echo "============================================" >> ${MAIN_SUMMARY_LOG}

# Function to get algorithm-specific parameters for ablation
get_algorithm_params() {
    local arch=$1
    local adapter_layer=$2
    local params=""
    
    # Determine reduction_factor based on architecture and layer
    local reduction_factor
    
    if [ "${arch}" == "deit_base" ] || [ "${arch}" == "vit_base" ]; then
        # DeiT-Base and ViT-Base: all layers have embed_dim=768
        reduction_factor=384  # 768/2 = 384 for bottleneck=2
    elif [ "${arch}" == "swin_tiny" ]; then
        # Swin-Tiny: different dimensions per stage
        # Layer 0-1: 96 → reduction_factor=48
        # Layer 2-3: 192 → reduction_factor=96
        # Layer 4-9: 384 → reduction_factor=192
        # Layer 10-11: 768 → reduction_factor=384
        if [ ${adapter_layer} -le 1 ]; then
            reduction_factor=48
        elif [ ${adapter_layer} -le 3 ]; then
            reduction_factor=96
        elif [ ${adapter_layer} -le 9 ]; then
            reduction_factor=192
        else
            reduction_factor=384
        fi
    elif [ "${arch}" == "resnet50" ]; then
        # ResNet-50: typically use smaller reduction factor
        reduction_factor=48
    else
        # Default fallback
        reduction_factor=384
    fi
    
    # CAZO parameters with variable adapter_layer and reduction_factor
    params="--arch ${arch} --lr 0.01 --pertub 20 --adapter_layer ${adapter_layer} --reduction_factor ${reduction_factor} --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1 --nu 0.8 --fitness_lambda 0.4"
    
    echo "${params}"
}

# Function to run experiments on a single GPU (serial execution)
run_gpu_experiments() {
    local gpu_id=$1
    local exp_list=("${@:2}")  # Get remaining arguments as experiment list
    
    echo "GPU ${gpu_id}: Starting ${#exp_list[@]} experiments"
    
    for exp_info in "${exp_list[@]}"; do
        # Parse experiment info: "arch:adapter_layer:seed:exp_id"
        IFS=':' read -r arch adapter_layer seed exp_id <<< "${exp_info}"
        
        echo "GPU ${gpu_id}: Starting experiment ${exp_id} - ${algorithm} (${arch}, layer ${adapter_layer}) with seed ${seed}"
        
        # Create directories: ablation/{arch}/layer_{adapter_layer}/{algorithm}
        local output_dir="${BASE_OUTPUT_DIR}/${arch}/layer_${adapter_layer}/${algorithm}"
        local log_dir="${BASE_LOG_DIR}/${arch}/layer_${adapter_layer}/${algorithm}"
        mkdir -p ${output_dir}
        mkdir -p ${log_dir}
        
        # Get algorithm parameters
        local algo_params=$(get_algorithm_params ${arch} ${adapter_layer})
        local tag="_seed${seed}_bs${batch_size}_${arch}_layer${adapter_layer}"
        
        # Log experiment start
        local start_time=$(date)
        echo "Experiment ${exp_id} (${algorithm}, ${arch}, layer ${adapter_layer}, seed ${seed}) started at: ${start_time} on GPU ${gpu_id}" >> ${MAIN_SUMMARY_LOG}
        
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
            echo "Experiment ${exp_id} (${algorithm}, ${arch}, layer ${adapter_layer}, seed ${seed}) completed successfully at: ${end_time}" >> ${MAIN_SUMMARY_LOG}
            echo "GPU ${gpu_id}: ✓ Experiment ${exp_id} completed - ${algorithm} (${arch}, layer ${adapter_layer}) seed ${seed}"
        else
            echo "ERROR: Experiment ${exp_id} (${algorithm}, ${arch}, layer ${adapter_layer}, seed ${seed}) failed at: ${end_time}" >> ${MAIN_SUMMARY_LOG}
            echo "GPU ${gpu_id}: ✗ Experiment ${exp_id} failed - ${algorithm} (${arch}, layer ${adapter_layer}) seed ${seed}"
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
    for adapter_layer in "${adapter_layers[@]}"; do
        for seed in "${seeds[@]}"; do
            experiment_list+=("${arch}:${adapter_layer}:${seed}:${experiment_id}")
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
echo "ADAPTER LAYER ABLATION STUDY COMPLETED!"
echo "=========================================="
echo "Completed at: $(date)"
echo "Summary saved to: ${MAIN_SUMMARY_LOG}"
echo ""
echo "Results location: ${BASE_OUTPUT_DIR}"
echo "Logs location: ${BASE_LOG_DIR}"
echo ""
echo "Directory structure:"
echo "  ${BASE_OUTPUT_DIR}/{arch}/layer_{0-11}/${algorithm}/"
echo ""
echo "To analyze results:"
echo "  cd ${BASE_LOG_DIR}"
echo "  python analyze_layer_ablation.py"

# Final log entry
echo "" >> ${MAIN_SUMMARY_LOG}
echo "All experiments completed at $(date)" >> ${MAIN_SUMMARY_LOG}
echo "Results saved to: ${BASE_OUTPUT_DIR}" >> ${MAIN_SUMMARY_LOG}
echo "Logs saved to: ${BASE_LOG_DIR}" >> ${MAIN_SUMMARY_LOG}
echo "" >> ${MAIN_SUMMARY_LOG}
echo "Experiment breakdown:" >> ${MAIN_SUMMARY_LOG}
echo "  Architectures: ${#archs[@]}" >> ${MAIN_SUMMARY_LOG}
echo "  Adapter layers: ${#adapter_layers[@]}" >> ${MAIN_SUMMARY_LOG}
echo "  Seeds: ${#seeds[@]}" >> ${MAIN_SUMMARY_LOG}
echo "  Total: $((${#archs[@]} * ${#adapter_layers[@]} * ${#seeds[@]})) experiments" >> ${MAIN_SUMMARY_LOG}
