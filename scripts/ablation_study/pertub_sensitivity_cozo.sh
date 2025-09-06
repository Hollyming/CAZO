#!/bin/bash

# COZO Perturbation Sensitivity Analysis
# GPU并行执行不同pertub值的敏感性分析
# 基于main_experiment_simplified.sh的GPU分配策略

PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO

cd ${PROJECT_ROOT}

# 创建输出和日志目录
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/pertub_sensitivity_cozo"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/pertub_sensitivity_cozo"

mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${BASE_LOG_DIR}

# ======================== 配置 ========================
# 实验参数
batch_size=64
workers=16

# 固定参数（控制变量）
lr=0.01
adapter_layer=3
reduction_factor=384
adapter_style="parallel"
fitness_lambda=0.4
epsilon=0.1
optimizer="sgd"
beta=0.9
mode="cov_only"

# 测试的pertub值（2到30，步长2）
pertub_values=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30)

# GPU配置 - 根据需要修改GPU数量和编号
GPU_COUNT=7  # 修改为所需的GPU数量
GPU_IDS=(0 1 2 3 4 5 6)  # 修改为可用的GPU编号

# ================================================================

# 创建汇总日志
MAIN_SUMMARY_LOG="${BASE_LOG_DIR}/pertub_sensitivity_cozo_summary.log"
echo "COZO Perturbation Sensitivity Analysis Started at $(date)" > ${MAIN_SUMMARY_LOG}
echo "Testing pertub values: ${pertub_values[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Note: Higher perturbation may provide better gradient estimation but requires more computation" >> ${MAIN_SUMMARY_LOG}
echo "GPU Count: ${GPU_COUNT}" >> ${MAIN_SUMMARY_LOG}
echo "GPU IDs: ${GPU_IDS[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Total experiments: ${#pertub_values[@]}" >> ${MAIN_SUMMARY_LOG}
echo "============================================" >> ${MAIN_SUMMARY_LOG}

# 运行单个GPU上实验的函数
run_gpu_experiments() {
    local gpu_id=$1
    local exp_list=("${@:2}")  # 获取剩余参数作为实验列表
    
    echo "GPU ${gpu_id}: Starting ${#exp_list[@]} experiments"
    
    for exp_info in "${exp_list[@]}"; do
        # 解析实验信息: "pertub:exp_id"
        IFS=':' read -r pertub exp_id <<< "${exp_info}"
        
        echo "GPU ${gpu_id}: Starting experiment ${exp_id} - pertub ${pertub}"
        
        # 创建特定输出和日志目录
        local output_dir="${BASE_OUTPUT_DIR}/pertub_${pertub}"
        local log_dir="${BASE_LOG_DIR}/pertub_${pertub}"
        mkdir -p ${output_dir}
        mkdir -p ${log_dir}
        
        local tag="_pertub_sensitivity_${pertub}_bs${batch_size}_lr${lr}_adapter${adapter_layer}_rf${reduction_factor}_${adapter_style}_eps${epsilon}"
        
        # 记录实验开始
        local start_time=$(date)
        echo "Experiment ${exp_id} (pertub=${pertub}) started at: ${start_time} on GPU ${gpu_id}" >> ${MAIN_SUMMARY_LOG}
        
        # 运行实验
        CUDA_VISIBLE_DEVICES=${gpu_id} python ${PROJECT_ROOT}/main.py \
            --batch_size ${batch_size} \
            --workers ${workers} \
            --data /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet \
            --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
            --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
            --data_corruption /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-c/imagenet-c \
            --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
            --output ${output_dir} \
            --root_log_dir ${log_dir} \
            --algorithm "cozo" \
            --lr ${lr} \
            --pertub ${pertub} \
            --adapter_layer ${adapter_layer} \
            --reduction_factor ${reduction_factor} \
            --adapter_style ${adapter_style} \
            --optimizer ${optimizer} \
            --beta ${beta} \
            --epsilon ${epsilon} \
            --fitness_lambda ${fitness_lambda} \
            --mode ${mode} \
            --tag "${tag}"
        
        local exit_code=$?
        local end_time=$(date)
        
        # 记录实验完成情况
        if [ ${exit_code} -eq 0 ]; then
            echo "Experiment ${exp_id} (pertub=${pertub}) completed successfully at: ${end_time}" >> ${MAIN_SUMMARY_LOG}
            echo "GPU ${gpu_id}: ✓ Experiment ${exp_id} completed - pertub ${pertub}"
        else
            echo "ERROR: Experiment ${exp_id} (pertub=${pertub}) failed at: ${end_time}" >> ${MAIN_SUMMARY_LOG}
            echo "GPU ${gpu_id}: ✗ Experiment ${exp_id} failed - pertub ${pertub}"
        fi
    done
    
    echo "GPU ${gpu_id}: All experiments completed"
}

# 在GPU间分配实验
echo "Distributing experiments across ${GPU_COUNT} GPUs..."

# 创建实验列表
experiment_list=()
experiment_id=0

for pertub in "${pertub_values[@]}"; do
    experiment_list+=("${pertub}:${experiment_id}")
    experiment_id=$((experiment_id + 1))
done

echo "Total experiments to run: ${#experiment_list[@]}"

# 在GPU间平均分配实验
declare -a gpu_experiment_lists
for ((i=0; i<GPU_COUNT; i++)); do
    gpu_experiment_lists[i]=""
done

# 轮询分配
for ((i=0; i<${#experiment_list[@]}; i++)); do
    gpu_idx=$((i % GPU_COUNT))
    if [ -z "${gpu_experiment_lists[gpu_idx]}" ]; then
        gpu_experiment_lists[gpu_idx]="${experiment_list[i]}"
    else
        gpu_experiment_lists[gpu_idx]="${gpu_experiment_lists[gpu_idx]} ${experiment_list[i]}"
    fi
done

# 打印实验分配情况
echo ""
echo "Experiment distribution:"
for ((i=0; i<GPU_COUNT; i++)); do
    exp_count=$(echo ${gpu_experiment_lists[i]} | wc -w)
    echo "GPU ${GPU_IDS[i]}: ${exp_count} experiments"
done
echo ""

# 在每个GPU上并行启动实验
pids=()
for ((i=0; i<GPU_COUNT; i++)); do
    gpu_id=${GPU_IDS[i]}
    if [ -n "${gpu_experiment_lists[i]}" ]; then
        # 将空格分隔的字符串转换为数组
        IFS=' ' read -ra gpu_experiments <<< "${gpu_experiment_lists[i]}"
        
        echo "Starting experiments on GPU ${gpu_id}..."
        run_gpu_experiments ${gpu_id} "${gpu_experiments[@]}" &
        pids+=($!)
    fi
done

# 等待所有GPU进程完成
echo ""
echo "Waiting for all GPU processes to complete..."
for pid in "${pids[@]}"; do
    wait ${pid}
done

# 最终汇总
echo ""
echo "=========================================="
echo "ALL PERTURBATION SENSITIVITY EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Completed at: $(date)"
echo "Summary saved to: ${MAIN_SUMMARY_LOG}"
echo ""
echo "Results location: ${BASE_OUTPUT_DIR}"
echo "Logs location: ${BASE_LOG_DIR}"
echo ""

# 最终日志条目
echo "" >> ${MAIN_SUMMARY_LOG}
echo "All experiments completed at $(date)" >> ${MAIN_SUMMARY_LOG}
echo "Results saved to: ${BASE_OUTPUT_DIR}" >> ${MAIN_SUMMARY_LOG}
echo "Logs saved to: ${BASE_LOG_DIR}" >> ${MAIN_SUMMARY_LOG}

echo "Note: pertub值影响梯度估计的质量和计算成本"
echo "较低的pertub值计算更快但梯度估计可能不够准确"
echo "较高的pertub值提供更好的梯度估计但需要更多计算"
