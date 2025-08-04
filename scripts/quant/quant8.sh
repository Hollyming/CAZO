#!/bin/bash

# Quantized Model Experiments for Multiple Algorithms
# Testing NoAdapt, LAME, T3A, FOA, ZO, CAZO with 8-bit quantization
# Supports running on any specified GPUs with sequential execution per GPU

PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO

# ====================================================================
# GPU配置部分 - 用户可以指定要使用的GPU
# ====================================================================
# 指定要使用的GPU ID列表（可以是0-7中的任意组合）
# 示例：AVAILABLE_GPUS=(0 1 2 3) 或 AVAILABLE_GPUS=(1 3 5 7)
AVAILABLE_GPUS=(0 2 3 4)  # 默认使用所有8张卡，用户可以修改这里

# 如果通过命令行参数指定GPU，优先使用命令行参数
if [ $# -gt 0 ]; then
    AVAILABLE_GPUS=($@)
    echo "使用命令行指定的GPU: ${AVAILABLE_GPUS[@]}"
else
    echo "使用默认GPU配置: ${AVAILABLE_GPUS[@]}"
fi

# 验证GPU数量
if [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; then
    echo "错误：必须指定至少一张GPU"
    exit 1
fi

# Create quantization experiment directories
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/quant"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/quant"

mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${BASE_LOG_DIR}

# Common parameters
batch_size=64
workers=16

# Create summary log
SUMMARY_LOG="${BASE_LOG_DIR}/quant_experiments_summary.log"
echo "Quantized Model Experiments Started at $(date)" > ${SUMMARY_LOG}
echo "Algorithms: NoAdapt, LAME, T3A, FOA, ZO_Base, CAZO" >> ${SUMMARY_LOG}
echo "Quantization: 8-bit (PTQ4ViT)" >> ${SUMMARY_LOG}
echo "Using GPUs: ${AVAILABLE_GPUS[@]}" >> ${SUMMARY_LOG}
echo "GPU Strategy: Sequential execution per GPU" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}

# Change to project root directory
cd ${PROJECT_ROOT}

echo "Starting quantized model experiments..."
echo "Using batch_size=${batch_size} for memory efficiency"
echo "Available GPUs: ${AVAILABLE_GPUS[@]}"
echo "Strategy: Each GPU runs experiments sequentially"

# ====================================================================
# 实验运行函数
# ====================================================================
run_experiment() {
    local algorithm=$1
    local gpu_id=$2
    local algo_params=$3
    local tag_suffix=$4
    
    local output_dir="${BASE_OUTPUT_DIR}/${algorithm}"
    local log_dir="${BASE_LOG_DIR}/${algorithm}"
    mkdir -p ${output_dir}
    mkdir -p ${log_dir}
    
    local tag="_quant8_bs${batch_size}${tag_suffix}"
    local start_time=$(date)
    
    echo "Starting ${algorithm} on GPU ${gpu_id} at ${start_time}"
    echo "Experiment ${algorithm} started at: ${start_time} on GPU ${gpu_id}" >> ${SUMMARY_LOG}
    
    # 同步执行（不使用后台运行&）
    CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
        --batch_size ${batch_size} \
        --workers ${workers} \
        --quant \
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
    
    if [ ${exit_code} -eq 0 ]; then
        echo "✓ ${algorithm} completed successfully on GPU ${gpu_id} at ${end_time}"
        echo "Experiment ${algorithm} completed successfully at: ${end_time} on GPU ${gpu_id}" >> ${SUMMARY_LOG}
    else
        echo "✗ ${algorithm} failed on GPU ${gpu_id} at ${end_time}"
        echo "Experiment ${algorithm} FAILED at: ${end_time} on GPU ${gpu_id}" >> ${SUMMARY_LOG}
    fi
    
    # GPU memory cleanup
    nvidia-smi -i ${gpu_id} --gpu-reset || true
    sleep 10  # Recovery time after each experiment
    
    return ${exit_code}
}

# ====================================================================
# GPU任务管理函数
# ====================================================================
run_experiments_on_gpu() {
    local gpu_id=$1
    shift
    local experiments=("$@")
    
    echo ""
    echo "=========================================="
    echo "GPU ${gpu_id}: Starting sequential experiments"
    echo "Experiments: ${experiments[@]}"
    echo "=========================================="
    
    for algorithm in "${experiments[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "GPU ${gpu_id}: Running ${algorithm}"
        echo "----------------------------------------"
        
        run_experiment ${algorithm} ${gpu_id} "${ALGO_PARAMS[$algorithm]}" "${TAG_SUFFIX[$algorithm]}"
        local result=$?
        
        if [ ${result} -ne 0 ]; then
            echo "Warning: ${algorithm} failed on GPU ${gpu_id}, continuing with next experiment..."
        fi
        
        echo "GPU ${gpu_id}: ${algorithm} finished, preparing for next experiment..."
        sleep 5  # Brief pause between experiments on same GPU
    done
    
    echo ""
    echo "GPU ${gpu_id}: All experiments completed"
}

# Define algorithm-specific parameters
declare -A ALGO_PARAMS
ALGO_PARAMS["no_adapt"]=""
ALGO_PARAMS["lame"]=""
ALGO_PARAMS["t3a"]=""
ALGO_PARAMS["foa"]="--num_prompts 3 --fitness_lambda 0.4"
ALGO_PARAMS["zo_base"]="--lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor 384 --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1"
ALGO_PARAMS["cazo"]="--lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor 384 --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1 --nu 0.8 --fitness_lambda 0.4"

declare -A TAG_SUFFIX
TAG_SUFFIX["no_adapt"]="_baseline"
TAG_SUFFIX["lame"]="_calibration"
TAG_SUFFIX["t3a"]="_classification"
TAG_SUFFIX["foa"]="_prompts3_lambda0.4"
TAG_SUFFIX["zo_base"]="_adapter3_rf384"
TAG_SUFFIX["cazo"]="_adapter3_rf384_nu0.8"

# ====================================================================
# 任务分配策略
# ====================================================================
algorithms=(no_adapt lame t3a foa zo_base cazo)
num_gpus=${#AVAILABLE_GPUS[@]}
num_algorithms=${#algorithms[@]}

echo ""
echo "Task Distribution Strategy:"
echo "- Total algorithms: ${num_algorithms}"
echo "- Available GPUs: ${num_gpus}"

# 创建GPU任务分配
declare -A GPU_TASKS

# 平均分配算法到GPU
for i in "${!algorithms[@]}"; do
    gpu_index=$((i % num_gpus))
    gpu_id=${AVAILABLE_GPUS[$gpu_index]}
    
    if [ -z "${GPU_TASKS[$gpu_id]}" ]; then
        GPU_TASKS[$gpu_id]="${algorithms[$i]}"
    else
        GPU_TASKS[$gpu_id]="${GPU_TASKS[$gpu_id]} ${algorithms[$i]}"
    fi
done

# 显示任务分配
echo ""
echo "Task Assignment:"
for gpu_id in "${AVAILABLE_GPUS[@]}"; do
    if [ -n "${GPU_TASKS[$gpu_id]}" ]; then
        echo "- GPU ${gpu_id}: ${GPU_TASKS[$gpu_id]}"
    fi
done

# ====================================================================
# 开始并行执行（每张卡内串行，多张卡间并行）
# ====================================================================
echo ""
echo "Starting experiments with parallel GPU execution..."

declare -a gpu_pids=()

# 为每张GPU启动一个后台进程来运行其分配的实验
for gpu_id in "${AVAILABLE_GPUS[@]}"; do
    if [ -n "${GPU_TASKS[$gpu_id]}" ]; then
        # 将字符串转换为数组
        IFS=' ' read -ra gpu_experiments <<< "${GPU_TASKS[$gpu_id]}"
        
        # 为每张GPU启动后台进程
        run_experiments_on_gpu ${gpu_id} "${gpu_experiments[@]}" &
        local pid=$!
        gpu_pids+=($pid)
        
        echo "Started GPU ${gpu_id} worker process (PID: ${pid})"
        sleep 2  # 避免同时启动造成的资源竞争
    fi
done

# 等待所有GPU完成实验
echo ""
echo "Waiting for all GPUs to complete their experiments..."

for i in "${!gpu_pids[@]}"; do
    local pid=${gpu_pids[$i]}
    local gpu_id=${AVAILABLE_GPUS[$i]}
    
    echo "Waiting for GPU ${gpu_id} (PID: ${pid}) to complete..."
    wait ${pid}
    local exit_code=$?
    
    if [ ${exit_code} -eq 0 ]; then
        echo "✓ GPU ${gpu_id} completed all experiments successfully"
    else
        echo "✗ GPU ${gpu_id} encountered errors during execution"
    fi
done

echo ""
echo "All GPUs have completed their experiments!"

# ====================================================================
# 结果提取脚本生成
# ====================================================================
# Generate results extraction script
EXTRACT_SCRIPT="${BASE_LOG_DIR}/extract_quant_results.sh"
cat > ${EXTRACT_SCRIPT} << 'EOF'
#!/bin/bash

# Extract results from quantized model experiments
PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/quant"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/quant"
RESULTS_FILE="${BASE_LOG_DIR}/quant_experiments_results.csv"

# Create CSV header
echo "algorithm,accuracy,ece,model_type,gpu_used" > ${RESULTS_FILE}

algorithms=(no_adapt lame t3a foa zo_base cazo)

for algorithm in "${algorithms[@]}"; do
    echo "Extracting results for ${algorithm}..."
    
    # Find log files
    LOG_PATTERN="${BASE_OUTPUT_DIR}/${algorithm}/${algorithm}_quant8_bs64"*"/"*"-log.txt"
    
    if ls ${LOG_PATTERN} 1> /dev/null 2>&1; then
        LOG_FILE=$(ls ${LOG_PATTERN} | head -1)
        
        # Extract final results
        ACC_LINE=$(tail -30 ${LOG_FILE} | grep -E "(mean acc|Top-1 Accuracy)" | tail -1)
        ECE_LINE=$(tail -30 ${LOG_FILE} | grep -E "(mean ece|ECE)" | tail -1)
        
        ACC_VALUE=$(echo "$ACC_LINE" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        ECE_VALUE=$(echo "$ECE_LINE" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        
        # Try to extract GPU info from log
        GPU_INFO=$(grep -E "CUDA_VISIBLE_DEVICES|Starting.*on GPU" ${LOG_FILE} | head -1 | grep -oE '[0-9]+' | head -1)
        
        echo "${algorithm},${ACC_VALUE:-N/A},${ECE_VALUE:-N/A},quantized_8bit,${GPU_INFO:-N/A}" >> ${RESULTS_FILE}
        echo "  ✓ ${algorithm}: ACC=${ACC_VALUE:-N/A}, ECE=${ECE_VALUE:-N/A}, GPU=${GPU_INFO:-N/A}"
    else
        echo "  ✗ No log file found for ${algorithm}"
        echo "${algorithm},N/A,N/A,quantized_8bit,N/A" >> ${RESULTS_FILE}
    fi
done

echo ""
echo "Quantized results extracted to: ${RESULTS_FILE}"
echo "Preview:"
cat ${RESULTS_FILE}
EOF

chmod +x ${EXTRACT_SCRIPT}

# ====================================================================
# 分析脚本生成（同原始版本）
# ====================================================================
# Generate analysis script
ANALYSIS_SCRIPT="${BASE_LOG_DIR}/analyze_quant_results.py"
cat > ${ANALYSIS_SCRIPT} << 'EOF'
#!/usr/bin/env python3
"""
Quantized Model Results Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_quant_results(csv_file):
    """Analyze quantized model results"""
    if not os.path.exists(csv_file):
        print(f"Results file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['ece'] = pd.to_numeric(df['ece'], errors='coerce')
    
    print("Quantized Model Results Summary:")
    print("=" * 50)
    
    for _, row in df.iterrows():
        acc = row['accuracy']
        ece = row['ece']
        alg = row['algorithm']
        
        acc_str = f"{acc:.2f}%" if pd.notna(acc) else "N/A"
        ece_str = f"{ece:.2f}%" if pd.notna(ece) else "N/A"
        
        print(f"{alg:10s}: Accuracy = {acc_str:8s}, ECE = {ece_str}")
    
    # Generate comparison plot
    valid_data = df.dropna(subset=['accuracy', 'ece'])
    
    if len(valid_data) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy plot
        ax1.bar(valid_data['algorithm'], valid_data['accuracy'], color='lightblue', edgecolor='navy')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Quantized Models: Accuracy Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # ECE plot
        ax2.bar(valid_data['algorithm'], valid_data['ece'], color='lightcoral', edgecolor='darkred')
        ax2.set_ylabel('ECE (%)')
        ax2.set_title('Quantized Models: ECE Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        output_dir = os.path.dirname(csv_file)
        plot_path = os.path.join(output_dir, 'quant_results_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        plt.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, 'quant_experiments_results.csv')
    analyze_quant_results(csv_file)
EOF

chmod +x ${ANALYSIS_SCRIPT}

# Final summary
echo "" >> ${SUMMARY_LOG}
echo "All quantized experiments completed at $(date)" >> ${SUMMARY_LOG}
echo "" >> ${SUMMARY_LOG}
echo "GPU Assignment Summary:" >> ${SUMMARY_LOG}
for gpu_id in "${AVAILABLE_GPUS[@]}"; do
    if [ -n "${GPU_TASKS[$gpu_id]}" ]; then
        echo "- GPU ${gpu_id}: ${GPU_TASKS[$gpu_id]}" >> ${SUMMARY_LOG}
    fi
done
echo "" >> ${SUMMARY_LOG}
echo "Next steps:" >> ${SUMMARY_LOG}
echo "1. Extract results: bash ${EXTRACT_SCRIPT}" >> ${SUMMARY_LOG}
echo "2. Analyze results: python3 ${ANALYSIS_SCRIPT}" >> ${SUMMARY_LOG}

echo ""
echo "=========================================="
echo "QUANTIZED MODEL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Summary saved to: ${SUMMARY_LOG}"
echo ""
echo "To extract and analyze results:"
echo "1. bash ${EXTRACT_SCRIPT}"
echo "2. python3 ${ANALYSIS_SCRIPT}"
echo ""
echo "GPU Task Summary:"
for gpu_id in "${AVAILABLE_GPUS[@]}"; do
    if [ -n "${GPU_TASKS[$gpu_id]}" ]; then
        echo "- GPU ${gpu_id}: ${GPU_TASKS[$gpu_id]}"
    fi
done