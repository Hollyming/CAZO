#!/bin/bash

# Main Experiments Script for CAZO Project
# Running 9 algorithms with 5 different seeds each (45 experiments total)
# Using 6 GPUs (0-5) for parallel execution

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO

# Create main experiment directories
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/main_experiments"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/main_experiments"

mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${BASE_LOG_DIR}

# Experiment parameters
batch_size=64
workers=16

# Seeds for 5 experiments
seeds=(42 2020 2025 1234 888)

# Algorithms to test
algorithms=(no_adapt lame t3a tent cotta sar foa zo_base cazo)

# GPU assignment strategy: distribute experiments across 6 GPUs
GPU_COUNT=6

# Create summary log
MAIN_SUMMARY_LOG="${BASE_LOG_DIR}/main_experiments_summary.log"
echo "Main Experiments Started at $(date)" > ${MAIN_SUMMARY_LOG}
echo "Algorithms: ${algorithms[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Seeds: ${seeds[@]}" >> ${MAIN_SUMMARY_LOG}
echo "Total experiments: $((${#algorithms[@]} * ${#seeds[@]}))" >> ${MAIN_SUMMARY_LOG}
echo "============================================" >> ${MAIN_SUMMARY_LOG}

# Function to run single experiment
run_experiment() {
    local algorithm=$1
    local seed=$2
    local gpu_id=$3
    local exp_id=$4
    
    echo "Starting experiment ${exp_id}: ${algorithm} with seed ${seed} on GPU ${gpu_id}"
    
    # Create algorithm-specific directories
    local output_dir="${BASE_OUTPUT_DIR}/${algorithm}"
    local log_dir="${BASE_LOG_DIR}/${algorithm}"
    mkdir -p ${output_dir}
    mkdir -p ${log_dir}
    
    # Set algorithm-specific parameters
    local algo_params=""
    local tag="_seed${seed}_bs${batch_size}"
    
    case ${algorithm} in
        "no_adapt")
            algo_params=""
            ;;
        "lame")
            algo_params=""
            ;;
        "t3a")
            algo_params=""
            ;;
        "tent")
            algo_params=""
            ;;
        "cotta")
            algo_params=""
            ;;
        "sar")
            algo_params="--margin_e0 0.4"
            ;;
        "foa")
            algo_params="--num_prompts 3 --fitness_lambda 0.4"
            ;;
        "zo_base")
            algo_params="--lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor 384 --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1"
            ;;
        "cazo")
            algo_params="--lr 0.01 --pertub 20 --adapter_layer 3 --reduction_factor 384 --adapter_style parallel --optimizer sgd --beta 0.9 --epsilon 0.1 --nu 0.8 --fitness_lambda 0.4"
            ;;
    esac
    
    # Run experiment on specified GPU
    local start_time=$(date)
    echo "Experiment ${exp_id} (${algorithm}, seed ${seed}) started at: ${start_time}" >> ${MAIN_SUMMARY_LOG}
    
    # Execute with single GPU visibility
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
        --tag "${tag}" &
    
    local pid=$!
    echo "Experiment ${exp_id} PID: ${pid}" >> ${MAIN_SUMMARY_LOG}
    
    # Wait for experiment to complete
    wait ${pid}
    local exit_code=$?
    
    local end_time=$(date)
    if [ ${exit_code} -eq 0 ]; then
        echo "Experiment ${exp_id} (${algorithm}, seed ${seed}) completed successfully at: ${end_time}" >> ${MAIN_SUMMARY_LOG}
        echo "✓ Experiment ${exp_id} completed: ${algorithm} seed ${seed}"
    else
        echo "ERROR: Experiment ${exp_id} (${algorithm}, seed ${seed}) failed at: ${end_time}" >> ${MAIN_SUMMARY_LOG}
        echo "✗ Experiment ${exp_id} failed: ${algorithm} seed ${seed}"
    fi
}

# GPU load balancing: assign experiments to GPUs in round-robin fashion
experiment_id=0
gpu_pids=()

# Initialize GPU tracking
for gpu in $(seq 0 $((GPU_COUNT-1))); do
    gpu_pids[${gpu}]=""
done

echo "Starting parallel experiments across ${GPU_COUNT} GPUs..."

for algorithm in "${algorithms[@]}"; do
    for seed in "${seeds[@]}"; do
        # Find least loaded GPU
        gpu_id=$((experiment_id % GPU_COUNT))
        
        # Wait if GPU has too many running processes (max 2 per GPU)
        while [ $(jobs -r | wc -l) -ge $((GPU_COUNT * 2)) ]; do
            echo "Waiting for GPU slots to free up..."
            sleep 30
        done
        
        # Run experiment
        run_experiment ${algorithm} ${seed} ${gpu_id} ${experiment_id} &
        
        experiment_id=$((experiment_id + 1))
        
        # Small delay to avoid overwhelming the system
        sleep 10
    done
done

# Wait for all experiments to complete
echo "Waiting for all experiments to complete..."
wait

echo "All experiments completed!"

# Generate results extraction script
EXTRACT_SCRIPT="${BASE_LOG_DIR}/extract_all_results.sh"
cat > ${EXTRACT_SCRIPT} << 'EOF'
#!/bin/bash

# Extract results from all main experiments
PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/main_experiments"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/main_experiments"
RESULTS_FILE="${BASE_LOG_DIR}/main_experiments_results.csv"

# Create CSV header
echo "algorithm,seed,accuracy,ece,dataset" > ${RESULTS_FILE}

algorithms=(no_adapt lame t3a tent cotta sar foa zo_base cazo)
seeds=(42 2020 2025 1234 888)

for algorithm in "${algorithms[@]}"; do
    echo "Extracting results for ${algorithm}..."
    for seed in "${seeds[@]}"; do
        # Find log files
        LOG_PATTERN="${BASE_OUTPUT_DIR}/${algorithm}/${algorithm}_seed${seed}_bs64/"*"-log.txt"
        
        if ls ${LOG_PATTERN} 1> /dev/null 2>&1; then
            # Extract accuracy and ECE
            LOG_FILE=$(ls ${LOG_PATTERN} | head -1)
            
            # Extract final results (looking for mean accuracy and ECE)
            ACC_LINE=$(tail -20 ${LOG_FILE} | grep -E "(mean acc)" | tail -1)
            ECE_LINE=$(tail -20 ${LOG_FILE} | grep -E "(mean ece)" | tail -1)
            
            if [[ -n "$ACC_LINE" ]]; then
                ACC_VALUE=$(echo "$ACC_LINE" | grep -oE '[0-9]+\.[0-9]+' | head -1)
                ECE_VALUE=$(echo "$ECE_LINE" | grep -oE '[0-9]+\.[0-9]+' | head -1)
                
                # Add to CSV (you might need to extract per-dataset results separately)
                echo "${algorithm},${seed},${ACC_VALUE:-N/A},${ECE_VALUE:-N/A},overall" >> ${RESULTS_FILE}
            else
                echo "${algorithm},${seed},N/A,N/A,overall" >> ${RESULTS_FILE}
            fi
        else
            echo "Warning: No log file found for ${algorithm} seed ${seed}"
            echo "${algorithm},${seed},N/A,N/A,overall" >> ${RESULTS_FILE}
        fi
    done
done

echo "Results extracted to: ${RESULTS_FILE}"
EOF

chmod +x ${EXTRACT_SCRIPT}

# Generate analysis script
ANALYSIS_SCRIPT="${BASE_LOG_DIR}/analyze_main_results.py"
cat > ${ANALYSIS_SCRIPT} << 'EOF'
#!/usr/bin/env python3
"""
Main Experiments Analysis Script
Generate error bars and summary statistics for all algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_results(csv_file):
    """Load experiment results from CSV"""
    df = pd.read_csv(csv_file)
    # Convert to numeric, handling N/A values
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['ece'] = pd.to_numeric(df['ece'], errors='coerce')
    return df

def calculate_stats(df):
    """Calculate mean, std, and confidence intervals"""
    stats = df.groupby('algorithm').agg({
        'accuracy': ['mean', 'std', 'count'],
        'ece': ['mean', 'std', 'count']
    }).round(3)
    
    # Calculate 95% confidence intervals
    stats[('accuracy', 'ci_95')] = 1.96 * stats[('accuracy', 'std')] / np.sqrt(stats[('accuracy', 'count')])
    stats[('ece', 'ci_95')] = 1.96 * stats[('ece', 'std')] / np.sqrt(stats[('ece', 'count')])
    
    return stats

def plot_results(df, output_dir):
    """Generate plots with error bars"""
    plt.style.use('seaborn-v0_8')
    
    # Accuracy plot
    plt.figure(figsize=(14, 8))
    stats = df.groupby('algorithm')['accuracy'].agg(['mean', 'std']).reset_index()
    
    plt.bar(stats['algorithm'], stats['mean'], yerr=stats['std'], 
            capsize=5, alpha=0.8, color='skyblue', edgecolor='navy')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Main Experiments: Algorithm Comparison (Accuracy)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (alg, acc, std) in enumerate(zip(stats['algorithm'], stats['mean'], stats['std'])):
        plt.text(i, acc + std + 0.5, f'{acc:.2f}±{std:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_results_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ECE plot
    plt.figure(figsize=(14, 8))
    stats_ece = df.groupby('algorithm')['ece'].agg(['mean', 'std']).reset_index()
    
    plt.bar(stats_ece['algorithm'], stats_ece['mean'], yerr=stats_ece['std'], 
            capsize=5, alpha=0.8, color='lightcoral', edgecolor='darkred')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('ECE (%)', fontsize=12)
    plt.title('Main Experiments: Algorithm Comparison (ECE)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (alg, ece, std) in enumerate(zip(stats_ece['algorithm'], stats_ece['mean'], stats_ece['std'])):
        plt.text(i, ece + std + 0.1, f'{ece:.2f}±{std:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_results_ece.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_latex_table(stats, output_dir):
    """Generate LaTeX table for paper"""
    latex_content = """
\\begin{table}[h]
\\centering
\\caption{Main Experimental Results}
\\label{tab:main_results}
\\begin{tabular}{lcc}
\\hline
Algorithm & Accuracy (\\%) & ECE (\\%) \\\\
\\hline
"""
    
    for alg in stats.index:
        acc_mean = stats.loc[alg, ('accuracy', 'mean')]
        acc_std = stats.loc[alg, ('accuracy', 'std')]
        ece_mean = stats.loc[alg, ('ece', 'mean')]
        ece_std = stats.loc[alg, ('ece', 'std')]
        
        latex_content += f"{alg.replace('_', '\\_')} & ${acc_mean:.2f} \\pm {acc_std:.2f}$ & ${ece_mean:.2f} \\pm {ece_std:.2f}$ \\\\\n"
    
    latex_content += """\\hline
\\end{tabular}
\\end{table}
"""
    
    with open(os.path.join(output_dir, 'main_results_table.tex'), 'w') as f:
        f.write(latex_content)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, 'main_experiments_results.csv')
    
    if os.path.exists(csv_file):
        print("Loading results...")
        df = load_results(csv_file)
        
        print("Calculating statistics...")
        stats = calculate_stats(df)
        print(stats)
        
        print("Generating plots...")
        plot_results(df, base_dir)
        
        print("Generating LaTeX table...")
        generate_latex_table(stats, base_dir)
        
        print("Analysis complete!")
    else:
        print(f"Results file not found: {csv_file}")
        print("Please run extract_all_results.sh first")
EOF

chmod +x ${ANALYSIS_SCRIPT}

echo "" >> ${MAIN_SUMMARY_LOG}
echo "All experiments completed at $(date)" >> ${MAIN_SUMMARY_LOG}
echo "" >> ${MAIN_SUMMARY_LOG}
echo "Next steps:" >> ${MAIN_SUMMARY_LOG}
echo "1. Run: ${EXTRACT_SCRIPT}" >> ${MAIN_SUMMARY_LOG}
echo "2. Run: python3 ${ANALYSIS_SCRIPT}" >> ${MAIN_SUMMARY_LOG}

echo ""
echo "=========================================="
echo "ALL MAIN EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Summary saved to: ${MAIN_SUMMARY_LOG}"
echo ""
echo "To extract and analyze results:"
echo "1. bash ${EXTRACT_SCRIPT}"
echo "2. python3 ${ANALYSIS_SCRIPT}"