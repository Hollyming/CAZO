#!/bin/bash

# CAZO Adapter Layer Sensitivity Analysis
# This script tests the effect of different adapter layers (1, 2, 3, ..., 12) in ViT-Base/16

PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO

# Fixed parameters (control variables)
batch_size=64
lr=0.01
pertub=20  # Using the baseline pertub value
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
export CUDA_VISIBLE_DEVICES=4

# Create base output and log directories
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/adapter_layer_sensitivity"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/adapter_layer_sensitivity"

mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${BASE_LOG_DIR}

# Log file for summary
SUMMARY_LOG="${BASE_LOG_DIR}/adapter_layer_sensitivity_summary.log"
echo "CAZO Adapter Layer Sensitivity Analysis Started at $(date)" > ${SUMMARY_LOG}
echo "Testing adapter_layer values: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11" >> ${SUMMARY_LOG}
echo "Corresponding to ViT-Base/16 Transformer Blocks: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}

# Array to store results for final summary
declare -a RESULTS=()

# Loop through adapter_layer values from 0 to 11 (corresponding to 12 transformer layers)
for adapter_layer in $(seq 0 11); do
    echo "=========================================="
    echo "Starting experiment with adapter_layer=${adapter_layer} (Transformer Block $((adapter_layer + 1)))"
    echo "=========================================="
    
    # Record start time
    start_time=$(date)
    echo "Experiment adapter_layer=${adapter_layer} started at: ${start_time}" >> ${SUMMARY_LOG}
    
    # Create specific output and log directories for this adapter_layer value
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/adapter_layer_${adapter_layer}"
    LOG_DIR="${BASE_LOG_DIR}/adapter_layer_${adapter_layer}"
    
    mkdir -p ${OUTPUT_DIR}
    mkdir -p ${LOG_DIR}
    
    # Generate tag for this experiment
    tag="_adapter_layer_sensitivity_${adapter_layer}_bs${batch_size}_lr${lr}_pertub${pertub}_rf${reduction_factor}_${adapter_style}_eps${epsilon}_nu${nu}"
    
    # Run the experiment
    python ${PROJECT_ROOT}/main.py \
        --batch_size ${batch_size} \
        --workers 16 \
        --data /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet \
        --data_v2 /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenetv2 \
        --data_sketch /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-sketch/sketch \
        --data_corruption /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-c/imagenet-c \
        --data_rendition /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-r/imagenet-r \
        --output ${OUTPUT_DIR} \
        --root_log_dir ${LOG_DIR} \
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
        echo "Experiment adapter_layer=${adapter_layer} completed successfully at: ${end_time}" >> ${SUMMARY_LOG}
        
        # Try to extract accuracy results from log (adjust path as needed)
        LOG_FILE="${LOG_DIR}/cazo${tag}/"*"-log.txt"
        if ls ${LOG_FILE} 1> /dev/null 2>&1; then
            # Extract final accuracy results
            RESULT=$(tail -20 ${LOG_FILE} | grep -E "(mean acc|Top-1 Accuracy)" | tail -1)
            RESULTS+=("adapter_layer=${adapter_layer}: ${RESULT}")
            echo "Result - adapter_layer=${adapter_layer}: ${RESULT}" >> ${SUMMARY_LOG}
        else
            echo "Warning: Log file not found for adapter_layer=${adapter_layer}" >> ${SUMMARY_LOG}
        fi
    else
        echo "ERROR: Experiment adapter_layer=${adapter_layer} failed!" >> ${SUMMARY_LOG}
        echo "ERROR: Experiment with adapter_layer=${adapter_layer} failed!"
    fi
    
    echo "Experiment adapter_layer=${adapter_layer} finished. Moving to next..."
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
echo "ALL ADAPTER LAYER SENSITIVITY EXPERIMENTS COMPLETED!"
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

# Generate analysis script
ANALYSIS_SCRIPT="${BASE_LOG_DIR}/analyze_adapter_layer_results.py"
cat > ${ANALYSIS_SCRIPT} << 'EOF'
#!/usr/bin/env python3
"""
Adapter Layer Sensitivity Analysis Script
This script parses the experimental results and generates plots.
"""

import re
import matplotlib.pyplot as plt
import os

def parse_results(summary_file):
    """Parse results from summary log file"""
    results = {}
    
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('adapter_layer='):
            # Extract adapter_layer number and accuracy
            match = re.search(r'adapter_layer=(\d+):.*?(\d+\.\d+)', line)
            if match:
                layer = int(match.group(1))
                accuracy = float(match.group(2))
                results[layer] = accuracy
    
    return results

def plot_results(results, output_dir):
    """Generate plots for adapter layer sensitivity"""
    if not results:
        print("No results found to plot")
        return
    
    # Sort by layer number
    layers = sorted(results.keys())
    accuracies = [results[layer] for layer in layers]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(layers, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Adapter Layer Position', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('CAZO: Effect of Adapter Layer Position on Performance', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(layers)
    
    # Add value labels on points
    for layer, acc in zip(layers, accuracies):
        plt.annotate(f'{acc:.2f}%', (layer, acc), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Save plot
    plot_path = os.path.join(output_dir, 'adapter_layer_sensitivity.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {plot_path}")
    
    # Print statistics
    best_layer = max(results, key=results.get)
    best_accuracy = results[best_layer]
    worst_layer = min(results, key=results.get)
    worst_accuracy = results[worst_layer]
    
    print(f"\nAnalysis Results:")
    print(f"Best layer: {best_layer} with accuracy {best_accuracy:.2f}%")
    print(f"Worst layer: {worst_layer} with accuracy {worst_accuracy:.2f}%")
    print(f"Performance gap: {best_accuracy - worst_accuracy:.2f}%")

if __name__ == "__main__":
    import sys
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_file = os.path.join(base_dir, 'adapter_layer_sensitivity_summary.log')
    
    if os.path.exists(summary_file):
        results = parse_results(summary_file)
        plot_results(results, base_dir)
    else:
        print(f"Summary file not found: {summary_file}")
EOF

chmod +x ${ANALYSIS_SCRIPT}

echo ""
echo "Analysis script created: ${ANALYSIS_SCRIPT}"
echo "Run it after experiments complete: python3 ${ANALYSIS_SCRIPT}"