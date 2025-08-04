#!/bin/bash

# CAZO Reduction Factor Sensitivity Analysis
# This script tests the effect of different adapter reduction factors (384, 256, 192, 128, 96, 48)

PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO

# Fixed parameters (control variables)
batch_size=64
lr=0.01
pertub=20  # Using the baseline pertub value
adapter_layer=3  # Using the baseline adapter layer
adapter_style="parallel"
fitness_lambda=0.4

# Hessian related parameters
epsilon=0.1
nu=0.8

# Optimizer related parameters
optimizer="sgd"
beta=0.9

# GPU setting
export CUDA_VISIBLE_DEVICES=1

# Create base output and log directories
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/reduction_factor_sensitivity"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/reduction_factor_sensitivity"

mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${BASE_LOG_DIR}

# Log file for summary
SUMMARY_LOG="${BASE_LOG_DIR}/reduction_factor_sensitivity_summary.log"
echo "CAZO Reduction Factor Sensitivity Analysis Started at $(date)" > ${SUMMARY_LOG}
echo "Testing reduction_factor values: 384, 256, 192, 128, 96, 48" >> ${SUMMARY_LOG}
echo "Note: Larger reduction factor = smaller adapter bottleneck dimension" >> ${SUMMARY_LOG}
echo "Adapter bottleneck dim = hidden_size / reduction_factor = 768 / reduction_factor" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}

# Array to store results for final summary
declare -a RESULTS=()

# Array of reduction factor values to test
reduction_factors=(384 256 192 128 96 48)

# Loop through reduction_factor values
for reduction_factor in "${reduction_factors[@]}"; do
    # Calculate actual bottleneck dimension
    bottleneck_dim=$((768 / reduction_factor))
    
    echo "=========================================="
    echo "Starting experiment with reduction_factor=${reduction_factor}"
    echo "Adapter bottleneck dimension: ${bottleneck_dim}"
    echo "=========================================="
    
    # Record start time
    start_time=$(date)
    echo "Experiment reduction_factor=${reduction_factor} (bottleneck_dim=${bottleneck_dim}) started at: ${start_time}" >> ${SUMMARY_LOG}
    
    # Create specific output and log directories for this reduction_factor value
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/reduction_factor_${reduction_factor}"
    LOG_DIR="${BASE_LOG_DIR}/reduction_factor_${reduction_factor}"
    
    mkdir -p ${OUTPUT_DIR}
    mkdir -p ${LOG_DIR}
    
    # Generate tag for this experiment
    tag="_reduction_factor_sensitivity_${reduction_factor}_bs${batch_size}_lr${lr}_pertub${pertub}_adapter${adapter_layer}_${adapter_style}_eps${epsilon}_nu${nu}"
    
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
        echo "Experiment reduction_factor=${reduction_factor} completed successfully at: ${end_time}" >> ${SUMMARY_LOG}
        
        # Try to extract accuracy results from log (adjust path as needed)
        LOG_FILE="${LOG_DIR}/cazo${tag}/"*"-log.txt"
        if ls ${LOG_FILE} 1> /dev/null 2>&1; then
            # Extract final accuracy results
            RESULT=$(tail -20 ${LOG_FILE} | grep -E "(mean acc|Top-1 Accuracy)" | tail -1)
            RESULTS+=("reduction_factor=${reduction_factor} (bottleneck=${bottleneck_dim}): ${RESULT}")
            echo "Result - reduction_factor=${reduction_factor} (bottleneck=${bottleneck_dim}): ${RESULT}" >> ${SUMMARY_LOG}
        else
            echo "Warning: Log file not found for reduction_factor=${reduction_factor}" >> ${SUMMARY_LOG}
        fi
    else
        echo "ERROR: Experiment reduction_factor=${reduction_factor} failed!" >> ${SUMMARY_LOG}
        echo "ERROR: Experiment with reduction_factor=${reduction_factor} failed!"
    fi
    
    echo "Experiment reduction_factor=${reduction_factor} finished. Moving to next..."
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
echo "ALL REDUCTION FACTOR SENSITIVITY EXPERIMENTS COMPLETED!"
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
ANALYSIS_SCRIPT="${BASE_LOG_DIR}/analyze_reduction_factor_results.py"
cat > ${ANALYSIS_SCRIPT} << 'EOF'
#!/usr/bin/env python3
"""
Reduction Factor Sensitivity Analysis Script
This script parses the experimental results and generates plots.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_results(summary_file):
    """Parse results from summary log file"""
    results = {}
    
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('reduction_factor='):
            # Extract reduction_factor and accuracy
            match = re.search(r'reduction_factor=(\d+).*?(\d+\.\d+)', line)
            if match:
                rf = int(match.group(1))
                accuracy = float(match.group(2))
                results[rf] = accuracy
    
    return results

def plot_results(results, output_dir):
    """Generate plots for reduction factor sensitivity"""
    if not results:
        print("No results found to plot")
        return
    
    # Sort by reduction factor
    rfs = sorted(results.keys(), reverse=True)  # From large to small
    accuracies = [results[rf] for rf in rfs]
    bottleneck_dims = [768 // rf for rf in rfs]
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot accuracy vs reduction factor
    color = 'tab:blue'
    ax1.set_xlabel('Reduction Factor', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', color=color, fontsize=14)
    line1 = ax1.plot(rfs, accuracies, 'bo-', linewidth=2, markersize=8, color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for bottleneck dimension
    ax2 = ax1.twiny()
    color = 'tab:red'
    ax2.set_xlabel('Adapter Bottleneck Dimension', color=color, fontsize=14)
    ax2.set_xticks(rfs)
    ax2.set_xticklabels(bottleneck_dims)
    ax2.tick_params(axis='x', labelcolor=color)
    
    # Add value labels on points
    for rf, acc in zip(rfs, accuracies):
        ax1.annotate(f'{acc:.2f}%', (rf, acc), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title('CAZO: Effect of Adapter Reduction Factor on Performance', fontsize=16, pad=20)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'reduction_factor_sensitivity.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create second plot: accuracy vs bottleneck dimension
    plt.figure(figsize=(12, 8))
    plt.plot(bottleneck_dims, accuracies, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Adapter Bottleneck Dimension', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('CAZO: Accuracy vs Adapter Bottleneck Dimension', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for dim, acc in zip(bottleneck_dims, accuracies):
        plt.annotate(f'{acc:.2f}%', (dim, acc), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plot_path2 = os.path.join(output_dir, 'bottleneck_dimension_sensitivity.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to:")
    print(f"  {plot_path}")
    print(f"  {plot_path2}")
    
    # Print statistics
    best_rf = max(results, key=results.get)
    best_accuracy = results[best_rf]
    worst_rf = min(results, key=results.get)
    worst_accuracy = results[worst_rf]
    
    print(f"\nAnalysis Results:")
    print(f"Best reduction factor: {best_rf} (bottleneck dim: {768//best_rf}) with accuracy {best_accuracy:.2f}%")
    print(f"Worst reduction factor: {worst_rf} (bottleneck dim: {768//worst_rf}) with accuracy {worst_accuracy:.2f}%")
    print(f"Performance gap: {best_accuracy - worst_accuracy:.2f}%")
    
    # Print detailed mapping
    print(f"\nReduction Factor -> Bottleneck Dimension -> Accuracy:")
    for rf in sorted(results.keys(), reverse=True):
        bottleneck = 768 // rf
        acc = results[rf]
        print(f"  {rf:3d} -> {bottleneck:2d} -> {acc:.2f}%")

if __name__ == "__main__":
    import sys
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_file = os.path.join(base_dir, 'reduction_factor_sensitivity_summary.log')
    
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