#!/usr/bin/env python3
"""
Reduction Factor Sensitivity Analysis Script
This script parses the experimental results and generates plots for both accuracy and ECE.
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
            # Extract reduction_factor, accuracy, and ECE
            # Format: reduction_factor=384 (bottleneck=2): accuracy=75.23%, ece=12.45%
            rf_match = re.search(r'reduction_factor=(\d+)', line)
            acc_match = re.search(r'accuracy=([0-9]+\.?[0-9]*)', line)
            ece_match = re.search(r'ece=([0-9]+\.?[0-9]*)', line)
            
            if rf_match and acc_match and ece_match:
                rf = int(rf_match.group(1))
                accuracy = float(acc_match.group(1))
                ece = float(ece_match.group(1))
                results[rf] = {'accuracy': accuracy, 'ece': ece}
            elif rf_match:
                rf = int(rf_match.group(1))
                accuracy = float(acc_match.group(1)) if acc_match else None
                ece = float(ece_match.group(1)) if ece_match else None
                results[rf] = {'accuracy': accuracy, 'ece': ece}
    
    return results

def plot_results(results, output_dir):
    """Generate plots for reduction factor sensitivity"""
    if not results:
        print("No results found to plot")
        return
    
    # Sort by reduction factor
    rfs = sorted(results.keys(), reverse=True)  # From large to small
    accuracies = [results[rf]['accuracy'] for rf in rfs if results[rf]['accuracy'] is not None]
    eces = [results[rf]['ece'] for rf in rfs if results[rf]['ece'] is not None]
    bottleneck_dims = [768 // rf for rf in rfs]
    
    # Filter out None values
    valid_rfs = [rf for rf in rfs if results[rf]['accuracy'] is not None and results[rf]['ece'] is not None]
    valid_accuracies = [results[rf]['accuracy'] for rf in valid_rfs]
    valid_eces = [results[rf]['ece'] for rf in valid_rfs]
    valid_bottleneck_dims = [768 // rf for rf in valid_rfs]
    
    if not valid_rfs:
        print("No valid data found to plot")
        return
    
    # Create dual-metric plot with shared x-axis
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    # Plot accuracy
    color1 = 'tab:blue'
    ax1.set_xlabel('Reduction Factor', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', color=color1, fontsize=14)
    line1 = ax1.plot(valid_rfs, valid_accuracies, 'bo-', linewidth=3, markersize=10, 
                     color=color1, label='Accuracy', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for ECE
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('ECE (%)', color=color2, fontsize=14)
    line2 = ax2.plot(valid_rfs, valid_eces, 'rs-', linewidth=3, markersize=10, 
                     color=color2, label='ECE', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add value labels on points with smart positioning
    acc_max = max(valid_accuracies)
    acc_min = min(valid_accuracies)
    acc_range = acc_max - acc_min
    
    for i, (rf, acc) in enumerate(zip(valid_rfs, valid_accuracies)):
        # 智能调整标注位置：顶部数据点向下偏移，避免超出边界
        if acc > acc_max - 0.2 * acc_range:  # 接近顶部的点
            xytext = (0, -25)  # 向下偏移
            va = 'top'
        else:
            xytext = (0, 15)   # 向上偏移
            va = 'bottom'
        
        # 左右边界处理
        if rf == valid_rfs[0]:  # 最左边的点
            xytext = (10, xytext[1])
            ha = 'left'
        elif rf == valid_rfs[-1]:  # 最右边的点
            xytext = (-10, xytext[1])
            ha = 'right'
        else:
            ha = 'center'
            
        ax1.annotate(f'{acc:.2f}%', (rf, acc), 
                    textcoords="offset points", xytext=xytext, ha=ha, va=va,
                    color=color1, fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
    
    ece_max = max(valid_eces)
    ece_min = min(valid_eces)
    ece_range = ece_max - ece_min
    
    for i, (rf, ece) in enumerate(zip(valid_rfs, valid_eces)):
        # 智能调整ECE标注位置：底部数据点向上偏移
        if ece < ece_min + 0.2 * ece_range:  # 接近底部的点
            xytext = (0, 25)   # 向上偏移
            va = 'bottom'
        else:
            xytext = (0, -20)  # 向下偏移
            va = 'top'
        
        # 左右边界处理
        if rf == valid_rfs[0]:  # 最左边的点
            xytext = (10, xytext[1])
            ha = 'left'
        elif rf == valid_rfs[-1]:  # 最右边的点
            xytext = (-10, xytext[1])
            ha = 'right'
        else:
            ha = 'center'
            
        ax2.annotate(f'{ece:.2f}%', (rf, ece), 
                    textcoords="offset points", xytext=xytext, ha=ha, va=va,
                    color=color2, fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Add bottleneck dimension labels on top
    ax3 = ax1.twiny()
    ax3.set_xlabel('Adapter Bottleneck Dimension', color='tab:green', fontsize=14)
    ax3.set_xticks(valid_rfs)
    ax3.set_xticklabels(valid_bottleneck_dims)
    ax3.tick_params(axis='x', labelcolor='tab:green')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # 将图例放在中间靠右位置，确保不遮挡曲线
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, 
                       loc='center right', fontsize=11,
                       frameon=True, fancybox=True, shadow=True,
                       facecolor='white', edgecolor='gray',
                       borderpad=0.8, handletextpad=0.5)
    legend.set_zorder(1000)  # 确保图例在最顶层
    
    # plt.title('CAZO: Effect of Adapter Reduction Factor on Performance\n(Accuracy and ECE)', 
    #           fontsize=16, pad=30)
    
    # Save combined plot
    plot_path = os.path.join(output_dir, 'reduction_factor_accuracy_ece.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate plots for clarity
    # Plot 1: Accuracy vs Bottleneck Dimension
    plt.figure(figsize=(12, 8))
    plt.plot(valid_bottleneck_dims, valid_accuracies, 'bo-', linewidth=3, markersize=10)
    plt.xlabel('Adapter Bottleneck Dimension', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('CAZO: Accuracy vs Adapter Bottleneck Dimension', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    for dim, acc in zip(valid_bottleneck_dims, valid_accuracies):
        plt.annotate(f'{acc:.2f}%', (dim, acc), 
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontweight='bold')
    
    plot_path2 = os.path.join(output_dir, 'bottleneck_dimension_accuracy.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: ECE vs Bottleneck Dimension
    plt.figure(figsize=(12, 8))
    plt.plot(valid_bottleneck_dims, valid_eces, 'rs-', linewidth=3, markersize=10)
    plt.xlabel('Adapter Bottleneck Dimension', fontsize=14)
    plt.ylabel('ECE (%)', fontsize=14)
    plt.title('CAZO: ECE vs Adapter Bottleneck Dimension', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    for dim, ece in zip(valid_bottleneck_dims, valid_eces):
        plt.annotate(f'{ece:.2f}%', (dim, ece), 
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontweight='bold')
    
    plot_path3 = os.path.join(output_dir, 'bottleneck_dimension_ece.png')
    plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to:")
    print(f"  {plot_path}")
    print(f"  {plot_path2}")
    print(f"  {plot_path3}")
    
    # Print statistics
    if valid_accuracies:
        best_acc_rf = valid_rfs[np.argmax(valid_accuracies)]
        best_accuracy = max(valid_accuracies)
        worst_acc_rf = valid_rfs[np.argmin(valid_accuracies)]
        worst_accuracy = min(valid_accuracies)
        
        best_ece_rf = valid_rfs[np.argmin(valid_eces)]  # Lower ECE is better
        best_ece = min(valid_eces)
        worst_ece_rf = valid_rfs[np.argmax(valid_eces)]
        worst_ece = max(valid_eces)
        
        print(f"\nAnalysis Results:")
        print(f"Best accuracy: {best_acc_rf} (bottleneck dim: {768//best_acc_rf}) with {best_accuracy:.2f}%")
        print(f"Worst accuracy: {worst_acc_rf} (bottleneck dim: {768//worst_acc_rf}) with {worst_accuracy:.2f}%")
        print(f"Accuracy gap: {best_accuracy - worst_accuracy:.2f}%")
        print(f"")
        print(f"Best ECE: {best_ece_rf} (bottleneck dim: {768//best_ece_rf}) with {best_ece:.2f}%")
        print(f"Worst ECE: {worst_ece_rf} (bottleneck dim: {768//worst_ece_rf}) with {worst_ece:.2f}%")
        print(f"ECE gap: {worst_ece - best_ece:.2f}%")
        
        # Print detailed mapping
        print(f"\nDetailed Results:")
        print(f"{'RF':>3} -> {'Bottleneck':>10} -> {'Accuracy':>8} -> {'ECE':>6}")
        print("-" * 40)
        for rf in sorted(results.keys(), reverse=True):
            bottleneck = 768 // rf
            acc = results[rf]['accuracy']
            ece = results[rf]['ece']
            acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
            ece_str = f"{ece:.2f}%" if ece is not None else "N/A"
            print(f"{rf:3d} -> {bottleneck:10d} -> {acc_str:>8} -> {ece_str:>6}")

if __name__ == "__main__":
    import sys
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 优先使用修复后的文件
    summary_file_fixed = os.path.join(base_dir, 'reduction_factor_sensitivity_summary_fixed.log')
    summary_file_original = os.path.join(base_dir, 'reduction_factor_sensitivity_summary.log')
    
    if os.path.exists(summary_file_fixed):
        print(f"Using fixed summary file: {summary_file_fixed}")
        results = parse_results(summary_file_fixed)
        plot_results(results, base_dir)
    elif os.path.exists(summary_file_original):
        print(f"Using original summary file: {summary_file_original}")
        results = parse_results(summary_file_original)
        plot_results(results, base_dir)
    else:
        print(f"Summary file not found. Searched:")
        print(f"  {summary_file_fixed}")
        print(f"  {summary_file_original}")
