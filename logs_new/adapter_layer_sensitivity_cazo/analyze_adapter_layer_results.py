#!/usr/bin/env python3
"""
Adapter Layer Sensitivity Analysis Script
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
        if line.startswith('adapter_layer='):
            # Extract adapter_layer, accuracy, and ECE
            # Format: adapter_layer=3 (transformer_block=4): accuracy=75.23%, ece=12.45%
            layer_match = re.search(r'adapter_layer=(\d+)', line)
            acc_match = re.search(r'accuracy=([0-9]+\.?[0-9]*)', line)
            ece_match = re.search(r'ece=([0-9]+\.?[0-9]*)', line)
            
            if layer_match and acc_match and ece_match:
                layer = int(layer_match.group(1))
                accuracy = float(acc_match.group(1))
                ece = float(ece_match.group(1))
                results[layer] = {'accuracy': accuracy, 'ece': ece}
            elif layer_match:
                layer = int(layer_match.group(1))
                accuracy = float(acc_match.group(1)) if acc_match else None
                ece = float(ece_match.group(1)) if ece_match else None
                results[layer] = {'accuracy': accuracy, 'ece': ece}
    
    return results

def plot_results(results, output_dir):
    """Generate plots for adapter layer sensitivity"""
    if not results:
        print("No results found to plot")
        return
    
    # Sort by adapter layer
    layers = sorted(results.keys())
    
    # Filter out None values
    valid_layers = [l for l in layers if results[l]['accuracy'] is not None and results[l]['ece'] is not None]
    valid_accuracies = [results[l]['accuracy'] for l in valid_layers]
    valid_eces = [results[l]['ece'] for l in valid_layers]
    valid_transformer_blocks = [l + 1 for l in valid_layers]  # Convert to 1-based indexing
    
    if not valid_layers:
        print("No valid data found to plot")
        return
    
    # Create dual-metric plot with shared x-axis
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    # Plot accuracy
    color1 = 'tab:blue'
    ax1.set_xlabel('Adapter Layer Position (0-based indexing)', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', color=color1, fontsize=14)
    line1 = ax1.plot(valid_layers, valid_accuracies, 'bo-', linewidth=3, markersize=10, 
                     color=color1, label='Accuracy', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis ticks
    ax1.set_xticks(valid_layers)
    ax1.set_xticklabels(valid_layers)
    
    # Create second y-axis for ECE
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('ECE (%)', color=color2, fontsize=14)
    line2 = ax2.plot(valid_layers, valid_eces, 'rs-', linewidth=3, markersize=10, 
                     color=color2, label='ECE', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add value labels on points with smart positioning
    acc_max = max(valid_accuracies)
    acc_min = min(valid_accuracies)
    acc_range = acc_max - acc_min
    
    for i, (layer, acc) in enumerate(zip(valid_layers, valid_accuracies)):
        # 智能调整标注位置：顶部数据点向下偏移，避免超出边界
        if acc > acc_max - 0.2 * acc_range:  # 接近顶部的点
            xytext = (0, -25)  # 向下偏移
            va = 'top'
        else:
            xytext = (0, 15)   # 向上偏移
            va = 'bottom'
        
        # 左右边界处理
        if layer == valid_layers[0]:  # 最左边的点
            xytext = (10, xytext[1])
            ha = 'left'
        elif layer == valid_layers[-1]:  # 最右边的点
            xytext = (-10, xytext[1])
            ha = 'right'
        else:
            ha = 'center'
            
        ax1.annotate(f'{acc:.2f}%', (layer, acc), 
                    textcoords="offset points", xytext=xytext, ha=ha, va=va,
                    color=color1, fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
    
    ece_max = max(valid_eces)
    ece_min = min(valid_eces)
    ece_range = ece_max - ece_min
    
    for i, (layer, ece) in enumerate(zip(valid_layers, valid_eces)):
        # 智能调整ECE标注位置：底部数据点向上偏移
        if ece < ece_min + 0.2 * ece_range:  # 接近底部的点
            xytext = (0, 25)   # 向上偏移
            va = 'bottom'
        else:
            xytext = (0, -20)  # 向下偏移
            va = 'top'
        
        # 左右边界处理
        if layer == valid_layers[0]:  # 最左边的点
            xytext = (10, xytext[1])
            ha = 'left'
        elif layer == valid_layers[-1]:  # 最右边的点
            xytext = (-10, xytext[1])
            ha = 'right'
        else:
            ha = 'center'
            
        ax2.annotate(f'{ece:.2f}%', (layer, ece), 
                    textcoords="offset points", xytext=xytext, ha=ha, va=va,
                    color=color2, fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Add transformer block labels on top
    ax3 = ax1.twiny()
    ax3.set_xlabel('Corresponding Transformer Block (1-based indexing)', color='tab:green', fontsize=14)
    ax3.set_xticks(valid_layers)
    ax3.set_xticklabels(valid_transformer_blocks)
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
    
    # plt.title('CAZO: Effect of Adapter Layer Position on Performance\n(Accuracy and ECE)', 
    #           fontsize=16, pad=30)
    
    # Save combined plot
    plot_path = os.path.join(output_dir, 'adapter_layer_accuracy_ece.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate plots for clarity
    # Plot 1: Accuracy vs Adapter Layer
    plt.figure(figsize=(12, 8))
    plt.plot(valid_layers, valid_accuracies, 'bo-', linewidth=3, markersize=10)
    plt.xlabel('Adapter Layer Position (0-based)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('CAZO: Accuracy vs Adapter Layer Position', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(valid_layers)
    
    # Add transformer block numbers as secondary x-axis
    ax_sec = plt.gca().secondary_xaxis('top')
    ax_sec.set_xlabel('Transformer Block (1-based)', fontsize=12, color='tab:green')
    ax_sec.set_xticks(valid_layers)
    ax_sec.set_xticklabels(valid_transformer_blocks)
    ax_sec.tick_params(axis='x', labelcolor='tab:green')
    
    for layer, acc in zip(valid_layers, valid_accuracies):
        plt.annotate(f'{acc:.2f}%', (layer, acc), 
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontweight='bold')
    
    plot_path2 = os.path.join(output_dir, 'adapter_layer_accuracy.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: ECE vs Adapter Layer
    plt.figure(figsize=(12, 8))
    plt.plot(valid_layers, valid_eces, 'rs-', linewidth=3, markersize=10)
    plt.xlabel('Adapter Layer Position (0-based)', fontsize=14)
    plt.ylabel('ECE (%)', fontsize=14)
    plt.title('CAZO: ECE vs Adapter Layer Position', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(valid_layers)
    
    # Add transformer block numbers as secondary x-axis
    ax_sec = plt.gca().secondary_xaxis('top')
    ax_sec.set_xlabel('Transformer Block (1-based)', fontsize=12, color='tab:green')
    ax_sec.set_xticks(valid_layers)
    ax_sec.set_xticklabels(valid_transformer_blocks)
    ax_sec.tick_params(axis='x', labelcolor='tab:green')
    
    for layer, ece in zip(valid_layers, valid_eces):
        plt.annotate(f'{ece:.2f}%', (layer, ece), 
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontweight='bold')
    
    plot_path3 = os.path.join(output_dir, 'adapter_layer_ece.png')
    plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to:")
    print(f"  {plot_path}")
    print(f"  {plot_path2}")
    print(f"  {plot_path3}")
    
    # Print statistics
    if valid_accuracies:
        best_acc_layer = valid_layers[np.argmax(valid_accuracies)]
        best_accuracy = max(valid_accuracies)
        worst_acc_layer = valid_layers[np.argmin(valid_accuracies)]
        worst_accuracy = min(valid_accuracies)
        
        best_ece_layer = valid_layers[np.argmin(valid_eces)]  # Lower ECE is better
        best_ece = min(valid_eces)
        worst_ece_layer = valid_layers[np.argmax(valid_eces)]
        worst_ece = max(valid_eces)
        
        print(f"\nAnalysis Results:")
        print(f"Best accuracy: layer {best_acc_layer} (transformer block {best_acc_layer + 1}) with {best_accuracy:.2f}%")
        print(f"Worst accuracy: layer {worst_acc_layer} (transformer block {worst_acc_layer + 1}) with {worst_accuracy:.2f}%")
        print(f"Accuracy gap: {best_accuracy - worst_accuracy:.2f}%")
        print(f"")
        print(f"Best ECE: layer {best_ece_layer} (transformer block {best_ece_layer + 1}) with {best_ece:.2f}%")
        print(f"Worst ECE: layer {worst_ece_layer} (transformer block {worst_ece_layer + 1}) with {worst_ece:.2f}%")
        print(f"ECE gap: {worst_ece - best_ece:.2f}%")
        
        # Check for optimal layer (balance between accuracy and ECE)
        # Normalize scores (0-1 range)
        norm_acc = [(acc - min(valid_accuracies)) / (max(valid_accuracies) - min(valid_accuracies)) for acc in valid_accuracies]
        norm_ece = [(max(valid_eces) - ece) / (max(valid_eces) - min(valid_eces)) for ece in valid_eces]  # Inverted for ECE
        combined_scores = [0.6 * acc + 0.4 * ece for acc, ece in zip(norm_acc, norm_ece)]  # Weight accuracy more
        
        best_combined_idx = np.argmax(combined_scores)
        best_combined_layer = valid_layers[best_combined_idx]
        best_combined_acc = valid_accuracies[best_combined_idx]
        best_combined_ece = valid_eces[best_combined_idx]
        
        print(f"")
        print(f"Best combined (60% acc + 40% ece): layer {best_combined_layer} (transformer block {best_combined_layer + 1}) with acc={best_combined_acc:.2f}%, ece={best_combined_ece:.2f}%")
        
        # Print detailed mapping
        print(f"\nDetailed Results:")
        print(f"{'Layer':>5} -> {'T-Block':>7} -> {'Accuracy':>8} -> {'ECE':>6} -> {'Combined Score':>14}")
        print("-" * 55)
        for i, layer in enumerate(valid_layers):
            acc = valid_accuracies[i]
            ece = valid_eces[i]
            combined = combined_scores[i]
            t_block = layer + 1
            print(f"{layer:5d} -> {t_block:7d} -> {acc:7.2f}% -> {ece:5.2f}% -> {combined:13.3f}")

def analyze_layer_trends(results):
    """Analyze performance trends across different layers"""
    valid_layers = [l for l in sorted(results.keys()) if results[l]['accuracy'] is not None and results[l]['ece'] is not None]
    
    if len(valid_layers) < 3:
        return
    
    valid_accuracies = [results[l]['accuracy'] for l in valid_layers]
    valid_eces = [results[l]['ece'] for l in valid_layers]
    
    print(f"\nLayer Position Analysis:")
    
    # Divide layers into categories
    early_layers = [l for l in valid_layers if l <= 3]  # Layers 0-3 (shallow)
    middle_layers = [l for l in valid_layers if 4 <= l <= 7]  # Layers 4-7 (middle)
    late_layers = [l for l in valid_layers if l >= 8]  # Layers 8-11 (deep)
    
    if early_layers:
        early_acc = np.mean([results[l]['accuracy'] for l in early_layers])
        print(f"Early layers (0-3) average accuracy: {early_acc:.2f}%")
    
    if middle_layers:
        middle_acc = np.mean([results[l]['accuracy'] for l in middle_layers])
        print(f"Middle layers (4-7) average accuracy: {middle_acc:.2f}%")
    
    if late_layers:
        late_acc = np.mean([results[l]['accuracy'] for l in late_layers])
        print(f"Late layers (8-11) average accuracy: {late_acc:.2f}%")
    
    # Calculate trend
    if len(valid_layers) >= 6:
        trend_corr = np.corrcoef(valid_layers, valid_accuracies)[0, 1]
        print(f"\nAccuracy trend correlation with layer depth: {trend_corr:.3f}")
        
        if trend_corr > 0.3:
            print("Trend: Performance improves in deeper layers")
        elif trend_corr < -0.3:
            print("Trend: Performance degrades in deeper layers")
        else:
            print("Trend: No clear layer depth preference")

if __name__ == "__main__":
    import sys
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 优先使用修复后的文件
    summary_file_fixed = os.path.join(base_dir, 'adapter_layer_sensitivity_summary_fixed.log')
    summary_file_original = os.path.join(base_dir, 'adapter_layer_sensitivity_summary.log')
    
    if os.path.exists(summary_file_fixed):
        print(f"Using fixed summary file: {summary_file_fixed}")
        results = parse_results(summary_file_fixed)
        plot_results(results, base_dir)
        analyze_layer_trends(results)
    elif os.path.exists(summary_file_original):
        print(f"Using original summary file: {summary_file_original}")
        results = parse_results(summary_file_original)
        plot_results(results, base_dir)
        analyze_layer_trends(results)
    else:
        print(f"Summary file not found. Searched:")
        print(f"  {summary_file_fixed}")
        print(f"  {summary_file_original}")
