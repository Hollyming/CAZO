#!/usr/bin/env python3
"""
Multi-Model Adapter Layer Sensitivity Analysis Plot
This script plots accuracy and ECE for multiple models (VitBase, DeiT, Swin) in a single figure.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_multi_model_results(summary_file):
    """Parse results from summary file containing multiple models"""
    all_results = {}
    current_model = None
    
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Check if we're entering a new model section
        if 'Ablation Results for' in line:
            # Extract model name
            model_match = re.search(r'Ablation Results for (\w+)', line)
            if model_match:
                current_model = model_match.group(1)
                all_results[current_model] = {}
        
        # Parse adapter layer results
        elif line.startswith('adapter_layer=') and current_model:
            # Extract adapter_layer, accuracy, and ECE
            layer_match = re.search(r'adapter_layer=(\d+)', line)
            acc_match = re.search(r'accuracy=([0-9]+\.?[0-9]*)', line)
            ece_match = re.search(r'ece=([0-9]+\.?[0-9]*)', line)
            
            if layer_match and acc_match and ece_match:
                layer = int(layer_match.group(1))
                accuracy = float(acc_match.group(1))
                ece = float(ece_match.group(1))
                all_results[current_model][layer] = {'accuracy': accuracy, 'ece': ece}
    
    return all_results

def plot_multi_model_results(all_results, output_dir):
    """Generate combined plot for multiple models"""
    if not all_results:
        print("No results found to plot")
        return
    
    # Define model configurations (marker, color for accuracy, color for ECE, display name)
    model_configs = {
        'vitbase': {
            'marker': 'o',
            'acc_color': '#1f77b4',  # Deep blue
            'ece_color': '#d62728',  # Deep red
            'display_name': 'ViT-Base',
            'linestyle': '-'
        },
        'deit_base': {
            'marker': 's',  # square
            'acc_color': '#3498db',  # Medium blue
            'ece_color': '#e74c3c',  # Medium red
            'display_name': 'DeiT-Base',
            'linestyle': '--'
        },
        'swin_tiny': {
            'marker': '^',  # triangle
            'acc_color': '#5dade2',  # Light blue
            'ece_color': '#f1948a',  # Light red
            'display_name': 'Swin-Tiny',
            'linestyle': '-.'
        }
    }
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Setup accuracy axis (left y-axis)
    ax1.set_xlabel('Adapter Layer Position', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Create second y-axis for ECE (right y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('ECE (%)', fontsize=16, fontweight='bold', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=12)
    
    # Store all lines for legend
    lines_acc = []
    lines_ece = []
    labels_acc = []
    labels_ece = []
    
    # Collect all layers for x-axis
    all_layers = set()
    for model_name, results in all_results.items():
        all_layers.update(results.keys())
    all_layers = sorted(list(all_layers))
    
    # Plot each model
    for model_name, results in sorted(all_results.items()):
        if model_name not in model_configs:
            print(f"Warning: No config for model {model_name}, skipping...")
            continue
        
        config = model_configs[model_name]
        
        # Extract data for this model
        layers = sorted(results.keys())
        accuracies = [results[l]['accuracy'] for l in layers]
        eces = [results[l]['ece'] for l in layers]
        
        # Plot accuracy
        line_acc = ax1.plot(layers, accuracies, 
                           marker=config['marker'], 
                           color=config['acc_color'],
                           linestyle=config['linestyle'],
                           linewidth=2.5, 
                           markersize=9,
                           label=f"{config['display_name']} (Acc)",
                           alpha=0.85)[0]
        lines_acc.append(line_acc)
        labels_acc.append(f"{config['display_name']} (Acc)")
        
        # Plot ECE
        line_ece = ax2.plot(layers, eces,
                           marker=config['marker'],
                           color=config['ece_color'],
                           linestyle=config['linestyle'],
                           linewidth=2.5,
                           markersize=9,
                           label=f"{config['display_name']} (ECE)",
                           alpha=0.85)[0]
        lines_ece.append(line_ece)
        labels_ece.append(f"{config['display_name']} (ECE)")
    
    # Set x-axis ticks (convert to 1-based indexing)
    ax1.set_xticks(all_layers)
    layer_labels_1based = [l + 1 for l in all_layers]
    ax1.set_xticklabels(layer_labels_1based)
    
    # Set y-axis limit for accuracy (start from 30%)
    ax1.set_ylim(bottom=30)
    
    # Add transformer block labels on top
    ax3 = ax1.twiny()
    ax3.set_xlabel('Corresponding Transformer Block', 
                   fontsize=14, fontweight='bold', color='tab:green')
    ax3.set_xticks(all_layers)
    transformer_blocks = [l + 1 for l in all_layers]
    ax3.set_xticklabels(transformer_blocks)
    ax3.tick_params(axis='x', labelcolor='tab:green', labelsize=11)
    
    # Create legend with two columns (accuracy and ECE)
    # Combine all lines and labels
    all_lines = lines_acc + lines_ece
    all_labels = labels_acc + labels_ece
    
    # Create a single legend with proper organization
    legend = ax1.legend(all_lines, all_labels,
                       loc='center left',
                       fontsize=11,
                       ncol=2,
                       frameon=True,
                       fancybox=True,
                       shadow=True,
                       facecolor='white',
                       edgecolor='gray',
                       borderpad=1.0,
                       columnspacing=1.5,
                       handletextpad=0.5)
    legend.set_zorder(1000)
    
    # Set title
    # plt.title('CAZO: Multi-Model Adapter Layer Sensitivity Analysis\n(Accuracy and ECE vs. Adapter Position)', 
    #           fontsize=18, fontweight='bold', pad=35)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'multi_model_adapter_layer_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()
    
    # Print statistics for each model
    print("\n" + "="*80)
    print("Multi-Model Analysis Summary")
    print("="*80)
    
    for model_name, results in sorted(all_results.items()):
        if model_name not in model_configs:
            continue
            
        config = model_configs[model_name]
        layers = sorted(results.keys())
        accuracies = [results[l]['accuracy'] for l in layers]
        eces = [results[l]['ece'] for l in layers]
        
        print(f"\n{config['display_name']}:")
        print(f"  Layers analyzed: {len(layers)} (layer {min(layers)} to {max(layers)})")
        print(f"  Accuracy range: {min(accuracies):.2f}% - {max(accuracies):.2f}% (Δ={max(accuracies)-min(accuracies):.2f}%)")
        print(f"  ECE range: {min(eces):.2f}% - {max(eces):.2f}% (Δ={max(eces)-min(eces):.2f}%)")
        
        best_acc_idx = np.argmax(accuracies)
        best_ece_idx = np.argmin(eces)  # Lower ECE is better
        
        print(f"  Best accuracy: layer {layers[best_acc_idx]} (block {layers[best_acc_idx]+1}) = {accuracies[best_acc_idx]:.2f}%")
        print(f"  Best ECE: layer {layers[best_ece_idx]} (block {layers[best_ece_idx]+1}) = {eces[best_ece_idx]:.2f}%")
        
        # Calculate optimal layer (balanced score)
        norm_acc = [(acc - min(accuracies)) / (max(accuracies) - min(accuracies) + 1e-8) for acc in accuracies]
        norm_ece = [(max(eces) - ece) / (max(eces) - min(eces) + 1e-8) for ece in eces]
        combined_scores = [0.6 * acc + 0.4 * ece for acc, ece in zip(norm_acc, norm_ece)]
        
        best_combined_idx = np.argmax(combined_scores)
        print(f"  Optimal layer (60% acc + 40% ece): layer {layers[best_combined_idx]} "
              f"(block {layers[best_combined_idx]+1}) = "
              f"acc {accuracies[best_combined_idx]:.2f}%, ece {eces[best_combined_idx]:.2f}%")
    
    # Compare models
    print("\n" + "="*80)
    print("Cross-Model Comparison")
    print("="*80)
    
    # Find best performing model overall
    best_acc_overall = -1
    best_acc_model = None
    best_acc_layer = None
    
    best_ece_overall = float('inf')
    best_ece_model = None
    best_ece_layer = None
    
    for model_name, results in all_results.items():
        if model_name not in model_configs:
            continue
        
        for layer, data in results.items():
            if data['accuracy'] > best_acc_overall:
                best_acc_overall = data['accuracy']
                best_acc_model = model_name
                best_acc_layer = layer
            
            if data['ece'] < best_ece_overall:
                best_ece_overall = data['ece']
                best_ece_model = model_name
                best_ece_layer = layer
    
    if best_acc_model and best_ece_model:
        print(f"\nBest accuracy overall: {model_configs[best_acc_model]['display_name']} "
              f"at layer {best_acc_layer} (block {best_acc_layer+1}) = {best_acc_overall:.2f}%")
        print(f"Best ECE overall: {model_configs[best_ece_model]['display_name']} "
              f"at layer {best_ece_layer} (block {best_ece_layer+1}) = {best_ece_overall:.2f}%")

def create_separate_subplots(all_results, output_dir):
    """Create a version with separate subplots for accuracy and ECE"""
    
    model_configs = {
        'vitbase': {
            'marker': 'o',
            'color': '#1f77b4',
            'display_name': 'ViT-Base',
            'linestyle': '-'
        },
        'deit_base': {
            'marker': 's',
            'color': '#ff7f0e',
            'display_name': 'DeiT-Base',
            'linestyle': '--'
        },
        'swin_tiny': {
            'marker': '^',
            'color': '#2ca02c',
            'display_name': 'Swin-Tiny',
            'linestyle': '-.'
        }
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Collect all layers for x-axis
    all_layers = set()
    for model_name, results in all_results.items():
        all_layers.update(results.keys())
    all_layers = sorted(list(all_layers))
    
    # Plot 1: Accuracy
    for model_name, results in sorted(all_results.items()):
        if model_name not in model_configs:
            continue
        
        config = model_configs[model_name]
        layers = sorted(results.keys())
        accuracies = [results[l]['accuracy'] for l in layers]
        
        ax1.plot(layers, accuracies,
                marker=config['marker'],
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=2.5,
                markersize=9,
                label=config['display_name'],
                alpha=0.85)
    
    ax1.set_xlabel('Adapter Layer Position', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Accuracy vs. Adapter Layer Position', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax1.set_xticks(all_layers)
    ax1.set_ylim(bottom=30)  # Set y-axis to start from 30%
    
    # Plot 2: ECE
    for model_name, results in sorted(all_results.items()):
        if model_name not in model_configs:
            continue
        
        config = model_configs[model_name]
        layers = sorted(results.keys())
        eces = [results[l]['ece'] for l in layers]
        
        ax2.plot(layers, eces,
                marker=config['marker'],
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=2.5,
                markersize=9,
                label=config['display_name'],
                alpha=0.85)
    
    ax2.set_xlabel('Adapter Layer Position', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ECE (%)', fontsize=14, fontweight='bold')
    ax2.set_title('ECE vs. Adapter Layer Position', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax2.set_xticks(all_layers)
    
    # plt.suptitle('CAZO: Multi-Model Adapter Layer Sensitivity Analysis', 
    #              fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'multi_model_adapter_layer_subplots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Subplot version saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Set the path to the summary file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_file = os.path.join(base_dir, 'ablation_results_summary.txt')
    
    if not os.path.exists(summary_file):
        print(f"Error: Summary file not found at {summary_file}")
        exit(1)
    
    print(f"Reading results from: {summary_file}")
    
    # Parse results for all models
    all_results = parse_multi_model_results(summary_file)
    
    if not all_results:
        print("No results found in summary file")
        exit(1)
    
    print(f"\nFound {len(all_results)} model(s): {', '.join(all_results.keys())}")
    
    # Generate plots
    plot_multi_model_results(all_results, base_dir)
    create_separate_subplots(all_results, base_dir)
    
    print("\n✓ Analysis complete!")
