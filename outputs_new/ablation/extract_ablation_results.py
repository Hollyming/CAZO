#!/usr/bin/env python3
"""
Extract ablation study results from log files.
Reads mean acc of corruption and mean ece of corruption from the last occurrence in each log file.
"""

import os
import re
from pathlib import Path


def extract_last_metrics_from_log(log_file_path):
    """
    Extract the last occurrence of mean acc and mean ece from a log file.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        tuple: (mean_acc, mean_ece) or (None, None) if not found
    """
    mean_acc = None
    mean_ece = None
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Look for mean acc of corruption
                if 'mean acc of corruption:' in line:
                    match = re.search(r'mean acc of corruption:\s*([\d.]+)', line)
                    if match:
                        mean_acc = float(match.group(1))
                
                # Look for mean ece of corruption
                if 'mean ece of corruption:' in line:
                    match = re.search(r'mean ece of corruption:\s*([\d.]+)', line)
                    if match:
                        mean_ece = float(match.group(1))
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return None, None
    
    return mean_acc, mean_ece


def find_log_file(layer_dir):
    """
    Find the log file in the layer directory.
    
    Args:
        layer_dir: Path to the layer directory
        
    Returns:
        Path to the log file or None if not found
    """
    # Navigate through the expected structure: layer_X/cazo/cazo_seed*/log.txt
    cazo_dir = layer_dir / 'cazo'
    if not cazo_dir.exists():
        return None
    
    # Find the first subdirectory (should be cazo_seed42_bs64_deit_base_layerX or similar)
    subdirs = [d for d in cazo_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    
    # Find .txt files in the subdirectory
    log_dir = subdirs[0]
    log_files = list(log_dir.glob('*.txt'))
    
    if log_files:
        return log_files[0]  # Return the first log file found
    
    return None


def process_ablation_results(base_dir, model_name, num_layers=12):
    """
    Process ablation results for a specific model.
    
    Args:
        base_dir: Base directory containing ablation results
        model_name: Name of the model (e.g., 'deit_base', 'swin_tiny')
        num_layers: Number of layers to process (default: 12)
    """
    model_dir = Path(base_dir) / model_name
    
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Ablation Results for {model_name}")
    print(f"{'='*80}\n")
    
    results = []
    
    # Process each layer from 0 to num_layers-1
    for layer_idx in range(num_layers):
        layer_dir = model_dir / f'layer_{layer_idx}'
        
        if not layer_dir.exists():
            print(f"Warning: Layer directory not found: {layer_dir}")
            continue
        
        log_file = find_log_file(layer_dir)
        
        if log_file is None:
            print(f"Warning: No log file found for layer {layer_idx}")
            continue
        
        mean_acc, mean_ece = extract_last_metrics_from_log(log_file)
        
        if mean_acc is not None and mean_ece is not None:
            transformer_block = layer_idx + 1
            results.append((layer_idx, transformer_block, mean_acc, mean_ece))
            print(f"adapter_layer={layer_idx} (transformer_block={transformer_block}): "
                  f"accuracy={mean_acc}%, ece={mean_ece}%")
        else:
            print(f"Warning: Could not extract metrics for layer {layer_idx}")
    
    return results


def main():
    """Main function to process all models."""
    # Base directory containing ablation results
    base_dir = Path(__file__).parent / 'outputs_new' / 'ablation'
    
    # Check if base directory exists
    if not base_dir.exists():
        print(f"Error: Ablation directory not found: {base_dir}")
        return
    
    # List all model directories
    model_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("No model directories found in ablation folder")
        return
    
    # Process each model
    all_results = {}
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        results = process_ablation_results(base_dir, model_name)
        all_results[model_name] = results
    
    # Optional: Save results to a file
    output_file = Path(__file__).parent / 'ablation_results_summary.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for model_name, results in all_results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Ablation Results for {model_name}\n")
            f.write(f"{'='*80}\n\n")
            
            if results:
                for layer_idx, transformer_block, mean_acc, mean_ece in results:
                    f.write(f"adapter_layer={layer_idx} (transformer_block={transformer_block}): "
                           f"accuracy={mean_acc}%, ece={mean_ece}%\n")
            else:
                f.write("No results found\n")
    
    print(f"\n\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
