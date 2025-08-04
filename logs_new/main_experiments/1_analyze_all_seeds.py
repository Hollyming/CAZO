#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析main_experiments中所有seed的实验结果
计算每种corruption数据集下的acc和ece的均值与标准差
以及综合15种corruption的acc与ece的均值和标准差
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
from collections import defaultdict

def parse_log_file(log_file_path):
    """
    解析单个日志文件，提取数据集性能指标
    
    Args:
        log_file_path: 日志文件路径
    
    Returns:
        list: 包含解析数据的字典列表
    """
    results = []
    
    # 正则表达式模式，匹配目标行
    pattern = r'Under shift type (\w+) After (\w+) Top-1 Accuracy: ([\d.]+) and Top-5 Accuracy: ([\d.]+) and ECE: ([\d.]+)'
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                match = re.search(pattern, line)
                if match:
                    dataset = match.group(1)        # 数据集名称
                    method = match.group(2)         # 方法名称
                    top1_acc = float(match.group(3))  # Top-1 Accuracy
                    top5_acc = float(match.group(4))  # Top-5 Accuracy  
                    ece = float(match.group(5))     # ECE
                    
                    results.append({
                        'dataset': dataset,
                        'method': method,
                        'top1_accuracy': top1_acc,
                        'top5_accuracy': top5_acc,
                        'ece': ece
                    })
    
    except FileNotFoundError:
        print(f"警告：找不到文件 {log_file_path}")
        return []
    except Exception as e:
        print(f"解析文件 {log_file_path} 时出错：{e}")
        return []
    
    return results

def find_log_files(method_name, base_dir="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO/outputs_new/main_experiments"):
    """
    查找指定方法的所有seed的日志文件
    
    Args:
        method_name: 方法名称 (如 'no_adapt', 'lame' 等)
        base_dir: 基础目录
    
    Returns:
        dict: seed -> log_file_path 的映射
    """
    log_files = {}
    method_dir = os.path.join(base_dir, method_name)
    
    if not os.path.exists(method_dir):
        print(f"方法目录不存在: {method_dir}")
        return log_files
    
    # 查找所有seed目录
    seed_pattern = os.path.join(method_dir, f"{method_name}_seed*_bs64")
    seed_dirs = glob.glob(seed_pattern)
    
    for seed_dir in seed_dirs:
        # 提取seed号
        seed_match = re.search(r'seed(\d+)', seed_dir)
        if seed_match:
            seed = seed_match.group(1)
            
            # 查找日志文件
            log_pattern = os.path.join(seed_dir, "*.txt")
            log_file_candidates = glob.glob(log_pattern)
            
            if log_file_candidates:
                log_files[seed] = log_file_candidates[0]  # 取第一个匹配的日志文件
            else:
                print(f"警告：在 {seed_dir} 中未找到日志文件")
    
    return log_files

def analyze_method_results(method_name, base_dir="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO/outputs_new/main_experiments"):
    """
    分析指定方法所有seed的结果
    
    Args:
        method_name: 方法名称
        base_dir: 基础目录
    
    Returns:
        tuple: (per_dataset_stats, overall_stats, all_results_df)
    """
    print(f"\n正在分析方法: {method_name}")
    print("="*60)
    
    # 查找所有日志文件
    log_files = find_log_files(method_name, base_dir)
    
    if not log_files:
        print(f"未找到方法 {method_name} 的日志文件")
        return None, None, None
    
    print(f"找到 {len(log_files)} 个seed的日志文件:")
    for seed, log_file in log_files.items():
        print(f"  Seed {seed}: {log_file}")
    
    # 收集所有结果
    all_results = []
    seed_overall_stats = []  # 存储每个seed的综合统计
    
    for seed, log_file in log_files.items():
        print(f"\n解析 Seed {seed}...")
        results = parse_log_file(log_file)
        
        if results:
            # 添加seed信息到每个结果
            for result in results:
                result['seed'] = seed
                all_results.append(result)
            
            # 计算该seed的综合统计（15个corruption的平均值）
            seed_df = pd.DataFrame(results)
            seed_mean_acc = seed_df['top1_accuracy'].mean()
            seed_mean_ece = seed_df['ece'].mean()
            
            # 添加该seed的综合指标到all_results
            overall_result = {
                'dataset': 'ImageNet-C_Overall',
                'method': results[0]['method'],  # 从第一个结果获取方法名
                'top1_accuracy': seed_mean_acc,
                'top5_accuracy': seed_df['top5_accuracy'].mean(),  # 也计算top5的平均
                'ece': seed_mean_ece,
                'seed': seed
            }
            all_results.append(overall_result)
            
            # 保存seed级别的统计用于计算overall
            seed_overall_stats.append({
                'seed': seed,
                'top1_accuracy': seed_mean_acc,
                'ece': seed_mean_ece
            })
            
            print(f"  找到 {len(results)} 个数据集的结果")
            print(f"  该seed的综合指标: Top-1 Acc = {seed_mean_acc:.1f}, ECE = {seed_mean_ece*100:.1f}")
        else:
            print(f"  警告：Seed {seed} 没有找到结果")
    
    if not all_results:
        print(f"方法 {method_name} 没有找到任何结果")
        return None, None, None
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 按数据集分组计算统计信息（排除overall行）
    per_dataset_stats = {}
    
    # 只对真正的corruption数据集计算跨seed统计
    corruption_data = df[df['dataset'] != 'ImageNet-C_Overall']
    datasets = corruption_data['dataset'].unique()
    
    for dataset in datasets:
        dataset_data = corruption_data[corruption_data['dataset'] == dataset]
        
        # 将ECE转换为百分比数值后再计算统计信息
        ece_percentage_values = dataset_data['ece'] * 100
        
        per_dataset_stats[dataset] = {
            'count': len(dataset_data),
            'top1_accuracy': {
                'mean': dataset_data['top1_accuracy'].mean(),
                'std': dataset_data['top1_accuracy'].std(),
                'values': dataset_data['top1_accuracy'].tolist()
            },
            'ece': {
                'mean': ece_percentage_values.mean(),  # 基于百分比数值计算均值
                'std': ece_percentage_values.std(),    # 基于百分比数值计算标准差
                'values': ece_percentage_values.tolist()  # 百分比数值列表
            }
        }
    
    # 计算overall统计信息（基于seed级别的综合指标）
    if seed_overall_stats:
        seed_overall_df = pd.DataFrame(seed_overall_stats)
        
        # 将ECE转换为百分比数值后再计算统计信息
        seed_ece_percentage = seed_overall_df['ece'] * 100
        
        overall_stats = {
            'total_experiments': len(corruption_data),  # corruption实验总数
            'datasets_count': len(datasets),
            'seeds_count': len(seed_overall_stats),
            'top1_accuracy': {
                'mean': seed_overall_df['top1_accuracy'].mean(),  # 基于seed综合指标的均值
                'std': seed_overall_df['top1_accuracy'].std(),   # 基于seed综合指标的标准差
                'min': seed_overall_df['top1_accuracy'].min(),
                'max': seed_overall_df['top1_accuracy'].max(),
                'seed_values': seed_overall_df['top1_accuracy'].tolist()  # 各seed的综合值
            },
            'ece': {
                'mean': seed_ece_percentage.mean(),       # 基于百分比数值计算均值
                'std': seed_ece_percentage.std(),         # 基于百分比数值计算标准差
                'min': seed_ece_percentage.min(),
                'max': seed_ece_percentage.max(),
                'seed_values': seed_ece_percentage.tolist()  # 各seed的百分比数值
            }
        }
    else:
        overall_stats = {}
    
    return per_dataset_stats, overall_stats, df

def print_analysis_results(method_name, per_dataset_stats, overall_stats, df):
    """
    打印分析结果
    """
    print(f"\n{'='*80}")
    print(f"方法 {method_name.upper()} 的分析结果")
    print('='*80)
    
    # 打印总体统计
    print(f"\n总体统计信息:")
    print(f"  总实验数: {overall_stats['total_experiments']}")
    print(f"  数据集数: {overall_stats['datasets_count']}")
    print(f"  Seed数: {overall_stats['seeds_count']}")
    
    print(f"\nImageNet-C综合性能 (基于{overall_stats['seeds_count']}个seed的综合指标):")
    print(f"  Top-1 Accuracy - 均值: {overall_stats['top1_accuracy']['mean']:.1f} ± {overall_stats['top1_accuracy']['std']:.1f}")
    print(f"  ECE - 均值: {overall_stats['ece']['mean']:.1f} ± {overall_stats['ece']['std']:.1f}")
    print(f"  各seed的Top-1综合值: {[f'{v:.1f}' for v in overall_stats['top1_accuracy']['seed_values']]}")
    print(f"  各seed的ECE综合值: {[f'{v:.1f}' for v in overall_stats['ece']['seed_values']]}")
    
    # 打印每个数据集的统计
    print(f"\n各corruption数据集统计 (基于{overall_stats['seeds_count']}个seed):")
    print(f"{'数据集':<20} {'Top-1 Acc (均值±std)':<25} {'ECE (均值±std)':<20} {'样本数':<8}")
    print("-" * 80)
    
    for dataset, stats in per_dataset_stats.items():
        top1_mean = stats['top1_accuracy']['mean']
        top1_std = stats['top1_accuracy']['std']
        ece_mean = stats['ece']['mean']  # 已经是百分比数值
        ece_std = stats['ece']['std']    # 已经是百分比数值
        count = stats['count']
        
        print(f"{dataset:<20} {top1_mean:.1f}±{top1_std:.1f}{'':>14} {ece_mean:.1f}±{ece_std:.1f}{'':>8} {count:<8}")
    
    # 创建详细的DataFrame表格
    print(f"\n详细结果表格:")
    summary_data = []
    for dataset, stats in per_dataset_stats.items():
        summary_data.append({
            'dataset': dataset,
            'top1_acc_mean': stats['top1_accuracy']['mean'],
            'top1_acc_std': stats['top1_accuracy']['std'],
            'ece_mean': stats['ece']['mean'],  # 已经是百分比数值
            'ece_std': stats['ece']['std'],    # 已经是百分比数值
            'count': stats['count']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, float_format='%.1f'))

def save_results_to_csv(method_name, per_dataset_stats, overall_stats, df):
    """
    保存结果到CSV文件
    """
    # 确保results_csv目录存在
    # os.makedirs("results_csv", exist_ok=True)
    
    # 保存原始数据（包含每个corruption结果和每个seed的综合指标）
    raw_output_file = f"results_csv/{method_name}_raw_results.csv"
    df.to_csv(raw_output_file, index=False)
    
    # 保存统计摘要
    summary_data = []
    
    # 添加各个corruption数据集的统计
    for dataset, stats in per_dataset_stats.items():
        summary_data.append({
            'method': method_name,
            'dataset': dataset,
            'top1_acc_mean': stats['top1_accuracy']['mean'],
            'top1_acc_std': stats['top1_accuracy']['std'],
            'ece_mean': stats['ece']['mean'],  # 已经是百分比数值
            'ece_std': stats['ece']['std'],    # 已经是百分比数值
            'count': stats['count'],
            'note': 'per_corruption_across_seeds'
        })
    
    # 添加总体统计（基于seed综合指标的统计）
    summary_data.append({
        'method': method_name,
        'dataset': 'ImageNet-C_Overall',
        'top1_acc_mean': overall_stats['top1_accuracy']['mean'],
        'top1_acc_std': overall_stats['top1_accuracy']['std'],
        'ece_mean': overall_stats['ece']['mean'],  # 已经是百分比数值
        'ece_std': overall_stats['ece']['std'],    # 已经是百分比数值
        'count': overall_stats['seeds_count'],
        'note': 'overall_across_seeds'
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_output_file = f"results_csv/{method_name}_summary_stats.csv"
    summary_df.to_csv(summary_output_file, index=False)
    
    print(f"\n结果已保存:")
    print(f"  原始数据: {raw_output_file}")
    print(f"    - 包含每个corruption的结果")
    print(f"    - 包含每个seed的ImageNet-C综合指标")
    print(f"  统计摘要: {summary_output_file}")
    print(f"    - 各corruption数据集跨seed的统计")
    print(f"    - ImageNet-C综合性能跨seed的统计（用于error bar）")

def main():
    """
    主函数：分析指定方法的所有seed结果
    """
    # 可以修改这里来分析不同的方法
    methods_to_analyze = ['no_adapt', 'lame', 'foa', 'tent', 't3a','sar','cotta','cazo','zo_base']  # 添加您想分析的方法
    # methods_to_analyze = ['no_adapt', 'lame', 'foa', 'tent', 't3a','sar','cotta','cazo'] 
    
    for method in methods_to_analyze:
        per_dataset_stats, overall_stats, df = analyze_method_results(method)
        
        if per_dataset_stats is not None:
            print_analysis_results(method, per_dataset_stats, overall_stats, df)
            save_results_to_csv(method, per_dataset_stats, overall_stats, df)
        else:
            print(f"跳过方法 {method}：无结果数据")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 