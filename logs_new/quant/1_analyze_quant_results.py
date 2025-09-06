#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析quant量化实验的结果
提取8bit和6bit量化模型在各种corruption数据集下的acc和ece结果
由于量化实验只有一组数据，不需要计算均值和方差
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
                    ece = float(match.group(5))     # ECE (保持原始小数格式)
                    
                    results.append({
                        'dataset': dataset,
                        'method': method,
                        'top1_accuracy': top1_acc,
                        'top5_accuracy': top5_acc,
                        'ece': ece
                    })
    
    except FileNotFoundError:
        print(f"文件未找到: {log_file_path}")
    except Exception as e:
        print(f"解析文件 {log_file_path} 时出错: {e}")
    
    return results

def analyze_quant_experiments():
    """
    分析量化实验结果
    """
    print("开始分析量化实验结果...")
    
    # 基础路径
    base_output_dir = "outputs_new/quant"
    
    # 量化类型 (只处理bs64)
    # quant_types = ["quant8_bs64", "quant6_bs64"]
    quant_types = ["quant8_bs32"]
    
    # 存储所有结果
    all_results = []
    
    for quant_type in quant_types:
        print(f"\n处理 {quant_type} 实验...")
        
        quant_dir = Path(base_output_dir) / quant_type
        if not quant_dir.exists():
            print(f"目录不存在: {quant_dir}")
            continue
        
        # 查找所有算法目录
        method_dirs = [d for d in quant_dir.iterdir() if d.is_dir()]
        
        for method_dir in method_dirs:
            method_name = method_dir.name
            print(f"  处理方法: {method_name}")
            
            # 查找具体实验目录
            exp_dirs = [d for d in method_dir.iterdir() if d.is_dir()]
            
            for exp_dir in exp_dirs:
                # 查找日志文件
                log_files = list(exp_dir.glob("*.txt"))
                
                for log_file in log_files:
                    print(f"    解析日志: {log_file.name}")
                    
                    # 解析日志文件
                    results = parse_log_file(log_file)
                    
                    # 添加量化类型信息
                    for result in results:
                        result['quant_type'] = quant_type
                        result['exp_name'] = exp_dir.name
                    
                    all_results.extend(results)
    
    if not all_results:
        print("未找到任何实验结果!")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 创建输出目录
    results_dir = Path("results_csv/quant_bs32")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存原始结果
    print("\n保存原始结果...")
    raw_results_file = results_dir / "raw_results.csv"
    df.to_csv(raw_results_file, index=False, encoding='utf-8-sig')
    print(f"原始结果已保存到: {raw_results_file}")
    
    # 生成各方法的结果文件
    print("\n生成各方法的结果文件...")
    generate_method_results(df, results_dir)
    
    print(f"\n分析完成！结果保存在: {results_dir}")

def generate_method_results(df, results_dir):
    """
    为每个方法生成结果文件
    
    Args:
        df: 原始结果DataFrame
        results_dir: 结果保存目录
    """
    
    # 获取所有方法
    methods = df['method'].unique()
    
    for method in methods:
        method_df = df[df['method'] == method]
        
        # 为每个量化类型生成结果
        method_results = []
        
        for quant_type in method_df['quant_type'].unique():
            quant_df = method_df[method_df['quant_type'] == quant_type]
            
            # 按数据集分组
            dataset_results = {}
            for dataset in quant_df['dataset'].unique():
                dataset_df = quant_df[quant_df['dataset'] == dataset]
                if len(dataset_df) > 0:
                    # 只有一组数据，直接取值
                    row = dataset_df.iloc[0]
                    dataset_results[dataset] = {
                        'top1_accuracy': row['top1_accuracy'],
                        'ece': row['ece']
                    }
            
            # 计算ImageNet-C综合指标（15个corruption的平均值）
            if dataset_results:
                avg_acc = np.mean([v['top1_accuracy'] for v in dataset_results.values()])
                avg_ece = np.mean([v['ece'] for v in dataset_results.values()])
                
                # 添加各个corruption的结果
                for dataset, metrics in dataset_results.items():
                    method_results.append({
                        'method': method,
                        'quant_type': quant_type,
                        'dataset': dataset,
                        'top1_accuracy': metrics['top1_accuracy'],
                        'ece': metrics['ece']
                    })
                
                # 添加综合结果
                method_results.append({
                    'method': method,
                    'quant_type': quant_type,
                    'dataset': 'ImageNet-C_Overall',
                    'top1_accuracy': avg_acc,
                    'ece': avg_ece
                })
        
        # 保存方法结果
        if method_results:
            method_results_df = pd.DataFrame(method_results)
            method_file = results_dir / f"{method}_results.csv"
            method_results_df.to_csv(method_file, index=False, encoding='utf-8-sig')
            print(f"  {method} 结果已保存到: {method_file}")

def print_statistics(df):
    """
    打印统计信息
    
    Args:
        df: 结果DataFrame
    """
    print("\n=== 实验统计信息 ===")
    
    quant_types = df['quant_type'].unique()
    methods = df['method'].unique()
    datasets = df['dataset'].unique()
    
    print(f"量化类型数量: {len(quant_types)}")
    print(f"算法数量: {len(methods)}")
    print(f"数据集数量: {len(datasets)}")
    print(f"总记录数: {len(df)}")
    
    print(f"\n量化类型: {list(quant_types)}")
    print(f"算法: {list(methods)}")
    print(f"数据集: {list(datasets)}")

if __name__ == "__main__":
    analyze_quant_experiments()