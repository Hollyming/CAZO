#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取outputs_new/other_datasets中所有算法在三种数据集(rendition, v2, sketch)的acc1和ece结果
计算每种算法在三种数据集的均值
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_method_name_from_dir(dir_name):
    """
    从目录名提取方法名称
    """
    method_mapping = {
        'cotta_bs64_lr0.01_other_datasets': 'cotta',
        'lame_bs64_other_datasets': 'lame',
        'no_adapt_bs64_other_datasets': 'no_adapt',
        'sar_bs64_other_datasets': 'sar',
        't3a_bs64_other_datasets': 't3a',
        'tent_bs64_other_datasets': 'tent',
        'foa_bs64_other_datasets': 'foa',
        'zo_base_bs64_lr0.01_pertub20_adapter_layer3_epsilon0.1_other_datasets': 'zo_base',
        'cazo_bs64_lr0.01_pertub20_adapter_layer3_reduction_factor384_parallel_epsilon0.1_nu0.8_other_datasets': 'cazo',
        'cozo_bs64_lr0.01_pertub20_adapter_layer3_reduction_factor384_parallel_optsgd0.9_other_datasets': 'cozo'
    }
    return method_mapping.get(dir_name, dir_name)

def get_display_name(method_name):
    """
    获取方法的显示名称
    """
    display_mapping = {
        'no_adapt': 'NoAdapt',
        'lame': 'LAME',
        't3a': 'T3A',
        'tent': 'TENT',
        'cotta': 'CoTTA',
        'sar': 'SAR',
        'foa': 'FOA',
        'zo_base': 'ZO',
        'cazo': 'CAZO',
        'cozo': 'COZO'
    }
    return display_mapping.get(method_name, method_name.upper())

def extract_results_from_log(log_file_path):
    """
    从日志文件中提取结果
    
    返回格式: {dataset: {'acc1': value, 'ece': value}}
    """
    results = {}
    
    # 正则表达式匹配结果行
    pattern = r'Under shift type (\w+) After (\w+) Top-1 Accuracy: ([\d.]+) and Top-5 Accuracy: ([\d.]+) and ECE: ([\d.]+)'
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    dataset = match.group(1)  # rendition, v2, sketch
                    method = match.group(2)   # 方法名称
                    acc1 = float(match.group(3))  # Top-1 Accuracy
                    ece = float(match.group(5))   # ECE
                    
                    results[dataset] = {
                        'acc1': acc1,
                        'ece': ece,
                        'method': method
                    }
    
    except FileNotFoundError:
        print(f"警告：找不到文件 {log_file_path}")
    except Exception as e:
        print(f"解析文件 {log_file_path} 时出错：{e}")
    
    return results

def collect_all_results(base_dir="./"):
    """
    收集所有方法的结果
    """
    all_results = {}
    
    # 获取所有方法目录
    if not os.path.exists(base_dir):
        print(f"基础目录不存在: {base_dir}")
        return all_results
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        # 解析方法名
        method_name = parse_method_name_from_dir(item)
        
        # 查找日志文件
        log_files = glob.glob(os.path.join(item_path, "*.txt"))
        
        if not log_files:
            print(f"警告：在 {item_path} 中未找到日志文件")
            continue
        
        log_file = log_files[0]  # 取第一个日志文件
        print(f"正在处理 {method_name}: {log_file}")
        
        # 提取结果
        results = extract_results_from_log(log_file)
        
        if results:
            all_results[method_name] = results
            print(f"  找到 {len(results)} 个数据集的结果")
            
            # 检查是否有异常值
            for dataset, data in results.items():
                if data['acc1'] < 1.0:  # 检查是否是百分比格式
                    print(f"  注意：{dataset} 的 acc1={data['acc1']:.6f} 可能是异常值")
        else:
            print(f"  未找到有效结果")
    
    return all_results

def create_summary_table(all_results):
    """
    创建汇总表格
    """
    datasets = ['rendition', 'v2', 'sketch']
    methods = list(all_results.keys())
    
    # 创建详细结果表
    detailed_data = []
    
    for method in methods:
        method_results = all_results[method]
        
        for dataset in datasets:
            if dataset in method_results:
                detailed_data.append({
                    'Method': get_display_name(method),
                    'Dataset': dataset,
                    'Acc1': method_results[dataset]['acc1'],
                    'ECE': method_results[dataset]['ece']
                })
            else:
                detailed_data.append({
                    'Method': get_display_name(method),
                    'Dataset': dataset,
                    'Acc1': np.nan,
                    'ECE': np.nan
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # 创建汇总统计表
    summary_data = []
    
    for method in methods:
        method_results = all_results[method]
        
        # 收集该方法在所有数据集的结果
        acc1_values = []
        ece_values = []
        
        for dataset in datasets:
            if dataset in method_results:
                acc1 = method_results[dataset]['acc1']
                ece = method_results[dataset]['ece']
                
                # 处理可能的异常值
                if acc1 > 1.0:  # 正常的百分比值
                    acc1_values.append(acc1)
                    ece_values.append(ece)
                else:
                    print(f"跳过异常值：{method} 在 {dataset} 的 acc1={acc1}")
        
        if acc1_values:  # 如果有有效数据
            summary_data.append({
                'Method': get_display_name(method),
                'Avg_Acc1': np.mean(acc1_values),
                'Std_Acc1': np.std(acc1_values, ddof=1) if len(acc1_values) > 1 else 0,
                'Avg_ECE': np.mean(ece_values),
                'Std_ECE': np.std(ece_values, ddof=1) if len(ece_values) > 1 else 0,
                'N_Datasets': len(acc1_values)
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    return detailed_df, summary_df

def create_pivot_table(detailed_df):
    """
    创建数据透视表，便于对比
    """
    # Acc1 透视表
    acc1_pivot = detailed_df.pivot(index='Method', columns='Dataset', values='Acc1')
    
    # ECE 透视表
    ece_pivot = detailed_df.pivot(index='Method', columns='Dataset', values='ECE')
    
    return acc1_pivot, ece_pivot

def print_results(detailed_df, summary_df, acc1_pivot, ece_pivot):
    """
    打印结果
    """
    print("\n" + "="*80)
    print("其他数据集(Rendition, V2, Sketch)实验结果分析")
    print("="*80)
    
    print(f"\n1. 详细结果表格:")
    print(detailed_df.to_string(index=False, float_format='%.3f'))
    
    print(f"\n2. Top-1 Accuracy 对比表:")
    print(acc1_pivot.to_string(float_format='%.2f'))
    
    print(f"\n3. ECE 对比表:")
    print(ece_pivot.to_string(float_format='%.3f'))
    
    print(f"\n4. 各方法在三个数据集的平均性能:")
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    # 找出最佳性能
    if not summary_df.empty:
        best_acc_idx = summary_df['Avg_Acc1'].idxmax()
        best_ece_idx = summary_df['Avg_ECE'].idxmin()
        
        print(f"\n5. 性能总结:")
        print(f"  最佳平均准确率: {summary_df.loc[best_acc_idx, 'Method']} "
              f"({summary_df.loc[best_acc_idx, 'Avg_Acc1']:.2f}±{summary_df.loc[best_acc_idx, 'Std_Acc1']:.2f})")
        print(f"  最佳平均校准: {summary_df.loc[best_ece_idx, 'Method']} "
              f"({summary_df.loc[best_ece_idx, 'Avg_ECE']:.3f}±{summary_df.loc[best_ece_idx, 'Std_ECE']:.3f})")

def save_results(detailed_df, summary_df, acc1_pivot, ece_pivot, output_dir="other_datasets_results"):
    """
    保存结果到文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    detailed_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    
    # 保存汇总结果
    summary_df.to_csv(f"{output_dir}/summary_results.csv", index=False)
    
    # 保存透视表
    acc1_pivot.to_csv(f"{output_dir}/acc1_comparison.csv")
    ece_pivot.to_csv(f"{output_dir}/ece_comparison.csv")
    
    print(f"\n结果已保存到 {output_dir}/ 目录:")
    print(f"  - detailed_results.csv: 详细结果")
    print(f"  - summary_results.csv: 汇总统计")
    print(f"  - acc1_comparison.csv: Acc1对比表")
    print(f"  - ece_comparison.csv: ECE对比表")

def main():
    """
    主函数
    """
    print("正在提取其他数据集的实验结果...")
    
    # 收集所有结果
    all_results = collect_all_results()
    
    if not all_results:
        print("未找到任何结果！")
        return
    
    print(f"\n成功收集了 {len(all_results)} 个方法的结果")
    
    # 创建汇总表格
    detailed_df, summary_df = create_summary_table(all_results)
    
    # 创建透视表
    acc1_pivot, ece_pivot = create_pivot_table(detailed_df)
    
    # 打印结果
    print_results(detailed_df, summary_df, acc1_pivot, ece_pivot)
    
    # 保存结果
    save_results(detailed_df, summary_df, acc1_pivot, ece_pivot)

if __name__ == "__main__":
    main()