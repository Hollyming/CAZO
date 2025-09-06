#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取量化实验结果，生成类似论文的结果表格
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

def load_quant_results(results_dir="results_csv/quant"):
    """
    加载量化实验的所有方法结果
    
    Returns:
        dict: method_name -> results_data 的映射
    """
    all_results = {}
    
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return all_results
    
    # 查找所有方法结果文件
    result_files = glob.glob(os.path.join(results_dir, "*_results.csv"))
    
    print(f"找到 {len(result_files)} 个结果文件:")
    
    for file_path in result_files:
        # 从文件名提取方法名
        filename = os.path.basename(file_path)
        method_name = filename.replace("_results.csv", "")
        
        try:
            df = pd.read_csv(file_path)
            all_results[method_name] = df
            print(f"  {method_name}: {len(df)} 行数据")
        except Exception as e:
            print(f"  读取 {file_path} 失败: {e}")
    
    return all_results

def organize_quant_data(all_results):
    """
    整理量化实验数据为表格格式
    
    Returns:
        tuple: (table_data_8bit, table_data_6bit)
    """
    # 定义corruption数据集的分组和顺序
    corruption_groups = {
        'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
        'Blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'], 
        'Weather': ['snow', 'frost', 'fog'],
        'Digital': ['brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    }
    
    # 将数据集名称映射到简短名称
    dataset_name_mapping = {
        'gaussian_noise': 'Gauss.',
        'shot_noise': 'Shot',
        'impulse_noise': 'Impul.',
        'defocus_blur': 'Defoc.',
        'glass_blur': 'Glass',
        'motion_blur': 'Motion', 
        'zoom_blur': 'Zoom',
        'snow': 'Snow',
        'frost': 'Frost',
        'fog': 'Fog',
        'brightness': 'Brit.',
        'contrast': 'Contr.',
        'elastic_transform': 'Elas.',
        'pixelate': 'Pix.',
        'jpeg_compression': 'JPEG'
    }
    
    # 方法名映射
    method_display_names = {
        'no_adapt': 'NoAdapt',
        't3a': 'T3A',
        'foa': 'FOA',
        'lame': 'LAME',
        'cazo': 'CAZO',
        'zo_base': 'ZO(RGE)'
    }
    
    # 为8bit和6bit分别组织数据
    data_8bit = []
    data_6bit = []
    
    # 方法排序 (优先显示主要方法)
    method_order = ['no_adapt', 't3a', 'foa', 'lame', 'cazo', 'zo_base']
    
    for method_name in method_order:
        if method_name not in all_results:
            continue
            
        df = all_results[method_name]
        method_display = method_display_names.get(method_name, method_name)
        
        # 处理8bit数据
        df_8bit = df[df['quant_type'] == 'quant8_bs64']
        if not df_8bit.empty:
            row_8bit = {'Method': method_display}
            
            # 添加各个corruption的结果
            for group_name, datasets in corruption_groups.items():
                for dataset in datasets:
                    dataset_row = df_8bit[df_8bit['dataset'] == dataset]
                    if not dataset_row.empty:
                        short_name = dataset_name_mapping.get(dataset, dataset)
                        row_8bit[short_name] = dataset_row.iloc[0]['top1_accuracy']
            
            # 添加平均结果
            overall_row = df_8bit[df_8bit['dataset'] == 'ImageNet-C_Overall']
            if not overall_row.empty:
                row_8bit['Acc.'] = overall_row.iloc[0]['top1_accuracy']
                row_8bit['ECE'] = overall_row.iloc[0]['ece'] * 100  # 转换为百分比
            
            data_8bit.append(row_8bit)
        
        # 处理6bit数据
        df_6bit = df[df['quant_type'] == 'quant6_bs64']
        if not df_6bit.empty:
            row_6bit = {'Method': method_display}
            
            # 添加各个corruption的结果
            for group_name, datasets in corruption_groups.items():
                for dataset in datasets:
                    dataset_row = df_6bit[df_6bit['dataset'] == dataset]
                    if not dataset_row.empty:
                        short_name = dataset_name_mapping.get(dataset, dataset)
                        row_6bit[short_name] = dataset_row.iloc[0]['top1_accuracy']
            
            # 添加平均结果
            overall_row = df_6bit[df_6bit['dataset'] == 'ImageNet-C_Overall']
            if not overall_row.empty:
                row_6bit['Acc.'] = overall_row.iloc[0]['top1_accuracy']
                row_6bit['ECE'] = overall_row.iloc[0]['ece'] * 100  # 转换为百分比
            
            data_6bit.append(row_6bit)
    
    return data_8bit, data_6bit, corruption_groups, dataset_name_mapping

def create_quant_table(data_8bit, data_6bit, corruption_groups, dataset_name_mapping):
    """
    创建量化实验的结果表格
    """
    # 创建表头
    header = ['Model', 'Method']
    
    # 添加各组corruption的列头
    for group_name, datasets in corruption_groups.items():
        for dataset in datasets:
            short_name = dataset_name_mapping.get(dataset, dataset)
            header.append(short_name)
    
    # 添加Average列
    header.extend(['Average', 'Acc.', 'ECE'])
    
    # 创建表格数据
    table_data = []
    table_data.append(header)
    
    # 添加8-bit数据
    for row in data_8bit:
        method_name = row['Method']
        data_row = ['8-bit', method_name]
        
        # 添加各个corruption的acc数据
        for group_name, datasets in corruption_groups.items():
            for dataset in datasets:
                short_name = dataset_name_mapping.get(dataset, dataset)
                if short_name in row:
                    data_row.append(f"{row[short_name]:.1f}")
                else:
                    data_row.append('-')
        
        # 添加平均数据
        data_row.append('')  # Average列分隔
        if 'Acc.' in row:
            data_row.append(f"{row['Acc.']:.1f}")
        else:
            data_row.append('-')
        
        if 'ECE' in row:
            data_row.append(f"{row['ECE']:.1f}")
        else:
            data_row.append('-')
        
        table_data.append(data_row)
    
    # 添加6-bit数据
    for row in data_6bit:
        method_name = row['Method']
        data_row = ['6-bit', method_name]
        
        # 添加各个corruption的acc数据
        for group_name, datasets in corruption_groups.items():
            for dataset in datasets:
                short_name = dataset_name_mapping.get(dataset, dataset)
                if short_name in row:
                    data_row.append(f"{row[short_name]:.1f}")
                else:
                    data_row.append('-')
        
        # 添加平均数据
        data_row.append('')  # Average列分隔
        if 'Acc.' in row:
            data_row.append(f"{row['Acc.']:.1f}")
        else:
            data_row.append('-')
        
        if 'ECE' in row:
            data_row.append(f"{row['ECE']:.1f}")
        else:
            data_row.append('-')
        
        table_data.append(data_row)
    
    return table_data

def save_table_to_csv(table_data, output_file="quant_results_table.csv"):
    """
    保存表格到CSV文件
    """
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"表格已保存到: {output_file}")

def generate_latex_table(table_data, output_file="quant_results_table.tex"):
    """
    生成LaTeX格式的表格
    """
    header = table_data[0]
    data_rows = table_data[1:]
    
    # 创建LaTeX表格
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Quantization Experiments Results on ImageNet-C}")
    latex_content.append("\\label{tab:quant_imagenet_c_results}")
    latex_content.append("\\resizebox{\\textwidth}{!}{")
    
    # 构建表格列格式
    num_cols = len(header)
    col_format = "l|l|" + "c" * (num_cols - 5) + "|c|c|c"
    latex_content.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_content.append("\\hline")
    
    # 添加表头组
    latex_content.append("\\multirow{2}{*}{Model} & \\multirow{2}{*}{Method} & \\multicolumn{3}{c|}{Noise} & \\multicolumn{4}{c|}{Blur} & \\multicolumn{3}{c|}{Weather} & \\multicolumn{5}{c|}{Digital} & \\multicolumn{3}{c}{Average} \\\\")
    latex_content.append("\\cline{3-19}")
    
    # 添加具体列名
    header_line = " & ".join(header[2:])  # 跳过Model和Method
    latex_content.append(f"& & {header_line} \\\\")
    latex_content.append("\\hline")
    
    # 添加数据行
    for row in data_rows:
        latex_content.append(" & ".join(str(cell) for cell in row) + " \\\\")
    
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("}")
    latex_content.append("\\end{table}")
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX表格已保存到: {output_file}")

def print_console_table(table_data):
    """
    在控制台打印表格
    """
    # 计算每列的最大宽度
    col_widths = []
    for i in range(len(table_data[0])):
        max_width = max(len(str(row[i])) for row in table_data)
        col_widths.append(max(max_width, 6))  # 最小宽度6
    
    # 打印表头
    print("\n" + "="*150)
    print("Quantization Experiments Results on ImageNet-C")
    print("="*150)
    
    # 打印表格
    for i, row in enumerate(table_data):
        row_str = " | ".join(str(cell).ljust(col_widths[j]) for j, cell in enumerate(row))
        print(row_str)
        
        if i == 0:  # 表头后添加分隔线
            print("-" * len(row_str))
    
    print("="*150)

def main():
    """
    主函数
    """
    print("正在读取量化实验结果...")
    
    # 确保输出目录存在
    os.makedirs("results_csv/quant_bs32", exist_ok=True)
    
    # 加载所有结果
    all_results = load_quant_results("results_csv/quant_bs32")
    
    if not all_results:
        print("未找到任何结果文件！请先运行 1_analyze_quant_results.py")
        return
    
    print(f"\n成功加载 {len(all_results)} 个方法的结果")
    
    # 整理数据
    data_8bit, data_6bit, corruption_groups, dataset_name_mapping = organize_quant_data(all_results)
    
    # 创建表格
    table_data = create_quant_table(data_8bit, data_6bit, corruption_groups, dataset_name_mapping)
    
    # 在控制台显示表格
    print_console_table(table_data)
    
    # 保存为不同格式
    save_table_to_csv(table_data, "quant_results_table_bs32.csv")
    generate_latex_table(table_data, "quant_results_table_bs32.tex")
    
    print(f"\n量化结果表格已生成:")
    print(f"  - CSV格式: quant_results_table_bs32.csv")
    print(f"  - LaTeX格式: quant_results_table_bs32.tex")
    
    # 显示统计信息
    print(f"\n统计信息:")
    print(f"  8-bit方法数: {len(data_8bit)}")
    print(f"  6-bit方法数: {len(data_6bit)}")
    print(f"  corruption数据集数: {sum(len(datasets) for datasets in corruption_groups.values())}")

if __name__ == "__main__":
    main() 