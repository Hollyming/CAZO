#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取results_csv中的所有算法结果，生成类似论文的结果表格
显示均值±标准差格式，支持导出为LaTeX和CSV
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.table import Table
import seaborn as sns

def load_all_method_results(results_dir="results_csv"):
    """
    加载所有方法的结果
    
    Returns:
        dict: method_name -> summary_data 的映射
    """
    all_results = {}
    
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return all_results
    
    # 查找所有summary_stats.csv文件
    summary_files = glob.glob(os.path.join(results_dir, "*_summary_stats.csv"))
    
    print(f"找到 {len(summary_files)} 个结果文件:")
    
    for file_path in summary_files:
        # 从文件名提取方法名
        filename = os.path.basename(file_path)
        method_name = filename.replace("_summary_stats.csv", "")
        
        try:
            df = pd.read_csv(file_path)
            all_results[method_name] = df
            print(f"  {method_name}: {len(df)} 行数据")
        except Exception as e:
            print(f"  读取 {file_path} 失败: {e}")
    
    return all_results

def organize_results_data(all_results):
    """
    整理结果数据为表格格式
    
    Returns:
        tuple: (corruption_results_df, overall_results_df)
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
    
    # 方法名映射和是否使用BP
    method_info = {
        'no_adapt': {'display_name': 'NoAdapt', 'uses_bp': False},
        'lame': {'display_name': 'LAME', 'uses_bp': False},
        't3a': {'display_name': 'T3A', 'uses_bp': False},
        'tent': {'display_name': 'TENT', 'uses_bp': True},
        'cotta': {'display_name': 'CoTTA', 'uses_bp': True},
        'sar': {'display_name': 'SAR', 'uses_bp': True},
        'foa': {'display_name': 'FOA', 'uses_bp': False},
        'zo_base': {'display_name': 'ZO(RGE)', 'uses_bp': False},
        'cazo': {'display_name': 'CAZO', 'uses_bp': False}
    }
    
    # 收集各个corruption的结果
    corruption_results = []
    overall_results = []
    
    for method_name, df in all_results.items():
        if method_name not in method_info:
            print(f"警告: 未知方法 {method_name}")
            continue
            
        method_display = method_info[method_name]['display_name']
        uses_bp = method_info[method_name]['uses_bp']
        
        # 提取overall结果
        overall_row = df[df['dataset'] == 'ImageNet-C_Overall']
        if not overall_row.empty:
            overall_results.append({
                'Method': method_display,
                'BP': '✓' if uses_bp else '✗',
                'Acc_mean': overall_row.iloc[0]['top1_acc_mean'],
                'Acc_std': overall_row.iloc[0]['top1_acc_std'],
                'ECE_mean': overall_row.iloc[0]['ece_mean'],
                'ECE_std': overall_row.iloc[0]['ece_std']
            })
        
        # 提取各个corruption的结果
        method_corruption_data = {'Method': method_display, 'BP': '✓' if uses_bp else '✗'}
        
        # 按组处理corruption数据集
        for group_name, datasets in corruption_groups.items():
            for dataset in datasets:
                dataset_row = df[df['dataset'] == dataset]
                if not dataset_row.empty:
                    short_name = dataset_name_mapping.get(dataset, dataset)
                    method_corruption_data[f'{short_name}_acc'] = dataset_row.iloc[0]['top1_acc_mean']
                    method_corruption_data[f'{short_name}_acc_std'] = dataset_row.iloc[0]['top1_acc_std']
                    method_corruption_data[f'{short_name}_ece'] = dataset_row.iloc[0]['ece_mean']
                    method_corruption_data[f'{short_name}_ece_std'] = dataset_row.iloc[0]['ece_std']
        
        corruption_results.append(method_corruption_data)
    
    corruption_df = pd.DataFrame(corruption_results)
    overall_df = pd.DataFrame(overall_results)
    
    return corruption_df, overall_df, corruption_groups, dataset_name_mapping

def format_value_with_std(mean, std, decimal_places=1):
    """
    格式化数值为 mean±std 的形式
    """
    if pd.isna(mean) or pd.isna(std):
        return "-"
    
    if decimal_places == 1:
        return f"{mean:.1f}±{std:.1f}"
    elif decimal_places == 2:
        return f"{mean:.2f}±{std:.2f}"
    elif decimal_places == 3:
        return f"{mean:.3f}±{std:.3f}"
    else:
        return f"{mean:.{decimal_places}f}±{std:.{decimal_places}f}"

def create_results_table(corruption_df, overall_df, corruption_groups, dataset_name_mapping):
    """
    创建结果表格
    """
    # 创建最终的表格数据
    table_data = []
    
    # 表头
    header = ['Method', 'BP']
    
    # 添加各组corruption的列头
    for group_name, datasets in corruption_groups.items():
        for dataset in datasets:
            short_name = dataset_name_mapping.get(dataset, dataset)
            header.append(short_name)
    
    # 添加Average列
    header.extend(['Average Acc.', 'Average ECE'])
    
    table_data.append(header)
    
    # 按方法添加数据行
    for _, row in corruption_df.iterrows():
        method_name = row['Method']
        bp_status = row['BP']
        
        # 找到对应的overall数据
        overall_row = overall_df[overall_df['Method'] == method_name]
        
        data_row = [method_name, bp_status]
        
        # 添加各个corruption的acc数据（只显示accuracy，ECE太多会很拥挤）
        for group_name, datasets in corruption_groups.items():
            for dataset in datasets:
                short_name = dataset_name_mapping.get(dataset, dataset)
                acc_col = f'{short_name}_acc'
                acc_std_col = f'{short_name}_acc_std'
                
                if acc_col in row and acc_std_col in row:
                    formatted_acc = format_value_with_std(row[acc_col], row[acc_std_col], 1)
                    data_row.append(formatted_acc)
                else:
                    data_row.append('-')
        
        # 添加overall数据
        if not overall_row.empty:
            overall_acc = format_value_with_std(overall_row.iloc[0]['Acc_mean'], 
                                              overall_row.iloc[0]['Acc_std'], 1)
            overall_ece = format_value_with_std(overall_row.iloc[0]['ECE_mean'], 
                                              overall_row.iloc[0]['ECE_std'], 1)
            data_row.extend([overall_acc, overall_ece])
        else:
            data_row.extend(['-', '-'])
        
        table_data.append(data_row)
    
    return table_data

def save_table_to_csv(table_data, output_file="results_table.csv"):
    """
    保存表格到CSV文件
    """
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    df.to_csv(output_file, index=False)
    print(f"表格已保存到: {output_file}")

def generate_latex_table(table_data, output_file="results_table.tex"):
    """
    生成LaTeX格式的表格
    """
    header = table_data[0]
    data_rows = table_data[1:]
    
    # 创建LaTeX表格
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Test-time Adaptation Results on ImageNet-C}")
    latex_content.append("\\label{tab:imagenet_c_results}")
    latex_content.append("\\resizebox{\\textwidth}{!}{")
    
    # 构建表格列格式
    num_cols = len(header)
    col_format = "l|c|" + "c" * (num_cols - 4) + "|c|c"
    latex_content.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_content.append("\\hline")
    
    # 添加表头组
    latex_content.append("\\multirow{2}{*}{Method} & \\multirow{2}{*}{BP} & \\multicolumn{3}{c|}{Noise} & \\multicolumn{4}{c|}{Blur} & \\multicolumn{3}{c|}{Weather} & \\multicolumn{5}{c|}{Digital} & \\multicolumn{2}{c}{Average} \\\\")
    latex_content.append("\\cline{3-18}")
    
    # 添加具体列名
    header_line = " & ".join(header[2:])  # 跳过Method和BP
    latex_content.append(f"& & {header_line} \\\\")
    latex_content.append("\\hline")
    
    # 添加数据行
    for row in data_rows:
        escaped_row = [str(cell).replace('±', '$\\pm$') for cell in row]  # 转义±符号
        latex_content.append(" & ".join(escaped_row) + " \\\\")
    
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("}")
    latex_content.append("\\end{table}")
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX表格已保存到: {output_file}")

def create_visual_table(table_data, output_file="results_table.png"):
    """
    创建可视化表格图片
    """
    # 设置图表大小
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(cellText=table_data[1:], 
                    colLabels=table_data[0],
                    cellLoc='center',
                    loc='center')
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 2)
    
    # 设置表头样式
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置Method列样式
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#f1f1f2')
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Test-time Adaptation Results on ImageNet-C', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化表格已保存到: {output_file}")

def print_console_table(table_data):
    """
    在控制台打印表格
    """
    # 计算每列的最大宽度
    col_widths = []
    for i in range(len(table_data[0])):
        max_width = max(len(str(row[i])) for row in table_data)
        col_widths.append(max(max_width, 8))  # 最小宽度8
    
    # 打印表头
    print("\n" + "="*120)
    print("Test-time Adaptation Results on ImageNet-C")
    print("="*120)
    
    # 打印表格
    for i, row in enumerate(table_data):
        row_str = " | ".join(str(cell).ljust(col_widths[j]) for j, cell in enumerate(row))
        print(row_str)
        
        if i == 0:  # 表头后添加分隔线
            print("-" * len(row_str))
    
    print("="*120)

def main():
    """
    主函数
    """
    print("正在读取所有算法结果...")
    
    # 确保输出目录存在
    os.makedirs("results_csv", exist_ok=True)
    
    # 加载所有结果
    all_results = load_all_method_results("results_csv")
    
    if not all_results:
        print("未找到任何结果文件！")
        return
    
    print(f"\n成功加载 {len(all_results)} 个方法的结果")
    
    # 整理数据
    corruption_df, overall_df, corruption_groups, dataset_name_mapping = organize_results_data(all_results)
    
    # 创建表格
    table_data = create_results_table(corruption_df, overall_df, corruption_groups, dataset_name_mapping)
    
    # 在控制台显示表格
    print_console_table(table_data)
    
    # 保存为不同格式
    save_table_to_csv(table_data, "results_table.csv")
    generate_latex_table(table_data, "results_table.tex")
    create_visual_table(table_data, "results_table.png")
    
    print(f"\n结果表格已生成:")
    print(f"  - CSV格式: results_table.csv")
    print(f"  - LaTeX格式: results_table.tex")  
    print(f"  - 图片格式: results_table.png")
    
    # 显示一些统计信息
    print(f"\n统计信息:")
    print(f"  分析的方法数: {len(overall_df)}")
    print(f"  corruption数据集数: {sum(len(datasets) for datasets in corruption_groups.values())}")
    
    # 显示最佳性能
    if not overall_df.empty:
        best_acc_method = overall_df.loc[overall_df['Acc_mean'].idxmax()]
        best_ece_method = overall_df.loc[overall_df['ECE_mean'].idxmin()]
        
        print(f"\n性能总结:")
        print(f"  最佳准确率: {best_acc_method['Method']} ({best_acc_method['Acc_mean']:.1f}±{best_acc_method['Acc_std']:.1f})")
        print(f"  最佳校准: {best_ece_method['Method']} ({best_ece_method['ECE_mean']:.1f}±{best_ece_method['ECE_std']:.1f})")

if __name__ == "__main__":
    main() 