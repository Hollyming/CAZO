#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取main_experiments中tensorboard数据进行绘图
可以选择绘制哪些算法，在指定seed下的某种corruption类型的acc1或loss曲线
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
from typing import List, Dict, Optional

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 方法名映射：用于图例显示
METHOD_DISPLAY_NAMES = {
    'no_adapt': 'NoAdapt',
    'lame': 'LAME', 
    't3a': 'T3A',
    'tent': 'TENT',
    'cotta': 'CoTTA',
    'sar': 'SAR',
    'foa': 'FOA',
    'zo_base': 'ZO',
    'cazo': 'CAZO'
}

def get_display_name(method_name: str) -> str:
    """
    获取方法的显示名称
    """
    return METHOD_DISPLAY_NAMES.get(method_name, method_name.upper())

def get_available_methods(base_dir: str) -> List[str]:
    """
    获取所有可用的方法
    """
    methods = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            methods.append(item)
    return sorted(methods)

def get_available_seeds(method_dir: str) -> List[str]:
    """
    获取指定方法的所有可用seed
    """
    seeds = []
    if os.path.exists(method_dir):
        for item in os.listdir(method_dir):
            if os.path.isdir(os.path.join(method_dir, item)):
                # 提取seed号
                if 'seed' in item:
                    seed_num = item.split('seed')[1].split('_')[0]
                    seeds.append(seed_num)
    return sorted(seeds)

def read_tensorboard_scalars(log_dir: str) -> Dict[str, pd.DataFrame]:
    """
    读取tensorboard日志文件中的scalar数据
    
    Args:
        log_dir: 包含events文件的目录路径
    
    Returns:
        dict: tag_name -> DataFrame(step, value) 的映射
    """
    # 查找events文件
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        print(f"在 {log_dir} 中未找到tensorboard事件文件")
        return {}
    
    event_file = event_files[0]  # 取第一个文件
    
    # 创建EventAccumulator
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # 获取所有scalar标签
    scalar_tags = ea.Tags()['scalars']
    
    scalar_data = {}
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        
        # 转换为DataFrame
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        
        scalar_data[tag] = pd.DataFrame({
            'step': steps,
            'value': values
        })
    
    return scalar_data

def extract_corruption_data(scalar_data: Dict[str, pd.DataFrame], 
                          corruption_type: str, 
                          metric: str = 'acc1') -> Optional[pd.DataFrame]:
    """
    提取指定corruption类型和指标的数据
    
    Args:
        scalar_data: 从tensorboard读取的scalar数据
        corruption_type: corruption类型名称（如'brightness', 'jpeg_compression'等）
        metric: 指标类型 ('acc1' 或 'loss')
    
    Returns:
        DataFrame or None: 包含step和value列的数据，如果没找到则返回None
    """
    # 构建标签模式
    if metric.lower() == 'acc1':
        tag_pattern = f"Accuracy/Top1/{corruption_type}"
    elif metric.lower() == 'loss':
        tag_pattern = f"Loss/{corruption_type}"
    else:
        print(f"不支持的指标类型: {metric}")
        return None
    
    # 查找匹配的标签
    for tag in scalar_data.keys():
        if tag_pattern in tag:
            return scalar_data[tag]
    
    print(f"未找到标签: {tag_pattern}")
    return None

def plot_curves(data_dict: Dict[str, pd.DataFrame], 
                title: str, 
                ylabel: str, 
                output_file: str = None,
                figsize: tuple = (8, 6),#大小可以修改
                smooth_window: int = 10) -> None:
    """
    绘制多条曲线的对比图
    
    Args:
        data_dict: method_name -> DataFrame 的映射
        title: 图表标题
        ylabel: y轴标签
        output_file: 输出文件路径（可选）
        figsize: 图像大小
        smooth_window: 平滑窗口大小
    """
    plt.figure(figsize=figsize)
    
    # 定义颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    
    for i, (method_name, df) in enumerate(data_dict.items()):
        if df is None or df.empty:
            continue
            
        steps = df['step'].values
        values = df['value'].values
        
        # 平滑处理
        if smooth_window > 1 and len(values) > smooth_window:
            smoothed_values = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='valid')
            smoothed_steps = steps[smooth_window-1:]
        else:
            smoothed_values = values
            smoothed_steps = steps
        
        # 使用自定义的显示名称
        display_name = get_display_name(method_name)
        
        plt.plot(smoothed_steps, smoothed_values, 
                label=display_name,  # 修改这里：使用映射后的显示名称
                color=colors[i], 
                linewidth=2)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)  # 修改这行：放到图内左上角，添加半透明背景
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {output_file}")
    
    plt.show()

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='绘制tensorboard曲线图')
    parser.add_argument('--base_dir', type=str, 
                       default='./',
                       help='实验基础目录')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['cazo', 'cotta', 'zo_base', 'tent', 'sar', 'foa', 'no_adapt', 'lame', 't3a'],
                       help='要比较的方法列表')
    parser.add_argument('--seed', type=str, default='42',
                       help='种子编号')
    parser.add_argument('--corruption', type=str, default='brightness',
                       help='corruption类型')
    parser.add_argument('--metric', type=str, default='acc1',
                       choices=['acc1', 'loss'],
                       help='绘制的指标类型')
    parser.add_argument('--smooth', type=int, default=10,
                       help='平滑窗口大小')
    parser.add_argument('--output', type=str, default='./3_output/tensorboard_curves.png',
                       help='输出图像文件路径')
    
    args = parser.parse_args()
    
    print(f"正在绘制 {args.corruption} corruption 的 {args.metric.upper()} 曲线...")
    print(f"比较方法: {', '.join(args.methods)}")
    print(f"使用seed: {args.seed}")
    print("-" * 60)
    
    # 检查基础目录
    if not os.path.exists(args.base_dir):
        print(f"基础目录不存在: {args.base_dir}")
        return
    
    # 收集数据
    data_dict = {}
    
    for method in args.methods:
        method_dir = os.path.join(args.base_dir, method)
        seed_dir = os.path.join(method_dir, f"{method}_seed{args.seed}_bs64")
        
        if not os.path.exists(seed_dir):
            print(f"警告: 目录不存在 {seed_dir}")
            continue
        
        print(f"读取 {method} (seed {args.seed}) 的数据...")
        
        # 读取tensorboard数据
        scalar_data = read_tensorboard_scalars(seed_dir)
        
        if not scalar_data:
            print(f"  未找到scalar数据")
            continue
        
        # 提取指定corruption和指标的数据
        curve_data = extract_corruption_data(scalar_data, args.corruption, args.metric)
        
        if curve_data is not None:
            data_dict[method] = curve_data
            print(f"  成功提取 {len(curve_data)} 个数据点")
        else:
            print(f"  未找到 {args.corruption}/{args.metric} 数据")
    
    if not data_dict:
        print("没有找到任何有效数据！")
        return
    
    # 生成图表
    ylabel = 'Acc1' if args.metric == 'acc1' else 'Loss'
    title = f'{ylabel}/{args.corruption.title()}'
    
    plot_curves(data_dict, 
                title=title,
                ylabel=ylabel,
                output_file=args.output,
                smooth_window=args.smooth)
    
    print(f"\n成功绘制了 {len(data_dict)} 个方法的曲线")

def interactive_mode():
    """
    交互式模式
    """
    base_dir = './'
    
    print("=== Tensorboard曲线绘制工具 ===")
    print()
    
    # 显示可用方法
    available_methods = get_available_methods(base_dir)
    print("可用的方法:")
    for i, method in enumerate(available_methods, 1):
        display_name = get_display_name(method)
        print(f"  {i}. {method} ({display_name})")
    
    # 选择方法
    method_input = input(f"\n请选择要比较的方法 (用逗号分隔编号，如1,2,3): ").strip()
    try:
        method_indices = [int(x.strip()) - 1 for x in method_input.split(',')]
        selected_methods = [available_methods[i] for i in method_indices if 0 <= i < len(available_methods)]
    except:
        print("输入格式错误，使用默认方法")
        selected_methods = ['cazo', 'cotta', 'tent']
    
    print(f"选择的方法: {', '.join([f'{m}({get_display_name(m)})' for m in selected_methods])}")
    
    # 选择seed
    if selected_methods:
        available_seeds = get_available_seeds(os.path.join(base_dir, selected_methods[0]))
        print(f"\n可用的seed: {', '.join(available_seeds)}")
        seed = input("请输入seed编号 (默认42): ").strip() or '42'
    else:
        seed = '42'
    
    # 选择corruption类型
    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    print(f"\n可用的corruption类型:")
    for i, corruption in enumerate(corruption_types, 1):
        print(f"  {i:2d}. {corruption}")
    
    corruption_input = input("请选择corruption类型编号 (默认11=brightness): ").strip() or '11'
    try:
        corruption_idx = int(corruption_input) - 1
        corruption = corruption_types[corruption_idx] if 0 <= corruption_idx < len(corruption_types) else 'brightness'
    except:
        corruption = 'brightness'
    
    # 选择指标
    metric = input("请选择指标 (acc1/loss，默认acc1): ").strip().lower() or 'acc1'
    
    # 执行绘图
    print(f"\n开始绘制...")
    print(f"方法: {', '.join([f'{m}({get_display_name(m)})' for m in selected_methods])}")
    print(f"Seed: {seed}")
    print(f"Corruption: {corruption}")
    print(f"指标: {metric}")
    
    # 收集数据
    data_dict = {}
    
    for method in selected_methods:
        method_dir = os.path.join(base_dir, method)
        seed_dir = os.path.join(method_dir, f"{method}_seed{seed}_bs64")
        
        if not os.path.exists(seed_dir):
            print(f"警告: 目录不存在 {seed_dir}")
            continue
        
        print(f"读取 {method} 的数据...")
        scalar_data = read_tensorboard_scalars(seed_dir)
        
        if scalar_data:
            curve_data = extract_corruption_data(scalar_data, corruption, metric)
            if curve_data is not None:
                data_dict[method] = curve_data
                print(f"  成功提取 {len(curve_data)} 个数据点")
    
    if data_dict:
        ylabel = 'Acc1' if metric == 'acc1' else 'Loss'
        title = f'{ylabel}/{corruption.title()}'
        
        # 确保输出目录存在
        os.makedirs("3_output", exist_ok=True)
        output_file = f"3_output/curve_{corruption}_{metric}_seed{seed}.png"
        plot_curves(data_dict, 
                    title=title,
                    ylabel=ylabel,
                    output_file=output_file)
    else:
        print("没有找到任何有效数据！")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 没有命令行参数，启动交互模式
        interactive_mode()
    else:
        # 有命令行参数，使用命令行模式
        main() 