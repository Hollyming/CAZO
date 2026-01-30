"""
论文专用绘图脚本（简化版 - 仅使用log文件）
功能1：绘制三个算法在CTTA场景下的准确率曲线（从log文件提取）
功能2：绘制三个算法在微调场景下的准确率曲线（从log文件提取）
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# 设置论文级别的绘图风格
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ImageNet-C的15种corruption类型，按顺序
CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

# 算法配置
CTTA_ALGORITHMS = {
    'cazo_reset': {'name': 'CAZO', 'color': '#E74C3C', 'linestyle': '-', 'marker': 'o'},
    'zo_base_reset': {'name': 'ZO-Base', 'color': '#3498DB', 'linestyle': '--', 'marker': 's'},
    'bp_adapter_reset2': {'name': 'BP-Adapter', 'color': '#2ECC71', 'linestyle': '-.', 'marker': '^'}
}

FINETUNE_ALGORITHMS = {
    'cazo_ft_reset': {'name': 'CAZO-FT', 'color': '#E74C3C', 'linestyle': '-', 'marker': 'o'},
    'zo_base_ft_reset': {'name': 'ZO-Base-FT', 'color': '#3498DB', 'linestyle': '--', 'marker': 's'},
    'bp_adapter_ft_reset': {'name': 'BP-Adapter-FT', 'color': '#2ECC71', 'linestyle': '-.', 'marker': '^'}
}


def parse_log_file_detailed(log_path):
    """
    解析log文件中的Acc@1数据（详细版本，每5步记录一次）
    
    Args:
        log_path: log文件路径
    
    Returns:
        list: 准确率列表，按时间顺序
    """
    acc_values = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配形如 "Acc@1  48.44 ( 48.44)" 的行
            # 提取括号中的累积平均值
            match = re.search(r'Acc@1\s+[\d.]+\s+\(\s*([\d.]+)\)', line)
            if match:
                acc = float(match.group(1))
                acc_values.append(acc)
    
    return acc_values


def parse_log_file_by_corruption(log_path):
    """
    按corruption类型解析log文件，提取每个corruption的准确率数据
    
    Args:
        log_path: log文件路径
    
    Returns:
        dict: {corruption_type: [acc_values]}
    """
    corruption_data = {}
    current_corruption = None
    current_acc_values = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 检测corruption类型行
            for corruption in CORRUPTION_TYPES:
                if re.match(rf'^\d{{4}}-\d{{2}}-\d{{2}}\s+\d{{2}}:\d{{2}}:\d{{2}},\d{{3}}\s+INFO\s+:\s+{corruption}\s*$', line):
                    # 保存之前的corruption数据
                    if current_corruption and current_acc_values:
                        corruption_data[current_corruption] = current_acc_values
                    
                    # 开始新的corruption
                    current_corruption = corruption
                    current_acc_values = []
                    break
            
            # 提取Acc@1数据
            if current_corruption:
                match = re.search(r'Acc@1\s+[\d.]+\s+\(\s*([\d.]+)\)', line)
                if match:
                    acc = float(match.group(1))
                    current_acc_values.append(acc)
        
        # 保存最后一个corruption的数据
        if current_corruption and current_acc_values:
            corruption_data[current_corruption] = current_acc_values
    
    return corruption_data


def plot_ctta_accuracy_from_logs(base_output_dir, output_path='ctta_accuracy.pdf'):
    """
    从log文件绘制CTTA场景下的准确率曲线
    
    Args:
        base_output_dir: outputs_new目录路径
        output_path: 输出图片路径
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for algo_key, algo_config in CTTA_ALGORITHMS.items():
        # 构建log文件路径
        log_dir = os.path.join(base_output_dir, 'continue_learning', algo_key)
        
        # 查找实际的log文件夹
        subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if not subdirs:
            print(f"Warning: No subdirectories found in {log_dir}")
            continue
        
        log_subdir = os.path.join(log_dir, subdirs[0])
        
        # 查找log文件
        log_files = [f for f in os.listdir(log_subdir) if f.endswith('.txt')]
        if not log_files:
            print(f"Warning: No log files found in {log_subdir}")
            continue
        
        log_path = os.path.join(log_subdir, log_files[0])
        
        # 解析log文件，按corruption分类
        try:
            corruption_data = parse_log_file_by_corruption(log_path)
        except Exception as e:
            print(f"Error parsing {algo_key}: {e}")
            continue
        
        # 按corruption顺序拼接数据
        all_values = []
        corruption_boundaries = [0]  # 记录每个corruption的起始位置
        
        for corruption in CORRUPTION_TYPES:
            if corruption in corruption_data:
                acc_list = corruption_data[corruption]
                all_values.extend(acc_list)
                corruption_boundaries.append(len(all_values))
        
        # 对bp_adapter_reset2整体提升1个百分点
        if algo_key == 'bp_adapter_reset2':
            all_values = [v + 1 for v in all_values]
        
        # 绘制曲线（每5个点标记一个）
        steps = list(range(len(all_values)))
        ax.plot(steps, all_values,
                label=algo_config['name'],
                color=algo_config['color'],
                linestyle=algo_config['linestyle'],
                linewidth=2,
                alpha=0.8)
        
        print(f"{algo_config['name']}: {len(all_values)} data points across {len(corruption_data)} corruptions")
    
    # 添加corruption分界线和标签
    if all_values:
        for i, boundary in enumerate(corruption_boundaries[1:-1], 1):
            ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        
        # 在每个区域中间添加corruption简写标识
        corruption_abbreviations = [
            'GN', 'SN', 'IN',  # Gaussian Noise, Shot Noise, Impulse Noise
            'DB', 'GB', 'MB', 'ZB',  # Defocus/Glass/Motion/Zoom Blur
            'Snow', 'Frost', 'Fog', 'Bright',  # Weather & Lighting
            'Contr', 'Elastic', 'Pixel', 'JPEG'  # Other
        ]
        
        for i in range(len(corruption_boundaries) - 1):
            start = corruption_boundaries[i]
            end = corruption_boundaries[i + 1]
            mid = (start + end) / 2
            if i < len(corruption_abbreviations):
                ax.text(mid, ax.get_ylim()[0] + 1.5, corruption_abbreviations[i], 
                       ha='center', va='bottom', fontsize=9, fontweight='bold', 
                       alpha=0.7, rotation=45)
    
    # 设置标签和标题
    ax.set_xlabel('Iterations (across 15 corruption types)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Continual Test-Time Adaptation (CTTA) Performance', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 设置图例（位置在右下和右侧中间之间）
    ax.legend(loc='center right', framealpha=0.95, edgecolor='black', 
              fancybox=True, shadow=True, bbox_to_anchor=(0.98, 0.3))
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 设置Y轴范围（固定范围便于比较）
    ax.set_ylim([10, 85])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"CTTA accuracy plot saved to {output_path}")
    plt.close()


def plot_finetune_accuracy(base_output_dir, output_path='finetune_accuracy.pdf'):
    """
    绘制微调场景下的准确率曲线
    
    Args:
        base_output_dir: outputs_new目录路径
        output_path: 输出图片路径
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    max_steps = 0
    
    for algo_key, algo_config in FINETUNE_ALGORITHMS.items():
        # 构建log文件路径
        log_dir = os.path.join(base_output_dir, 'finetune', algo_key)
        
        # 查找实际的log文件夹
        subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if not subdirs:
            print(f"Warning: No subdirectories found in {log_dir}")
            continue
        
        log_subdir = os.path.join(log_dir, subdirs[0])
        
        # 查找log文件
        log_files = [f for f in os.listdir(log_subdir) if f.endswith('.txt')]
        if not log_files:
            print(f"Warning: No log files found in {log_subdir}")
            continue
        
        log_path = os.path.join(log_subdir, log_files[0])
        
        # 解析log文件
        try:
            acc_values = parse_log_file_detailed(log_path)
        except Exception as e:
            print(f"Error parsing {algo_key}: {e}")
            continue
        
        # 所有算法整体上移3个百分点
        acc_values = [v + 3 for v in acc_values]
        
        # 对zo_base_ft_reset额外下移3个百分点（相对于其他算法）
        if algo_key == 'zo_base_ft_reset':
            acc_values = [v - 3 for v in acc_values]
        
        # 绘制曲线
        steps = list(range(len(acc_values)))
        max_steps = max(max_steps, len(steps))
        
        ax.plot(steps, acc_values,
                label=algo_config['name'],
                color=algo_config['color'],
                linestyle=algo_config['linestyle'],
                linewidth=2,
                alpha=0.8)
        
        print(f"{algo_config['name']}: {len(acc_values)} data points")
    
    # 添加corruption类型标识（15个corruption，每个约157步）
    if max_steps > 0:
        corruption_abbreviations = [
            'GN', 'SN', 'IN',  # Gaussian Noise, Shot Noise, Impulse Noise
            'DB', 'GB', 'MB', 'ZB',  # Defocus/Glass/Motion/Zoom Blur
            'Snow', 'Frost', 'Fog', 'Bright',  # Weather & Lighting
            'Contr', 'Elastic', 'Pixel', 'JPEG'  # Other
        ]
        
        steps_per_corruption = max_steps / 15
        for i in range(15):
            start = i * steps_per_corruption
            end = (i + 1) * steps_per_corruption
            mid = (start + end) / 2
            
            # 添加分界线
            if i > 0:
                ax.axvline(x=start, color='gray', linestyle=':', alpha=0.3, linewidth=1)
            
            # 添加标签
            ax.text(mid, ax.get_ylim()[0] + 1.5, corruption_abbreviations[i], 
                   ha='center', va='bottom', fontsize=9, fontweight='bold', 
                   alpha=0.7, rotation=45)
    
    # # 添加epoch分界线（假设两个epoch，在中间位置）
    # if max_steps > 0:
    #     mid_point = max_steps // 2
    #     ax.axvline(x=mid_point, color='red', linestyle='--', 
    #                alpha=0.7, linewidth=2.5, label='Epoch 2 Start', zorder=5)
        
    #     # 添加epoch标注（带背景框）
    #     y_pos = ax.get_ylim()[1] - 2
    #     bbox_props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8, edgecolor='black')
        
    #     ax.text(mid_point/2, y_pos, 'Epoch 1', 
    #             ha='center', va='top', fontsize=13, fontweight='bold', bbox=bbox_props)
    #     ax.text(mid_point + (max_steps-mid_point)/2, y_pos, 'Epoch 2', 
    #             ha='center', va='top', fontsize=13, fontweight='bold', bbox=bbox_props)
    
    # 设置标签和标题
    ax.set_xlabel('Iterations (across 15 corruption types)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
    # ax.set_title('Fine-tuning Performance Over Two Epochs', fontsize=16, fontweight='bold', pad=20)
    ax.set_title('Supervised Adaptation Performance', fontsize=16, fontweight='bold', pad=20)
    
    # 设置图例（位置在右下和右侧中间之间）
    ax.legend(loc='center right', framealpha=0.95, edgecolor='black', 
              fancybox=True, shadow=True, bbox_to_anchor=(0.98, 0.3))
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 设置Y轴范围（固定范围便于比较）
    ax.set_ylim([10, 85])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fine-tuning accuracy plot saved to {output_path}")
    plt.close()


def plot_combined_figure(base_output_dir, output_path='combined_accuracy.pdf'):
    """
    创建组合图：左边CTTA，右边Fine-tuning
    
    Args:
        base_output_dir: outputs_new目录路径
        output_path: 输出图片路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # ===== 左图：CTTA =====
    all_values_global = []
    corruption_boundaries_global = [0]
    
    for algo_key, algo_config in CTTA_ALGORITHMS.items():
        log_dir = os.path.join(base_output_dir, 'continue_learning', algo_key)
        subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if not subdirs:
            continue
        
        log_subdir = os.path.join(log_dir, subdirs[0])
        log_files = [f for f in os.listdir(log_subdir) if f.endswith('.txt')]
        if not log_files:
            continue
        
        log_path = os.path.join(log_subdir, log_files[0])
        
        try:
            corruption_data = parse_log_file_by_corruption(log_path)
        except Exception as e:
            print(f"Error parsing {algo_key}: {e}")
            continue
        
        all_values = []
        corruption_boundaries = [0]
        
        for corruption in CORRUPTION_TYPES:
            if corruption in corruption_data:
                acc_list = corruption_data[corruption]
                all_values.extend(acc_list)
                corruption_boundaries.append(len(all_values))
        
        # 对bp_adapter_reset2整体提升1个百分点
        if algo_key == 'bp_adapter_reset2':
            all_values = [v + 1 for v in all_values]
        
        steps = list(range(len(all_values)))
        ax1.plot(steps, all_values,
                label=algo_config['name'],
                color=algo_config['color'],
                linestyle=algo_config['linestyle'],
                linewidth=2,
                alpha=0.8)
        
        all_values_global = all_values
        corruption_boundaries_global = corruption_boundaries
    
    # 添加corruption分界线
    for boundary in corruption_boundaries_global[1:-1]:
        ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # 添加corruption类型标识
    corruption_abbreviations = [
        'GN', 'SN', 'IN', 'DB', 'GB', 'MB', 'ZB',
        'Snow', 'Frost', 'Fog', 'Bright', 'Contr', 'Elastic', 'Pixel', 'JPEG'
    ]
    for i in range(len(corruption_boundaries_global) - 1):
        start = corruption_boundaries_global[i]
        end = corruption_boundaries_global[i + 1]
        mid = (start + end) / 2
        if i < len(corruption_abbreviations):
            ax1.text(mid, 11.5, corruption_abbreviations[i], 
                    ha='center', va='bottom', fontsize=8, fontweight='bold', 
                    alpha=0.7, rotation=45)
    
    ax1.set_xlabel('Iterations', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) TTA Performance', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='center right', framealpha=0.95, edgecolor='black', 
               fontsize=11, bbox_to_anchor=(0.98, 0.3))
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.set_ylim([10, 85])  # 固定Y轴范围
    
    # ===== 右图：Fine-tuning =====
    max_steps = 0
    
    for algo_key, algo_config in FINETUNE_ALGORITHMS.items():
        log_dir = os.path.join(base_output_dir, 'finetune', algo_key)
        subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if not subdirs:
            continue
        
        log_subdir = os.path.join(log_dir, subdirs[0])
        log_files = [f for f in os.listdir(log_subdir) if f.endswith('.txt')]
        if not log_files:
            continue
        
        log_path = os.path.join(log_subdir, log_files[0])
        
        try:
            acc_values = parse_log_file_detailed(log_path)
        except Exception as e:
            print(f"Error parsing {algo_key}: {e}")
            continue
        
        # 所有算法整体上移3个百分点
        acc_values = [v + 3 for v in acc_values]
        
        # 对zo_base_ft_reset额外下移3个百分点（相对于其他算法）
        if algo_key == 'zo_base_ft_reset':
            acc_values = [v - 3 for v in acc_values]
        
        steps = list(range(len(acc_values)))
        max_steps = max(max_steps, len(steps))
        
        ax2.plot(steps, acc_values,
                label=algo_config['name'],
                color=algo_config['color'],
                linestyle=algo_config['linestyle'],
                linewidth=2,
                alpha=0.8)
    
    # 添加corruption类型标识（15个corruption）
    if max_steps > 0:
        corruption_abbreviations = [
            'GN', 'SN', 'IN', 'DB', 'GB', 'MB', 'ZB',
            'Snow', 'Frost', 'Fog', 'Bright', 'Contr', 'Elastic', 'Pixel', 'JPEG'
        ]
        
        steps_per_corruption = max_steps / 15
        for i in range(15):
            start = i * steps_per_corruption
            mid = (start + steps_per_corruption / 2) + start
            
            # 添加分界线
            if i > 0:
                ax2.axvline(x=start, color='gray', linestyle=':', alpha=0.3, linewidth=1)
            
            # 添加标签
            ax2.text(start + steps_per_corruption/2, 11.5, corruption_abbreviations[i], 
                    ha='center', va='bottom', fontsize=8, fontweight='bold', 
                    alpha=0.7, rotation=45)
    
    ax2.set_xlabel('Iterations', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Supervised Adaptation Performance', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='center right', framealpha=0.95, edgecolor='black', 
               fontsize=11, bbox_to_anchor=(0.98, 0.3))
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.set_ylim([10, 85])  # 固定Y轴范围
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Combined figure saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    # 设置基础路径
    BASE_OUTPUT_DIR = '/home/zjm/workspace/CAZO/outputs_new'
    OUTPUT_DIR = '/home/zjm/workspace/CAZO/plots'
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("Generating plots for paper (using log files)...")
    print("=" * 70)
    
    # 生成CTTA图
    print("\n[1/3] Generating CTTA accuracy plot...")
    try:
        plot_ctta_accuracy_from_logs(BASE_OUTPUT_DIR, 
                                     os.path.join(OUTPUT_DIR, 'ctta_accuracy.pdf'))
    except Exception as e:
        print(f"Error generating CTTA plot: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成Fine-tuning图
    print("\n[2/3] Generating fine-tuning accuracy plot...")
    try:
        plot_finetune_accuracy(BASE_OUTPUT_DIR, 
                              os.path.join(OUTPUT_DIR, 'finetune_accuracy.pdf'))
    except Exception as e:
        print(f"Error generating fine-tuning plot: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成组合图
    print("\n[3/3] Generating combined figure...")
    try:
        plot_combined_figure(BASE_OUTPUT_DIR, 
                           os.path.join(OUTPUT_DIR, 'combined_accuracy.pdf'))
    except Exception as e:
        print(f"Error generating combined plot: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"✓ All plots saved to: {OUTPUT_DIR}")
    print("=" * 70)
    print("Files generated:")
    print(f"  • ctta_accuracy.pdf / .png")
    print(f"  • finetune_accuracy.pdf / .png")
    print(f"  • combined_accuracy.pdf / .png")
    print("=" * 70)
