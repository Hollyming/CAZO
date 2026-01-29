#!/usr/bin/env python3
"""
从真实实验数据绘制Cumulative Explained Variance和Eigenvalue Decay Pattern重叠图
使用双Y轴，特征值衰减使用普通坐标（非对数）

数据来源：analysis_data.npz (从run_experiment3.py生成的结果)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# matplotlib 样式设置
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# 配色方案：左Y轴（累积解释方差）使用蓝色系，右Y轴（特征值）使用红色系
COLORS_VAR = ['#1f77b4', '#ff7f0e', '#2ca02c']      # 蓝、橙、绿
COLORS_EIGEN = ['#d62728', '#9467bd', '#8c564b']     # 红、紫、棕


def get_display_steps(history_len):
    """获取要显示的代表性步骤"""
    desired = [0, 25, 50, 75, 99]
    if history_len == 0:
        return []
    if history_len > max(desired):
        return desired
    # 均匀采样
    count = min(5, history_len)
    indices = np.linspace(0, history_len - 1, count).astype(int).tolist()
    return sorted(list(set(indices)))


def load_analysis_data(data_path):
    """从npz文件加载分析数据"""
    print(f"从 {data_path} 加载数据...")
    data = np.load(data_path, allow_pickle=True)
    
    eigenvalues_history = data['eigenvalues_history']
    steps = data['steps']
    
    print(f"✓ 成功加载数据")
    print(f"  - 步骤数: {len(eigenvalues_history)}")
    print(f"  - 特征值维度: {eigenvalues_history[0].shape if len(eigenvalues_history) > 0 else 'N/A'}")
    
    return eigenvalues_history, steps


def plot_overlapped_figure(eigenvalues_history, steps, output_path):
    """绘制重叠图：Cumulative Explained Variance + Eigenvalue Decay"""
    
    if len(eigenvalues_history) == 0:
        print("错误：没有可用的特征值数据")
        return
    
    # 获取要显示的步骤
    display_steps = get_display_steps(len(eigenvalues_history))
    print(f"将显示以下步骤: {display_steps}")
    
    # 创建图形
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))
    
    # ========== 左Y轴：Cumulative Explained Variance (%) ==========
    ax1.set_xlabel('Number of Components / Eigenvalue Index', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Explained Variance (%)', fontsize=14, fontweight='bold', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    
    # 绘制累积解释方差曲线
    for i, step_idx in enumerate(display_steps):  # 显示所有选定的步骤
        if step_idx < len(eigenvalues_history):
            eigenvals = eigenvalues_history[step_idx]
            
            # 计算累积解释方差
            total_variance = np.sum(eigenvals)
            cumulative_variance = np.cumsum(eigenvals) / (total_variance + 1e-12)
            cumulative_variance = cumulative_variance[:50]  # 只显示前50个
            
            x = np.arange(1, len(cumulative_variance) + 1)
            color = COLORS_VAR[i % len(COLORS_VAR)]
            
            ax1.plot(x, cumulative_variance * 100,
                    color=color,
                    linewidth=2.5,
                    marker='o',
                    markersize=5,
                    markevery=5,
                    label=f'Cumulative Var. (Step {step_idx})',
                    alpha=0.85,
                    linestyle='-',
                    zorder=10)
    
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # ========== 右Y轴：Eigenvalue Magnitude (普通坐标，非对数) ==========
    ax2 = ax1.twinx()
    ax2.set_ylabel('Eigenvalue Magnitude', fontsize=14, fontweight='bold', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    
    # 绘制特征值衰减曲线（使用普通坐标）
    for i, step_idx in enumerate(display_steps):  # 显示所有选定的步骤
        if step_idx < len(eigenvalues_history):
            eigenvals = eigenvalues_history[step_idx]
            display_eigenvals = eigenvals[:min(50, len(eigenvals))]
            
            x = np.arange(1, len(display_eigenvals) + 1)
            color = COLORS_EIGEN[i % len(COLORS_EIGEN)]
            
            # 注意：这里使用普通plot，不是semilogy
            ax2.plot(x, np.abs(display_eigenvals),
                    color=color,
                    linewidth=2.5,
                    marker='s',
                    markersize=5,
                    markevery=5,
                    label=f'Eigenvalue (Step {step_idx})',
                    alpha=0.85,
                    linestyle='--',
                    zorder=5)
    
    # 设置右Y轴范围，确保从0开始
    y_max = ax2.get_ylim()[1]
    ax2.set_ylim([0, y_max * 1.05])
    
    # ========== 合并图例 ==========
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
              loc='center right',
              frameon=True,
              shadow=True,
              fancybox=True,
              fontsize=10,
              ncol=1)
    
    # 标题
    plt.title('Hessian Low-Rank Analysis: Cumulative Variance & Eigenvalue Decay',
             fontsize=16, fontweight='bold', pad=15)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 重叠图已保存到: {output_path}")
    
    # 同时保存PDF格式
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 重叠图已保存到: {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='从真实数据绘制Cumulative Explained Variance和Eigenvalue Decay重叠图'
    )
    parser.add_argument('--data', type=str,
                       default='/home/zjm/Workspace/CAZO/hessian_analysis/results/hessian_separate_experiment_v3_20260129_173644/analysis_data.npz',
                       help='analysis_data.npz文件路径')
    parser.add_argument('--output', type=str,
                       default='/home/zjm/Workspace/CAZO/hessian_analysis/results/hessian_separate_experiment_v3_20260129_173644/figures/overlapped_real_data.png',
                       help='输出图形路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.data).exists():
        print(f"错误: 数据文件不存在: {args.data}")
        print("\n提示: 请确保已运行run_experiment3.py生成analysis_data.npz")
        return
    
    # 加载数据
    eigenvalues_history, steps = load_analysis_data(args.data)
    
    # 绘制重叠图
    print("\n开始绘制重叠图...")
    plot_overlapped_figure(eigenvalues_history, steps, args.output)
    
    print("\n完成！")
    print(f"\n图形说明：")
    print(f"  - 左Y轴（蓝色系）：累积解释方差百分比 (0-100%)")
    print(f"  - 右Y轴（红色系）：特征值大小（普通坐标，非对数）")
    print(f"  - X轴：特征值索引/主成分数量 (1-50)")
    print(f"  - 显示了不同优化步骤的曲线变化")


if __name__ == '__main__':
    main()
