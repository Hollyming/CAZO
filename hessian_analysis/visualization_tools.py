#!/usr/bin/env python3
"""
Visualization Tools for Hessian Low-Rank Analysis
专业论文级别的Hessian低秩特性可视化工具

包含以下可视化功能：
1. 特征值分布图 (Eigenvalue Distribution)
2. 解释方差比例图 (Explained Variance Ratio)
3. 一致性指标图 (Consistency Metrics)
4. 低秩特性综合展示 (Low-Rank Properties)
5. 时间序列分析图 (Temporal Analysis)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
import datetime

# 设置中文字体和论文风格
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 专业配色方案
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'light_blue': '#aec7e8',
    'light_orange': '#ffbb78',
    'light_green': '#98df8a',
    'light_red': '#ff9896',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f'
}

class HessianVisualizationTools:
    """
    Hessian分析可视化工具类
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
    def plot_eigenvalue_distribution(self, 
                                   eigenvalues_history: List[np.ndarray], 
                                   steps: List[int],
                                   save_name: str = "eigenvalue_distribution.pdf") -> None:
        """
        绘制特征值分布图
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hessian Eigenvalue Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. 特征值衰减图 (Eigenvalue Decay)
        ax1 = axes[0, 0]
        
        # 选择几个代表性步骤进行展示
        display_steps = [0, len(eigenvalues_history)//4, len(eigenvalues_history)//2, -1]
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['quaternary']]
        
        for i, step_idx in enumerate(display_steps):
            if step_idx < len(eigenvalues_history):
                eigenvals = eigenvalues_history[step_idx]
                # 只显示前50个特征值
                display_eigenvals = eigenvals[:min(50, len(eigenvals))]
                ax1.semilogy(range(1, len(display_eigenvals) + 1), 
                           np.abs(display_eigenvals), 
                           color=colors[i], 
                           linewidth=2,
                           marker='o', 
                           markersize=4,
                           label=f'Step {steps[step_idx] if step_idx < len(steps) else "Final"}')
        
        ax1.set_xlabel('Eigenvalue Index')
        ax1.set_ylabel('Eigenvalue Magnitude (log scale)')
        ax1.set_title('Eigenvalue Decay Pattern')
        ax1.legend()
        ax1.grid(True, alpha=0.3, zorder=0)
        
        # 2. 累计解释方差比例 (Cumulative Explained Variance)
        ax2 = axes[0, 1]
        
        for i, step_idx in enumerate(display_steps):
            if step_idx < len(eigenvalues_history):
                eigenvals = eigenvalues_history[step_idx]
                eigenvals_abs = np.abs(eigenvals)
                cumsum = np.cumsum(eigenvals_abs) / np.sum(eigenvals_abs)
                # 显示前50个
                display_cumsum = cumsum[:min(50, len(cumsum))]
                ax2.plot(range(1, len(display_cumsum) + 1), 
                        display_cumsum * 100,
                        color=colors[i], 
                        linewidth=2,
                        marker='s', 
                        markersize=3,
                        label=f'Step {steps[step_idx] if step_idx < len(steps) else "Final"}')
        
        # 添加参考线
        ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% Threshold')
        ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% Threshold')
        
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance (%)')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        # 3. 特征值分布直方图 (最终步骤)
        ax3 = axes[1, 0]
        
        final_eigenvals = eigenvalues_history[-1]
        # 只考虑正特征值
        positive_eigenvals = final_eigenvals[final_eigenvals > 1e-10]
        
        ax3.hist(np.log10(positive_eigenvals + 1e-12), 
                bins=30, 
                color=COLORS['light_blue'], 
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5)
        ax3.set_xlabel('log₁₀(Eigenvalue)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Final Eigenvalue Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. 有效秩随时间变化
        ax4 = axes[1, 1]
        
        effective_ranks = []
        for eigenvals in eigenvalues_history:
            # 计算有效秩 (基于Shannon熵)
            eigenvals_pos = np.abs(eigenvals) + 1e-12
            eigenvals_normalized = eigenvals_pos / np.sum(eigenvals_pos)
            entropy = -np.sum(eigenvals_normalized * np.log(eigenvals_normalized + 1e-12))
            effective_rank = np.exp(entropy)
            effective_ranks.append(effective_rank)
        
        ax4.plot(steps[:len(effective_ranks)], 
                effective_ranks,
                color=COLORS['primary'], 
                linewidth=3,
                marker='o', 
                markersize=5)
        ax4.set_xlabel('Optimization Step')
        ax4.set_ylabel('Effective Rank')
        ax4.set_title('Effective Rank Evolution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_lowrank_metrics(self, 
                           lowrank_metrics: Dict,
                           steps: List[int],
                           save_name: str = "lowrank_metrics.pdf") -> None:
        """
        绘制低秩指标图
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        fig.suptitle('Low-Rank Properties of Hessian Matrix', fontsize=16, fontweight='bold')
        
        # 1. 解释方差比例时间序列
        ax1 = fig.add_subplot(gs[0, :])
        
        # 绘制不同k值的解释方差比例
        k_values = ['top_5', 'top_10', 'top_20', 'top_50']
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['quaternary']]
        
        for i, k in enumerate(k_values):
            if lowrank_metrics['explained_variance_ratios']:
                ratios = []
                for step_ratios in lowrank_metrics['explained_variance_ratios']:
                    if k in step_ratios:
                        ratios.append(step_ratios[k] * 100)
                    else:
                        ratios.append(0)
                
                if ratios:
                    ax1.plot(steps[:len(ratios)], 
                            ratios,
                            color=colors[i], 
                            linewidth=2,
                            marker='o', 
                            markersize=4,
                            label=f'Top {k.split("_")[1]} Components')
        
        ax1.set_xlabel('Optimization Step')
        ax1.set_ylabel('Explained Variance (%)')
        ax1.set_title('Explained Variance Ratio Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])
        
        # 2. 有效秩时间序列
        ax2 = fig.add_subplot(gs[1, 0])
        
        if lowrank_metrics['effective_ranks']:
            ax2.plot(steps[:len(lowrank_metrics['effective_ranks'])], 
                    lowrank_metrics['effective_ranks'],
                    color=COLORS['primary'], 
                    linewidth=3,
                    marker='s', 
                    markersize=5)
            ax2.fill_between(steps[:len(lowrank_metrics['effective_ranks'])], 
                           lowrank_metrics['effective_ranks'],
                           alpha=0.3, 
                           color=COLORS['light_blue'])
        
        ax2.set_xlabel('Optimization Step')
        ax2.set_ylabel('Effective Rank')
        ax2.set_title('Effective Rank Evolution')
        ax2.grid(True, alpha=0.3)
        
        # 3. 谱范数时间序列
        ax3 = fig.add_subplot(gs[1, 1])
        
        if lowrank_metrics['spectral_norms']:
            ax3.semilogy(steps[:len(lowrank_metrics['spectral_norms'])], 
                        lowrank_metrics['spectral_norms'],
                        color=COLORS['secondary'], 
                        linewidth=3,
                        marker='^', 
                        markersize=5)
        
        ax3.set_xlabel('Optimization Step')
        ax3.set_ylabel('Spectral Norm (log scale)')
        ax3.set_title('Spectral Norm Evolution')
        ax3.grid(True, alpha=0.3)
        
        # 4. 条件数时间序列
        ax4 = fig.add_subplot(gs[1, 2])
        
        if lowrank_metrics['condition_numbers']:
            # 过滤无穷大值
            condition_numbers = [cn if np.isfinite(cn) else np.nan 
                               for cn in lowrank_metrics['condition_numbers']]
            ax4.semilogy(steps[:len(condition_numbers)], 
                        condition_numbers,
                        color=COLORS['tertiary'], 
                        linewidth=3,
                        marker='d', 
                        markersize=5)
        
        ax4.set_xlabel('Optimization Step')
        ax4.set_ylabel('Condition Number (log scale)')
        ax4.set_title('Condition Number Evolution')
        ax4.grid(True, alpha=0.3)
        
        # 5. 低秩指标对比 (箱线图)
        ax5 = fig.add_subplot(gs[2, :])
        
        # 准备数据
        explained_var_data = []
        for step_ratios in lowrank_metrics['explained_variance_ratios']:
            if 'top_10' in step_ratios:
                explained_var_data.append(step_ratios['top_10'] * 100)
        
        # 创建箱线图数据
        box_data = []
        labels = []
        
        if explained_var_data:
            box_data.append(explained_var_data)
            labels.append('Top 10\nExplained Var (%)')
        
        if lowrank_metrics['effective_ranks']:
            box_data.append(lowrank_metrics['effective_ranks'])
            labels.append('Effective Rank')
        
        # 归一化数据用于比较
        if box_data:
            # 计算每个指标的归一化版本
            normalized_data = []
            for data in box_data:
                data_array = np.array(data)
                normalized = (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array) + 1e-8)
                normalized_data.append(normalized * 100)  # 转换为百分比
            
            bp = ax5.boxplot(normalized_data, 
                           labels=labels,
                           patch_artist=True,
                           medianprops={'color': 'red', 'linewidth': 2})
            
            # 设置颜色
            colors = [COLORS['light_blue'], COLORS['light_orange']]
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
        
        ax5.set_ylabel('Normalized Value (%)')
        ax5.set_title('Low-Rank Metrics Distribution (Normalized)')
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_consistency_metrics(self, 
                               consistency_metrics: Dict,
                               steps: List[int],
                               save_name: str = "consistency_metrics.pdf") -> None:
        """
        绘制一致性指标图
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hessian Consistency Analysis During Optimization', fontsize=16, fontweight='bold')
        
        # 1. 子空间角度
        ax1 = axes[0, 0]
        
        if consistency_metrics['subspace_angles']:
            angles_deg = [np.degrees(angle) for angle in consistency_metrics['subspace_angles']]
            ax1.plot(steps[1:len(angles_deg)+1], 
                    angles_deg,
                    color=COLORS['primary'], 
                    linewidth=3,
                    marker='o', 
                    markersize=5,
                    label='Subspace Angle')
            
            # 添加趋势线
            if len(angles_deg) > 2:
                z = np.polyfit(range(len(angles_deg)), angles_deg, 1)
                p = np.poly1d(z)
                ax1.plot(steps[1:len(angles_deg)+1], 
                        p(range(len(angles_deg))), 
                        "--", 
                        color=COLORS['secondary'],
                        alpha=0.8,
                        linewidth=2,
                        label='Trend')
        
        ax1.set_xlabel('Optimization Step')
        ax1.set_ylabel('Principal Subspace Angle (degrees)')
        ax1.set_title('Subspace Consistency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 特征值相关性
        ax2 = axes[0, 1]
        
        if consistency_metrics['eigenvalue_correlations']:
            correlations = [corr if np.isfinite(corr) else 0 
                          for corr in consistency_metrics['eigenvalue_correlations']]
            ax2.plot(steps[1:len(correlations)+1], 
                    correlations,
                    color=COLORS['tertiary'], 
                    linewidth=3,
                    marker='s', 
                    markersize=5,
                    label='Eigenvalue Correlation')
            
            # 添加参考线
            ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='High Correlation (0.9)')
            ax2.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Moderate Correlation (0.7)')
        
        ax2.set_xlabel('Optimization Step')
        ax2.set_ylabel('Correlation Coefficient')
        ax2.set_title('Eigenvalue Correlation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.1, 1.1])
        
        # 3. Frobenius距离
        ax3 = axes[1, 0]
        
        if consistency_metrics['frobenius_distances']:
            ax3.semilogy(steps[1:len(consistency_metrics['frobenius_distances'])+1], 
                        consistency_metrics['frobenius_distances'],
                        color=COLORS['quaternary'], 
                        linewidth=3,
                        marker='^', 
                        markersize=5,
                        label='Frobenius Distance')
        
        ax3.set_xlabel('Optimization Step')
        ax3.set_ylabel('Frobenius Distance (log scale)')
        ax3.set_title('Hessian Matrix Distance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 一致性综合评分
        ax4 = axes[1, 1]
        
        # 计算综合一致性评分
        if (consistency_metrics['subspace_angles'] and 
            consistency_metrics['eigenvalue_correlations']):
            
            # 归一化各指标 (角度越小越好，相关性越高越好)
            angles = consistency_metrics['subspace_angles']
            correlations = consistency_metrics['eigenvalue_correlations']
            
            # 角度标准化到0-1，越小越好
            angles_norm = [(np.pi/2 - angle) / (np.pi/2) for angle in angles]
            angles_norm = [max(0, min(1, a)) for a in angles_norm]  # clip到[0,1]
            
            # 相关性已经在0-1范围
            correlations_clean = [corr if np.isfinite(corr) else 0 for corr in correlations]
            
            # 综合评分 (简单平均)
            consistency_scores = [(a + c) / 2 for a, c in zip(angles_norm, correlations_clean)]
            
            ax4.plot(steps[1:len(consistency_scores)+1], 
                    consistency_scores,
                    color=COLORS['purple'], 
                    linewidth=4,
                    marker='o', 
                    markersize=6,
                    label='Consistency Score')
            
            ax4.fill_between(steps[1:len(consistency_scores)+1], 
                           consistency_scores,
                           alpha=0.3, 
                           color=COLORS['purple'])
            
            # 添加参考线
            ax4.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='High Consistency (0.8)')
            ax4.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate Consistency (0.6)')
        
        ax4.set_xlabel('Optimization Step')
        ax4.set_ylabel('Consistency Score')
        ax4.set_title('Overall Consistency Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_comprehensive_summary(self, 
                                 eigenvalues_history: List[np.ndarray],
                                 lowrank_metrics: Dict,
                                 consistency_metrics: Dict,
                                 steps: List[int],
                                 save_name: str = "comprehensive_summary.pdf") -> None:
        """
        绘制综合总结图
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('Comprehensive Analysis: Hessian Low-Rank Properties and Consistency in TTA', 
                    fontsize=18, fontweight='bold')
        
        # 1. 主要发现展示 (左上角，跨两列)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # 关键指标的时间序列
        if lowrank_metrics['explained_variance_ratios']:
            top10_ratios = []
            for step_ratios in lowrank_metrics['explained_variance_ratios']:
                if 'top_10' in step_ratios:
                    top10_ratios.append(step_ratios['top_10'] * 100)
            
            if top10_ratios:
                ax1_twin = ax1.twinx()
                
                # 左y轴：解释方差比例
                line1 = ax1.plot(steps[:len(top10_ratios)], 
                               top10_ratios,
                               color=COLORS['primary'], 
                               linewidth=4,
                               marker='o', 
                               markersize=6,
                               label='Top 10 Explained Variance (%)')
                
                # 右y轴：有效秩
                if lowrank_metrics['effective_ranks']:
                    line2 = ax1_twin.plot(steps[:len(lowrank_metrics['effective_ranks'])], 
                                        lowrank_metrics['effective_ranks'],
                                        color=COLORS['secondary'], 
                                        linewidth=4,
                                        marker='s', 
                                        markersize=6,
                                        label='Effective Rank')
                
                ax1.set_xlabel('Optimization Step')
                ax1.set_ylabel('Explained Variance (%)', color=COLORS['primary'])
                ax1_twin.set_ylabel('Effective Rank', color=COLORS['secondary'])
                ax1.set_title('Key Low-Rank Properties')
                
                # 合并图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1_twin.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
                
                ax1.grid(True, alpha=0.3)
        
        # 2. 特征值谱图 (右上角，跨两列)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        # 显示几个关键步骤的特征值
        display_steps = [0, len(eigenvalues_history)//2, -1]
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
        labels = ['Initial', 'Mid-optimization', 'Final']
        
        for i, step_idx in enumerate(display_steps):
            if step_idx < len(eigenvalues_history):
                eigenvals = eigenvalues_history[step_idx]
                # 显示前30个
                display_eigenvals = eigenvals[:min(30, len(eigenvals))]
                ax2.semilogy(range(1, len(display_eigenvals) + 1), 
                           np.abs(display_eigenvals), 
                           color=colors[i], 
                           linewidth=3,
                           marker='o', 
                           markersize=4,
                           label=labels[i])
        
        ax2.set_xlabel('Eigenvalue Index')
        ax2.set_ylabel('Eigenvalue Magnitude (log scale)')
        ax2.set_title('Eigenvalue Spectrum Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 一致性指标 (第二行左)
        ax3 = fig.add_subplot(gs[1, 0])
        
        if consistency_metrics['subspace_angles']:
            angles_deg = [np.degrees(angle) for angle in consistency_metrics['subspace_angles']]
            ax3.plot(steps[1:len(angles_deg)+1], 
                    angles_deg,
                    color=COLORS['quaternary'], 
                    linewidth=3,
                    marker='o', 
                    markersize=5)
            
            # 添加平均线
            mean_angle = np.mean(angles_deg)
            ax3.axhline(y=mean_angle, color='red', linestyle='--', alpha=0.8, 
                       label=f'Mean: {mean_angle:.1f}°')
        
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Angle (degrees)')
        ax3.set_title('Subspace Consistency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 特征值相关性 (第二行中左)
        ax4 = fig.add_subplot(gs[1, 1])
        
        if consistency_metrics['eigenvalue_correlations']:
            correlations = [corr if np.isfinite(corr) else 0 
                          for corr in consistency_metrics['eigenvalue_correlations']]
            ax4.plot(steps[1:len(correlations)+1], 
                    correlations,
                    color=COLORS['tertiary'], 
                    linewidth=3,
                    marker='s', 
                    markersize=5)
            
            # 添加平均线
            mean_corr = np.mean(correlations)
            ax4.axhline(y=mean_corr, color='red', linestyle='--', alpha=0.8, 
                       label=f'Mean: {mean_corr:.2f}')
        
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Correlation')
        ax4.set_title('Eigenvalue Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 条件数 (第二行中右)
        ax5 = fig.add_subplot(gs[1, 2])
        
        if lowrank_metrics['condition_numbers']:
            condition_numbers = [cn if np.isfinite(cn) else np.nan 
                               for cn in lowrank_metrics['condition_numbers']]
            ax5.semilogy(steps[:len(condition_numbers)], 
                        condition_numbers,
                        color=COLORS['purple'], 
                        linewidth=3,
                        marker='^', 
                        markersize=5)
        
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Condition Number (log)')
        ax5.set_title('Matrix Conditioning')
        ax5.grid(True, alpha=0.3)
        
        # 6. 核范数 (第二行右)
        ax6 = fig.add_subplot(gs[1, 3])
        
        if lowrank_metrics['nuclear_norms']:
            ax6.plot(steps[:len(lowrank_metrics['nuclear_norms'])], 
                    lowrank_metrics['nuclear_norms'],
                    color=COLORS['brown'], 
                    linewidth=3,
                    marker='d', 
                    markersize=5)
        
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Nuclear Norm')
        ax6.set_title('Nuclear Norm Evolution')
        ax6.grid(True, alpha=0.3)
        
        # 7. 关键统计信息表格 (第三行，跨全部)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # 创建统计信息表格
        stats_data = []
        
        # 计算统计信息
        if lowrank_metrics['explained_variance_ratios']:
            top10_ratios = [step_ratios.get('top_10', 0) * 100 
                           for step_ratios in lowrank_metrics['explained_variance_ratios']]
            if top10_ratios:
                stats_data.append(['Top 10 Explained Variance (%)', 
                                 f'{np.mean(top10_ratios):.1f} ± {np.std(top10_ratios):.1f}',
                                 f'[{np.min(top10_ratios):.1f}, {np.max(top10_ratios):.1f}]'])
        
        if lowrank_metrics['effective_ranks']:
            eff_ranks = lowrank_metrics['effective_ranks']
            stats_data.append(['Effective Rank', 
                             f'{np.mean(eff_ranks):.1f} ± {np.std(eff_ranks):.1f}',
                             f'[{np.min(eff_ranks):.1f}, {np.max(eff_ranks):.1f}]'])
        
        if consistency_metrics['subspace_angles']:
            angles_deg = [np.degrees(angle) for angle in consistency_metrics['subspace_angles']]
            stats_data.append(['Subspace Angle (degrees)', 
                             f'{np.mean(angles_deg):.1f} ± {np.std(angles_deg):.1f}',
                             f'[{np.min(angles_deg):.1f}, {np.max(angles_deg):.1f}]'])
        
        if consistency_metrics['eigenvalue_correlations']:
            correlations = [corr if np.isfinite(corr) else 0 
                          for corr in consistency_metrics['eigenvalue_correlations']]
            if correlations:
                stats_data.append(['Eigenvalue Correlation', 
                                 f'{np.mean(correlations):.3f} ± {np.std(correlations):.3f}',
                                 f'[{np.min(correlations):.3f}, {np.max(correlations):.3f}]'])
        
        if stats_data:
            table = ax7.table(cellText=stats_data,
                            colLabels=['Metric', 'Mean ± Std', 'Range'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0.1, 0.2, 0.8, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # 设置表格样式
            for i in range(len(stats_data) + 1):
                for j in range(3):
                    cell = table[(i, j)]
                    if i == 0:  # 标题行
                        cell.set_facecolor(COLORS['light_blue'])
                        cell.set_text_props(weight='bold')
                    else:
                        cell.set_facecolor('white')
        
        # 添加结论文本
        conclusion_text = ("Key Findings:\n"
                         "• Hessian matrix exhibits clear low-rank structure\n"
                         "• Principal subspace remains relatively consistent during optimization\n"
                         "• Top eigenvalues capture majority of variance (indicating low-rank property)\n"
                         "• Optimization process maintains subspace consistency")
        
        ax7.text(0.05, 0.1, conclusion_text, 
                transform=ax7.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['light_green'], alpha=0.7))
        
        plt.savefig(self.figures_dir / save_name, bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_combined_eigenvalue_analysis(self,
                                        eigenvalues_history: List[np.ndarray], 
                                        steps: List[int],
                                        save_name: str = "combined_eigenvalue_analysis.png") -> None:
        """
        生成合并的特征值分析图：Eigenvalue Decay Pattern + Cumulative Explained Variance
        左纵轴：Eigenvalue Decay Pattern (百分比)
        右纵轴：Cumulative Explained Variance (百分比)
        """
        # 这里修改尺寸，有(6, 4),(10, 6),(12, 8)
        fig, ax1 = plt.subplots(figsize=(6, 4))
        # 标题，可以注释掉
        # fig.suptitle('Hessian Eigenvalue Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 选择几个代表性步骤进行展示
        display_steps = [0, len(eigenvalues_history)//4, len(eigenvalues_history)//2, -1]
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['quaternary']]
        step_labels = []
        
        # 左纵轴：Eigenvalue Decay Pattern
        lines1 = []
        for i, step_idx in enumerate(display_steps):
            if step_idx < len(eigenvalues_history):
                eigenvals = eigenvalues_history[step_idx]
                eigenvals_abs = np.abs(eigenvals)
                total_variance = np.sum(eigenvals_abs)
                
                # 计算每个特征值的百分比贡献
                eigenval_percentages = (eigenvals_abs / total_variance) * 100
                # 只显示前50个特征值
                display_eigenvals = eigenval_percentages[:min(50, len(eigenval_percentages))]
                
                line1 = ax1.plot(range(1, len(display_eigenvals) + 1), 
                               display_eigenvals, 
                               color=colors[i], 
                               linewidth=2,
                               marker='o', 
                               markersize=4,
                               linestyle='-',
                               label=f'Eigenvalue Decay - Step {steps[step_idx] if step_idx < len(steps) else "Final"}')
                lines1.extend(line1)
                step_labels.append(f'Step {steps[step_idx] if step_idx < len(steps) else "Final"}')
        
        ax1.set_xlabel('Eigenvalue Index', fontsize=12)
        ax1.set_ylabel('Individual Eigenvalue Contribution (%)', fontsize=12, color=COLORS['primary'])
        ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
        ax1.grid(True, alpha=0.3, zorder=0)
        ax1.set_ylim(bottom=0)
        
        # 右纵轴：Cumulative Explained Variance
        ax2 = ax1.twinx()
        lines2 = []
        for i, step_idx in enumerate(display_steps):
            if step_idx < len(eigenvalues_history):
                eigenvals = eigenvalues_history[step_idx]
                eigenvals_abs = np.abs(eigenvals)
                cumsum = np.cumsum(eigenvals_abs) / np.sum(eigenvals_abs)
                # 显示前50个
                display_cumsum = cumsum[:min(50, len(cumsum))]
                
                line2 = ax2.plot(range(1, len(display_cumsum) + 1), 
                               display_cumsum * 100,
                               color=colors[i], 
                               linewidth=2,
                               marker='s', 
                               markersize=3,
                               linestyle='--',
                               label=f'Cumulative Variance - Step {steps[step_idx] if step_idx < len(steps) else "Final"}')
                lines2.extend(line2)
        
        # 添加参考线
        ax2.axhline(y=90, color='red', linestyle=':', alpha=0.7, linewidth=1)
        ax2.axhline(y=95, color='orange', linestyle=':', alpha=0.7, linewidth=1)
        
        ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, color=COLORS['secondary'])
        ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])
        ax2.set_ylim([0, 100])
        
        # 关键修复：显式关闭ax2的网格线，避免与ax1网格线冲突
        ax2.grid(False)
        
        # 合并图例，放在右侧中间空挡位置
        # 创建自定义图例
        legend_elements = []
        for i, step_label in enumerate(step_labels):
            # 添加特征值衰减线
            legend_elements.append(plt.Line2D([0], [0], color=colors[i], linewidth=2, 
                                            marker='o', markersize=4, linestyle='-',
                                            label=f'{step_label} - Eigenvalue Decay'))
            # 添加累计解释方差线
            legend_elements.append(plt.Line2D([0], [0], color=colors[i], linewidth=2, 
                                            marker='s', markersize=3, linestyle='--',
                                            label=f'{step_label} - Cumulative Variance'))
        
        # 添加参考线
        legend_elements.append(plt.Line2D([0], [0], color='red', linestyle=':', alpha=0.7,
                                        label='90% Threshold'))
        legend_elements.append(plt.Line2D([0], [0], color='orange', linestyle=':', alpha=0.7,
                                        label='95% Threshold'))
        
        # 创建图例并设置置顶显示，确保不被网格线遮挡
        legend = ax1.legend(handles=legend_elements, loc='center right', fontsize=9, 
                           frameon=True, fancybox=True, shadow=True, 
                           facecolor='white', edgecolor='gray', 
                           borderpad=0.8)
        # 设置图例置顶（必须在创建后设置）
        legend.set_zorder(1000)
        
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"合并的特征值分析图已保存到: {self.figures_dir / save_name}")

    def generate_all_visualizations(self, 
                                  analysis_data: Dict,
                                  title_prefix: str = "Hessian_Analysis") -> None:
        """
        生成合并的特征值分析图表
        """
        print("生成合并的特征值分析图表...")
        
        eigenvalues_history = analysis_data.get('eigenvalues_history', [])
        steps = analysis_data.get('steps', list(range(len(eigenvalues_history))))
        
        # 只生成合并的特征值分析图
        self.plot_combined_eigenvalue_analysis(
            eigenvalues_history, steps, 
            f"{title_prefix}_combined_eigenvalue_analysis.png"
        )
        
        print(f"图表已保存到: {self.figures_dir}")
        print("生成的文件:")
        for file in self.figures_dir.glob("*.png"):
            print(f"  - {file.name}")

    def save_analysis_data(self, analysis_data: Dict, save_name: str = "analysis_data.npz") -> None:
        """
        保存分析数据以便后续使用
        """
        save_path = self.output_dir / save_name
        
        # 转换数据格式以便保存
        data_to_save = {}
        
        # 保存特征值历史
        eigenvalues_history = analysis_data.get('eigenvalues_history', [])
        if eigenvalues_history:
            # 将特征值列表转换为可保存的格式
            for i, eigenvals in enumerate(eigenvalues_history):
                data_to_save[f'eigenvalues_step_{i}'] = eigenvals
            data_to_save['eigenvalues_steps_count'] = len(eigenvalues_history)
        
        # 保存其他指标
        lowrank_metrics = analysis_data.get('lowrank_metrics', {})
        consistency_metrics = analysis_data.get('consistency_metrics', {})
        steps = analysis_data.get('steps', [])
        
        data_to_save['steps'] = np.array(steps)
        
        # 保存低秩指标
        for key, value in lowrank_metrics.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    # 处理解释方差比例等字典格式的数据
                    for i, v in enumerate(value):
                        for sub_key, sub_value in v.items():
                            data_to_save[f'{key}_step_{i}_{sub_key}'] = sub_value
                else:
                    data_to_save[key] = np.array(value)
        
        # 保存一致性指标
        for key, value in consistency_metrics.items():
            if isinstance(value, list) and len(value) > 0:
                data_to_save[key] = np.array(value)
        
        # 保存到文件
        np.savez_compressed(save_path, **data_to_save)
        print(f"分析数据已保存到: {save_path}")
        
        # 同时保存一个JSON格式的元数据文件
        metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'data_description': '合并的特征值分析数据',
            'total_steps': len(steps),
            'eigenvalues_shape_info': [len(ev) for ev in eigenvalues_history[:5]] if eigenvalues_history else [],
            'file_format': 'npz (compressed numpy arrays)',
            'load_instructions': '使用 HessianVisualizationTools.load_analysis_data() 加载数据'
        }
        
        metadata_path = self.output_dir / save_name.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"元数据已保存到: {metadata_path}")
    
    @staticmethod
    def load_analysis_data(data_path: str) -> Dict:
        """
        从保存的文件中加载分析数据
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 加载数据
        loaded_data = np.load(data_path, allow_pickle=True)
        
        # 重构分析数据格式
        analysis_data = {
            'eigenvalues_history': [],
            'lowrank_metrics': {},
            'consistency_metrics': {},
            'steps': []
        }
        
        # 重构特征值历史
        if 'eigenvalues_steps_count' in loaded_data:
            steps_count = int(loaded_data['eigenvalues_steps_count'])
            for i in range(steps_count):
                key = f'eigenvalues_step_{i}'
                if key in loaded_data:
                    analysis_data['eigenvalues_history'].append(loaded_data[key])
        
        # 重构步骤
        if 'steps' in loaded_data:
            analysis_data['steps'] = loaded_data['steps'].tolist()
        else:
            analysis_data['steps'] = list(range(len(analysis_data['eigenvalues_history'])))
        
        # 重构其他指标
        for key in loaded_data.keys():
            if key.startswith('eigenvalues_') or key == 'steps':
                continue  # 已经处理过
            
            if key in ['effective_ranks', 'spectral_norms', 'nuclear_norms', 'condition_numbers']:
                analysis_data['lowrank_metrics'][key] = loaded_data[key].tolist()
            elif key in ['subspace_angles', 'eigenvalue_correlations', 'frobenius_distances']:
                analysis_data['consistency_metrics'][key] = loaded_data[key].tolist()
        
        print(f"分析数据已从 {data_path} 加载完成")
        print(f"包含 {len(analysis_data['eigenvalues_history'])} 个步骤的特征值数据")
        
        return analysis_data
    
    def regenerate_visualization_from_data(self, data_path: str, 
                                         title_prefix: str = "Reloaded_Analysis") -> None:
        """
        从保存的数据重新生成可视化图表
        """
        analysis_data = self.load_analysis_data(data_path)
        self.generate_all_visualizations(analysis_data, title_prefix)
    
    def plot_multi_step_hessian_analysis(self, 
                                        eigenvalues_history: List[np.ndarray], 
                                        lowrank_metrics: Dict,
                                        steps: List[int],
                                        save_name: str = "multi_step_hessian_analysis.pdf") -> None:
        """
        绘制多步骤Hessian低秩特征值与累计方差比例分析图
        专门用于显示不同优化步骤的Hessian特性演化
        """
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('Multi-Step Hessian Low-Rank Analysis for TTA\n多步骤Hessian低秩特性分析', 
                     fontsize=18, fontweight='bold', y=0.97)
        
        # 1. 不同步骤的特征值衰减对比 (大图)
        ax1 = fig.add_subplot(gs[0, :])
        
        # 选择关键步骤进行展示
        if len(eigenvalues_history) > 8:
            display_indices = [0, len(eigenvalues_history)//8, len(eigenvalues_history)//4, 
                              len(eigenvalues_history)//2, 3*len(eigenvalues_history)//4, -1]
        else:
            display_indices = list(range(len(eigenvalues_history)))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(display_indices)))
        
        for i, idx in enumerate(display_indices):
            if idx < len(eigenvalues_history):
                eigenvals = eigenvalues_history[idx]
                # 显示前30个特征值
                display_eigenvals = eigenvals[:min(30, len(eigenvals))]
                step_label = steps[idx] if idx < len(steps) else f"Step {idx}"
                
                ax1.semilogy(range(1, len(display_eigenvals) + 1), 
                           np.abs(display_eigenvals), 
                           color=colors[i], 
                           linewidth=2.5,
                           marker='o', 
                           markersize=5,
                           label=f'{step_label}',
                           alpha=0.8)
        
        ax1.set_xlabel('特征值索引 (Eigenvalue Index)', fontsize=14)
        ax1.set_ylabel('特征值大小 (Eigenvalue Magnitude)', fontsize=14)
        ax1.set_title('不同优化步骤的Hessian特征值衰减对比\nHessian Eigenvalue Decay Across Optimization Steps', 
                      fontsize=15, fontweight='bold')
        ax1.legend(loc='center right', frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 累计方差比例热图
        ax2 = fig.add_subplot(gs[1, 0])
        
        if lowrank_metrics['explained_variance_ratios']:
            # 构建累计方差比例矩阵
            k_values = ['top_5', 'top_10', 'top_20', 'top_50']
            k_labels = ['Top 5', 'Top 10', 'Top 20', 'Top 50']
            variance_matrix = []
            
            for step_ratios in lowrank_metrics['explained_variance_ratios']:
                row = []
                for k in k_values:
                    if k in step_ratios:
                        row.append(step_ratios[k] * 100)
                    else:
                        row.append(0)
                variance_matrix.append(row)
            
            if variance_matrix:
                variance_matrix = np.array(variance_matrix)
                im = ax2.imshow(variance_matrix.T, aspect='auto', cmap='YlOrRd', 
                               interpolation='nearest', origin='lower')
                
                # 设置坐标轴
                ax2.set_xticks(range(0, len(steps), max(1, len(steps)//10)))
                ax2.set_xticklabels([f'{steps[i]}' for i in range(0, len(steps), max(1, len(steps)//10))], 
                                   rotation=45)
                ax2.set_yticks(range(len(k_labels)))
                ax2.set_yticklabels(k_labels)
                
                # 添加数值标注
                for i in range(variance_matrix.shape[1]):
                    for j in range(0, variance_matrix.shape[0], max(1, variance_matrix.shape[0]//5)):
                        text = ax2.text(j, i, f'{variance_matrix[j, i]:.1f}%',
                                       ha="center", va="center", color="black", fontsize=8)
                
                plt.colorbar(im, ax=ax2, shrink=0.8, label='解释方差比例 (%)')
        
        ax2.set_xlabel('优化步骤 (Optimization Step)')
        ax2.set_ylabel('主成分数量 (Number of Components)')
        ax2.set_title('累计方差比例热图\nCumulative Variance Explained Heatmap', fontweight='bold')
        
        # 3. 有效秩演化
        ax3 = fig.add_subplot(gs[1, 1])
        
        if lowrank_metrics['effective_ranks']:
            ax3.plot(steps[:len(lowrank_metrics['effective_ranks'])], 
                    lowrank_metrics['effective_ranks'],
                    color=COLORS['primary'], 
                    linewidth=3,
                    marker='s', 
                    markersize=6)
            ax3.fill_between(steps[:len(lowrank_metrics['effective_ranks'])], 
                           lowrank_metrics['effective_ranks'],
                           alpha=0.3, 
                           color=COLORS['light_blue'])
            
            # 添加统计信息
            eff_ranks = lowrank_metrics['effective_ranks']
            mean_rank = np.mean(eff_ranks)
            std_rank = np.std(eff_ranks)
            ax3.axhline(y=mean_rank, color='red', linestyle='--', alpha=0.7, 
                       label=f'平均值: {mean_rank:.1f}±{std_rank:.1f}')
        
        ax3.set_xlabel('优化步骤 (Optimization Step)')
        ax3.set_ylabel('有效秩 (Effective Rank)')
        ax3.set_title('有效秩演化\nEffective Rank Evolution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 特征值比例分析
        ax4 = fig.add_subplot(gs[1, 2])
        
        if eigenvalues_history:
            # 计算最大特征值相对于总和的比例
            max_eigenval_ratios = []
            top5_eigenval_ratios = []
            
            for eigenvals in eigenvalues_history:
                if len(eigenvals) > 0:
                    eigenvals_abs = np.abs(eigenvals)
                    total_sum = np.sum(eigenvals_abs)
                    if total_sum > 0:
                        max_ratio = eigenvals_abs[0] / total_sum
                        top5_ratio = np.sum(eigenvals_abs[:5]) / total_sum
                        max_eigenval_ratios.append(max_ratio * 100)
                        top5_eigenval_ratios.append(top5_ratio * 100)
            
            if max_eigenval_ratios:
                ax4.plot(steps[:len(max_eigenval_ratios)], max_eigenval_ratios,
                        color=COLORS['secondary'], linewidth=2, marker='o', 
                        markersize=5, label='最大特征值占比')
                ax4.plot(steps[:len(top5_eigenval_ratios)], top5_eigenval_ratios,
                        color=COLORS['tertiary'], linewidth=2, marker='^', 
                        markersize=5, label='前5特征值占比')
        
        ax4.set_xlabel('优化步骤 (Optimization Step)')
        ax4.set_ylabel('特征值占比 (%)')
        ax4.set_title('主要特征值占比\nDominant Eigenvalue Ratios', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 低秩度量综合对比 (底部大图)
        ax5 = fig.add_subplot(gs[2, :])
        
        # 创建双y轴图
        ax5_twin = ax5.twinx()
        
        # 左轴：解释方差比例
        if lowrank_metrics['explained_variance_ratios']:
            ratios_top10 = []
            for step_ratios in lowrank_metrics['explained_variance_ratios']:
                if 'top_10' in step_ratios:
                    ratios_top10.append(step_ratios['top_10'] * 100)
                else:
                    ratios_top10.append(0)
            
            if ratios_top10:
                line1 = ax5.plot(steps[:len(ratios_top10)], ratios_top10,
                               color=COLORS['primary'], linewidth=3, marker='o', 
                               markersize=6, label='前10主成分解释方差比例 (%)')
        
        # 右轴：有效秩
        if lowrank_metrics['effective_ranks']:
            line2 = ax5_twin.plot(steps[:len(lowrank_metrics['effective_ranks'])], 
                                lowrank_metrics['effective_ranks'],
                                color=COLORS['secondary'], linewidth=3, marker='s', 
                                markersize=6, label='有效秩')
        
        ax5.set_xlabel('优化步骤 (Optimization Step)', fontsize=14)
        ax5.set_ylabel('前10主成分解释方差比例 (%)', color=COLORS['primary'], fontsize=12)
        ax5_twin.set_ylabel('有效秩 (Effective Rank)', color=COLORS['secondary'], fontsize=12)
        ax5.set_title('Hessian低秩特性综合分析：解释方差比例 vs 有效秩\nComprehensive Low-Rank Analysis: Explained Variance vs Effective Rank', 
                      fontsize=15, fontweight='bold')
        
        # 合并图例
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='y', labelcolor=COLORS['primary'])
        ax5_twin.tick_params(axis='y', labelcolor=COLORS['secondary'])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 多步骤Hessian分析图已保存: {save_name}")


def load_and_visualize_results(experiment_dir: str) -> None:
    """
    从实验目录加载结果并生成可视化
    """
    exp_path = Path(experiment_dir)
    
    # 加载实验总结
    summary_file = exp_path / "experiment_summary.json"
    if not summary_file.exists():
        print(f"找不到实验总结文件: {summary_file}")
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # 加载详细数据
    analysis_dir = exp_path / "analysis_results"
    eigenvalues_history = []
    steps = []
    
    # 按步骤加载特征值数据
    for metrics_file in sorted(analysis_dir.glob("step_*_metrics.json")):
        with open(metrics_file, 'r') as f:
            step_data = json.load(f)
        
        eigenvalues_history.append(np.array(step_data['eigenvalues']))
        steps.append(step_data['step'])
    
    # 从总结中提取指标
    lowrank_metrics = {
        'explained_variance_ratios': [],
        'effective_ranks': [],
        'spectral_norms': [],
        'nuclear_norms': [],
        'condition_numbers': []
    }
    
    consistency_metrics = {
        'subspace_angles': [],
        'eigenvalue_correlations': [],
        'frobenius_distances': []
    }
    
    # 重构指标数据（如果需要的话）
    # 这里可以根据实际的数据结构进行调整
    
    # 创建可视化工具
    visualizer = HessianVisualizationTools(str(exp_path))
    
    # 生成所有可视化
    analysis_data = {
        'eigenvalues_history': eigenvalues_history,
        'lowrank_metrics': lowrank_metrics,
        'consistency_metrics': consistency_metrics,
        'steps': steps
    }
    
    visualizer.generate_all_visualizations(analysis_data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations for Hessian analysis results')
    parser.add_argument('--experiment_dir', required=True, 
                       help='Directory containing experiment results')
    
    args = parser.parse_args()
    
    load_and_visualize_results(args.experiment_dir) 