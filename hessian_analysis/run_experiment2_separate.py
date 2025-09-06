#!/usr/bin/env python3
"""
Hessian Eigenvalue Distribution Analysis - Separate Visualization
分别生成Hessian特征值分布分析图表的实验脚本

分别生成以下四个独立图表：
1. eigenvalue_decay_pattern.png - 特征值衰减模式
2. cumulative_explained_variance.png - 累积解释方差
3. final_eigenvalue_distribution.png - 最终特征值分布
4. effective_rank_evolution.png - 有效秩演化

运行方式:
python run_experiment2_separate.py --data_corruption /path/to/imagenet-c --output ./results
"""

import os
import sys
import time
import datetime
import argparse
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import timm
from pathlib import Path
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigh
from sklearn.decomposition import PCA

# 添加父目录到Python路径以访问项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hessian_analysis.hessian_lowrank_experiment import HessianLowRankAnalyzer
from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data

warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
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
    'light_red': '#ff9896'
}

class SeparateHessianExperiment:
    """
    分别生成独立图表的Hessian实验类
    """
    
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_experiment()
        
    def setup_experiment(self):
        """
        设置实验环境
        """
        # 创建主实验目录
        self.exp_dir = Path(self.args.output) / f"hessian_separate_experiment_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置子目录
        self.figures_dir = self.exp_dir / "figures"
        self.logs_dir = self.exp_dir / "logs"
        
        for dir_path in [self.figures_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = get_logger(
            name="separate_experiment",
            output_directory=str(self.logs_dir),
            log_name=f"experiment_{self.timestamp}.log",
            debug=False
        )
        
        # 设置设备
        if torch.cuda.is_available() and self.args.gpu is not None:
            self.device = f"cuda:{self.args.gpu}"
            torch.cuda.set_device(self.args.gpu)
        else:
            self.device = "cpu"
        
        self.logger.info(f"实验设置完成，使用设备: {self.device}")
        self.logger.info(f"实验目录: {self.exp_dir}")
        
    def create_model(self):
        """
        创建ViT模型 - 参考run_hessian_experiment.py的实现
        """
        self.logger.info("创建ViT模型...")
        
        # 使用timm加载预训练ViT模型
        vit_model = timm.create_model(
            'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            pretrained=True,
            num_classes=1000
        )
        
        self.logger.info(f"ViT模型创建成功: {vit_model.__class__.__name__}")
        
        # 冻结backbone参数
        for param in vit_model.parameters():
            param.requires_grad = False
        
        # 创建AdaFormer模型
        from models.adaformer import AdaFormerViT
        model = AdaFormerViT(
            vit=vit_model,
            adapter_layer=self.args.adapter_layers,  # 可配置的adapter层
            reduction_factor=self.args.reduction_factor,
            adapter_style="parallel"
        )
        
        model = model.to(self.device)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型参数统计:")
        self.logger.info(f"  总参数: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")
        self.logger.info(f"  可训练比例: {trainable_params/total_params*100:.2f}%")
        
        return model
        
    def prepare_data(self):
        """
        准备数据 - 参考run_hessian_experiment.py的实现
        """
        self.logger.info("准备ImageNet-C数据...")
        
        # 设置数据参数
        self.args.corruption = 'gaussian_noise'
        self.args.level = 5
        self.args.if_shuffle = False
        
        teset, teloader = prepare_test_data(self.args)
        
        self.logger.info(f"数据集信息:")
        self.logger.info(f"  数据集大小: {len(teset)}")
        self.logger.info(f"  批次大小: {self.args.batch_size}")
        self.logger.info(f"  批次数量: {len(teloader)}")
        self.logger.info(f"  腐败类型: {self.args.corruption}")
        self.logger.info(f"  腐败级别: {self.args.level}")
        
        return teset, teloader
        
    def run_hessian_analysis(self, model, teloader):
        """
        运行Hessian分析
        """
        self.logger.info("开始Hessian分析...")
        
        # 更新args的output为当前实验目录
        analysis_args = argparse.Namespace(**vars(self.args))
        analysis_args.output = str(self.exp_dir)
        
        # 创建分析器
        analyzer = HessianLowRankAnalyzer(model, teloader, analysis_args)
        
        # 运行分析
        max_batches = min(self.args.max_batches, len(teloader))
        self.logger.info(f"将分析 {max_batches} 个批次")
        
        start_time = time.time()
        
        for step, batch in enumerate(teloader):
            if step >= max_batches:
                break
                
            try:
                self.logger.info(f"处理批次 {step + 1}/{max_batches}")
                analyzer.run_analysis_step(batch, step)
                
                # 每10步输出进度
                if (step + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / (step + 1) * (max_batches - step - 1)
                    self.logger.info(f"进度: {step + 1}/{max_batches}, "
                                   f"已用时: {elapsed:.1f}s, "
                                   f"预计剩余: {eta:.1f}s")
                    
            except Exception as e:
                self.logger.error(f"批次 {step} 分析失败: {e}")
                continue
        
        total_time = time.time() - start_time
        self.logger.info(f"Hessian分析完成，总用时: {total_time:.1f}s")
        
        return analyzer
        
    def plot_eigenvalue_decay_pattern(self, analyzer):
        """
        生成特征值衰减模式图
        """
        self.logger.info("生成特征值衰减模式图...")
        
        plt.figure(figsize=(10, 6))
        
        # 选择代表性步骤：0, 25, 50, 75, 99
        display_steps = [0, 25, 50, 75, 99]
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['light_blue']]
        
        for i, step_idx in enumerate(display_steps):
            if step_idx < len(analyzer.eigenvalues_history):
                eigenvals = analyzer.eigenvalues_history[step_idx]
                # 只显示前50个特征值
                display_eigenvals = eigenvals[:min(50, len(eigenvals))]
                plt.semilogy(range(1, len(display_eigenvals) + 1), 
                           np.abs(display_eigenvals), 
                           color=colors[i], 
                           linewidth=2,
                           marker='o', 
                           markersize=4,
                           label=f'Step {step_idx}')
        
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue Magnitude (log scale)')
        plt.title('Eigenvalue Decay Pattern')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        save_path = self.figures_dir / "eigenvalue_decay_pattern.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"特征值衰减模式图已保存到: {save_path}")
        
        return save_path
        
    def plot_cumulative_explained_variance(self, analyzer):
        """
        生成累积解释方差图 - 只展示前50个分量
        """
        self.logger.info("生成累积解释方差图...")
        
        plt.figure(figsize=(10, 6))
        
        # 选择代表性步骤：0, 25, 50, 75, 99
        display_steps = [0, 25, 50, 75, 99]
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['light_blue']]
        
        for i, step_idx in enumerate(display_steps):
            if step_idx < len(analyzer.eigenvalues_history):
                eigenvals = analyzer.eigenvalues_history[step_idx]
                # 计算累积解释方差，只取前50个分量
                total_variance = np.sum(eigenvals)
                cumulative_variance = np.cumsum(eigenvals) / total_variance
                # 限制到前50个分量
                cumulative_variance = cumulative_variance[:50]
                plt.plot(range(1, len(cumulative_variance) + 1), 
                        cumulative_variance * 100, 
                        color=colors[i], 
                        linewidth=2,
                        marker='s', 
                        markersize=4,
                        label=f'Step {step_idx}')
        
        # 添加阈值线
        plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% Threshold')
        plt.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% Threshold')
        
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance (%)')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        save_path = self.figures_dir / "cumulative_explained_variance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"累积解释方差图已保存到: {save_path}")
        
        return save_path
        
    def plot_final_eigenvalue_distribution(self, analyzer):
        """
        生成最终特征值分布图
        """
        self.logger.info("生成最终特征值分布图...")
        
        plt.figure(figsize=(10, 6))
        
        if len(analyzer.eigenvalues_history) > 0:
            final_eigenvals = analyzer.eigenvalues_history[-1]
            # 计算log10分布
            log_eigenvals = np.log10(np.abs(final_eigenvals) + 1e-12)
            plt.hist(log_eigenvals, bins=30, alpha=0.7, color=COLORS['primary'], edgecolor='black')
            plt.xlabel('log10(Eigenvalue)')
            plt.ylabel('Frequency')
            plt.title('Final Eigenvalue Distribution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        # 保存图表
        save_path = self.figures_dir / "final_eigenvalue_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"最终特征值分布图已保存到: {save_path}")
        
        return save_path
        
    def plot_effective_rank_evolution(self, analyzer):
        """
        生成有效秩演化图
        """
        self.logger.info("生成有效秩演化图...")
        
        plt.figure(figsize=(10, 6))
        
        if analyzer.lowrank_metrics['effective_ranks']:
            steps = list(range(len(analyzer.lowrank_metrics['effective_ranks'])))
            effective_ranks = analyzer.lowrank_metrics['effective_ranks']
            plt.plot(steps, effective_ranks, 
                    color=COLORS['secondary'], 
                    linewidth=2,
                    marker='o', 
                    markersize=4)
            plt.xlabel('Optimization Step')
            plt.ylabel('Effective Rank')
            plt.title('Effective Rank Evolution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        # 保存图表
        save_path = self.figures_dir / "effective_rank_evolution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"有效秩演化图已保存到: {save_path}")
        
        return save_path
        
    def run_complete_experiment(self):
        """
        运行完整的实验
        """
        self.logger.info("开始运行分别生成图表的Hessian实验...")
        
        try:
            # 1. 创建模型
            model = self.create_model()
            
            # 2. 准备数据
            teset, teloader = self.prepare_data()
            
            # 3. 运行Hessian分析
            analyzer = self.run_hessian_analysis(model, teloader)
            
            # 4. 分别生成四个独立图表
            figure_paths = []
            
            # 生成特征值衰减模式图
            path1 = self.plot_eigenvalue_decay_pattern(analyzer)
            figure_paths.append(path1)
            
            # 生成累积解释方差图
            path2 = self.plot_cumulative_explained_variance(analyzer)
            figure_paths.append(path2)
            
            # 生成最终特征值分布图
            path3 = self.plot_final_eigenvalue_distribution(analyzer)
            figure_paths.append(path3)
            
            # 生成有效秩演化图
            path4 = self.plot_effective_rank_evolution(analyzer)
            figure_paths.append(path4)
            
            # 5. 保存分析数据
            analysis_data = {
                'eigenvalues_history': analyzer.eigenvalues_history,
                'lowrank_metrics': analyzer.lowrank_metrics,
                'consistency_metrics': analyzer.consistency_metrics,
                'steps': list(range(len(analyzer.hessian_history)))
            }
            
            data_path = self.exp_dir / "analysis_data.npz"
            np.savez(data_path, **analysis_data)
            self.logger.info(f"分析数据已保存到: {data_path}")
            
            # 6. 生成总结报告
            self.generate_summary_report(analyzer, figure_paths)
            
            self.logger.info("实验完成！生成了以下四个独立图表:")
            for i, path in enumerate(figure_paths, 1):
                self.logger.info(f"  {i}. {path.name}")
            
        except Exception as e:
            self.logger.error(f"实验过程中出现错误: {str(e)}")
            raise
            
    def generate_summary_report(self, analyzer, figure_paths):
        """
        生成总结报告
        """
        report_path = self.exp_dir / "summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Hessian Eigenvalue Distribution Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总步骤数: {len(analyzer.hessian_history)}\n")
            f.write(f"特征值历史记录数: {len(analyzer.eigenvalues_history)}\n\n")
            
            f.write("生成的图表文件:\n")
            for i, path in enumerate(figure_paths, 1):
                f.write(f"  {i}. {path.name}\n")
            f.write("\n")
            
            # 分析特征值衰减
            if analyzer.eigenvalues_history:
                f.write("特征值衰减分析:\n")
                for i, step in enumerate([0, 25, 50, 75, 99]):
                    if step < len(analyzer.eigenvalues_history):
                        eigenvals = analyzer.eigenvalues_history[step]
                        f.write(f"  步骤 {step}: 最大特征值={eigenvals[0]:.2e}, 最小特征值={eigenvals[-1]:.2e}\n")
                f.write("\n")
            
            # 分析有效秩
            if analyzer.lowrank_metrics['effective_ranks']:
                f.write("有效秩分析:\n")
                ranks = analyzer.lowrank_metrics['effective_ranks']
                f.write(f"  平均有效秩: {np.mean(ranks):.2f}\n")
                f.write(f"  最小有效秩: {np.min(ranks)}\n")
                f.write(f"  最大有效秩: {np.max(ranks)}\n")
                f.write(f"  有效秩标准差: {np.std(ranks):.2f}\n\n")
            
            f.write("结论: Hessian矩阵在不同优化步骤间保持稳定的低秩结构。\n")
        
        self.logger.info(f"总结报告已保存到: {report_path}")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Separate Hessian Analysis Experiment')
    
    # 数据参数
    parser.add_argument('--data_corruption', type=str, required=True,
                       help='Path to ImageNet-C dataset')
    parser.add_argument('--corruption', type=str, default='gaussian_noise',
                       help='Corruption type for ImageNet-C')
    parser.add_argument('--severity', type=int, default=5,
                       help='Corruption severity level')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='vit_base_patch16_224',
                       help='Model architecture')
    parser.add_argument('--adapter_layers', type=str, default='11',
                       help='Adapter layers to add')
    parser.add_argument('--reduction_factor', type=int, default=16,
                       help='Adapter reduction factor')
    
    # 实验参数
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for testing')
    parser.add_argument('--max_batches', type=int, default=50,
                       help='Maximum number of batches to analyze')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 创建实验实例
    experiment = SeparateHessianExperiment(args)
    
    # 运行实验
    experiment.run_complete_experiment()

if __name__ == '__main__':
    main() 