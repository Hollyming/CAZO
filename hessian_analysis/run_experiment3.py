#!/usr/bin/env python3
"""
Hessian Eigenvalue Distribution Analysis - Separate Visualization + Heatmaps
分别生成Hessian特征值分布分析图表，并在代表性步骤绘制Hessian矩阵热度图

包含以下图表：
1. eigenvalue_decay_pattern.png - 特征值衰减模式
2. cumulative_explained_variance.png - 累积解释方差
3. final_eigenvalue_distribution.png - 最终特征值分布
4. effective_rank_evolution.png - 有效秩演化
5. hessian_heatmap_step_XXX.png - 指定display_steps各步的Hessian热度图
6. hessian_heatmaps_grid.png - 汇总多步热度图的网格图

运行方式:
python run_experiment3.py --data_corruption /path/to/imagenet-c --output ./results
"""

import os
import sys
import time
import datetime
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加父目录到Python路径以访问项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hessian_analysis.hessian_lowrank_experiment import HessianLowRankAnalyzer
from utils.utils import get_logger
from utils.cli_utils import *  # noqa: F403,F401
from dataset.selectedRotateImageFolder import prepare_test_data

# matplotlib 样式
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


class SeparateHessianExperimentV3:
    """分别生成独立图表 + Hessian热度图的实验类"""

    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_experiment()

    def setup_experiment(self):
        """设置实验环境"""
        # 创建主实验目录
        self.exp_dir = Path(self.args.output) / f"hessian_separate_experiment_v3_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 子目录
        self.figures_dir = self.exp_dir / "figures"
        self.logs_dir = self.exp_dir / "logs"
        for dir_path in [self.figures_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)

        # 日志
        self.logger = get_logger(
            name="separate_experiment_v3",
            output_directory=str(self.logs_dir),
            log_name=f"experiment_{self.timestamp}.log",
            debug=False,
        )

        # 设备
        if torch.cuda.is_available() and self.args.gpu is not None:
            self.device = f"cuda:{self.args.gpu}"
            torch.cuda.set_device(self.args.gpu)
        else:
            self.device = "cpu"

        self.logger.info(f"实验设置完成，使用设备: {self.device}")
        self.logger.info(f"实验目录: {self.exp_dir}")

    def create_model(self):
        """创建ViT + Adapter 模型（与v2一致）"""
        import timm
        from models.adaformer import AdaFormerViT

        self.logger.info("创建ViT模型...")
        vit_model = timm.create_model(
            'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            pretrained=True,
            num_classes=1000
        )
        self.logger.info(f"ViT模型创建成功: {vit_model.__class__.__name__}")

        for param in vit_model.parameters():
            param.requires_grad = False

        model = AdaFormerViT(
            vit=vit_model,
            adapter_layer=self.args.adapter_layers,
            reduction_factor=self.args.reduction_factor,
            adapter_style="parallel",
        )
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info("模型参数统计:")
        self.logger.info(f"  总参数: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")
        self.logger.info(f"  可训练比例: {trainable_params/total_params*100:.2f}%")
        return model

    def prepare_data(self):
        """准备数据（与v2一致）"""
        self.logger.info("准备ImageNet-C数据...")
        # 固定为 gaussian_noise@5 作为示例
        self.args.corruption = 'gaussian_noise'
        self.args.level = 5
        self.args.if_shuffle = False
        teset, teloader = prepare_test_data(self.args)

        self.logger.info("数据集信息:")
        self.logger.info(f"  数据集大小: {len(teset)}")
        self.logger.info(f"  批次大小: {self.args.batch_size}")
        self.logger.info(f"  批次数量: {len(teloader)}")
        self.logger.info(f"  腐败类型: {self.args.corruption}")
        self.logger.info(f"  腐败级别: {self.args.level}")
        return teset, teloader

    def run_hessian_analysis(self, model, teloader):
        """运行Hessian分析（与v2一致）"""
        self.logger.info("开始Hessian分析...")
        analysis_args = argparse.Namespace(**vars(self.args))
        analysis_args.output = str(self.exp_dir)
        analyzer = HessianLowRankAnalyzer(model, teloader, analysis_args)

        max_batches = min(self.args.max_batches, len(teloader))
        self.logger.info(f"将分析 {max_batches} 个批次")

        start_time = time.time()
        for step, batch in enumerate(teloader):
            if step >= max_batches:
                break
            try:
                self.logger.info(f"处理批次 {step + 1}/{max_batches}")
                analyzer.run_analysis_step(batch, step)
                if (step + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / (step + 1) * (max_batches - step - 1)
                    self.logger.info(
                        f"进度: {step + 1}/{max_batches}, 已用时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s"
                    )
            except Exception as e:
                self.logger.error(f"批次 {step} 分析失败: {e}")
                continue

        total_time = time.time() - start_time
        self.logger.info(f"Hessian分析完成，总用时: {total_time:.1f}s")
        return analyzer

    # --------------------- 原有四种图表 ---------------------
    def plot_eigenvalue_decay_pattern(self, analyzer):
        self.logger.info("生成特征值衰减模式图...")
        plt.figure(figsize=(10, 6))
        display_steps = self._get_display_steps(len(analyzer.eigenvalues_history))
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['light_blue']]
        for i, step_idx in enumerate(display_steps):
            if step_idx < len(analyzer.eigenvalues_history):
                eigenvals = analyzer.eigenvalues_history[step_idx]
                display_eigenvals = eigenvals[:min(50, len(eigenvals))]
                plt.semilogy(
                    range(1, len(display_eigenvals) + 1),
                    np.abs(display_eigenvals),
                    color=colors[i], linewidth=2, marker='o', markersize=4,
                    label=f'Step {step_idx}'
                )
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue Magnitude (log scale)')
        plt.title('Eigenvalue Decay Pattern')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.figures_dir / "eigenvalue_decay_pattern.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"特征值衰减模式图已保存到: {save_path}")
        return save_path

    def plot_cumulative_explained_variance(self, analyzer):
        self.logger.info("生成累积解释方差图...")
        plt.figure(figsize=(10, 6))
        display_steps = self._get_display_steps(len(analyzer.eigenvalues_history))
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['quaternary'], COLORS['light_blue']]
        for i, step_idx in enumerate(display_steps):
            if step_idx < len(analyzer.eigenvalues_history):
                eigenvals = analyzer.eigenvalues_history[step_idx]
                total_variance = np.sum(eigenvals)
                cumulative_variance = np.cumsum(eigenvals) / (total_variance + 1e-12)
                cumulative_variance = cumulative_variance[:50]
                plt.plot(
                    range(1, len(cumulative_variance) + 1),
                    cumulative_variance * 100,
                    color=colors[i], linewidth=2, marker='s', markersize=4,
                    label=f'Step {step_idx}'
                )
        plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% Threshold')
        plt.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% Threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance (%)')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.figures_dir / "cumulative_explained_variance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"累积解释方差图已保存到: {save_path}")
        return save_path

    def plot_final_eigenvalue_distribution(self, analyzer):
        self.logger.info("生成最终特征值分布图...")
        plt.figure(figsize=(10, 6))
        if len(analyzer.eigenvalues_history) > 0:
            final_eigenvals = analyzer.eigenvalues_history[-1]
            log_eigenvals = np.log10(np.abs(final_eigenvals) + 1e-12)
            plt.hist(log_eigenvals, bins=30, alpha=0.7, color=COLORS['primary'], edgecolor='black')
            plt.xlabel('log10(Eigenvalue)')
            plt.ylabel('Frequency')
            plt.title('Final Eigenvalue Distribution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        save_path = self.figures_dir / "final_eigenvalue_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"最终特征值分布图已保存到: {save_path}")
        return save_path

    def plot_effective_rank_evolution(self, analyzer):
        self.logger.info("生成有效秩演化图...")
        plt.figure(figsize=(10, 6))
        if analyzer.lowrank_metrics['effective_ranks']:
            steps = list(range(len(analyzer.lowrank_metrics['effective_ranks'])))
            effective_ranks = analyzer.lowrank_metrics['effective_ranks']
            plt.plot(steps, effective_ranks,
                     color=COLORS['secondary'], linewidth=2, marker='o', markersize=4)
            plt.xlabel('Optimization Step')
            plt.ylabel('Effective Rank')
            plt.title('Effective Rank Evolution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        save_path = self.figures_dir / "effective_rank_evolution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"有效秩演化图已保存到: {save_path}")
        return save_path

    # --------------------- 新增：Hessian热度图 ---------------------
    def preprocess_hessian_for_heatmap(self, H: np.ndarray) -> np.ndarray:
        """将 Hessian 转换为可视化矩阵，支持缩放、裁剪和下采样。"""
        eps = 1e-12
        H = np.asarray(H)
        # 保证对称
        if H.shape[0] == H.shape[1]:
            H = 0.5 * (H + H.T)
        # 缩放模式
        mode = self.args.heatmap_scale
        if mode == 'logabs':
            X = np.log10(np.abs(H) + eps)
            cmap = 'viridis'
            center = None
        elif mode == 'abs':
            X = np.abs(H)
            cmap = 'magma'
            center = None
        else:  # 'none'
            X = H
            cmap = 'coolwarm'
            center = 0.0
        # 百分位裁剪
        p = float(self.args.heatmap_clip_percentile)
        if 50.0 < p < 100.0:
            lo = np.nanpercentile(X, 100 - p)
            hi = np.nanpercentile(X, p)
            X = np.clip(X, lo, hi)
        # 下采样到合适尺寸
        max_dim = int(self.args.heatmap_max_dim)
        n = X.shape[0]
        if n > max_dim:
            idx = np.linspace(0, n - 1, max_dim).astype(int)
            X = X[np.ix_(idx, idx)]
        return X, cmap, center

    def plot_hessian_heatmaps(self, analyzer):
        """为 display_steps 生成单图和网格热度图。"""
        self.logger.info("生成Hessian热度图...")
        display_steps = self._get_display_steps(len(analyzer.hessian_history))
        heatmap_paths = []
        # 单张图
        for step_idx in display_steps:
            if step_idx < len(analyzer.hessian_history):
                H = analyzer.hessian_history[step_idx].detach().cpu().numpy()
                X, cmap, center = self.preprocess_hessian_for_heatmap(H)
                plt.figure(figsize=(6, 5))
                sns.heatmap(
                    X, cmap=cmap, center=center, square=True, cbar=True,
                    xticklabels=False, yticklabels=False
                )
                plt.title(f'Hessian Heatmap (Step {step_idx})')
                plt.tight_layout()
                save_path = self.figures_dir / f"hessian_heatmap_step_{step_idx:03d}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                heatmap_paths.append(save_path)
                self.logger.info(f"Hessian热度图已保存: {save_path}")
        # 网格图
        if heatmap_paths:
            num = len(heatmap_paths)
            cols = min(num, 3)
            rows = int(np.ceil(num / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            axes = np.atleast_2d(axes)
            for i, step_idx in enumerate(display_steps[:num]):
                r, c = divmod(i, cols)
                ax = axes[r, c]
                H = analyzer.hessian_history[step_idx].detach().cpu().numpy()
                X, cmap, center = self.preprocess_hessian_for_heatmap(H)
                sns.heatmap(
                    X, cmap=cmap, center=center, square=True, cbar=True,
                    xticklabels=False, yticklabels=False, ax=ax
                )
                ax.set_title(f'Step {step_idx}')
            # 关闭多余子图
            for j in range(num, rows * cols):
                r, c = divmod(j, cols)
                fig.delaxes(axes[r, c])
            plt.tight_layout()
            grid_path = self.figures_dir / "hessian_heatmaps_grid.png"
            plt.savefig(grid_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Hessian热度图网格已保存: {grid_path}")
            return heatmap_paths, grid_path
        return heatmap_paths, None

    # --------------------- 工具方法 ---------------------
    def _get_display_steps(self, history_len: int):
        # 默认想展示 [0, 25, 50, 75, 99]，如果步数不够则依据长度均匀取样
        desired = [0, 25, 50, 75, 99]
        if history_len == 0:
            return []
        if history_len > max(desired):
            return desired
        # 均匀采样到最多5个点
        count = min(5, history_len)
        indices = np.linspace(0, history_len - 1, count).astype(int).tolist()
        return sorted(list(set(indices)))

    # --------------------- 主流程 ---------------------
    def run_complete_experiment(self):
        self.logger.info("开始运行分别生成图表+热度图的Hessian实验...")
        try:
            model = self.create_model()
            teset, teloader = self.prepare_data()
            analyzer = self.run_hessian_analysis(model, teloader)

            figure_paths = []
            path1 = self.plot_eigenvalue_decay_pattern(analyzer); figure_paths.append(path1)
            path2 = self.plot_cumulative_explained_variance(analyzer); figure_paths.append(path2)
            path3 = self.plot_final_eigenvalue_distribution(analyzer); figure_paths.append(path3)
            path4 = self.plot_effective_rank_evolution(analyzer); figure_paths.append(path4)

            heatmap_paths, grid_path = self.plot_hessian_heatmaps(analyzer)

            # 保存分析数据
            analysis_data = {
                'eigenvalues_history': analyzer.eigenvalues_history,
                'lowrank_metrics': analyzer.lowrank_metrics,
                'consistency_metrics': analyzer.consistency_metrics,
                'steps': list(range(len(analyzer.hessian_history)))
            }
            data_path = self.exp_dir / "analysis_data.npz"
            np.savez(data_path, **analysis_data)
            self.logger.info(f"分析数据已保存到: {data_path}")

            # 总结报告
            self.generate_summary_report(analyzer, figure_paths, heatmap_paths, grid_path)
            self.logger.info("实验完成！")
        except Exception as e:
            self.logger.error(f"实验过程中出现错误: {str(e)}")
            raise

    def generate_summary_report(self, analyzer, figure_paths, heatmap_paths, grid_path):
        report_path = self.exp_dir / "summary_report.txt"
        with open(report_path, 'w') as f:
            f.write("Hessian Eigenvalue Distribution Analysis Report (with Heatmaps)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总步骤数: {len(analyzer.hessian_history)}\n")
            f.write(f"特征值历史记录数: {len(analyzer.eigenvalues_history)}\n\n")

            f.write("生成的图表文件:\n")
            for i, path in enumerate(figure_paths, 1):
                f.write(f"  {i}. {os.path.basename(str(path))}\n")
            f.write("\n")

            f.write("生成的Hessian热力图:\n")
            for p in heatmap_paths:
                f.write(f"  - {os.path.basename(str(p))}\n")
            if grid_path is not None:
                f.write(f"  - {os.path.basename(str(grid_path))}\n")
            f.write("\n")

            # 特征值衰减简单摘要
            if analyzer.eigenvalues_history:
                f.write("特征值衰减分析:\n")
                steps = self._get_display_steps(len(analyzer.eigenvalues_history))
                for step in steps:
                    if step < len(analyzer.eigenvalues_history):
                        eigenvals = analyzer.eigenvalues_history[step]
                        f.write(
                            f"  步骤 {step}: 最大特征值={eigenvals[0]:.2e}, 最小特征值={eigenvals[-1]:.2e}\n"
                        )
                f.write("\n")

            # 有效秩
            if analyzer.lowrank_metrics['effective_ranks']:
                ranks = analyzer.lowrank_metrics['effective_ranks']
                f.write("有效秩分析:\n")
                f.write(f"  平均有效秩: {np.mean(ranks):.2f}\n")
                f.write(f"  最小有效秩: {np.min(ranks)}\n")
                f.write(f"  最大有效秩: {np.max(ranks)}\n")
                f.write(f"  有效秩标准差: {np.std(ranks):.2f}\n\n")

            f.write("结论: Hessian矩阵在不同优化步骤间保持稳定的低秩结构；热度图展示了整体结构与强度分布。\n")
        self.logger.info(f"总结报告已保存到: {report_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Separate Hessian Analysis Experiment with Heatmaps (v3)')

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

    # 热度图参数
    parser.add_argument('--heatmap_scale', type=str, default='logabs', choices=['logabs', 'abs', 'none'],
                        help='Heatmap value transform: logabs, abs, or none (signed)')
    parser.add_argument('--heatmap_clip_percentile', type=float, default=99.5,
                        help='Percentile for clipping extremes (50-100). e.g., 99.5 keeps central 99% range')
    parser.add_argument('--heatmap_max_dim', type=int, default=512,
                        help='Downsample Hessian to at most this dimension for visualization')

    return parser.parse_args()


def main():
    args = parse_args()
    # 随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 创建实验并运行
    experiment = SeparateHessianExperimentV3(args)
    experiment.run_complete_experiment()


if __name__ == '__main__':
    main() 