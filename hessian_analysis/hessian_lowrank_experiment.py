#!/usr/bin/env python3
"""
Hessian Low-Rank Property and Consistency Analysis for Test-Time Adaptation
验证TTA问题中Hessian矩阵的低秩特性和优化过程中的一致性

实验设计:
1. 使用timm预训练的ViT模型
2. 在ImageNet-C gaussian_noise (level=5)上进行TTA
3. 分析Hessian矩阵的低秩特性
4. 验证优化过程中Hessian的一致性
5. 生成专业的论文图表
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
from scipy.linalg import eigh
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle

# 添加父目录到Python路径以访问项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data
from models.adaformer import AdaFormerViT, freeze_vit_parameters
from tta_library.tent import Tent

class HessianLowRankAnalyzer:
    """
    Hessian低秩分析器
    用于分析TTA过程中Hessian矩阵的低秩特性和一致性
    """
    
    def __init__(self, model, data_loader, args):
        self.model = model
        self.data_loader = data_loader
        self.args = args
        self.device = next(model.parameters()).device
        
        # 创建日志和输出目录
        self.setup_directories()
        self.setup_logger()
        
        # 获取adapter参数
        self.setup_adapter_params()
        
        # 初始化存储容器
        self.init_storage()
        
        # 设置计算参数
        self.max_rank_to_analyze = min(50, self.total_params)  # 分析的最大秩
        self.eigenvalue_threshold = 1e-6  # 特征值阈值
        
    def setup_directories(self):
        """设置输出目录"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(self.args.output) / f"hessian_lowrank_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.hessian_dir = self.exp_dir / "hessian_matrices"
        self.analysis_dir = self.exp_dir / "analysis_results"
        self.figures_dir = self.exp_dir / "figures"
        
        for dir_path in [self.hessian_dir, self.analysis_dir, self.figures_dir]:
            dir_path.mkdir(exist_ok=True)
            
    def setup_logger(self):
        """设置日志器"""
        self.logger = get_logger(
            name="hessian_lowrank", 
            output_directory=str(self.exp_dir),
            log_name=f"hessian_lowrank_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            debug=False
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.exp_dir / "tensorboard"))
        
    def setup_adapter_params(self):
        """设置adapter参数"""
        self.adapter_params = []
        self.logger.info("收集adapter参数:")
        
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
                self.adapter_params.append(param)
                self.logger.info(f"添加参数: {name}, shape: {param.shape}")
        
        if not self.adapter_params:
            self.logger.error("没有找到adapter参数!")
            raise ValueError("没有找到adapter参数")
        
        self.param_shapes = [p.shape for p in self.adapter_params]
        self.param_sizes = [p.numel() for p in self.adapter_params]
        self.total_params = sum(self.param_sizes)
        
        self.logger.info(f"Adapter参数总数: {self.total_params}")
        
    def init_storage(self):
        """初始化存储容器"""
        self.hessian_history = []  # 存储Hessian矩阵
        self.eigenvalues_history = []  # 存储特征值
        self.eigenvectors_history = []  # 存储特征向量
        self.step_info = []  # 存储步骤信息
        
        # 低秩分析结果
        self.lowrank_metrics = {
            'explained_variance_ratios': [],
            'effective_ranks': [],
            'spectral_norms': [],
            'nuclear_norms': [],
            'condition_numbers': []
        }
        
        # 一致性分析结果
        self.consistency_metrics = {
            'subspace_angles': [],
            'eigenvalue_correlations': [],
            'frobenius_distances': []
        }
        
    def compute_hessian_fisher(self, batch):
        """
        使用Fisher信息矩阵作为Hessian的近似
        这是计算Hessian的高效方法
        """
        images, targets = batch[0].to(self.device), batch[1].to(self.device)
        batch_size = images.shape[0]
        
        self.model.train()
        self.model.zero_grad()
        
        # 前向传播
        outputs = self.model(images)
        
        # 计算每个样本的梯度
        grads_list = []
        for i in range(batch_size):
            self.model.zero_grad()
            
            # 单样本损失
            sample_loss = F.cross_entropy(outputs[i:i+1], targets[i:i+1])
            
            # 计算梯度
            sample_grads = torch.autograd.grad(
                sample_loss, self.adapter_params,
                create_graph=False, retain_graph=True,
                allow_unused=True
            )
            
            # 处理梯度并展平
            processed_grads = []
            for j, grad in enumerate(sample_grads):
                if grad is not None:
                    processed_grads.append(grad.detach().view(-1))
                else:
                    processed_grads.append(torch.zeros_like(self.adapter_params[j].view(-1)))
            
            flat_grad = torch.cat(processed_grads)
            grads_list.append(flat_grad)
        
        # 计算Fisher信息矩阵 F = E[grad⊗grad]
        grads_tensor = torch.stack(grads_list)
        hessian_approx = torch.matmul(grads_tensor.t(), grads_tensor) / batch_size
        
        # 确保对称性
        hessian_approx = 0.5 * (hessian_approx + hessian_approx.t())
        
        return hessian_approx
        
    def analyze_lowrank_properties(self, hessian, step):
        """
        分析Hessian矩阵的低秩特性
        """
        self.logger.info(f"Step {step}: 分析低秩特性")
        
        # 特征值分解
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
            eigenvalues = eigenvalues.cpu().numpy()
            eigenvectors = eigenvectors.cpu().numpy()
        except Exception as e:
            self.logger.warning(f"GPU特征值分解失败，使用CPU: {e}")
            hessian_np = hessian.cpu().numpy()
            eigenvalues, eigenvectors = eigh(hessian_np)
        
        # 按特征值大小排序（降序）
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 存储结果
        self.eigenvalues_history.append(eigenvalues)
        self.eigenvectors_history.append(eigenvectors)
        
        # 计算低秩指标
        metrics = self.calculate_lowrank_metrics(eigenvalues, eigenvectors)
        
        # 更新历史记录
        for key, value in metrics.items():
            self.lowrank_metrics[key].append(value)
        
        # 记录到TensorBoard
        self.log_lowrank_metrics(metrics, step)
        
        return eigenvalues, eigenvectors, metrics
    
    def calculate_lowrank_metrics(self, eigenvalues, eigenvectors):
        """
        计算低秩相关指标
        """
        # 1. 解释方差比例（前k个特征值的累计贡献）
        total_variance = np.sum(np.abs(eigenvalues))
        k_values = [5, 10, 20, 50]
        explained_ratios = {}
        
        for k in k_values:
            if k <= len(eigenvalues):
                ratio = np.sum(np.abs(eigenvalues[:k])) / total_variance
                explained_ratios[f'top_{k}'] = ratio
        
        # 2. 有效秩（根据特征值分布估计）
        # 使用Shannon熵估计有效秩
        eigenvals_pos = np.abs(eigenvalues) + 1e-12
        eigenvals_normalized = eigenvals_pos / np.sum(eigenvals_pos)
        entropy = -np.sum(eigenvals_normalized * np.log(eigenvals_normalized + 1e-12))
        effective_rank = np.exp(entropy)
        
        # 3. 谱范数（最大特征值）
        spectral_norm = np.max(np.abs(eigenvalues))
        
        # 4. 核范数（所有特征值绝对值之和）
        nuclear_norm = np.sum(np.abs(eigenvalues))
        
        # 5. 条件数
        min_eigenval = np.min(eigenvalues[eigenvalues > self.eigenvalue_threshold])
        max_eigenval = np.max(eigenvalues)
        condition_number = max_eigenval / min_eigenval if min_eigenval > 0 else float('inf')
        
        return {
            'explained_variance_ratios': explained_ratios,
            'effective_ranks': effective_rank,
            'spectral_norms': spectral_norm,
            'nuclear_norms': nuclear_norm,
            'condition_numbers': condition_number
        }
    
    def analyze_consistency(self, step):
        """
        分析Hessian矩阵在优化过程中的一致性
        """
        if step == 0:
            return  # 第一步没有历史数据比较
        
        self.logger.info(f"Step {step}: 分析一致性")
        
        current_eigenvals = self.eigenvalues_history[-1]
        current_eigenvecs = self.eigenvectors_history[-1]
        
        prev_eigenvals = self.eigenvalues_history[-2]
        prev_eigenvecs = self.eigenvectors_history[-2]
        
        # 1. 主子空间角度
        subspace_angle = self.compute_subspace_angles(
            prev_eigenvecs, current_eigenvecs, k=min(10, len(current_eigenvals))
        )
        
        # 2. 特征值相关性
        eigenval_corr = np.corrcoef(
            prev_eigenvals[:min(20, len(prev_eigenvals))],
            current_eigenvals[:min(20, len(current_eigenvals))]
        )[0, 1]
        
        # 3. Frobenius距离
        current_hessian = self.hessian_history[-1]
        prev_hessian = self.hessian_history[-2]
        frobenius_dist = torch.norm(current_hessian - prev_hessian, 'fro').item()
        
        # 存储一致性指标
        self.consistency_metrics['subspace_angles'].append(subspace_angle)
        self.consistency_metrics['eigenvalue_correlations'].append(eigenval_corr)
        self.consistency_metrics['frobenius_distances'].append(frobenius_dist)
        
        # 记录到TensorBoard
        self.writer.add_scalar('Consistency/SubspaceAngle', subspace_angle, step)
        self.writer.add_scalar('Consistency/EigenvalueCorrelation', eigenval_corr, step)
        self.writer.add_scalar('Consistency/FrobeniusDistance', frobenius_dist, step)
        
    def compute_subspace_angles(self, U1, U2, k=10):
        """
        计算两个子空间之间的主角度
        """
        # 取前k个主成分
        U1_k = U1[:, :k]
        U2_k = U2[:, :k]
        
        # 计算奇异值分解 U1^T @ U2
        try:
            _, s, _ = np.linalg.svd(U1_k.T @ U2_k)
            # 主角度是arccos(奇异值)
            angles = np.arccos(np.clip(s, 0, 1))
            # 返回最大角度（最不一致的方向）
            return np.max(angles)
        except:
            return float('nan')
    
    def log_lowrank_metrics(self, metrics, step):
        """
        记录低秩指标到TensorBoard
        """
        # 解释方差比例
        for k, ratio in metrics['explained_variance_ratios'].items():
            self.writer.add_scalar(f'LowRank/ExplainedVariance_{k}', ratio, step)
        
        # 其他指标
        self.writer.add_scalar('LowRank/EffectiveRank', metrics['effective_ranks'], step)
        self.writer.add_scalar('LowRank/SpectralNorm', metrics['spectral_norms'], step)
        self.writer.add_scalar('LowRank/NuclearNorm', metrics['nuclear_norms'], step)
        self.writer.add_scalar('LowRank/ConditionNumber', metrics['condition_numbers'], step)
    
    def run_analysis_step(self, batch, step):
        """
        运行单步分析
        """
        self.logger.info(f"=== Step {step}: 开始Hessian分析 ===")
        
        # 计算Hessian矩阵
        hessian = self.compute_hessian_fisher(batch)
        self.hessian_history.append(hessian)
        
        # 分析低秩特性
        eigenvals, eigenvecs, lowrank_metrics = self.analyze_lowrank_properties(hessian, step)
        
        # 分析一致性（从第二步开始）
        if step > 0:
            self.analyze_consistency(step)
        
        # 保存数据
        self.save_step_data(step, hessian, eigenvals, eigenvecs, lowrank_metrics)
        
        self.logger.info(f"Step {step}: 分析完成")
        
    def save_step_data(self, step, hessian, eigenvals, eigenvecs, metrics):
        """
        保存步骤数据
        """
        step_data = {
            'step': step,
            'eigenvalues': eigenvals,
            'metrics': metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # 保存特征值和指标
        with open(self.analysis_dir / f"step_{step:03d}_metrics.json", 'w') as f:
            json.dump(step_data, f, indent=2, default=str)
        
        # 保存Hessian矩阵（numpy格式）
        np.save(self.hessian_dir / f"hessian_step_{step:03d}.npy", hessian.cpu().numpy())
        
        # 保存特征向量（前20个）
        np.save(self.analysis_dir / f"eigenvecs_step_{step:03d}.npy", eigenvecs[:, :20])
    
    def generate_summary_report(self):
        """
        生成总结报告
        """
        self.logger.info("生成总结报告...")
        
        report = {
            'experiment_info': {
                'total_steps': len(self.hessian_history),
                'total_parameters': self.total_params,
                'max_rank_analyzed': self.max_rank_to_analyze,
                'timestamp': datetime.datetime.now().isoformat()
            },
            'lowrank_summary': self.summarize_lowrank_properties(),
            'consistency_summary': self.summarize_consistency()
        }
        
        # 保存报告
        with open(self.exp_dir / "experiment_summary.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"报告已保存到: {self.exp_dir / 'experiment_summary.json'}")
        return report
    
    def summarize_lowrank_properties(self):
        """
        总结低秩特性
        """
        if not self.lowrank_metrics['explained_variance_ratios']:
            return {}
        
        # 计算解释方差比例的平均值
        explained_var_summary = {}
        for step_ratios in self.lowrank_metrics['explained_variance_ratios']:
            for k, ratio in step_ratios.items():
                if k not in explained_var_summary:
                    explained_var_summary[k] = []
                explained_var_summary[k].append(ratio)
        
        for k in explained_var_summary:
            explained_var_summary[k] = {
                'mean': np.mean(explained_var_summary[k]),
                'std': np.std(explained_var_summary[k]),
                'min': np.min(explained_var_summary[k]),
                'max': np.max(explained_var_summary[k])
            }
        
        return {
            'explained_variance_ratios': explained_var_summary,
            'effective_rank': {
                'mean': np.mean(self.lowrank_metrics['effective_ranks']),
                'std': np.std(self.lowrank_metrics['effective_ranks'])
            },
            'condition_number': {
                'mean': np.mean(self.lowrank_metrics['condition_numbers']),
                'std': np.std(self.lowrank_metrics['condition_numbers'])
            }
        }
    
    def summarize_consistency(self):
        """
        总结一致性特性
        """
        if not self.consistency_metrics['subspace_angles']:
            return {}
        
        return {
            'subspace_angle': {
                'mean': np.mean(self.consistency_metrics['subspace_angles']),
                'std': np.std(self.consistency_metrics['subspace_angles'])
            },
            'eigenvalue_correlation': {
                'mean': np.mean(self.consistency_metrics['eigenvalue_correlations']),
                'std': np.std(self.consistency_metrics['eigenvalue_correlations'])
            }
        }


def create_model_with_adapter(args):
    """
    创建带adapter的ViT模型
    """
    # 使用timm加载预训练ViT模型
    vit_model = timm.create_model(
        'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        pretrained=True,
        num_classes=1000
    )
    
    # 冻结backbone参数
    freeze_vit_parameters(vit_model)
    
    # 创建AdaFormer模型
    model = AdaFormerViT(
        vit=vit_model,
        adapter_layer="11",  # 在最后一层添加adapter
        reduction_factor=16,
        adapter_style="parallel"
    )
    
    return model


def run_hessian_lowrank_experiment(args):
    """
    运行Hessian低秩特性实验
    """
    logger = get_logger(
        name="experiment", 
        output_directory=args.output,
        log_name=f"hessian_experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logger.info("开始Hessian低秩特性实验")
    logger.info(f"参数: {args}")
    
    # 设置设备
    if torch.cuda.is_available() and args.gpu is not None:
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cpu"
    
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    model = create_model_with_adapter(args)
    model = model.to(device)
    
    # 准备数据
    args.corruption = 'gaussian_noise'
    args.level = 5
    args.if_shuffle = False
    
    teset, teloader = prepare_test_data(args)
    logger.info(f"数据集大小: {len(teset)}")
    
    # 创建分析器
    analyzer = HessianLowRankAnalyzer(model, teloader, args)
    
    # 运行实验
    max_batches = min(args.max_batches, len(teloader))
    logger.info(f"将分析 {max_batches} 个批次")
    
    for step, batch in enumerate(teloader):
        if step >= max_batches:
            break
            
        try:
            analyzer.run_analysis_step(batch, step)
        except Exception as e:
            logger.error(f"Step {step} 分析失败: {e}")
            continue
        
        if step % 10 == 0:
            logger.info(f"已完成 {step + 1}/{max_batches} 步")
    
    # 生成报告
    report = analyzer.generate_summary_report()
    
    logger.info("实验完成！")
    logger.info(f"结果保存在: {analyzer.exp_dir}")
    
    return analyzer, report


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Hessian Low-Rank Property Analysis for TTA')
    
    # 数据路径
    parser.add_argument('--data_corruption', default='/dockerdata/imagenet-c', 
                       help='ImageNet-C数据集路径')
    parser.add_argument('--output', default='./outputs_hessian', 
                       help='输出目录')
    
    # 实验参数
    parser.add_argument('--batch_size', default=16, type=int, 
                       help='批次大小')
    parser.add_argument('--max_batches', default=50, type=int,
                       help='最大分析批次数')
    parser.add_argument('--gpu', default=0, type=int, 
                       help='GPU ID')
    parser.add_argument('--seed', default=2024, type=int,
                       help='随机种子')
    parser.add_argument('--workers', default=4, type=int,
                       help='数据加载进程数')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 运行实验
    analyzer, report = run_hessian_lowrank_experiment(args)
    
    print(f"实验完成！结果保存在: {analyzer.exp_dir}") 