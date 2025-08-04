#!/usr/bin/env python3
"""
Complete Hessian Low-Rank Experiment Runner
完整的Hessian低秩特性实验运行器

这个脚本集成了：
1. Hessian分析 (HessianLowRankAnalyzer)
2. 可视化生成 (HessianVisualizationTools)
3. 实验报告生成
4. timm ViT模型集成
5. ImageNet-C数据加载

运行方式:
python run_hessian_experiment.py --data_corruption /path/to/imagenet-c --output ./results
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

# 添加父目录到Python路径以访问项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hessian_analysis.hessian_lowrank_experiment import HessianLowRankAnalyzer, create_model_with_adapter
from hessian_analysis.visualization_tools import HessianVisualizationTools
from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data

warnings.filterwarnings('ignore')

class ComprehensiveHessianExperiment:
    """
    综合的Hessian实验类
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
        self.exp_dir = Path(self.args.output) / f"hessian_tta_experiment_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置子目录
        self.results_dir = self.exp_dir / "results"
        self.figures_dir = self.exp_dir / "figures"
        self.logs_dir = self.exp_dir / "logs"
        
        for dir_path in [self.results_dir, self.figures_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = get_logger(
            name="comprehensive_experiment",
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
        创建ViT模型
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
        准备ImageNet-C数据
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
        analysis_args.output = str(self.results_dir)
        
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
        
        # 生成分析报告
        report = analyzer.generate_summary_report()
        
        total_time = time.time() - start_time
        self.logger.info(f"Hessian分析完成，总用时: {total_time:.1f}s")
        
        return analyzer, report
        
    def generate_visualizations(self, analyzer):
        """
        生成可视化图表并保存数据
        """
        self.logger.info("生成可视化图表并保存数据...")
        
        # 创建可视化工具
        visualizer = HessianVisualizationTools(str(self.exp_dir))
        
        # 准备数据
        analysis_data = {
            'eigenvalues_history': analyzer.eigenvalues_history,
            'lowrank_metrics': analyzer.lowrank_metrics,
            'consistency_metrics': analyzer.consistency_metrics,
            'steps': list(range(len(analyzer.hessian_history)))
        }
        
        # 保存分析数据
        data_filename = f"hessian_analysis_data_{self.timestamp}.npz"
        visualizer.save_analysis_data(analysis_data, data_filename)
        
        # 生成合并的特征值分析图
        visualizer.generate_all_visualizations(
            analysis_data, 
            title_prefix=f"TTA_Hessian_{self.timestamp}"
        )
        
        # 生成多步骤Hessian分析图（用户特别要求的图表）
        visualizer.plot_multi_step_hessian_analysis(
            eigenvalues_history=analyzer.eigenvalues_history,
            lowrank_metrics=analyzer.lowrank_metrics,
            steps=list(range(len(analyzer.hessian_history))),
            save_name=f"TTA_Hessian_MultiStep_Analysis_{self.timestamp}.pdf"
        )
        
        self.logger.info(f"可视化图表已保存到: {self.figures_dir}")
        self.logger.info(f"✨ 特别生成了多步骤Hessian分析图，包含不同步骤的特征值与累计方差比例")
        self.logger.info(f"分析数据已保存，可随时使用以下方式重新加载:")
        self.logger.info(f"  data_path = '{self.exp_dir / data_filename}'")
        self.logger.info(f"  analysis_data = HessianVisualizationTools.load_analysis_data(data_path)")
        
        return data_filename
        
    def generate_final_report(self, analyzer, report, data_filename=None):
        """
        生成最终实验报告
        """
        self.logger.info("生成最终实验报告...")
        
        # 计算关键指标的总结统计
        final_report = {
            'experiment_info': {
                'timestamp': self.timestamp,
                'device': self.device,
                'model_type': 'ViT-Base + AdaFormer',
                'dataset': 'ImageNet-C (gaussian_noise, level=5)',
                'total_batches_analyzed': len(analyzer.hessian_history),
                'total_parameters': analyzer.total_params,
                'adapter_layers': self.args.adapter_layers,
                'batch_size': self.args.batch_size
            },
            'key_findings': self.extract_key_findings(analyzer),
            'detailed_results': report,
            'file_locations': {
                'figures': str(self.figures_dir.relative_to(self.exp_dir)),
                'hessian_matrices': str(analyzer.hessian_dir.relative_to(self.exp_dir)),
                'analysis_results': str(analyzer.analysis_dir.relative_to(self.exp_dir)),
                'tensorboard_logs': str(analyzer.exp_dir.relative_to(self.exp_dir) / "tensorboard"),
                'analysis_data': data_filename if data_filename else "analysis_data.npz"
            }
        }
        
        # 保存最终报告
        final_report_path = self.exp_dir / "FINAL_EXPERIMENT_REPORT.json"
        with open(final_report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成markdown格式的报告
        self.generate_markdown_report(final_report)
        
        self.logger.info(f"最终报告已保存到: {final_report_path}")
        
    def extract_key_findings(self, analyzer):
        """
        提取关键发现
        """
        findings = {}
        
        # 低秩特性分析
        if analyzer.lowrank_metrics['explained_variance_ratios']:
            # 计算前10个分量的平均解释方差
            top10_ratios = []
            for step_ratios in analyzer.lowrank_metrics['explained_variance_ratios']:
                if 'top_10' in step_ratios:
                    top10_ratios.append(step_ratios['top_10'])
            
            if top10_ratios:
                findings['low_rank_evidence'] = {
                    'top_10_explained_variance_mean': float(np.mean(top10_ratios)),
                    'top_10_explained_variance_std': float(np.std(top10_ratios)),
                    'low_rank_confirmed': np.mean(top10_ratios) > 0.8,  # 如果前10个分量解释80%以上方差
                    'interpretation': f"前10个主成分平均解释了{np.mean(top10_ratios)*100:.1f}%的方差，"
                                   + ("证实了低秩特性" if np.mean(top10_ratios) > 0.8 else "低秩特性不明显")
                }
        
        # 一致性分析
        if analyzer.consistency_metrics['subspace_angles']:
            angles_deg = [np.degrees(angle) for angle in analyzer.consistency_metrics['subspace_angles']]
            findings['consistency_evidence'] = {
                'mean_subspace_angle_deg': float(np.mean(angles_deg)),
                'std_subspace_angle_deg': float(np.std(angles_deg)),
                'consistency_confirmed': np.mean(angles_deg) < 30,  # 平均角度小于30度认为一致
                'interpretation': f"主子空间角度平均为{np.mean(angles_deg):.1f}度，"
                                + ("表明优化过程中Hessian结构相对一致" if np.mean(angles_deg) < 30 else "一致性有待改善")
            }
        
        # 有效秩分析
        if analyzer.lowrank_metrics['effective_ranks']:
            eff_ranks = analyzer.lowrank_metrics['effective_ranks']
            findings['effective_rank_analysis'] = {
                'mean_effective_rank': float(np.mean(eff_ranks)),
                'effective_rank_stability': float(np.std(eff_ranks)),
                'rank_ratio': float(np.mean(eff_ranks) / analyzer.total_params),
                'interpretation': f"有效秩平均为{np.mean(eff_ranks):.1f}，"
                                + f"占总参数的{np.mean(eff_ranks)/analyzer.total_params*100:.2f}%"
            }
        
        return findings
        
    def generate_markdown_report(self, final_report):
        """
        生成markdown格式的报告
        """
        md_content = f"""# Hessian Low-Rank Property Analysis Report

## 实验概览
- **时间**: {final_report['experiment_info']['timestamp']}
- **模型**: {final_report['experiment_info']['model_type']}
- **数据集**: {final_report['experiment_info']['dataset']}
- **分析批次**: {final_report['experiment_info']['total_batches_analyzed']}
- **总参数**: {final_report['experiment_info']['total_parameters']:,}

## 关键发现

### 1. 低秩特性验证
"""
        
        if 'low_rank_evidence' in final_report['key_findings']:
            evidence = final_report['key_findings']['low_rank_evidence']
            md_content += f"""
- **前10个主成分解释方差**: {evidence['top_10_explained_variance_mean']*100:.1f}% ± {evidence['top_10_explained_variance_std']*100:.1f}%
- **低秩特性**: {'✅ 确认' if evidence['low_rank_confirmed'] else '❌ 未确认'}
- **解释**: {evidence['interpretation']}
"""

        if 'consistency_evidence' in final_report['key_findings']:
            evidence = final_report['key_findings']['consistency_evidence']
            md_content += f"""
### 2. 一致性验证
- **平均子空间角度**: {evidence['mean_subspace_angle_deg']:.1f}° ± {evidence['std_subspace_angle_deg']:.1f}°
- **一致性**: {'✅ 确认' if evidence['consistency_confirmed'] else '❌ 未确认'}
- **解释**: {evidence['interpretation']}
"""

        if 'effective_rank_analysis' in final_report['key_findings']:
            evidence = final_report['key_findings']['effective_rank_analysis']
            md_content += f"""
### 3. 有效秩分析
- **平均有效秩**: {evidence['mean_effective_rank']:.1f}
- **有效秩稳定性**: {evidence['effective_rank_stability']:.2f}
- **秩占比**: {evidence['rank_ratio']*100:.2f}%
- **解释**: {evidence['interpretation']}
"""

        md_content += f"""
## 文件结构
- 📊 **可视化图表**: `{final_report['file_locations']['figures']}/`
- 🔢 **Hessian矩阵**: `{final_report['file_locations']['hessian_matrices']}/`
- 📈 **分析结果**: `{final_report['file_locations']['analysis_results']}/`
- 📝 **TensorBoard日志**: `{final_report['file_locations']['tensorboard_logs']}/`
- 💾 **分析数据**: `{final_report['file_locations']['analysis_data']}`

## 使用方法

### 查看可视化结果
```bash
# 查看生成的图表
open {final_report['file_locations']['figures']}/*.png
```

### 重新加载数据生成图表
```python
from hessian_analysis.visualization_tools import HessianVisualizationTools

# 加载保存的数据
data_path = "{final_report['file_locations']['analysis_data']}"
analysis_data = HessianVisualizationTools.load_analysis_data(data_path)

# 重新生成可视化
visualizer = HessianVisualizationTools("./output")
visualizer.generate_all_visualizations(analysis_data, "Custom_Analysis")
```

### 启动TensorBoard
```bash
tensorboard --logdir {final_report['file_locations']['tensorboard_logs']}
```

## 结论

本实验验证了以下观察：

> "Hessian在TTA问题中具有低秩特性且在优化过程中保持相对一致"

实验结果{'支持' if final_report['key_findings'].get('low_rank_evidence', {}).get('low_rank_confirmed', False) and final_report['key_findings'].get('consistency_evidence', {}).get('consistency_confirmed', False) else '部分支持'}了这一观察。
"""

        # 保存markdown报告
        md_path = self.exp_dir / "EXPERIMENT_REPORT.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        self.logger.info(f"Markdown报告已保存到: {md_path}")
        
    def run_complete_experiment(self):
        """
        运行完整实验
        """
        self.logger.info("="*60)
        self.logger.info("开始完整的Hessian低秩特性验证实验")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 1. 创建模型
            model = self.create_model()
            
            # 2. 准备数据
            teset, teloader = self.prepare_data()
            
            # 3. 运行Hessian分析
            analyzer, report = self.run_hessian_analysis(model, teloader)
            
            # 4. 生成可视化并保存数据
            data_filename = self.generate_visualizations(analyzer)
            
            # 5. 生成最终报告
            self.generate_final_report(analyzer, report, data_filename)
            
            total_time = time.time() - start_time
            
            self.logger.info("="*60)
            self.logger.info("实验完成！")
            self.logger.info(f"总用时: {total_time:.1f}s ({total_time/60:.1f}分钟)")
            self.logger.info(f"结果保存在: {self.exp_dir}")
            self.logger.info("="*60)
            
            # 打印关键结果
            self.print_key_results(analyzer)
            
            return analyzer, report
            
        except Exception as e:
            self.logger.error(f"实验失败: {e}")
            raise
    
    def print_key_results(self, analyzer):
        """
        打印关键结果
        """
        print("\n" + "="*60)
        print("关键实验结果")
        print("="*60)
        
        # 低秩特性
        if analyzer.lowrank_metrics['explained_variance_ratios']:
            top10_ratios = []
            for step_ratios in analyzer.lowrank_metrics['explained_variance_ratios']:
                if 'top_10' in step_ratios:
                    top10_ratios.append(step_ratios['top_10'])
            
            if top10_ratios:
                mean_ratio = np.mean(top10_ratios)
                print(f"📊 低秩特性: 前10个主成分平均解释 {mean_ratio*100:.1f}% 的方差")
                if mean_ratio > 0.8:
                    print("   ✅ 确认低秩特性")
                else:
                    print("   ❌ 低秩特性不明显")
        
        # 一致性
        if analyzer.consistency_metrics['subspace_angles']:
            angles_deg = [np.degrees(angle) for angle in analyzer.consistency_metrics['subspace_angles']]
            mean_angle = np.mean(angles_deg)
            print(f"🔄 一致性: 平均子空间角度 {mean_angle:.1f}°")
            if mean_angle < 30:
                print("   ✅ 确认优化过程中的一致性")
            else:
                print("   ❌ 一致性有待改善")
        
        # 有效秩
        if analyzer.lowrank_metrics['effective_ranks']:
            mean_rank = np.mean(analyzer.lowrank_metrics['effective_ranks'])
            rank_ratio = mean_rank / analyzer.total_params
            print(f"📐 有效秩: {mean_rank:.1f} (占总参数的 {rank_ratio*100:.2f}%)")
        
        print(f"\n📁 详细结果: {self.exp_dir}")
        print(f"📊 可视化图表: {self.figures_dir}")
        print("="*60)


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='Complete Hessian Low-Rank Property Analysis for TTA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据相关
    parser.add_argument('--data_corruption', default='/dockerdata/imagenet-c',
                       help='ImageNet-C数据集路径')
    parser.add_argument('--output', default='./hessian_experiments',
                       help='输出根目录')
    
    # 模型相关
    parser.add_argument('--adapter_layers', default='11', type=str,
                       help='Adapter层位置 (例如: "11" 或 "9,10,11")')
    parser.add_argument('--reduction_factor', default=16, type=int,
                       help='Adapter降维因子')
    
    # 实验相关
    parser.add_argument('--batch_size', default=16, type=int,
                       help='批次大小')
    parser.add_argument('--max_batches', default=50, type=int,
                       help='最大分析批次数')
    parser.add_argument('--workers', default=4, type=int,
                       help='数据加载进程数')
    
    # 系统相关
    parser.add_argument('--gpu', default=0, type=int,
                       help='GPU ID')
    parser.add_argument('--seed', default=2024, type=int,
                       help='随机种子')
    
    return parser.parse_args()


def main():
    """
    主函数
    """
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
    
    print("🚀 启动Hessian低秩特性验证实验")
    print(f"📋 参数配置: {vars(args)}")
    
    # 运行实验
    experiment = ComprehensiveHessianExperiment(args)
    analyzer, report = experiment.run_complete_experiment()
    
    print("🎉 实验成功完成！")


if __name__ == '__main__':
    main() 