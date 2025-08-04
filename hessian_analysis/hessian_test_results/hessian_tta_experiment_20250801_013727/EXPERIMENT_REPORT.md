# Hessian Low-Rank Property Analysis Report

## 实验概览
- **时间**: 20250801_013727
- **模型**: ViT-Base + AdaFormer
- **数据集**: ImageNet-C (gaussian_noise, level=5)
- **分析批次**: 100
- **总参数**: 5,378

## 关键发现

### 1. 低秩特性验证

- **前10个主成分解释方差**: 86.5% ± 4.6%
- **低秩特性**: ✅ 确认
- **解释**: 前10个主成分平均解释了86.5%的方差，证实了低秩特性

### 2. 一致性验证
- **平均子空间角度**: 88.6° ± 1.0°
- **一致性**: ❌ 未确认
- **解释**: 主子空间角度平均为88.6度，一致性有待改善

### 3. 有效秩分析
- **平均有效秩**: 12.0
- **有效秩稳定性**: 2.17
- **秩占比**: 0.22%
- **解释**: 有效秩平均为12.0，占总参数的0.22%

## 文件结构
- 📊 **可视化图表**: `figures/`
- 🔢 **Hessian矩阵**: `results/hessian_lowrank_20250801_013740/hessian_matrices/`
- 📈 **分析结果**: `results/hessian_lowrank_20250801_013740/analysis_results/`
- 📝 **TensorBoard日志**: `results/hessian_lowrank_20250801_013740/tensorboard/`
- 💾 **分析数据**: `hessian_analysis_data_20250801_013727.npz`

## 使用方法

### 查看可视化结果
```bash
# 查看生成的图表
open figures/*.png
```

### 重新加载数据生成图表
```python
from hessian_analysis.visualization_tools import HessianVisualizationTools

# 加载保存的数据
data_path = "hessian_analysis_data_20250801_013727.npz"
analysis_data = HessianVisualizationTools.load_analysis_data(data_path)

# 重新生成可视化
visualizer = HessianVisualizationTools("./output")
visualizer.generate_all_visualizations(analysis_data, "Custom_Analysis")
```

### 启动TensorBoard
```bash
tensorboard --logdir results/hessian_lowrank_20250801_013740/tensorboard
```

## 结论

本实验验证了以下观察：

> "Hessian在TTA问题中具有低秩特性且在优化过程中保持相对一致"

实验结果部分支持了这一观察。
