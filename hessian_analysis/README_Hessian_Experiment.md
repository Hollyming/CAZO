# Hessian低秩特性验证实验

本实验旨在验证以下观察：

> "我们提供了一个观察，即TTA问题的Hessian矩阵具有低秩特性，并在优化过程中保持相对一致。"

## 实验设计

### 核心思想

1. **低秩特性验证**：分析Hessian矩阵的特征值分布，验证前几个主成分是否能解释大部分方差
2. **一致性验证**：分析优化过程中Hessian矩阵的主子空间是否保持相对稳定
3. **定量指标**：使用多种数学指标量化低秩特性和一致性

### 实验配置

- **模型**：使用timm的ViT-Base预训练模型 + AdaFormer adapter
- **数据集**：ImageNet-C的gaussian_noise corruption (level=5)
- **优化方式**：Test-Time Adaptation (TTA)
- **分析对象**：adapter参数的Hessian矩阵（使用Fisher信息矩阵近似）

## 快速开始

### 1. 环境准备

确保您的环境中已安装所需依赖：

```bash
pip install torch torchvision timm scipy sklearn matplotlib seaborn pandas
```

### 2. 数据准备

确保ImageNet-C数据集可用：

```bash
# 假设ImageNet-C数据集位于 /path/to/imagenet-c
# 目录结构应该是：
# /path/to/imagenet-c/
#   ├── gaussian_noise/
#   │   ├── 1/
#   │   ├── 2/
#   │   ├── 3/
#   │   ├── 4/
#   │   └── 5/
#   └── ...
```

### 3. 运行实验

#### 基础运行

```bash
python run_hessian_experiment.py \
    --data_corruption /path/to/imagenet-c \
    --output ./hessian_results \
    --batch_size 16 \
    --max_batches 50
```

#### 高级配置

```bash
python run_hessian_experiment.py \
    --data_corruption /path/to/imagenet-c \
    --output ./hessian_results \
    --adapter_layers "9,10,11" \
    --reduction_factor 16 \
    --batch_size 32 \
    --max_batches 100 \
    --gpu 0 \
    --seed 2024
```

### 4. 查看结果

实验完成后，结果将保存在指定的输出目录中：

```
hessian_results/
└── hessian_tta_experiment_YYYYMMDD_HHMMSS/
    ├── figures/                          # 📊 专业论文图表
    │   ├── TTA_Hessian_*_eigenvalue_distribution.pdf
    │   ├── TTA_Hessian_*_lowrank_metrics.pdf
    │   ├── TTA_Hessian_*_consistency_metrics.pdf
    │   └── TTA_Hessian_*_comprehensive_summary.pdf
    ├── results/                          # 📈 分析结果
    │   └── hessian_lowrank_*/
    │       ├── hessian_matrices/        # 🔢 Hessian矩阵数据
    │       ├── analysis_results/        # 📊 数值分析结果
    │       └── tensorboard/            # 📝 TensorBoard日志
    ├── logs/                            # 📋 实验日志
    ├── FINAL_EXPERIMENT_REPORT.json    # 📄 完整JSON报告
    └── EXPERIMENT_REPORT.md            # 📖 Markdown摘要报告
```

## 实验指标

### 低秩特性指标

1. **解释方差比例 (Explained Variance Ratio)**
   - 前k个主成分解释的总方差比例
   - k ∈ {5, 10, 20, 50}
   - 用于验证低秩假设

2. **有效秩 (Effective Rank)**
   - 基于Shannon熵的有效维度估计
   - 公式：`effective_rank = exp(-Σ(p_i * log(p_i)))`
   - 其中p_i是归一化的特征值

3. **条件数 (Condition Number)**
   - 最大特征值与最小正特征值的比值
   - 反映矩阵的数值稳定性

### 一致性指标

1. **主子空间角度 (Principal Subspace Angle)**
   - 相邻步骤间主子空间的最大角度
   - 使用奇异值分解计算
   - 角度越小表示一致性越好

2. **特征值相关性 (Eigenvalue Correlation)**
   - 相邻步骤间前20个特征值的Pearson相关系数
   - 高相关性表示特征值结构稳定

3. **Frobenius距离 (Frobenius Distance)**
   - 相邻Hessian矩阵的Frobenius范数距离
   - 反映矩阵整体变化程度

## 可视化图表

### 1. 特征值分布分析 (`eigenvalue_distribution.pdf`)

- **特征值衰减图**：展示主要特征值的衰减模式
- **累计解释方差**：验证低秩特性的核心指标
- **特征值分布直方图**：最终特征值的统计分布
- **有效秩演化**：优化过程中有效秩的变化

### 2. 低秩指标分析 (`lowrank_metrics.pdf`)

- **解释方差比例时间序列**：不同k值的解释方差演化
- **有效秩、谱范数、条件数演化**：多维度低秩特性分析
- **指标分布箱线图**：统计特性总览

### 3. 一致性分析 (`consistency_metrics.pdf`)

- **子空间角度演化**：主要一致性指标
- **特征值相关性**：结构稳定性验证
- **Frobenius距离**：矩阵变化幅度
- **综合一致性评分**：多指标融合评估

### 4. 综合总结 (`comprehensive_summary.pdf`)

- **关键发现展示**：最重要的指标组合
- **特征值谱演化**：不同阶段的谱变化
- **统计信息表格**：定量结果总结
- **结论文本**：实验结论说明

## 预期结果

### 支持低秩假设的证据

1. **前10个主成分解释>80%方差**：强有力的低秩证据
2. **有效秩远小于总参数数量**：维度显著降低
3. **特征值快速衰减**：呈现明显的长尾分布

### 支持一致性假设的证据

1. **主子空间角度<30°**：相对稳定的主要方向
2. **特征值相关性>0.7**：结构保持一致
3. **相对较小的Frobenius距离变化**：整体矩阵变化可控

## 参数配置

### 关键参数说明

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--data_corruption` | `/dockerdata/imagenet-c` | ImageNet-C数据路径 |
| `--adapter_layers` | `"11"` | Adapter层位置，可设置多层如"9,10,11" |
| `--reduction_factor` | `16` | Adapter降维因子，影响参数数量 |
| `--batch_size` | `16` | 批次大小，影响Hessian计算精度 |
| `--max_batches` | `50` | 分析的最大批次数，影响实验时长 |
| `--gpu` | `0` | GPU设备ID |
| `--seed` | `2024` | 随机种子，确保实验可重复 |

### 性能调优建议

1. **内存优化**：
   - 减小`batch_size`以降低内存使用
   - 使用单层adapter (`--adapter_layers "11"`)

2. **精度提升**：
   - 增加`max_batches`以获得更稳定的统计
   - 使用多层adapter增加参数复杂度

3. **速度优化**：
   - 减少`max_batches`以快速验证
   - 使用更小的`batch_size`

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 解决方案：减小批次大小
   --batch_size 8
   ```

2. **数据路径错误**
   ```bash
   # 确保路径正确且包含gaussian_noise/5/目录
   ls /path/to/imagenet-c/gaussian_noise/5/
   ```

3. **依赖缺失**
   ```bash
   # 安装必要依赖
   pip install scipy sklearn matplotlib seaborn
   ```

### 性能基准

在典型硬件配置下的预期性能：

- **GPU**: RTX 3080 (10GB)
- **批次大小**: 16
- **分析批次**: 50
- **预期时间**: 15-30分钟
- **内存使用**: ~6GB GPU内存

## 扩展实验

### 不同corruption类型

```bash
# 可以修改hessian_lowrank_experiment.py中的corruption类型
# 目前支持：gaussian_noise, shot_noise, impulse_noise等
```

### 不同模型架构

```bash
# 在create_model_with_adapter函数中修改模型类型
# 支持其他timm预训练模型
```

### 不同adapter配置

```bash
# 尝试不同的adapter层组合
--adapter_layers "6,9,11"    # 多层adapter
--reduction_factor 8         # 更大的adapter容量
```

## 引用和参考

如果您在研究中使用了这个实验框架，请引用相关工作：

```bibtex
@article{hessian_lowrank_tta,
  title={Hessian Low-Rank Properties in Test-Time Adaptation},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2024}
}
```

## 联系方式

如有问题或建议，请联系：
- 邮箱：your.email@domain.com
- GitHub Issues：提交到项目仓库 