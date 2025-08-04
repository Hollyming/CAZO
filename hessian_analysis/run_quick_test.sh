#!/bin/bash

# 快速测试Hessian低秩特性验证实验
# Quick test for Hessian Low-Rank Property Verification Experiment

set -e

echo "🔬 开始Hessian低秩特性快速测试实验"
echo "============================================"

# 配置参数
DATA_CORRUPTION="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-c/imagenet-c"
OUTPUT_DIR="./hessian_test_results"
BATCH_SIZE=32
MAX_BATCHES=100  # 快速测试使用较少的批次
GPU=0
ADAPTER_LAYERS="3"  # 使用单层adapter
REDUCTION_FACTOR=384
SEED=42

echo "📋 实验配置:"
echo "  数据路径: $DATA_CORRUPTION"
echo "  输出目录: $OUTPUT_DIR"
echo "  批次大小: $BATCH_SIZE"
echo "  最大批次: $MAX_BATCHES"
echo "  GPU设备: $GPU"
echo "  Adapter层: $ADAPTER_LAYERS"
echo "  降维因子: $REDUCTION_FACTOR"

# 检查Python环境
echo "🔍 检查Python环境..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import timm; print(f'timm版本: {timm.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"

# 检查数据集
echo "🔍 检查数据集..."
if [ -d "$DATA_CORRUPTION/gaussian_noise/5" ]; then
    echo "✅ ImageNet-C数据集路径验证成功"
    echo "   高斯噪声level 5数据存在"
else
    echo "❌ 错误: ImageNet-C数据集路径不存在或结构不正确"
    echo "   期望路径: $DATA_CORRUPTION/gaussian_noise/5"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建并执行Python命令
echo "🚀 开始运行实验..."
python ./run_hessian_experiment.py \
    --data_corruption "$DATA_CORRUPTION" \
    --output "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --max_batches $MAX_BATCHES \
    --adapter_layers "$ADAPTER_LAYERS" \
    --reduction_factor $REDUCTION_FACTOR \
    --seed $SEED \
    --workers 4 \
    --gpu $GPU

if [ $? -eq 0 ]; then
    echo "✅ 实验成功完成！"
    echo "📊 结果保存在: $OUTPUT_DIR"
    echo "📈 可视化图表包括:"
    echo "   - 特征值分布分析"
    echo "   - 低秩指标分析"
    echo "   - 一致性指标分析"
    echo "   - 多步骤Hessian分析（新增）"
    echo "   - 综合总结报告"
else
    echo "❌ 实验执行失败"
    exit 1
fi 