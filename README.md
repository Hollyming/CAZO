# CAZO: Test-Time Adaptation with Curvature-Aware Zeroth-Order Optimization

This repository implements CAZO, a novel test-time adaptation (TTA) algorithm that leverages curvature information for zeroth-order optimization. CAZO is designed to adapt pre-trained models to test data with distributional shifts without requiring backpropagation.

## Key Features

* 🚀 **Forward-Only Adaptation**: CAZO performs adaptation using only forward passes, making it:
  - Memory efficient (reduced memory usage by ~80% compared to backpropagation-based methods)
  - Compatible with quantized models
  - Suitable for specialized hardware where parameters are non-modifiable

* 🔄 **Curvature-Aware Optimization**: CAZO incorporates covariance information into the zero-order optimization process, leading to:
  - More efficient parameter updates
  - Better adaptation to distribution shifts
  - Improved convergence properties

## Project Structure
```bash
CAZO/
├── tta_library/ # TTA算法库
│ ├── CAZO.py # CAZO主算法
│ ├── COZO_lit.py # COZO单扰动
│ ├── COZO_Ablation.py # COZO标准（附带消融）
│ ├── tent.py # Tent基线
│ ├── cotta.py # CoTTA基线
│ ├── sar.py # SAR基线
│ ├── t3a.py # T3A基线
│ ├── lame.py # LAME基线
│ └── foa_.py # FOA相关算法
├── models/ # 模型架构
│ ├── adaformer.py # AdaFormer架构 (带适配器层)
│ ├── vit_adapter.py # 支持适配器的Vision Transformer
│ ├── vpt.py # Vision Prompt Tuning实现
│ └── backbone/ # 各种骨干网络架构
├── dataset/ # 数据集加载和处理方法
├── hessian_analysis/ # 早期预实验：证明Hessian在不同步骤间具有低秩结构
├── scripts/ # 可执行脚本
├── utils/ # 工具函数
├── calibration_library/ # 校准库
├── quant_library/ # 量化库
│ └── config/PTQ4ViT.py # 量化配置，控制量化字节数
├── figures/ # 图表文件
└── main.py # 主程序入口
```

## TTA Library Overview

The `tta_library` contains several TTA algorithms:

### CAZO Algorithms
- `CAZO.py`: Main CAZO algorithm


### Baseline Methods
- `tent.py`: Tent baseline
- `cotta.py`: CoTTA baseline
- `sar.py`: SAR baseline
- `t3a.py`: T3A baseline
- `lame.py`: LAME baseline
- `foa.py`: FOA baseline
- `zo_base.py`: ZO baseline (has the same structure with CAZO)
- `COZO_Ablation.py`: COZO's method combined CMA-ES with ZO, you can choose "cov_only" or "full"
- `COZO_lit.py`: single point perturbation for COZO


## Models Overview

The `models` directory contains several model architectures:

### Core Models
- `adaformer.py`: AdaFormer architecture with adapter layers
- `vit_adapter.py`: Vision Transformer with adapter support
- `vpt.py`: Vision Prompt Tuning implementation

### Backbone Models
- `backbone/`: Contains various backbone architectures

## Dataset Loading

The `dataset/` folder contains methods for loading and processing various datasets used in our experiments. This includes support for ImageNet, ImageNet-C, ImageNet-R, ImageNet-Sketch, and other domain adaptation benchmarks.

## Hessian Analysis

The `hessian_analysis/` folder contains our early preliminary experiments that demonstrate the low-rank structure of Hessian matrices across different optimization steps. These experiments provide theoretical foundation for our curvature-aware optimization approach.

## Quantization Support

CAZO supports model quantization for memory-efficient deployment. To run quantized versions:

1. **Enable Quantization**: Add `--quant` flag to your command
2. **Configure Bit Width**: Modify `quant_library/config/PTQ4ViT.py` to control quantization bit width
3. **Example Usage**:
```bash
python main.py \
    --data path/to/imagenet \
    --algorithm cma_zo1_multiadapter \
    --quant \
    --batch_size 64
```

## Dependencies

```bash
pip install cma
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.9.10
pip install tensorboard
```

## Usage

### Basic Usage
```python
from tta_library.XXX import XXX
from models.adaformer import AdaFormerViT

# Initialize model
# model = TODO_model()
model = AdaFormerViT(backbone_model)
adapt_model = XXX(model, fitness_lambda=0.4, lr=0.01)

# Adapt to test data
outputs = adapt_model(test_inputs)
```

### Command Line Usage
```bash
python main.py \
    --data path/to/imagenet \
    --data_v2 path/to/imagenet-v2 \
    --data_sketch path/to/imagenet-sketch \
    --data_corruption path/to/imagenet-c \
    --data_rendition path/to/imagenet-r \
    --algorithm cazo \
    --batch_size 64 \
    --lr 0.01 \
    --pertub 20
```

## Performance

COZO achieves state-of-the-art performance on various benchmarks:

| Method | ImageNet-C (Acc/ECE) | Memory Usage |
|--------|---------------------|--------------|
| NoAdapt | 55.5% / 10.5% | - |
| Tent | 59.6% / 18.5% | 6,026MB |
| CoTTA | 61.6% / 6.7% | 17,588MB |
| SAR | 59.3% / 6.4% | 6,172MB |
| **CAZO** | **69.0% / 4.2%** | **1,645MB** |

## Scripts

The `scripts/` directory contains executable scripts for various experiments:

- **Training Scripts**: For model training and fine-tuning
- **Evaluation Scripts**: For testing and benchmarking
- **Quantization Scripts**: For quantized model experiments
- **Analysis Scripts**: For Hessian analysis and ablation studies


## Acknowledgments

This work is inspired by:
- [EATA](https://github.com/mr-eggplant/EATA)
- [VPT](https://github.com/KMnP/vpt)
- [FOA](https://github.com/mr-eggplant/FOA)