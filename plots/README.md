# 论文绘图说明

## 生成的图表

本脚本生成了三个用于论文的高质量图表，展示了CAZO、ZO-Base和BP-Adapter三种算法的性能对比。

### 1. CTTA准确率曲线 (ctta_accuracy.pdf/png)
- **数据来源**: `outputs_new/continue_learning/` 目录下的log文件
- **内容**: 展示三个算法在15种ImageNet-C corruption类型上的连续测试时适应（CTTA）性能
- **数据点**: 每个算法2355个数据点（每5个batch记录一次）
- **特点**:
  - 按corruption顺序拼接：gaussian_noise → shot_noise → ... → jpeg_compression
  - 灰色虚线标记每个corruption的分界
  - 显示算法在不同分布偏移下的持续学习能力

### 2. 微调准确率曲线 (finetune_accuracy.pdf/png)
- **数据来源**: `outputs_new/finetune/` 目录下的log文件
- **内容**: 展示三个算法在两轮epoch微调过程中的性能变化
- **数据点**: 每个算法4710个数据点（2 epochs × 15 corruptions × 每5步记录）
- **特点**:
  - 红色虚线标记第2个epoch的开始位置
  - 显示算法的收敛速度和稳定性
  - 对比有监督微调的效果

### 3. 组合图 (combined_accuracy.pdf/png)
- **内容**: 左边CTTA性能，右边微调性能，方便直接对比
- **适用场景**: 论文中需要并排展示两种场景的性能对比

## 算法配置

### CTTA算法 (continue_learning)
- **CAZO**: 红色实线，圆形标记
- **ZO-Base**: 蓝色虚线，方形标记  
- **BP-Adapter**: 绿色点划线，三角形标记

### 微调算法 (finetune)
- **CAZO-FT**: 红色实线，圆形标记
- **ZO-Base-FT**: 蓝色虚线，方形标记
- **BP-Adapter-FT**: 绿色点划线，三角形标记

## 数据统计

### CTTA场景
```
算法              数据点数    Corruption数    平均步数/Corruption
CAZO              2355       15              157
ZO-Base           2355       15              157
BP-Adapter        2355       15              157
```

### 微调场景
```
算法              数据点数    Epochs    总步数
CAZO-FT           4710       2         4710
ZO-Base-FT        4710       2         4710
BP-Adapter-FT     4710       2         4710
```

## 图表特点

1. **论文级别质量**
   - 300 DPI高分辨率
   - 清晰的字体和标签
   - 专业的配色方案

2. **易于解释**
   - 清晰的图例和标题
   - 网格辅助读数
   - 关键位置标注

3. **多格式支持**
   - PDF格式：用于LaTeX论文排版
   - PNG格式：用于演示文稿和预览

## 使用方法

### 生成图表
```bash
cd /home/zjm/workspace/CAZO
python plot_paper_figures_simple.py
```

### 在LaTeX中使用
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{plots/combined_accuracy.pdf}
    \caption{Performance comparison of CAZO, ZO-Base, and BP-Adapter 
             on (a) Continual Test-Time Adaptation and (b) Fine-tuning tasks.}
    \label{fig:performance}
\end{figure}
```

## 注意事项

1. **数据完整性**: 脚本会自动检测并跳过缺失的数据文件
2. **Epoch分界**: 微调图中的epoch分界线位于数据点的中间位置（假设两个epoch数据量相同）
3. **覆写问题**: 由于TensorBoard数据被覆写，微调图使用log文件而非TensorBoard数据

## 文件位置

- **绘图脚本**: `/home/zjm/workspace/CAZO/plot_paper_figures_simple.py`
- **输出目录**: `/home/zjm/workspace/CAZO/plots/`
- **源数据**: `/home/zjm/workspace/CAZO/outputs_new/`

## 定制化

如需修改图表样式，可在脚本中调整以下参数：
- `plt.rcParams`: 全局绘图参数
- `ALGORITHMS`: 算法名称、颜色、线型
- 图表尺寸: `figsize=(width, height)`
- 标题和标签: 在各个绘图函数中修改
