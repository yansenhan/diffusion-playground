# Swiss Roll Diffusion Model Example

这个示例展示了如何在Swiss roll数据上训练和推理扩散模型，并生成inference过程的GIF动画。

## 概述

Swiss roll是一个经典的2D数据集，数据点分布在一个螺旋形的1D流形上。这个示例使用扩散模型来学习这个数据分布，并能够从噪声中生成新的Swiss roll数据点。

## 文件结构

- `swiss_roll_example.py`: 主要的示例类，包含完整的训练和推理流程
- `run_swiss_roll_example.py`: 简化的运行脚本，一键执行完整流程
- `data.py`: 数据生成和数据集类
- `model.py`: 扩散模型的神经网络架构
- `README.md`: 本说明文件

## 功能特性

1. **训练扩散模型**: 在Swiss roll数据上训练Karras扩散模型
2. **生成样本**: 从训练好的模型中生成新的数据点
3. **可视化结果**: 比较训练数据和生成数据的分布
4. **生成GIF**: 创建inference过程的动画，展示从噪声到最终样本的完整过程

## 快速开始

### 1. 安装依赖

确保你已经安装了所有必要的依赖：

```bash
pip install -r requirements.txt
```

### 2. 运行完整示例

最简单的运行方式：

```bash
cd playground/points_2d
python run_swiss_roll_example.py
```

这将：
- 训练一个扩散模型（约30个epoch）
- 生成1000个样本
- 创建可视化比较图
- 生成inference过程的GIF动画

### 3. 自定义运行

你也可以使用命令行参数自定义训练：

```bash
python swiss_roll_example.py --epochs 50 --batch-size 64 --lr 2e-4
```

可用的参数：
- `--epochs`: 训练轮数（默认50）
- `--batch-size`: 批次大小（默认128）
- `--lr`: 学习率（默认1e-4）
- `--n-train`: 训练样本数（默认4096）
- `--n-val`: 验证样本数（默认512）
- `--hidden-dim`: 隐藏层维度（默认256）
- `--num-layers`: 网络层数（默认2）

### 4. 仅运行推理

如果你已经有训练好的模型：

```bash
python swiss_roll_example.py --inference-only --model-path path/to/model.pt
```

## 输出文件

运行完成后，在`swiss_roll_output`目录中会生成以下文件：

- `model.pt`: 训练好的模型权重
- `results_comparison.png`: 训练数据vs生成数据的对比图
- `inference.gif`: inference过程的动画
- `lightning_logs/`: PyTorch Lightning的训练日志

## 模型架构

这个示例使用了一个简单的MLP架构：

- **输入**: 2D坐标点 (x, y)
- **时间嵌入**: 位置编码的时间嵌入
- **网络**: 多层感知机，包含残差连接
- **输出**: 去噪后的2D坐标点

## 训练配置

- **损失函数**: Karras预处理的MSE损失
- **优化器**: AdamW
- **学习率调度**: 余弦退火
- **噪声采样**: 对数均匀分布
- **EMA**: 指数移动平均

## 推理配置

- **ODE求解器**: Euler方法
- **噪声调度**: Karras噪声调度
- **步数**: 50步（可调整）

## 可视化说明

### 1. 结果对比图

`results_comparison.png`包含三个子图：
- 左图：训练数据（蓝色点）
- 中图：生成数据（红色点）
- 右图：训练数据vs生成数据对比

### 2. Inference GIF

`inference.gif`展示了完整的推理过程：
- 蓝色点：训练数据（背景）
- 红色点：当前推理步骤的样本
- 动画从纯噪声开始，逐步去噪到最终的Swiss roll形状

## 技术细节

### 扩散过程

1. **前向过程**: 逐步向数据添加噪声
2. **反向过程**: 学习去噪函数
3. **采样**: 从噪声开始，逐步去噪生成新样本

### 关键组件

- **KarrasDenoiser**: 预处理的去噪器包装器
- **LightningDiffusion**: PyTorch Lightning模块
- **EulerODE**: ODE求解器
- **NoiseSchedule**: 噪声调度策略

## 扩展和修改

你可以轻松修改这个示例：

1. **改变数据**: 修改`data.py`中的`make_swiss_roll_dataframe`函数
2. **调整模型**: 修改`model.py`中的网络架构
3. **自定义训练**: 调整`setup_training_config`中的超参数
4. **修改推理**: 调整`setup_inference_config`中的参数

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减少batch_size或hidden_dim
2. **训练不收敛**: 调整学习率或增加训练轮数
3. **GIF生成失败**: 确保安装了pillow库

### 性能优化

- 使用GPU加速训练（自动检测）
- 调整batch_size以平衡内存和速度
- 减少n_steps可以加快推理速度

## 参考资料

- [Karras et al., 2022]: Elucidating the Design Space of Diffusion-Based Generative Models
- [Ho et al., 2020]: Denoising Diffusion Probabilistic Models
- [Hang et al., 2022]: Min-SNR: A New Perspective on Training Diffusion Models 