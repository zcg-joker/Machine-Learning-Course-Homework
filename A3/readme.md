# 基于变分自编码器（VAE）和条件变分自编码器（CVAE）的图像生成

> 本项目为 BUPT 2024年秋季学期《高级机器学习》课程的作业 A3

## 项目简介

本项目实现了基于变分自编码器（VAE）和条件变分自编码器（CVAE）的图像生成模型，使用 CIFAR-10 数据集进行训练和评估。项目的主要目标是通过 VAE 和 CVAE 生成高质量的图像，并能够根据条件生成特定类别的图像。

## 项目结构

- `config.py`: 包含模型、训练和数据集的配置。
- `main.py`: 主程序，负责加载数据、训练模型和评估结果。
- `model.py`: 定义了 VAE 和 CVAE 的结构。
- `train.py`: 包含训练过程的实现。
- `predict.py`: 用于加载训练好的模型并生成图像。
- `visualizer.py`: 包含可视化生成图像和重构图像的工具函数。

## 环境要求

确保安装了以下依赖项：

- Python 3.12
- PyTorch
- torchvision
- swanlab
- numpy
- matplotlib

## 使用方法

1. **配置模型**: 在 `config.py` 中修改模型参数、训练参数等配置。

2. **运行主程序**: 使用以下命令运行主程序，开始训练模型：

   ```bash
   python main.py --model vae --mode train
   ```

   或者使用 CVAE 模型：

   ```bash
   python main.py --model cvae --mode train
   ```

    **（由于未作命名区分，所以在训练完VAE后，CVAE的最佳checkpoint会覆盖VAE的最佳checkpoint，建议手动重命名进行区分。后续会进行改进。）**

3. **生成图像**: 训练完成后，可以使用以下命令生成图像：

   ```bash
   python predict.py --model_path <模型检查点路径> --model_type vae
   ```

   或者使用 CVAE 模型：

   ```bash
   python predict.py --model_path <模型检查点路径> --model_type cvae
   ```


4. **查看结果**: 生成的图像将保存在 `./samples` 目录中。您可以查看生成的样本和重构的图像。

    **（由于未作命名区分，所以在使用VAE进行预测后，CVAE预测会覆盖VAE预测结果图，建议手动重命名进行区分。后续会进行改进。）**

## 实验配置

在 `config.py` 中，您可以自定义以下配置：

- **数据集配置**:
  - `batch_size`: 批量大小（默认为 128）
  - `image_size`: 输入图像大小（默认为 32）

- **模型配置**:
  - `latent_dim`: 潜在空间维度（默认为 128）
  - `hidden_dims`: 隐藏层维度（默认为 `[32, 64, 128, 256, 512]`）

- **训练配置**:
  - `learning_rate`: 学习率（默认为 3e-4）
  - `num_epochs`: 训练轮数（默认为 100）
  - `beta`: KL 散度的权重（默认为 0.05）
  - `device`: 设备（默认为 `cuda:3`）

- **优化器配置**:
  - `weight_decay`: 权重衰减（默认为 1e-5）
  - `scheduler_gamma`: 学习率调度因子（默认为 0.98）

- **数据保存配置**:
  - `checkpoint_dir`: 检查点保存目录（默认为 `./checkpoints`）
  - `sample_dir`: 生成样本保存目录（默认为 `./samples`）

## 结果可视化

在训练和评估过程中，项目会生成样本的可视化图像，并使用 `matplotlib` 进行记录和保存。您可以查看生成的样本和重构的图像。

