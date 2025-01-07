# 基于全连接网络的cifar-10图像分类

> 本项目为 BUPT 2024年秋季学期《高级机器学习》课程的作业 A1

本项目是一个基于 PyTorch 的图像分类模型，使用 CIFAR-10 数据集进行训练和评估。项目的主要目标是实现一个全连接神经网络（FCNet），并通过不同的优化器、损失函数和正则化方法进行实验，以找到最佳的模型配置。

## 项目结构

- `config.py`: 包含模型、训练、优化器、损失函数和正则化的配置。
- `main.py`: 主程序，负责加载数据、训练模型和评估结果。
- `models.py`: 定义了全连接神经网络（FCNet）的结构。
- `utils.py`: 包含数据加载、模型训练、评估和结果保存的工具函数。

## 环境要求

确保安装了以下依赖项：

- Python 3.12
- PyTorch
- torchvision
- swanlab
- numpy
- scikit-learn
- matplotlib
- seaborn

可以使用以下命令安装所需的库：

```bash
pip install torch torchvision swanlab numpy scikit-learn matplotlib seaborn
```

## 使用方法

1. **配置模型**: 在 `config.py` 中修改模型参数、训练参数、优化器和损失函数等配置。

2. **运行主程序**: 使用以下命令运行主程序，开始训练和评估模型：

   ```bash
   python main.py
   ```

3. **查看结果**: 训练完成后，结果将保存在 `completed_experiments.json` 文件中。您可以查看每个实验的训练损失、验证损失、测试损失、准确率等指标。

## 实验配置

在 `config.py` 中，您可以自定义以下配置：

- **模型配置**:
  - `name`: 模型名称（默认为 `FCNet`）
  - `input_size`: 输入大小（默认为 3072）
  - `hidden_sizes`: 隐藏层大小（默认为 `[1024, 512, 256]`）
  - `num_classes`: 类别数量（默认为 10）

- **训练配置**:
  - `batch_size`: 批量大小（默认为 256）
  - `num_epochs`: 训练轮数（默认为 30）
  - `device`: 设备（默认为可用的 GPU 或 CPU）

- **优化器**: 支持多种优化器，如 SGD、Adam 等。

- **损失函数**: 支持多种损失函数，如交叉熵损失、均方误差损失等。

- **正则化**: 支持多种正则化方法，如 Dropout、L1 和 L2 正则化。

## 结果可视化

在训练和评估过程中，项目会生成混淆矩阵的可视化图像，并使用 swanlab 进行记录和保存。使用下面命令可以查看实验结果

```bash
swanlab watch swanlog
```
