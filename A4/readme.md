# 项目A4：英文名字生成器

> 本项目为 BUPT 2024年秋季学期《高级机器学习》课程的作业 A4

## 项目介绍

本项目是一个基于循环神经网络（RNN）的英文名字生成器。它使用LSTM模型来生成英文名字，支持从开头、结尾或中间生成名字，并提供多种解码策略（如贪婪解码、温度采样、集束搜索等）。项目还包括数据爬取、分析和可视化功能，能够从指定网站爬取英文名字并进行统计分析。

## 环境要求

- Python 3.12
- PyTorch
- 其他依赖库（如 `requests`, `beautifulsoup4`, `pandas`, `matplotlib`, `seaborn`, `nltk` 等）

## 使用方法

1. **配置模型**: 在 `config.py` 中修改模型参数、训练参数等配置。

2. **爬取英文名字**: 运行以下命令以爬取英文名字并保存到文件中：

   ```bash
   python en_name_spider.py
   ```

3. **训练模型**: 使用以下命令训练模型：

   ```bash
   python main.py --train --model-type forward
   ```

   或者使用反向模型：

   ```bash
   python main.py --train --model-type reverse
   ```

4. **生成名字**: 训练完成后，可以使用以下命令生成名字：

   ```bash
   python main.py --generate --model-type forward
   ```

   或者使用反向模型：

   ```bash
   python main.py --generate --model-type reverse
   ```

5. **查看结果**: 生成的名字和可视化结果将保存在 `results` 目录中。您可以查看生成的样本和训练过程的可视化图表。

## 目录结构

```
Machine-Learning-Course-Homework/A4/
├── config.py          # 配置文件
├── en_name_spider.py  # 爬虫脚本
├── main.py            # 主程序
├── model.py           # 模型定义
├── predict.py         # 名字生成器
├── train.py           # 训练过程
├── visualizer.py      # 可视化工具
└── requirements.txt    # 依赖库列表
```

## 注意事项

- 确保在运行爬虫脚本时网络连接正常。
- 训练模型可能需要一定时间，具体取决于数据集大小和计算资源。
- 生成名字时，可以根据需要选择不同的生成策略。


