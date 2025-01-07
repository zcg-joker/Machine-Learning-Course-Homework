import torch
import argparse
import os
from config import VAEConfig, CVAEConfig
from model import VAE, CVAE
from train import Trainer
from visualizer import Visualizer
import swanlab
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_environment(config):
    """设置运行环境"""
    # 创建必要的目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logging.info(f"使用GPU: {torch.cuda.get_device_name(device)}")
    else:
        logging.info("使用CPU进行训练")
    
    return device

def main():
    parser = argparse.ArgumentParser(description='VAE/CVAE Training and Generation')
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'cvae'],
                      help='选择模型类型 (vae 或 cvae)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'],
                      help='训练模型或生成图像')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='用于生成的模型检查点路径')
    args = parser.parse_args()
    
    # 选择配置和模型
    if args.model == 'vae':
        config = VAEConfig()
        logging.info("初始化VAE模型...")
        model = VAE(config)
    else:
        config = CVAEConfig()
        logging.info("初始化CVAE模型...")
        model = CVAE(config)
    
    # 设置环境
    device = setup_environment(config)
    model = model.to(device)
    
    # 初始化SwanLab
    run = swanlab.init(
        project="vae_cvae",
        mode="local",
        experiment_name=args.model,
        config=vars(config)
    )
    logging.info(f"SwanLab实验名称: {args.model}")
    
    # 如果提供了检查点，加载模型权重
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"找不到检查点文件：{args.checkpoint}")
        
        logging.info(f"加载检查点: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
            logging.info(f"已加载检查点，训练轮次：{state_dict['epoch']}")
        else:
            model.load_state_dict(state_dict)
            logging.info("已加载模型权重")
    
    if args.mode == 'train':
        # 训练模式
        logging.info("开始训练模型...")
        trainer = Trainer(model, config)
        trainer.train()
        logging.info("训练完成！")
    else:
        # 生成模式
        if not args.checkpoint:
            raise ValueError("生成模式需要提供检查点路径")
        
        logging.info("开始生成图像...")
        visualizer = Visualizer(model, config)
        # 获取一个数据加载器用于重构可视化
        trainer = Trainer(model, config)
        _, test_loader = trainer.get_data_loaders()
        
        visualizer.visualize_all(test_loader)
        logging.info(f"所有生成的图像已保存到 {config.sample_dir} 目录")

if __name__ == '__main__':
    main()
