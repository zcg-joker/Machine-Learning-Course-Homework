import torch
import argparse
import os
from config import VAEConfig, CVAEConfig
from model import VAE, CVAE
from visualizer import Visualizer
from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def load_model(model_path, model_type='vae'):
    """加载训练好的模型"""
    print(f"正在加载模型：{model_path}")
    
    # 初始化配置和模型
    if model_type == 'vae':
        config = VAEConfig()
        model = VAE(config)
    else:
        config = CVAEConfig()
        model = CVAE(config)
    
    # 加载模型权重
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
        print(f"已加载检查点，训练轮次：{state_dict['epoch']}")
    else:
        model.load_state_dict(state_dict)
        print("已加载模型权重")
    
    return model, config

def get_cifar10_samples(config, num_samples_per_class=2):
    """获取CIFAR-10数据集的样本"""
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=False, 
                             transform=transform, download=False)
    
    # 按类别整理数据
    class_samples = {i: [] for i in range(10)}
    class_indices = {i: [] for i in range(10)}
    
    for idx, (image, label) in enumerate(dataset):
        if len(class_samples[label]) < num_samples_per_class:
            class_samples[label].append(image)
            class_indices[label].append(idx)
    
    return class_samples, class_indices

def generate_class_samples(model, config, device, num_samples=2):
    """为每个类别生成样本"""
    print("\n生成每个类别的样本...")
    model.eval()
    
    all_samples = []
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    with torch.no_grad():
        if isinstance(model, CVAE):
            # CVAE：为每个类别生成样本
            for label in range(10):
                labels = torch.full((num_samples,), label, dtype=torch.long).to(device)
                samples = model.sample(num_samples, labels, device)
                all_samples.append(samples)
                print(f"已生成类别 {class_names[label]} 的样本")
        else:
            # VAE：生成随机样本
            for _ in range(20):
                samples = model.sample(num_samples, device)
                all_samples.append(samples)
    
    # 将所有样本拼接成一个大网格
    all_samples = torch.cat(all_samples, dim=0)
    grid = torchvision.utils.make_grid(all_samples, nrow=num_samples, normalize=True)
    
    plt.figure(figsize=(15, 75))
    plt.imshow(grid.cpu().numpy().transpose(1, 2, 0))
    plt.axis('off')
    plt.title('Generated Samples for Each Class' if isinstance(model, CVAE) else 'Random Generated Samples')
    plt.savefig(os.path.join(config.sample_dir, 'class_samples.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

def reconstruct_original_samples(model, config, device):
    """重构原始样本并进行比较"""
    print("\n重构原始样本...")
    model.eval()
    
    # 获取原始样本
    class_samples, class_indices = get_cifar10_samples(config, num_samples_per_class=3)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    original_images = []
    reconstructed_images = []
    
    with torch.no_grad():
        for label in range(10):
            # 获取原始图像
            images = torch.stack(class_samples[label]).to(device)
            original_images.append(images)
            
            # 重构图像
            if isinstance(model, CVAE):
                labels = torch.full((len(images),), label, dtype=torch.long).to(device)
                recon, _, _ = model(images, labels)
            else:
                recon, _, _ = model(images)
            reconstructed_images.append(recon)
            print(f"已重构类别 {class_names[label]} 的样本")
    
    # 将所有图像拼接在一起
    original_images = torch.cat(original_images, dim=0)
    reconstructed_images = torch.cat(reconstructed_images, dim=0)
    comparison = torch.cat([original_images, reconstructed_images], dim=0)
    
    # 创建网格
    grid = torchvision.utils.make_grid(comparison, nrow=2, normalize=True)
    
    plt.figure(figsize=(10, 50))
    plt.imshow(grid.cpu().numpy().transpose(1, 2, 0))
    plt.axis('off')
    plt.title('Original vs Reconstructed')
    plt.savefig(os.path.join(config.sample_dir, 'reconstruction_comparison.png'), 
                bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='VAE/CVAE图像生成')
    parser.add_argument('--model_path', type=str, required=True,
                      help='训练好的模型路径')
    parser.add_argument('--model_type', type=str, default='vae',
                      choices=['vae', 'cvae'], help='模型类型')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到模型文件：{args.model_path}")
    
    # 加载模型
    model, config = load_model(args.model_path, args.model_type)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 为每个类别生成样本
    generate_class_samples(model, config, device, num_samples=3)
    
    # 重构原始样本
    reconstruct_original_samples(model, config, device)
    
    print(f"\n所有生成的图像已保存到 {config.sample_dir} 目录")

if __name__ == '__main__':
    main() 