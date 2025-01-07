import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
from model import VAE, CVAE
class Visualizer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()
        
    def _to_image(self, tensor):
        """将tensor转换为numpy图像"""
        img = tensor.cpu().detach()
        img = img / 2 + 0.5  # 反归一化
        return img.numpy().transpose(1, 2, 0)
    
    def save_images(self, images, filename, nrow=8):
        """保存图像网格"""
        grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(self._to_image(grid))
        plt.axis('off')
        plt.savefig(os.path.join(self.config.sample_dir, filename))
        plt.close()
    
    def generate_samples(self, num_samples=64, labels=None):
        """生成样本图像"""
        with torch.no_grad():
            if isinstance(self.model, CVAE):
                if labels is None:
                    labels = torch.randint(0, self.config.num_classes, (num_samples,))
                labels = labels.to(self.device)
                samples = self.model.sample(num_samples, labels, self.device)
            else:
                samples = self.model.sample(num_samples, self.device)
            
            self.save_images(samples, f'samples_{num_samples}.png')
    
    def reconstruct_samples(self, data_loader):
        """重构测试集中的样本"""
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(data_loader):
                if batch_idx == 0:
                    data = data.to(self.device)
                    if isinstance(self.model, CVAE):
                        labels = labels.to(self.device)
                        recon, _, _ = self.model(data, labels)
                    else:
                        recon, _, _ = self.model(data)
                    
                    # 连接原始图像和重构图像
                    comparison = torch.cat([data[:8], recon[:8]])
                    self.save_images(comparison, 'reconstruction.png', nrow=8)
                    break
    
    def interpolate(self, num_samples=8):
        """在隐空间中进行插值"""
        with torch.no_grad():
            # 生成两个随机隐向量
            z1 = torch.randn(1, self.config.latent_dim).to(self.device)
            z2 = torch.randn(1, self.config.latent_dim).to(self.device)
            
            # 在两点之间进行线性插值
            alphas = torch.linspace(0, 1, num_samples).to(self.device)
            z_interp = torch.stack([z1 * (1 - alpha) + z2 * alpha for alpha in alphas])  # [num_samples, 1, latent_dim]
            z_interp = z_interp.squeeze(1)  # [num_samples, latent_dim]
            
            if isinstance(self.model, CVAE):
                # 对于CVAE，我们需要一个固定的标签
                label = torch.zeros(num_samples, dtype=torch.long).to(self.device)
                c_embed = self.model.condition_embedding(label)  # [num_samples, condition_dim]
                z_interp = torch.cat([z_interp, c_embed], dim=1)  # [num_samples, latent_dim + condition_dim]
            
            # 生成插值图像
            images = self.model.decoder(z_interp)
            self.save_images(images, 'interpolation.png', nrow=num_samples)
            
            
    
    def visualize_all(self, data_loader, num_samples=64):
        """生成所有可视化结果"""
        print("生成随机样本...")
        self.generate_samples(num_samples)
        
        print("生成重构样本...")
        self.reconstruct_samples(data_loader)
        
        print("生成插值结果...")
        self.interpolate()
        
        if isinstance(self.model, CVAE):
            print("生成条件样本...")
            # 为每个类别生成样本
            for label in range(self.config.num_classes):
                labels = torch.full((num_samples,), label, dtype=torch.long)
                self.generate_samples(num_samples, labels)
                print(f"已生成类别 {label} 的样本") 