import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
from model import VAE, CVAE
import swanlab
import logging

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # 使用带权重衰减的Adam优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.scheduler_gamma
        )
        
        logging.info(f"优化器：Adam, 初始学习率：{config.learning_rate}")
        logging.info(f"权重衰减：{config.weight_decay}")
        logging.info(f"学习率衰减因子：{config.scheduler_gamma}")
        
        # 创建保存目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 添加数据增强
            transforms.RandomRotation(10),      # 添加随机旋转
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def get_data_loaders(self):
        logging.info("加载本地CIFAR-10数据集...")
        
        # 检查数据集是否存在
        dataset_path = './data/cifar-10-batches-py'
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                "找不到CIFAR-10数据集！请确保数据集已解压到 ./data/cifar-10-batches-py 目录下"
            )
        
        try:
            train_dataset = datasets.CIFAR10(
                root='./data', train=True, transform=self.transform, download=False)
            test_dataset = datasets.CIFAR10(
                root='./data', train=False, transform=self.transform, download=False)
        except Exception as e:
            raise Exception(
                "加载CIFAR-10数据集失败！请确保数据集格式正确，并已正确解压。\n"
                "目录结构应该是：\n"
                "./data/cifar-10-batches-py/data_batch_1\n"
                "./data/cifar-10-batches-py/data_batch_2\n"
                "等等...\n"
                f"错误信息: {str(e)}"
            )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, 
            shuffle=True, num_workers=self.config.num_workers)
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers)
        
        logging.info(f"数据集大小 - 训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")
        logging.info("数据集加载成功！")
        return train_loader, test_loader
    
    def loss_function(self, recon_x, x, mu, log_var, train=True):
        # 重构损失（使用MSE和L1损失的组合）
        mse_loss = F.mse_loss(recon_x, x, reduction='sum')
        l1_loss = F.l1_loss(recon_x, x, reduction='sum')
        recon_loss = mse_loss + 0.1 * l1_loss
        
        if train:
            swanlab.log({"recon_loss": recon_loss.item() / x.size(0)})
            swanlab.log({"mse_loss": mse_loss.item() / x.size(0)})
            swanlab.log({"l1_loss": l1_loss.item() / x.size(0)})
        else:
            swanlab.log({"test_recon_loss": recon_loss.item() / x.size(0)})
            
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        if train:
            swanlab.log({"kl_loss": kl_loss.item() / x.size(0)})
        else:
            swanlab.log({"test_kl_loss": kl_loss.item() / x.size(0)})
            
        # 总损失
        beta = self.config.beta * self._get_beta_weight(self.current_epoch)
        total_loss = recon_loss + beta * kl_loss
        return total_loss
    
    def _get_beta_weight(self, epoch):
        """KL散度权重的预热策略"""
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}') as pbar:
            for batch_idx, (data, labels) in enumerate(pbar):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                
                if isinstance(self.model, CVAE):
                    labels = labels.to(self.device)
                    recon_batch, mu, log_var = self.model(data, labels)
                else:
                    recon_batch, mu, log_var = self.model(data)
                
                loss = self.loss_function(recon_batch, data, mu, log_var, train=True)
                loss.backward()
                
                # 记录损失
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                self.optimizer.step()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{avg_loss/len(data):.4f}',
                    'recon_loss': f'{recon_loss_sum/(batch_idx+1)/len(data):.4f}',
                    'kl_loss': f'{kl_loss_sum/(batch_idx+1)/len(data):.4f}'
                })
        
        return total_loss / len(train_loader.dataset)
    
    def test_epoch(self, test_loader):
        self.model.eval()
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        
        with torch.no_grad():
            with tqdm(test_loader, desc='Testing') as pbar:
                for batch_idx, (data, labels) in enumerate(pbar):
                    data = data.to(self.device)
                    
                    if isinstance(self.model, CVAE):
                        labels = labels.to(self.device)
                        recon_batch, mu, log_var = self.model(data, labels)
                    else:
                        recon_batch, mu, log_var = self.model(data)
                    
                    loss = self.loss_function(recon_batch, data, mu, log_var, train=False)
                    total_loss += loss.item()
                    
                    # 更新进度条
                    avg_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix({
                        'loss': f'{avg_loss/len(data):.4f}',
                        'recon_loss': f'{recon_loss_sum/(batch_idx+1)/len(data):.4f}',
                        'kl_loss': f'{kl_loss_sum/(batch_idx+1)/len(data):.4f}'
                    })
        
        return total_loss / len(test_loader.dataset)
    
    def train(self):
        train_loader, test_loader = self.get_data_loaders()
        best_test_loss = float('inf')
        patience = 20  # 早停的耐心值
        no_improve = 0
        self.current_epoch = 0
        
        logging.info("开始训练...")
        logging.info(f"总轮次: {self.config.num_epochs}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch
            
            # 训练一个轮次
            train_loss = self.train_epoch(train_loader, epoch)
            swanlab.log({"train_loss": train_loss})
            
            # 测试
            test_loss = self.test_epoch(test_loader)
            swanlab.log({"test_loss": test_loss})
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            swanlab.log({"learning_rate": current_lr})
            
            # 打印进度
            logging.info(f'Epoch {epoch}/{self.config.num_epochs}:')
            logging.info(f'  Train Loss: {train_loss:.4f}')
            logging.info(f'  Test Loss: {test_loss:.4f}')
            logging.info(f'  Learning Rate: {current_lr:.6f}')
            
            # 保存最佳模型
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                no_improve = 0
                checkpoint_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                }, checkpoint_path)
                logging.info(f"保存最佳模型，测试损失: {test_loss:.4f}")
            else:
                no_improve += 1
            
            # 早停检查
            if no_improve >= patience:
                logging.info(f"连续{patience}轮未改善，停止训练")
                break
        
        logging.info("训练完成！") 