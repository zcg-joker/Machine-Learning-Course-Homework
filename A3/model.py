import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, hidden_dims, latent_dim, in_channels=3):
        super().__init__()
        
        modules = []
        # 构建编码器卷积层
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # 计算展平后的特征维度
        # 对于32x32的输入图像，经过5次stride=2的卷积，特征图大小为1x1
        self.flatten_dim = hidden_dims[-1] * 1 * 1
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, hidden_dims, latent_dim):
        super().__init__()
        
        # 计算初始特征图大小
        # 对于32x32的目标图像，需要从1x1开始
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 1 * 1)
        
        modules = []
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                     kernel_size=3, padding=1),
            nn.Tanh())
        
    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, x.shape[1], 1, 1)  # 重塑为1x1特征图
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.encoder = Encoder(config.hidden_dims, config.latent_dim)
        self.decoder = Decoder(config.hidden_dims, config.latent_dim)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
    
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples

class CVAE(VAE):
    def __init__(self, config):
        super().__init__(config)
        self.condition_embedding = nn.Embedding(config.num_classes, 
                                             config.condition_embedding_dim)
        
        # 重新定义编码器和解码器，考虑条件信息
        self.encoder = Encoder(config.hidden_dims, 
                             config.latent_dim,
                             in_channels=3 + config.condition_embedding_dim)
        self.decoder = Decoder(config.hidden_dims,
                             config.latent_dim + config.condition_embedding_dim)
        
    def forward(self, x, c):
        # 处理条件信息
        c_embed = self.condition_embedding(c)  # [B, embed_dim]
        c_embed = c_embed.unsqueeze(-1).unsqueeze(-1)  # [B, embed_dim, 1, 1]
        c_embed = c_embed.expand(-1, -1, x.size(2), x.size(3))  # [B, embed_dim, H, W]
        
        # 将输入和条件连接
        x_c = torch.cat([x, c_embed], dim=1)
        
        # 编码
        mu, log_var = self.encoder(x_c)
        z = self.reparameterize(mu, log_var)
        
        # 解码（需要将隐变量和条件连接）
        c_embed_flat = self.condition_embedding(c)  # 直接使用嵌入结果
        z_c = torch.cat([z, c_embed_flat], dim=1)
        
        return self.decoder(z_c), mu, log_var
    
    def sample(self, num_samples, c, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        c_embed = self.condition_embedding(c)
        z_c = torch.cat([z, c_embed], dim=1)
        samples = self.decoder(z_c)
        return samples 