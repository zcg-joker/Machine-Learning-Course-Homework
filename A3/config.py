class VAEConfig:
    # 数据集配置
    batch_size = 128
    image_size = 32
    num_workers = 2
    
    # 模型配置
    latent_dim = 128
    hidden_dims = [32, 64, 128, 256, 512]
    
    # 训练配置
    learning_rate = 3e-4
    num_epochs = 100
    beta = 0.05
    device = 'cuda:3'
    
    # 优化器配置
    weight_decay = 1e-5
    scheduler_gamma = 0.98
    
    # 数据保存配置
    checkpoint_dir = './checkpoints'
    sample_dir = './samples'
    
class CVAEConfig(VAEConfig):
    # CVAE特有的配置
    num_classes = 10
    condition_embedding_dim = 64 