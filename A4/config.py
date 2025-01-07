import torch
import os
from datetime import datetime

class Config:
    def __init__(self):
        # 基础目录
        self.base_dir = 'results'
        self.latest_exp_file = os.path.join(self.base_dir, 'latest_exp.txt')
        
        # 创建基础目录
        os.makedirs(self.base_dir, exist_ok=True)
        
        # 如果是训练模式，创建新的实验目录
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = os.path.join(self.base_dir, self.timestamp)
        
        # 模型类型
        self.model_type = 'forward'  # 'forward' 或 'reverse'
        
        # 数据相关
        self.data_path = 'data/english_names.txt'
        
        # 模型相关
        self.hidden_size = 32
        self.n_layers = 3
        self.bidirectional = False  # 是否使用双向RNN
        
        # 训练相关
        self.batch_size = 16
        self.n_epochs = 200
        self.learning_rate = 0.0001
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        # 生成相关
        self.max_length = 15
        self.temperature = 0.5
        
        # 解码策略相关
        self.beam_size = 5
        self.top_k = 10
        self.top_p = 0.9
        
        # 可视化相关
        self.top_k_vis = 5  # 可视化时显示前k个候选
        self.fig_size = (15, 8)
        self.dark_theme = False
        self.font_size = {
            'title': 16,      # 标题字体大小
            'label': 14,      # 轴标签字体大小
            'tick': 12,       # 刻度字体大小
            'legend': 12,     # 图例字体大小
            'annotation': 10  # 注释字体大小
        }
        
        # 模型保存相关
        self.model_dir = os.path.join(self.result_dir,'models')
        self.model_save_path = os.path.join(self.model_dir, 'name_rnn.pth')
        
        # 图片保存相关
        self.plot_dir = os.path.join(self.result_dir, 'plots')
        
        # 日志相关
        self.log_dir = os.path.join(self.result_dir, 'logs')
        self.log_file = os.path.join(self.log_dir, 'training.log')
        
        # 特殊标记
        self.start_token = ' '
        self.end_token = '.'
        self.pad_token = ' '
    
    def create_dirs(self):
        """创建必要的目录"""
        dirs = [
            self.result_dir,
            os.path.join(self.result_dir, 'models'),
            os.path.join(self.result_dir, 'plots'),
            os.path.join(self.result_dir, 'logs')
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def save_exp_dir(self):
        """保存当前实验目录作为最新实验"""
        with open(self.latest_exp_file, 'w') as f:
            f.write(self.timestamp)
    
    def load_latest_exp_dir(self):
        """加载最新实验目录"""
        if os.path.exists(self.latest_exp_file):
            with open(self.latest_exp_file, 'r') as f:
                self.timestamp = f.read().strip()
            self.result_dir = os.path.join(self.base_dir, self.timestamp)
            self.model_dir = os.path.join(self.result_dir, 'models')
            self.model_save_path = os.path.join(self.model_dir, 'name_rnn.pth')
            self.plot_dir = os.path.join(self.result_dir, 'plots')
            self.log_dir = os.path.join(self.result_dir, 'logs')
            self.log_file = os.path.join(self.log_dir, 'training.log')
            return True
        return False 
    
    def get_training_params(self):
        """获取训练相关参数"""
        return {
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'bidirectional': self.bidirectional,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'learning_rate': self.learning_rate,
            'device': str(self.device)
        }
    
    def get_generation_params(self):
        """获取生成相关参数"""
        return {
            'model_type': self.model_type,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'beam_size': self.beam_size,
            'top_k': self.top_k,
            'top_p': self.top_p
        } 