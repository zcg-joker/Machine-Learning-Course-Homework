import torch
import torch.nn as nn
import random
import os
import logging
import numpy as np
from tqdm import tqdm
from config import Config
from model import NameRNN
import string
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def setup_logger(config):
    """设置日志"""
    logger = logging.getLogger('name_generator')
    
    # 如果logger已经有处理器，说明已经初始化过，直接返回
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(config.log_file)
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def create_char_maps():
    """创建字符到索引的映射"""
    # 只使用小写字母和特殊token
    chars = string.ascii_lowercase + " ."  # 26个小写字母，空格和点号
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    return char_to_idx, idx_to_char

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    将数据集分割为训练集、验证集和测试集
    使用固定的随机种子以确保可重复性
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 复制数据并打乱
    data = data.copy()
    random.shuffle(data)
    
    # 计算分割点
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # 分割数据
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

def calculate_bleu(reference, candidate):
    """
    计算单个样本的BLEU分数
    reference: 参考名字
    candidate: 生成的名字
    """
    # 将名字转换为字符列表
    reference = list(reference)
    candidate = list(candidate)
    
    # 使用平滑函数处理零概率情况
    smoothing = SmoothingFunction().method1
    
    # 计算1-4gram的BLEU分数
    weights = [(1,), (0.5, 0.5), (0.33, 0.33, 0.34), (0.25, 0.25, 0.25, 0.25)]
    bleu_scores = []
    
    for weight in weights:
        try:
            score = sentence_bleu([reference], candidate, 
                                weights=weight, 
                                smoothing_function=smoothing)
            bleu_scores.append(score)
        except Exception:
            bleu_scores.append(0.0)
    
    return bleu_scores

def is_valid_name(name, allowed_chars=None):
    """检查名字是否合法"""
    if allowed_chars is None:
        allowed_chars = set(string.ascii_lowercase + " .")
    
    # 检查长度
    if len(name) < 2 or len(name) > 20:
        return False
    
    # 检查字符
    if not all(c in allowed_chars for c in name.lower()):
        return False
    
    # 检查特殊字符使用是否合理
    if ".." in name or "  " in name:  # 避免重复的特殊字符
        return False
    
    if name.count(".") > 0 and not name.endswith("."):  # 点号只能出现在结尾
        return False
    
    return True

def preprocess_name(name):
    """预处理名字"""
    # 移除首尾空白
    name = name.strip()
    
    # 标准化空格（多个空格转换为单个空格）
    name = re.sub(r'\s+', ' ', name)
    
    # 转换为小写
    name = name.lower()
    
    return name

def load_data(filename, logger=None):
    """加载并过滤数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        names = f.read().splitlines()
    
    # 获取允许的字符集
    allowed_chars = set(string.ascii_lowercase + " .")
    
    # 过滤和预处理名字
    filtered_names = []
    invalid_names = []
    
    for name in names:
        # 预处理
        processed_name = preprocess_name(name)
        
        # 验证
        if is_valid_name(processed_name, allowed_chars):
            filtered_names.append(processed_name)
        else:
            invalid_names.append(name)
    
    # 记录过滤信息
    if logger:
        logger.info(f"总名字数量: {len(names)}")
        logger.info(f"有效名字数量: {len(filtered_names)}")
        logger.info(f"无效名字数量: {len(invalid_names)}")
        if invalid_names:
            logger.info("部分无效名字示例:")
            for name in invalid_names[:5]:
                logger.info(f"  - {name}")
    
    return filtered_names

def prepare_reverse_data(names):
    """准备反向训练数据"""
    return [''.join(reversed(name)) for name in names]

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # 设置日志
        self.logger = setup_logger(config)
        self.logger.info(f"初始化训练器 - 模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
    def prepare_batch(self, batch_names, char_to_idx):
        x = []
        y = []
        for name in batch_names:
            name = self.config.start_token + name + self.config.end_token
            indices = [char_to_idx[c] for c in name]
            x.append(indices[:-1])
            y.append(indices[1:])
        
        # 填充序列
        max_len = max(len(seq) for seq in x)
        x = [seq + [char_to_idx[self.config.pad_token]] * (max_len - len(seq)) for seq in x]
        y = [seq + [char_to_idx[self.config.pad_token]] * (max_len - len(seq)) for seq in y]
        
        return (torch.LongTensor(x).to(self.device),
                torch.LongTensor(y).to(self.device))
    
    def evaluate_generation(self, data, char_to_idx, num_samples=100):
        """评估生成质量"""
        self.model.eval()
        bleu_scores = {
            'bleu1_1char': [], 'bleu2_1char': [], 'bleu3_1char': [], 'bleu4_1char': [],
            'bleu1_2char': [], 'bleu2_2char': [], 'bleu3_2char': [], 'bleu4_2char': []
        }
        generated_samples = {'1char': [], '2char': []}
        
        # 创建反向映射
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        # 随机选择样本进行评估
        eval_data = random.sample(data, min(num_samples, len(data)))
        
        with torch.no_grad():
            for name in tqdm(eval_data, desc='Evaluating'):
                # 使用前1个字符生成
                prefix1 = name[0]
                x = torch.LongTensor([[char_to_idx[prefix1]]]).to(self.device)
                generated = [prefix1]
                hidden = None
                
                while len(generated) < self.config.max_length:
                    output, hidden = self.model(x, hidden)
                    # 重置hidden的batch size
                    if isinstance(hidden, tuple):
                        hidden = tuple(h[:, :1, :] for h in hidden)
                    else:
                        hidden = hidden[:, :1, :]
                    
                    probs = torch.softmax(output[0, -1], dim=0)
                    next_char_idx = torch.multinomial(probs, 1).item()
                    next_char = idx_to_char[next_char_idx]
                    
                    if next_char == self.config.end_token:
                        break
                    
                    generated.append(next_char)
                    x = torch.LongTensor([[next_char_idx]]).to(self.device)
                
                generated_name1 = ''.join(generated).strip()
                generated_samples['1char'].append(generated_name1)
                
                # 计算使用1个字符的BLEU分数
                scores = calculate_bleu(name, generated_name1)
                for i, score in enumerate(scores):
                    bleu_scores[f'bleu{i+1}_1char'].append(score)
                
                # 如果名字长度大于1，使用前2个字符生成
                if len(name) > 1:
                    prefix2 = name[:2]
                    x = torch.LongTensor([char_to_idx[c] for c in prefix2]).unsqueeze(0).to(self.device)
                    generated = list(prefix2)
                    hidden = None
                    
                    while len(generated) < self.config.max_length:
                        output, hidden = self.model(x, hidden)
                        # 重置hidden的batch size
                        if isinstance(hidden, tuple):
                            hidden = tuple(h[:, :1, :] for h in hidden)
                        else:
                            hidden = hidden[:, :1, :]
                        
                        probs = torch.softmax(output[0, -1], dim=0)
                        next_char_idx = torch.multinomial(probs, 1).item()
                        next_char = idx_to_char[next_char_idx]
                        
                        if next_char == self.config.end_token:
                            break
                        
                        generated.append(next_char)
                        x = torch.LongTensor([[next_char_idx]]).to(self.device)
                    
                    generated_name2 = ''.join(generated).strip()
                    generated_samples['2char'].append(generated_name2)
                    
                    # 计算使用2个字符的BLEU分数
                    scores = calculate_bleu(name, generated_name2)
                    for i, score in enumerate(scores):
                        bleu_scores[f'bleu{i+1}_2char'].append(score)
        
        # 计算平均BLEU分数
        avg_scores = {k: np.mean(v) for k, v in bleu_scores.items()}
        
        # 记录生成样本
        self.logger.info("\n生成样本示例:")
        self.logger.info("使用1个字符生成:")
        for ref, gen in list(zip(eval_data, generated_samples['1char']))[:3]:
            self.logger.info(f"参考: {ref} -> 生成: {gen}")
        
        self.logger.info("\n使用2个字符生成:")
        for ref, gen in list(zip(eval_data, generated_samples['2char']))[:3]:
            self.logger.info(f"参考: {ref} -> 生成: {gen}")
        
        return avg_scores, generated_samples

    def train_epoch(self, data, char_to_idx):
        self.model.train()
        total_loss = 0
        total_chars = 0
        correct_chars = 0
        
        # 准备批次数据
        indices = list(range(len(data)))
        random.shuffle(indices)
        
        # 创建进度条
        pbar = tqdm(range(0, len(indices), self.config.batch_size), desc='Training')
        for i in pbar:
            # 获取当前批次的索引
            batch_indices = indices[i:i + self.config.batch_size]
            batch_names = [data[idx] for idx in batch_indices]
            x, y = self.prepare_batch(batch_names, char_to_idx)
            
            # 前向传播
            self.optimizer.zero_grad()
            output, _ = self.model(x)
            
            # 计算损失
            loss = self.criterion(output.view(-1, output.size(-1)), y.view(-1))
            
            # 计算准确率
            predictions = output.argmax(dim=-1)
            mask = y != char_to_idx[self.config.pad_token]
            correct = (predictions == y)[mask].sum().item()
            chars = mask.sum().item()
            
            correct_chars += correct
            total_chars += chars
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            current_acc = correct / chars if chars > 0 else 0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / (len(data) // self.config.batch_size)
        accuracy = correct_chars / total_chars if total_chars > 0 else 0
        
        return avg_loss, accuracy
    
    def validate(self, data, char_to_idx):
        self.model.eval()
        total_loss = 0
        total_chars = 0
        correct_chars = 0
        
        with torch.no_grad():
            pbar = tqdm(range(0, len(data), self.config.batch_size), desc='Validating')
            for i in pbar:
                batch_names = data[i:i + self.config.batch_size]
                x, y = self.prepare_batch(batch_names, char_to_idx)
                
                output, _ = self.model(x)
                loss = self.criterion(output.view(-1, output.size(-1)), y.view(-1))
                
                predictions = output.argmax(dim=-1)
                mask = y != char_to_idx[self.config.pad_token]
                correct = (predictions == y)[mask].sum().item()
                chars = mask.sum().item()
                
                correct_chars += correct
                total_chars += chars
                total_loss += loss.item()
                
                # 更新进度条
                current_acc = correct / chars if chars > 0 else 0
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}'
                })
        
        avg_loss = total_loss / (len(data) // self.config.batch_size)
        accuracy = correct_chars / total_chars if total_chars > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, train_data, val_data=None, char_to_idx=None, reverse=False):
        """
        训练模型
        reverse: 是否使用反向数据进行训练
        """
        if char_to_idx is None:
            char_to_idx, _ = create_char_maps()
        
        # 如果是反向训练，对数据进行反向处理
        if reverse:
            train_data = prepare_reverse_data(train_data)
            if val_data is not None:
                val_data = prepare_reverse_data(val_data)
            self.logger.info("使用反向数据进行训练...")
            
        self.logger.info("开始训练...")
        self.logger.info(f"训练集大小: {len(train_data)}")
        if val_data:
            self.logger.info(f"验证集大小: {len(val_data)}")
            
        best_val_loss = float('inf')
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.config.n_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config.n_epochs}")
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(train_data, char_to_idx)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # 验证
            if val_data is not None:
                val_loss, val_acc = self.validate(val_data, char_to_idx)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(reverse)
                    self.logger.info("保存最佳模型")
                
                self.logger.info(
                    f'Epoch {epoch+1} - '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
                )
            else:
                self.logger.info(
                    f'Epoch {epoch+1} - '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}'
                )
        
        self.logger.info("训练完成!")
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def save_model(self, reverse=False):
        """保存模型，添加是否为反向模型的标记"""
        save_path = self.config.model_save_path
        if reverse:
            save_path = save_path.replace('.pth', '_reverse.pth')
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reverse': reverse
        }, save_path)
        self.logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, reverse=False):
        """加载模型，支持加载反向模型"""
        load_path = self.config.model_save_path
        if reverse:
            load_path = load_path.replace('.pth', '_reverse.pth')
            
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info(f"加载模型从: {load_path}")
            return True
        self.logger.error(f"找不到模型文件: {load_path}")
        return False 