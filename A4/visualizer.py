import matplotlib.pyplot as plt
import numpy as np
import os
from config import Config

class Visualizer:
    def __init__(self, config):
        self.config = config
        if config.dark_theme:
            plt.style.use('dark_background')
    
    def save_plot(self, name):
        """保存图片"""
        plt.savefig(os.path.join(self.config.plot_dir, f'{name}.png'))
    
    def plot_training_curves(self, history):
        """绘制训练过程的损失和准确率曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制损失曲线
        ax1.plot(history['train_losses'], label='Train Loss')
        if history['val_losses']:
            ax1.plot(history['val_losses'], label='Val Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 绘制准确率曲线
        ax2.plot(history['train_accuracies'], label='Train Accuracy')
        if history['val_accuracies']:
            ax2.plot(history['val_accuracies'], label='Val Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        self.save_plot('training_curves')
        plt.show()
    
    def plot_generation_process(self, name, probs_history, chars_history):
        """可视化名字生成过程"""
        plt.figure(figsize=self.config.fig_size)
        
        # 创建热力图数据
        data = np.array(probs_history)
        
        # 绘制热力图
        plt.imshow(data.T, aspect='auto', cmap='viridis')
        
        # 添加字符标注
        for i in range(len(chars_history)):
            for j in range(min(len(chars_history[i]), self.config.top_k_vis)):
                plt.text(i, j, chars_history[i][j], 
                        ha='center', va='center', color='white')
        
        plt.title(f'Generated Name: {name}', pad=20, color='white', size=14)
        plt.xlabel('Generation Step', color='white')
        plt.ylabel('Top K Candidates', color='white')
        
        # 设置y轴标签
        plt.yticks(range(self.config.top_k_vis), 
                  [f'{i+1}st' for i in range(self.config.top_k_vis)])
        
        # 添加颜色条
        plt.colorbar(label='Probability')
        
        if self.config.dark_theme:
            plt.gca().set_facecolor('black')
            plt.gcf().set_facecolor('black')
        
        plt.tight_layout()
        self.save_plot(f'generation_process_{name}')
        plt.show()
    
    def plot_character_distribution(self, names, position='first'):
        """绘制字符分布"""
        if position == 'first':
            chars = [name[0] for name in names]
        elif position == 'last':
            chars = [name[-1] for name in names]
        else:
            chars = [char for name in names for char in name]
        
        # 统计字符频率
        char_freq = {}
        for char in chars:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # 按频率排序
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        chars, freqs = zip(*sorted_chars)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(chars)), freqs)
        plt.xticks(range(len(chars)), chars, rotation=45)
        
        if position == 'first':
            title = 'First Letter Distribution'
        elif position == 'last':
            title = 'Last Letter Distribution'
        else:
            title = 'Character Distribution'
            
        plt.title(title)
        plt.xlabel('Characters')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        self.save_plot(f'char_dist_{position}')
        plt.show()
    
    def plot_name_length_distribution(self, names):
        """绘制名字长度分布"""
        lengths = [len(name) for name in names]
        
        plt.figure(figsize=(10, 5))
        plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2, 1),
                edgecolor='black')
        plt.title('Name Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        self.save_plot('name_length_dist')
        plt.show()
    
    def plot_generation_comparison(self, results):
        """比较不同生成策略的结果"""
        n_methods = len(results)
        fig, axes = plt.subplots(n_methods, 1, figsize=(15, 5*n_methods))
        
        if n_methods == 1:
            axes = [axes]
        
        for ax, (method, (name, probs, chars)) in zip(axes, results.items()):
            data = np.array(probs)
            im = ax.imshow(data.T, aspect='auto', cmap='viridis')
            
            # 添加字符标注
            for i in range(len(chars)):
                for j in range(min(len(chars[i]), self.config.top_k_vis)):
                    ax.text(i, j, chars[i][j], ha='center', va='center', color='white')
            
            ax.set_title(f'{method}: {name}')
            ax.set_xlabel('Generation Step')
            ax.set_ylabel('Top K Candidates')
            
            # 设置y轴标签
            ax.set_yticks(range(self.config.top_k_vis))
            ax.set_yticklabels([f'{i+1}st' for i in range(self.config.top_k_vis)])
            
            plt.colorbar(im, ax=ax, label='Probability')
        
        plt.tight_layout()
        self.save_plot('generation_comparison')
        plt.show() 