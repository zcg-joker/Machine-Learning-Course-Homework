import torch
import torch.nn as nn
from config import Config

class NameRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=False, config=None):
        super(NameRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.config = config
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, 
                          batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        
        if self.bidirectional:
            # 合并双向输出
            output = output.view(output.size(0), output.size(1), 2, -1)
            output = output.sum(dim=2)
            
        output = self.fc(output)
        return output, hidden
    
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h0 = torch.zeros(self.n_layers * self.num_directions, 
                        batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers * self.num_directions, 
                        batch_size, self.hidden_size).to(device)
        return (h0, c0)
    
    def forward_generate(self, start_seq, char_to_idx, idx_to_char, max_length=None):
        """从开头生成名字"""
        device = next(self.parameters()).device
        self.eval()
        
        if max_length is None:
            max_length = self.config.max_length if self.config else 20
        
        with torch.no_grad():
            # 将起始序列转换为索引
            indices = [char_to_idx[c] for c in start_seq]
            x = torch.LongTensor([indices]).to(device)
            hidden = None
            
            # 记录生成历史
            generated = list(start_seq)
            probs_history = []
            chars_history = []
            
            while len(generated) < max_length:
                output, hidden = self(x, hidden)
                probs = torch.softmax(output[0, -1], dim=0)
                
                # 记录top k个概率和字符
                top_k_vis = self.config.top_k_vis if self.config else 5
                top_probs, top_indices = torch.topk(probs, top_k_vis)
                probs_history.append(top_probs.cpu().numpy())
                chars_history.append([idx_to_char[idx.item()] for idx in top_indices])
                
                # 采样下一个字符
                next_char_idx = torch.multinomial(probs, 1).item()
                next_char = idx_to_char[next_char_idx]
                
                if next_char == (self.config.end_token if self.config else '.'):
                    break
                    
                generated.append(next_char)
                x = torch.LongTensor([[next_char_idx]]).to(device)
        
        return ''.join(generated), probs_history, chars_history
    
    def backward_generate(self, end_seq, char_to_idx, idx_to_char, max_length=None):
        """从结尾往前生成名字（使用反向训练的模型）"""
        device = next(self.parameters()).device
        self.eval()
        
        if max_length is None:
            max_length = self.config.max_length if self.config else 20
        
        with torch.no_grad():
            # 将结尾序列反转
            end_seq_reversed = ''.join(reversed(end_seq))
            indices = [char_to_idx[c] for c in end_seq_reversed]
            x = torch.LongTensor([indices]).to(device)
            hidden = None
            
            # 记录生成历史
            generated = list(end_seq_reversed)
            probs_history = []
            chars_history = []
            
            while len(generated) < max_length:
                output, hidden = self(x, hidden)
                probs = torch.softmax(output[0, -1], dim=0)
                
                # 记录top k个概率和字符
                top_k_vis = self.config.top_k_vis if self.config else 5
                top_probs, top_indices = torch.topk(probs, top_k_vis)
                probs_history.append(top_probs.cpu().numpy())
                chars_history.append([idx_to_char[idx.item()] for idx in top_indices])
                
                # 采样下一个字符
                next_char_idx = torch.multinomial(probs, 1).item()
                next_char = idx_to_char[next_char_idx]
                
                if next_char == (self.config.start_token if self.config else ' '):
                    break
                    
                generated.append(next_char)
                x = torch.LongTensor([[next_char_idx]]).to(device)
            
            # 反转生成的序列得到最终结果
            final_name = ''.join(reversed(generated))
            
            return final_name, probs_history, chars_history
    
    def bidirectional_generate(self, middle_seq, char_to_idx, idx_to_char, max_length=20):
        """从中间向两端生成名字"""
        if not self.bidirectional:
            raise ValueError("Bidirectional generation requires bidirectional RNN")
            
        # 先向后生成
        forward_name, forward_probs, forward_chars = self.forward_generate(
            middle_seq, char_to_idx, idx_to_char, max_length//2)
            
        # 再向前生成
        backward_name, backward_probs, backward_chars = self.backward_generate(
            middle_seq, char_to_idx, idx_to_char, max_length//2)
        
        # 合并结果
        full_name = backward_name[:-len(middle_seq)] + forward_name
        full_probs = backward_probs + forward_probs
        full_chars = backward_chars + forward_chars
        
        return full_name, full_probs, full_chars 