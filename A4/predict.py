import torch
import torch.nn.functional as F
import numpy as np
from config import Config

class NameGenerator:
    def __init__(self, model, char_to_idx, idx_to_char, config):
        self.model = model
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.config = config
        self.device = next(model.parameters()).device
        
    def _prepare_input(self, sequence):
        indices = [self.char_to_idx[c] for c in sequence]
        return torch.LongTensor([indices]).to(self.device)
    
    def _get_next_probs(self, output, temperature=1.0):
        logits = output[0, -1] / temperature
        return F.softmax(logits, dim=0)
    
    def _top_k_filtering(self, probs, k):
        top_k = torch.topk(probs, k)
        filtered_probs = torch.zeros_like(probs)
        filtered_probs[top_k.indices] = top_k.values
        return filtered_probs / filtered_probs.sum()
    
    def _top_p_filtering(self, probs, p):
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        filtered_probs = torch.zeros_like(probs)
        keep_probs = sorted_probs[~sorted_indices_to_remove]
        keep_indices = sorted_indices[~sorted_indices_to_remove]
        filtered_probs[keep_indices] = keep_probs
        return filtered_probs / filtered_probs.sum()
    
    def generate_greedy(self, start_seq):
        """贪婪解码策略"""
        self.model.eval()
        with torch.no_grad():
            generated = list(start_seq)
            x = self._prepare_input(start_seq)
            hidden = None
            probs_history = []
            chars_history = []
            
            while len(generated) < self.config.max_length:
                output, hidden = self.model(x, hidden)
                probs = self._get_next_probs(output)
                
                # 记录top k个概率和字符
                top_probs, top_indices = torch.topk(probs, self.config.top_k_vis)
                probs_history.append(top_probs.cpu().numpy())
                chars_history.append([self.idx_to_char[idx.item()] for idx in top_indices])
                
                # 选择概率最大的字符
                next_char_idx = probs.argmax().item()
                next_char = self.idx_to_char[next_char_idx]
                
                if next_char == self.config.end_token:
                    break
                    
                generated.append(next_char)
                x = torch.LongTensor([[next_char_idx]]).to(self.device)
        
        return ''.join(generated), probs_history, chars_history
    
    def generate_temperature(self, start_seq, temperature=0.5):
        """基于temperature的采样策略"""
        self.model.eval()
        with torch.no_grad():
            generated = list(start_seq)
            x = self._prepare_input(start_seq)
            hidden = None
            probs_history = []
            chars_history = []
            
            while len(generated) < self.config.max_length:
                output, hidden = self.model(x, hidden)
                probs = self._get_next_probs(output, temperature)
                
                # 记录top k个概率和字符
                top_probs, top_indices = torch.topk(probs, self.config.top_k_vis)
                probs_history.append(top_probs.cpu().numpy())
                chars_history.append([self.idx_to_char[idx.item()] for idx in top_indices])
                
                # 按概率采样
                next_char_idx = torch.multinomial(probs, 1).item()
                next_char = self.idx_to_char[next_char_idx]
                
                if next_char == self.config.end_token:
                    break
                    
                generated.append(next_char)
                x = torch.LongTensor([[next_char_idx]]).to(self.device)
        
        return ''.join(generated), probs_history, chars_history
    
    def generate_beam_search(self, start_seq, beam_size=5):
        """集束搜索策略"""
        self.model.eval()
        with torch.no_grad():
            x = self._prepare_input(start_seq)
            hidden = None
            
            # 初始化beam
            output, hidden = self.model(x, hidden)
            probs = self._get_next_probs(output)
            top_probs, top_indices = torch.topk(probs, beam_size)
            beams = [(start_seq + self.idx_to_char[idx.item()], 
                     hidden, 
                     prob.item(),
                     [probs.cpu().numpy()],
                     [[self.idx_to_char[i.item()] for i in top_indices]])
                    for idx, prob in zip(top_indices, top_probs)]
            
            # Beam search
            while len(beams[0][0]) < self.config.max_length:
                candidates = []
                for sequence, hidden, score, probs_hist, chars_hist in beams:
                    if sequence[-1] == self.config.end_token:
                        candidates.append((sequence, hidden, score, probs_hist, chars_hist))
                        continue
                        
                    x = self._prepare_input(sequence[-1])
                    output, new_hidden = self.model(x, hidden)
                    probs = self._get_next_probs(output)
                    
                    top_probs, top_indices = torch.topk(probs, beam_size)
                    for idx, prob in zip(top_indices, top_probs):
                        new_seq = sequence + self.idx_to_char[idx.item()]
                        new_score = score + np.log(prob.item())
                        new_probs_hist = probs_hist + [probs.cpu().numpy()]
                        new_chars_hist = chars_hist + [[self.idx_to_char[i.item()] 
                                                      for i in top_indices]]
                        candidates.append((new_seq, new_hidden, new_score, 
                                        new_probs_hist, new_chars_hist))
                
                # 选择最好的beam_size个候选
                beams = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_size]
                
                # 如果最好的序列已经结束，就停止生成
                if beams[0][0][-1] == self.config.end_token:
                    break
            
            # 返回得分最高的序列
            best_sequence, _, _, probs_history, chars_history = beams[0]
            return best_sequence, probs_history, chars_history
    
    def generate_top_k(self, start_seq, k=40):
        """Top-k采样策略"""
        self.model.eval()
        with torch.no_grad():
            generated = list(start_seq)
            x = self._prepare_input(start_seq)
            hidden = None
            probs_history = []
            chars_history = []
            
            while len(generated) < self.config.max_length:
                output, hidden = self.model(x, hidden)
                probs = self._get_next_probs(output)
                filtered_probs = self._top_k_filtering(probs, k)
                
                # 记录top k个概率和字符
                top_probs, top_indices = torch.topk(filtered_probs, self.config.top_k_vis)
                probs_history.append(top_probs.cpu().numpy())
                chars_history.append([self.idx_to_char[idx.item()] for idx in top_indices])
                
                # 按过滤后的概率采样
                next_char_idx = torch.multinomial(filtered_probs, 1).item()
                next_char = self.idx_to_char[next_char_idx]
                
                if next_char == self.config.end_token:
                    break
                    
                generated.append(next_char)
                x = torch.LongTensor([[next_char_idx]]).to(self.device)
        
        return ''.join(generated), probs_history, chars_history
    
    def generate_top_p(self, start_seq, p=0.9):
        """Top-p (nucleus) 采样策略"""
        self.model.eval()
        with torch.no_grad():
            generated = list(start_seq)
            x = self._prepare_input(start_seq)
            hidden = None
            probs_history = []
            chars_history = []
            
            while len(generated) < self.config.max_length:
                output, hidden = self.model(x, hidden)
                probs = self._get_next_probs(output)
                filtered_probs = self._top_p_filtering(probs, p)
                
                # 记录top k个概率和字符
                top_probs, top_indices = torch.topk(filtered_probs, self.config.top_k_vis)
                probs_history.append(top_probs.cpu().numpy())
                chars_history.append([self.idx_to_char[idx.item()] for idx in top_indices])
                
                # 按过滤后的概率采样
                next_char_idx = torch.multinomial(filtered_probs, 1).item()
                next_char = self.idx_to_char[next_char_idx]
                
                if next_char == self.config.end_token:
                    break
                    
                generated.append(next_char)
                x = torch.LongTensor([[next_char_idx]]).to(self.device)
        
        return ''.join(generated), probs_history, chars_history 