import torch
import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, input_size=3072, hidden_sizes=[1024, 512], num_classes=10,
                 dropout_rate=0.5, use_batch_norm=True, l1_lambda=0.0, l2_lambda=0.0):
        super(FCNet, self).__init__()
        self.flatten = nn.Flatten()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, num_classes))
        self.fc_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
    
    def get_l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss
    
    def get_l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param.pow(2))
        return self.l2_lambda * l2_loss 