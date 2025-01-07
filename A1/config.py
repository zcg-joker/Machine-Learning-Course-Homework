import torch

def get_base_config():
    return {
        'model': {
            'name': 'FCNet',
            'input_size': 3072,
            'hidden_sizes': [1024, 512, 256],
            'num_classes': 10
        },
        'training': {
            'batch_size': 256,
            'num_epochs': 30,
            'device': str(torch.device("cuda:5" if torch.cuda.is_available() else "cpu"))
        },
        'optimizers': {
            'SGD': {'lr': 0.01},
            'SGD_with_momentum': {'lr': 0.01, 'momentum': 0.9},  # 改为SGD_with_momentum
            'SGD_with_nesterov': {'lr': 0.01, 'momentum': 0.9, 'nesterov': True},  # 改为SGD_with_nesterov
            'Adam': {'lr': 0.001},
            'AdamW': {'lr': 0.001},
            'Adamax': {'lr': 0.001},
            'Adagrad': {'lr': 0.01},
            'RMSprop': {'lr': 0.001}
        },
        'loss_functions': ['CrossEntropyLoss', 'MSELoss_OneHot', 'L1Loss_OneHot', 'SmoothL1Loss_OneHot'],
        'regularization': {
            'No_Reg': {'dropout_rate': 0.0, 'use_batch_norm': False, 'l1_lambda': 0.0, 'l2_lambda': 0.0},
            'Dropout_BN': {'dropout_rate': 0.5, 'use_batch_norm': True, 'l1_lambda': 0.0, 'l2_lambda': 0.0},
            'L1': {'dropout_rate': 0.0, 'use_batch_norm': False, 'l1_lambda': 0.001, 'l2_lambda': 0.0},
            'L2': {'dropout_rate': 0.0, 'use_batch_norm': False, 'l1_lambda': 0.0, 'l2_lambda': 0.001},
            'Combined': {'dropout_rate': 0.3, 'use_batch_norm': True, 'l1_lambda': 0.0001, 'l2_lambda': 0.0001}
        }
    }
 