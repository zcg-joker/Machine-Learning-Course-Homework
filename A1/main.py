import numpy as np
import torch.optim as optim
import swanlab
from config import get_base_config
from models import FCNet
from utils import (
    load_cifar10, get_loss_function, train_model, evaluate_model,
    get_completed_experiments, save_completed_experiment
)




def main():
    # 加载基础配置
    base_config = get_base_config()
    
    # 加载已完成的实验
    completed_experiments = get_completed_experiments()
    
    # 加载数据
    trainloader, testloader = load_cifar10(batch_size=base_config['training']['batch_size'])
    
    results = {}
    for reg_name, reg_params in base_config['regularization'].items():
        for opt_name, opt_params in base_config['optimizers'].items():
            for loss_name in base_config['loss_functions']:
                experiment_name = f"{reg_name}_{opt_name}_{loss_name}"
                
                if experiment_name in completed_experiments:
                    print(f"跳过已完成的实验: {experiment_name}")
                    results[experiment_name] = completed_experiments[experiment_name]
                    continue

                try:
                    config = {
                        'regularization': reg_name,
                        'optimizer': opt_name,
                        'loss_function': loss_name,
                        'dropout_rate': reg_params['dropout_rate'],
                        'use_batch_norm': reg_params['use_batch_norm'],
                        'l1_lambda': reg_params['l1_lambda'],
                        'l2_lambda': reg_params['l2_lambda'],
                        'input_size': base_config['model']['input_size'],
                        'hidden_sizes': base_config['model']['hidden_sizes'],
                        'num_classes': base_config['model']['num_classes'],
                        'batch_size': base_config['training']['batch_size'],
                        'num_epochs': base_config['training']['num_epochs'],
                        'device': base_config['training']['device'],
                        'optimizer_params': opt_params,
                    }
                    # swanlab.login(api_key="wCJT6EVKLoPojZB6EV2fj")
                    run = swanlab.init(
                        project="cifar10_classification",
                        mode="local",
                        experiment_name=experiment_name,
                        config=config
                    )

                    # 初始化模型、优化器和损失函数
                    model = FCNet(**{k: config[k] for k in 
                        ['input_size', 'hidden_sizes', 'num_classes', 'dropout_rate', 
                         'use_batch_norm', 'l1_lambda', 'l2_lambda']
                    }).to(config['device'])
                    
                    optimizer_class = getattr(optim, opt_name.split('_with_')[0])
                    optimizer = optimizer_class(model.parameters(), **opt_params)
                    criterion = get_loss_function(loss_name)
                    
                    print(f"\n使用配置: {experiment_name}")
                    
                    # 训练和评估
                    train_losses, valid_losses = train_model(
                        model, trainloader, testloader, criterion, optimizer,
                        config['device'], num_epochs=config['num_epochs']
                    )
                    metrics = evaluate_model(model, testloader, config['device'])
                    
                    # 保存结果
                    results[experiment_name] = {
                        'train_losses': train_losses,
                        'valid_losses': valid_losses,
                        'test_loss': metrics['test_loss'],
                        'accuracy': metrics['accuracy'],
                        'macro_f1': metrics['macro_f1'],                   
                        'confusion_matrix': metrics['confusion_matrix'],
                        'mean_roc_auc': metrics['mean_roc_auc'],
                        'config': config
                    }
                    
                    save_completed_experiment(experiment_name, results[experiment_name])
                    accuracy = metrics['accuracy']
                    print(f'测试集准确率: {accuracy:.2f}%')

                    swanlab.finish()
                    
                except Exception as e:
                    import traceback
                    print(f"实验 {experiment_name} 发生错误: {str(e)}")
                    traceback.print_exc()  # 打印错误的详细堆栈信息
                    continue
    
    return results

if __name__ == "__main__":
    results = main()
