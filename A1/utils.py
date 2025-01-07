import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_curve, auc, precision_score, accuracy_score
import swanlab
import json
from copy import deepcopy
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

def load_cifar10(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=2)
    
    return trainloader, testloader

def get_loss_function(loss_name):
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif loss_name == 'MSELoss_OneHot':
        return lambda outputs, labels: nn.MSELoss()(outputs, nn.functional.one_hot(labels, 10).float())
    elif loss_name == 'L1Loss_OneHot':
        return lambda outputs, labels: nn.L1Loss()(outputs, nn.functional.one_hot(labels, 10).float())
    elif loss_name == 'SmoothL1Loss_OneHot':
        return lambda outputs, labels: nn.SmoothL1Loss()(outputs, nn.functional.one_hot(labels, 10).float())
    else:
        raise ValueError(f"未知的损失函数: {loss_name}")

def train_model(model, trainloader, validloader, criterion, optimizer, device, num_epochs=10, patience=5):
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    start_time = time.time()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        
        # 验证集评估
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data in validloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        
        valid_loss = valid_loss / len(validloader)
        valid_losses.append(valid_loss)
        
        scheduler.step(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break
            
        swanlab.log({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'valid_loss': valid_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'time_elapsed': time.time() - start_time
        })
        
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_loss:.4f}')
    
    return train_losses, valid_losses

def evaluate_model(model, testloader, device, num_classes=10):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # per_class_precision = precision_score(all_labels, all_preds, average=None)
    # per_class_recall = recall_score(all_labels, all_preds, average=None)
    # per_class_f1 = f1_score(all_labels, all_preds, average=None)
  
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    metrics = {
        'test_loss': test_loss / len(testloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_precision': precision_score(all_labels, all_preds, average='macro'),
        'macro_recall': recall_score(all_labels, all_preds, average='macro'),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'mean_roc_auc': np.mean(list(roc_auc.values())),
        'confusion_matrix': conf_matrix,
    }
   
    swanlab.log({
        'test_loss': metrics['test_loss'],
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'mean_roc_auc': metrics['mean_roc_auc'],
    })

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # 将图像转换为numpy数组
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    # 将图像转换为RGB格式的numpy数组
    fig.canvas.draw()
    confusion_matrix_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    confusion_matrix_image = confusion_matrix_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # 使用swanlab保存图像
    confusion_matrix_viz = swanlab.Image(confusion_matrix_image, caption="Confusion Matrix")
    swanlab.log({"confusion_matrix": confusion_matrix_viz})
    
    return metrics

def get_completed_experiments():
    try:
        with open('completed_experiments.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_completed_experiment(experiment_name, results):
    # 创建结果的深拷贝，避免修改原始数据
    completed = get_completed_experiments()
    results_copy = deepcopy(results)
    
    # 将NumPy数组转换为列表
    if 'confusion_matrix' in results_copy:
        results_copy['confusion_matrix'] = results_copy['confusion_matrix'].tolist()
    if 'train_losses' in results_copy:
        results_copy['train_losses'] = [float(x) for x in results_copy['train_losses']]
    if 'valid_losses' in results_copy:
        results_copy['valid_losses'] = [float(x) for x in results_copy['valid_losses']]
    
    # 确保所有数值都是Python原生类型
    for key in ['test_loss', 'accuracy', 'macro_f1', 'mean_roc_auc']:
        if key in results_copy:
            results_copy[key] = float(results_copy[key])
    
    completed[experiment_name] = results_copy
    with open('completed_experiments.json', 'w') as f:
        json.dump(completed, f) 