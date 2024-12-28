# utils/helpers.py

import torch

def save_checkpoint(state, filename="best_model.pth"):
    """
    保存模型检查点
    Args:
        state (dict): 模型状态字典
        filename (str): 保存文件名
    """
    torch.save(state, filename)



def load_checkpoint(filename, model, optimizer=None, device='gpu'):
    """
    加载检查点并将权重加载到模型中
    Args:
        filename (str): 检查点文件名
        model (torch.nn.Module): 需要加载权重的模型
        optimizer (torch.optim.Optimizer, optional): 优化器（如果需要恢复优化器状态）
        device (str): 设备，例如 'cpu' 或 'cuda'
    Returns:
        tuple: (epoch, step, best_f1)
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型权重
    if optimizer:  # 如果提供优化器，则加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    best_f1 = checkpoint['best_f1']
    return epoch, step, best_f1

