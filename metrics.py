# utils/metrics.py

from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import numpy as np

def compute_metrics(preds, truths):
    """
    计算准确率、F1 分数和 IoU
    Args:
        preds (list or numpy array): 预测值
        truths (list or numpy array): 真实值
    Returns:
        dict: 包含 accuracy, f1, iou 的字典
    """
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds)
    iou = jaccard_score(truths, preds)
    return {"accuracy": acc, "f1_score": f1, "iou": iou}

def compute_metrics_efficient(preds, truths):
    tp = np.sum((preds == 1) & (truths == 1))
    tn = np.sum((preds == 0) & (truths == 0))
    fp = np.sum((preds == 1) & (truths == 0))
    fn = np.sum((preds == 0) & (truths == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {"accuracy": accuracy, "f1_score": f1, "iou": iou}


