import torch
import numpy as np

# 防止除零错误的微小值
EPSILON = 1e-6

def mIoU(predicted, target, num_classes):
    """
    计算 mIoU (Mean Intersection over Union)。
    :param predicted: 模型预测的张量，形状为 (batch_size, height, width)。
    :param target: 真实标签张量，形状为 (batch_size, height, width)。
    :param num_classes: 类别数
    :return: 平均 IoU 值
    """
    iou_scores = []
    for cls in range(num_classes):
        pred_cls = (predicted == cls).float()
        tgt_cls = (target == cls).float()
        
        intersection = torch.logical_and(pred_cls, tgt_cls).float().sum()
        union = torch.logical_or(pred_cls, tgt_cls).float().sum() + EPSILON
        
        iou = intersection / union
        iou_scores.append(iou.item())
    return np.mean(iou_scores)

def precision(predicted, target, num_classes):
    """
    计算 Precision。
    :param predicted: 模型预测的张量，形状为 (batch_size, height, width)。
    :param target: 真实标签张量，形状为 (batch_size, height, width)。
    :param num_classes: 类别数
    :return: 平均 Precision 值
    """
    precision_scores = []
    for cls in range(num_classes):
        pred_cls = (predicted == cls).float()
        tgt_cls = (target == cls).float()
        
        true_positive = torch.logical_and(pred_cls, tgt_cls).float().sum()
        predicted_positive = pred_cls.float().sum() + EPSILON
        
        precision_score = true_positive / predicted_positive
        precision_scores.append(precision_score.item())
    return np.mean(precision_scores)

def recall(predicted, target, num_classes):
    """
    计算 Recall。
    :param predicted: 模型预测的张量，形状为 (batch_size, height, width)。
    :param target: 真实标签张量，形状为 (batch_size, height, width)。
    :param num_classes: 类别数
    :return: 平均 Recall 值
    """
    recall_scores = []
    for cls in range(num_classes):
        pred_cls = (predicted == cls).float()
        tgt_cls = (target == cls).float()
        
        true_positive = torch.logical_and(pred_cls, tgt_cls).float().sum()
        actual_positive = tgt_cls.float().sum() + EPSILON
        
        recall_score = true_positive / actual_positive
        recall_scores.append(recall_score.item())
    return np.mean(recall_scores)

def f1_score(predicted, target, num_classes):
    """
    计算 F1 Score。
    :param predicted: 模型预测的张量，形状为 (batch_size, height, width)。
    :param target: 真实标签张量，形状为 (batch_size, height, width)。
    :param num_classes: 类别数
    :return: 平均 F1 Score 值
    """
    precision_value = precision(predicted, target, num_classes)
    recall_value = recall(predicted, target, num_classes)
    
    if precision_value + recall_value == 0:
        return 0.0
    
    f1 = 2 * (precision_value * recall_value) / (precision_value + recall_value + EPSILON)
    return f1

# 用于 segmentation 任务的预设指标字典
PRESET_METRICS = {
    "mIoU": mIoU,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score
}
