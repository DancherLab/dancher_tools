import torch
import numpy as np

# 小值避免除零错误
EPSILON = 1e-6

def mIoU(predicted, target):
    """
    计算 mIoU (Mean Intersection over Union)。
    :param predicted: 模型预测的张量，形状为 (batch_size, height, width)。
    :param target: 真实标签张量，形状为 (batch_size, height, width)。
    :return: 平均 IoU 值
    """
    iou_scores = []
    for pred, tgt in zip(predicted, target):
        pred = (pred > 0.5).float()  # 阈值化
        tgt = (tgt > 0.5).float()    # 确保 target 也是二值的
        intersection = torch.logical_and(pred, tgt).float().sum()
        union = torch.logical_or(pred, tgt).float().sum() + EPSILON
        iou = intersection / union
        iou_scores.append(iou.item())
    return np.mean(iou_scores)

def precision(predicted, target):
    """
    计算 Precision。
    :param predicted: 模型预测的张量，形状为 (batch_size, height, width)。
    :param target: 真实标签张量，形状为 (batch_size, height, width)。
    :return: 平均 Precision 值
    """
    precision_scores = []
    for pred, tgt in zip(predicted, target):
        pred = (pred > 0.5).float()  # 阈值化
        tgt = (tgt > 0.5).float()    # 确保 target 也是二值的
        true_positive = torch.logical_and(pred, tgt).float().sum()
        predicted_positive = pred.float().sum() + EPSILON
        precision_score = true_positive / predicted_positive
        precision_scores.append(precision_score.item())
    return np.mean(precision_scores)

def recall(predicted, target):
    """
    计算 Recall。
    :param predicted: 模型预测的张量，形状为 (batch_size, height, width)。
    :param target: 真实标签张量，形状为 (batch_size, height, width)。
    :return: 平均 Recall 值
    """
    recall_scores = []
    for pred, tgt in zip(predicted, target):
        pred = (pred > 0.5).float()  # 阈值化
        tgt = (tgt > 0.5).float()    # 确保 target 也是二值的
        true_positive = torch.logical_and(pred, tgt).float().sum()
        actual_positive = tgt.float().sum() + EPSILON
        recall_score = true_positive / actual_positive
        
        # 调试信息：确保 recall 不超过 1
        assert recall_score <= 1, f"Recall score exceeds 1: {recall_score}"
        
        recall_scores.append(recall_score.item())
    return np.mean(recall_scores)

def f1_score(predicted, target):
    """
    计算 F1 Score。
    :param predicted: 模型预测的张量，形状为 (batch_size, height, width)。
    :param target: 真实标签张量，形状为 (batch_size, height, width)。
    :return: 平均 F1 Score 值
    """
    precision_value = precision(predicted, target)
    recall_value = recall(predicted, target)
    
    if precision_value + recall_value == 0:
        return 0.0
    
    f1 = 2 * (precision_value * recall_value) / (precision_value + recall_value + EPSILON)
    
    # 调试信息：确保 F1 score 不超过 1
    assert f1 <= 1, f"F1 score exceeds 1: {f1}"
    
    return f1

# 用于 segmentation 任务的预设指标字典
PRESET_METRICS = {
    "mIoU": mIoU,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score
}
