import torch
import numpy as np

# 容忍微小的数值避免除零错误
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
        intersection = torch.logical_and(pred, tgt).float().sum()
        union = torch.logical_or(pred, tgt).float().sum() + EPSILON
        iou = (intersection + EPSILON) / union
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
        true_positive = torch.logical_and(pred, tgt).float().sum()
        predicted_positive = pred.float().sum() + EPSILON
        precision_score = (true_positive + EPSILON) / predicted_positive
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
        true_positive = torch.logical_and(pred, tgt).float().sum()
        actual_positive = tgt.float().sum() + EPSILON
        recall_score = (true_positive + EPSILON) / actual_positive
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
    f1 = (2 * precision_value * recall_value) / (precision_value + recall_value + EPSILON)
    return f1

# 用于 segmentation 任务的预设指标字典
PRESET_METRICS = {
    "mIoU": mIoU,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score
}
