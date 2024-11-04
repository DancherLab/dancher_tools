import torch
import numpy as np

# 小值避免除零错误
EPSILON = 1e-6

def mse(predicted, target):
    """
    计算 Mean Squared Error (MSE)。
    :param predicted: 模型预测的张量，形状为 (batch_size, sequence_length)。
    :param target: 真实标签张量，形状为 (batch_size, sequence_length)。
    :return: MSE 值
    """
    return ((predicted - target) ** 2).mean().item()

def rmse(predicted, target):
    """
    计算 Root Mean Squared Error (RMSE)。
    :param predicted: 模型预测的张量，形状为 (batch_size, sequence_length)。
    :param target: 真实标签张量，形状为 (batch_size, sequence_length)。
    :return: RMSE 值
    """
    return torch.sqrt(((predicted - target) ** 2).mean() + EPSILON).item()

def mae(predicted, target):
    """
    计算 Mean Absolute Error (MAE)。
    :param predicted: 模型预测的张量，形状为 (batch_size, sequence_length)。
    :param target: 真实标签张量，形状为 (batch_size, sequence_length)。
    :return: MAE 值
    """
    return (torch.abs(predicted - target)).mean().item()

def r2_score(predicted, target):
    """
    计算 R-squared (R²)。
    :param predicted: 模型预测的张量，形状为 (batch_size, sequence_length)。
    :param target: 真实标签张量，形状为 (batch_size, sequence_length)。
    :return: R² 值
    """
    target_mean = target.mean()
    ss_total = ((target - target_mean) ** 2).sum()
    ss_residual = ((target - predicted) ** 2).sum()
    r2 = 1 - (ss_residual / (ss_total + EPSILON))
    return r2.item()

# 用于回归任务的预设指标字典
PRESET_METRICS = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2_score": r2_score
}
