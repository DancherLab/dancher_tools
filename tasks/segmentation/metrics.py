import torch
import numpy as np

# 防止除零错误的微小值
EPSILON = 1e-6
def mIoU(predicted, target, num_classes):
    iou_scores = []
    for cls in range(num_classes):
        pred_cls = (predicted == cls).float()
        tgt_cls = (target == cls).float()

        intersection = torch.logical_and(pred_cls, tgt_cls).float().sum()
        union = torch.logical_or(pred_cls, tgt_cls).float().sum()

        if union == 0:
            iou = 1.0  # 类别在预测和目标中均未出现，设为完美分数
            # print(f"Class {cls} not present in predictions and targets. Setting IoU to 1.0")
        else:
            iou = (intersection / (union + EPSILON)).item()
            # print(f"Class {cls}: Intersection = {intersection.item()}, Union = {union.item()}, IoU = {iou}")
        iou_scores.append(iou)
    return iou_scores


def precision(predicted, target, num_classes):
    precision_scores = []
    for cls in range(num_classes):
        pred_cls = (predicted == cls).float()
        tgt_cls = (target == cls).float()
        
        true_positive = torch.logical_and(pred_cls, tgt_cls).float().sum()
        predicted_positive = pred_cls.float().sum()
        
        if predicted_positive == 0 and tgt_cls.float().sum() == 0:
            precision_score = 1.0  # 完美表现
        else:
            precision_score = (true_positive / (predicted_positive + EPSILON)).item()
        precision_scores.append(precision_score)
    return precision_scores

def recall(predicted, target, num_classes):
    recall_scores = []
    for cls in range(num_classes):
        pred_cls = (predicted == cls).float()
        tgt_cls = (target == cls).float()
        
        true_positive = torch.logical_and(pred_cls, tgt_cls).float().sum()
        actual_positive = tgt_cls.float().sum()
        
        if actual_positive == 0:
            recall_score = 1.0  # 完美表现
        else:
            recall_score = (true_positive / (actual_positive + EPSILON)).item()
        recall_scores.append(recall_score)
    return recall_scores

def f1_score(predicted, target, num_classes):
    precision_scores = precision(predicted, target, num_classes)
    recall_scores = recall(predicted, target, num_classes)
    
    f1_scores = []
    for p, r in zip(precision_scores, recall_scores):
        if p + r == 0:
            f1 = 0.0
        else:
            f1 = 2 * (p * r) / (p + r + EPSILON)
        f1_scores.append(f1)
    return f1_scores

# 用于 segmentation 任务的预设指标字典
PRESET_METRICS = {
    "mIoU": mIoU,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score
}
