import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import glob
from datetime import datetime
from PIL import Image
import re

from models.Base import BaseModel


class SegModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(SegModel, self).__init__(*args, **kwargs)
        self.model_name = 'segmentation_model'
        

    def fit(
        self,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_classes,
        num_epochs=500,
        model_save_dir='./checkpoints/',
        patience=15,
        delta=0.01,
    ):
        early_stopping = EarlyStopping(patience=patience, delta=delta)
        best_val_miou = 0
        best_model_weights = self.state_dict()
        device = next(self.parameters()).device  # 获取模型所在设备

        current_epoch = getattr(self, 'last_epoch', 0)
        total_epochs = current_epoch + num_epochs

        print(f"Starting training from epoch {current_epoch + 1} to epoch {total_epochs}")

        for epoch in range(current_epoch + 1, total_epochs + 1):
            
            print(f"\nStarting epoch {epoch}/{total_epochs}")
            self.train()
            running_loss = 0.0
            for images, masks in tqdm(train_loader, desc='Training Batches', leave=False):
                images = images.to(device)
                if num_classes > 1:
                    masks = masks.to(device).squeeze(1)  # 多分类任务移除通道维
                else:
                    masks = masks.to(device).float()  # 二分类任务保持通道维度并转换为浮点数
                    
                optimizer.zero_grad()
                
                # 直接获取模型的原始输出
                outputs = self(images)
                
                if num_classes == 1:
                    outputs = outputs.squeeze(1)  # 去除输出中的多余通道维度

                # 直接计算损失
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch}/{total_epochs}, Loss: {epoch_loss:.4f}')

            self.save(epoch, model_dir=model_save_dir, mode=1)

            # fit 方法中的验证阶段
            self.eval()
            val_loss = 0.0
            total_ious, total_precisions, total_recalls, total_f1_scores = [], [], [], []
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc='Validation Batches', leave=False):
                    images = images.to(device)
                    if num_classes > 1:
                        masks = masks.to(device).squeeze(1)  # 多分类任务移除通道维
                    else:
                        masks = masks.to(device).float()  # 二分类任务保持通道维度并转换为浮点数
                    # 获取模型的原始输出
                    outputs = self(images)
                    if num_classes == 1:
                        outputs = outputs.squeeze(1)  # 去除输出中的多余通道维度

                    # 计算损失
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    # 调用 compute_metrics 时直接传入 outputs 和 masks
                    iou, precision, recall, f1_score = compute_metrics(outputs, masks)
                    total_ious.append(iou)
                    total_precisions.append(precision)
                    total_recalls.append(recall)
                    total_f1_scores.append(f1_score)


                val_loss /= len(val_loader)
                val_iou = np.mean(total_ious)
                val_precision = np.mean(total_precisions)
                val_recall = np.mean(total_recalls)
                val_f1_score = np.mean(total_f1_scores)

                print(f'Validation Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1_score:.4f}')

                if self.best_val is None or val_iou > self.best_val:
                    self.best_val = val_iou
                    self.save(epoch, model_dir=model_save_dir, mode=3)
                    print(f'Best model updated at epoch {epoch} with best_val: {self.best_val:.4f}')

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

        self.load(model_dir=model_save_dir, mode=3)
        print(f'Training complete. Best mIoU: {self.best_val:.4f}')


    def test(self, data_loader, num_classes, save_dir='.', export=False):
        """
        对测试集进行评估，并根据选择决定是否导出预测结果。
        """
        device = next(self.parameters()).device  # 获取模型所在设备
        self.eval()
        save_dir = os.path.join(save_dir, 'outputs')
        if export and not os.path.exists(save_dir):
            os.makedirs(save_dir)  # 确保输出目录存在

        total_ious, total_precisions, total_recalls, total_f1_scores = [], [], [], []
        # Validation stage in the fit method

        total_ious, total_precisions, total_recalls, total_f1_scores = [], [], [], []
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(data_loader, desc='Testing Batches', leave=False)):
                images = images.to(device)
                if num_classes > 1:
                    masks = masks.to(device).squeeze(1)  # 多分类任务移除通道维
                else:
                    masks = masks.to(device).float()  # 二分类任务保持通道维度并转换为浮点数
                # 获取模型的原始输出
                outputs = self(images)
                
                if num_classes == 1:
                    outputs = outputs.squeeze(1)  # 去除输出中的多余通道维度
                
                # 直接传入 outputs 和 masks 以计算评价指标
                iou, precision, recall, f1_score = compute_metrics(outputs, masks)
                total_ious.append(iou)
                total_precisions.append(precision)
                total_recalls.append(recall)
                total_f1_scores.append(f1_score)

                # 如果选择导出，将输出保存为图像
                if export:
                    predicted = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                    for i in range(predicted.shape[0]):
                        output_image = (predicted[i] * 255).astype(np.uint8)
                        output_path = os.path.join(save_dir, f'output_{batch_idx}_{i}.png')
                        Image.fromarray(output_image).save(output_path)

        # 计算平均评价指标
        avg_iou = np.mean(total_ious)
        avg_precision = np.mean(total_precisions)
        avg_recall = np.mean(total_recalls)
        avg_f1_score = np.mean(total_f1_scores)

        print(f'Test Results - mIoU: {avg_iou:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1_score:.4f}')


              

class EarlyStopping:
    def __init__(self, patience=15, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def compute_metrics(outputs, targets):
    smooth = 1e-6  # 防止除以零的小值

    # 如果是二分类并且 `outputs` 有一个额外的通道维度，先 squeeze
    if outputs.shape[1] == 1:
        outputs = outputs.squeeze(1)  # 变成 [batch_size, height, width]

    # 确保 outputs 和 targets 的批次大小匹配
    if outputs.dim() == 3:  # 二分类
        predicted = torch.sigmoid(outputs)  # 转换为概率
        predicted = (predicted > 0.5).float()  # 阈值化为二值图
    elif outputs.dim() == 4:  # 多分类
        predicted = torch.argmax(outputs, dim=1)  # 转换为类别索引 [batch_size, height, width]
    else:
        raise ValueError(f"Unexpected shape for outputs: {outputs.shape}")

    # 确保 predicted 和 targets 的形状一致
    if predicted.shape != targets.shape:
        raise ValueError(f"Predicted shape {predicted.shape} does not match target shape {targets.shape}")

    # 针对多分类问题，对每一类别单独计算 iou，并取平均
    num_classes = outputs.shape[1] if outputs.dim() == 4 else 2  # 二分类视为 2 类

    ious, precisions, recalls, f1_scores = [], [], [], []
    for cls in range(num_classes):
        # 计算每个类别的交集、总预测、总目标和并集
        cls_predicted = (predicted == cls).float()
        cls_target = (targets == cls).float()
        
        intersection = (cls_predicted * cls_target).sum(dim=(1, 2)).float()
        total_predicted = cls_predicted.sum(dim=(1, 2)).float()
        total_targets = cls_target.sum(dim=(1, 2)).float()
        union = total_predicted + total_targets - intersection

        # 使用 smooth 防止分母为零，确保指标不会出现负数
        iou = (intersection + smooth) / (union + smooth)
        precision = (intersection + smooth) / (total_predicted + smooth)
        recall = (intersection + smooth) / (total_targets + smooth)
        f1_score = (2 * precision * recall + smooth) / (precision + recall + smooth)

        ious.append(iou.mean().item())
        precisions.append(precision.mean().item())
        recalls.append(recall.mean().item())
        f1_scores.append(f1_score.mean().item())

    # 取所有类别上的平均值
    mean_iou = np.mean(ious)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1_score = np.mean(f1_scores)

    return mean_iou, mean_precision, mean_recall, mean_f1_score
