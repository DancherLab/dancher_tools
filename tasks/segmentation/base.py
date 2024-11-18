from dancher_tools.core import Core
from dancher_tools.utils import EarlyStopping
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

class SegModel(Core):
    def __init__(self, num_classes, in_channels, img_size, *args, **kwargs):
        super(SegModel, self).__init__(*args, **kwargs)
        self.model_name = None
        self.num_classes = num_classes
        self.img_size = img_size
        self.in_channels = in_channels

    def fit(self, train_loader, val_loader, num_epochs=500, model_save_dir='./checkpoints/', patience=15, delta=0.01, save_interval=1):
        """
        训练模型并根据保存间隔保存模型。
        
        :param save_interval: 每隔多少个 epoch 保存一次模型（除了最新和最佳模型）。
        """
        early_stopping = EarlyStopping(patience=patience, delta=delta)
        device = next(self.parameters()).device
        current_epoch = getattr(self, 'last_epoch', 0)
        total_epochs = num_epochs

        first_metric = list(self.metrics.keys())[0]
        best_val = None

        for epoch in range(current_epoch + 1, total_epochs + 1):
            print(f"\nStarting epoch {epoch}/{total_epochs}")
            self.last_epoch = epoch
            self.train()
            running_loss = 0.0

            for images, masks in tqdm(train_loader, desc='Training Batches', leave=False):
                images, masks = images.to(device), masks.to(device)

                self.optimizer.zero_grad()
                outputs = self(images)

                # 确保输出的通道数与类别数一致
                if outputs.size(1) != self.num_classes:
                    raise ValueError(f"Expected outputs with {self.num_classes} classes, got {outputs.size(1)}")

                loss = self.criterion(outputs, masks)
                loss.backward()

                # 增加梯度裁剪以防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch}/{total_epochs}, Loss: {epoch_loss:.4f}')

            # # 保存最新模型
            # self.save(model_dir=model_save_dir, mode='latest')

            # 按照保存间隔保存模型
            if save_interval > 0 and epoch % save_interval == 0:
                self.save(model_dir=model_save_dir, mode='latest')

            # 验证阶段
            val_loss, val_metrics, _ = self.evaluate(val_loader)
            val_first_metric = val_metrics.get(first_metric)

            if best_val is None or (val_first_metric is not None and val_first_metric > best_val):
                best_val = val_first_metric
                self.best_val = best_val
                self.save(model_dir=model_save_dir, mode='best')
                print(f"New best model saved with {first_metric}: {best_val:.4f}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        self.load(model_dir=model_save_dir, mode='best')
        print(f'Training complete. Best {first_metric}: {self.best_val:.4f}')


    def predict(self, images, masks=None):
        """
        对输入的图像和真实标签进行预测并计算指标。

        :param images: 输入的图像，张量格式。
        :param masks: 真实的标签，张量格式（可选）。
        :return: 预测的标签、损失（如果提供 masks），总体指标、每类别指标。
        """
        device = next(self.parameters()).device
        images = images.to(device)
        
        # 如果提供了 masks，则将其移动到设备上
        if masks is not None:
            masks = masks.to(device)

        outputs = self(images)

        # 确保输出的通道数与类别数一致
        if outputs.size(1) != self.num_classes:
            raise ValueError(f"Expected outputs with {self.num_classes} classes, got {outputs.size(1)}")

        # 预测标签
        predicted_labels = torch.argmax(outputs, dim=1)

        # 如果提供了 masks，则计算损失和指标
        if masks is not None:
            # 确保 masks 的格式为单通道（适用于多分类）
            if masks.dim() > 3 and masks.size(1) > 1:  # 多分类情况
                masks = masks.argmax(dim=1)  # 取最大值索引

        # 计算损失
            loss = self.criterion(outputs, masks)

            # 计算总体指标
            overall_metrics = {metric_name: metric_fn(predicted_labels, masks, self.num_classes)
                            for metric_name, metric_fn in self.metrics.items()}

            # 计算每个类别的指标
            per_class_metrics = {
                metric_name: {
                    cls: metric_fn((predicted_labels == cls).float(), (masks == cls).float(), 2)
                    for cls in range(self.num_classes)
                } for metric_name, metric_fn in self.metrics.items()
            }

            return predicted_labels, loss.item(), overall_metrics, per_class_metrics

        # 如果没有提供 masks，则仅返回预测结果
        return predicted_labels, None, None, None

    def evaluate(self, data_loader):
        """
        对整个数据集进行评估。

        :param data_loader: 数据加载器。
        :return: 平均损失、总体平均指标、每类别平均指标。
        """
        device = next(self.parameters()).device
        self.eval()
        total_loss = 0.0

        # 初始化存储总体和每类别指标的结构
        metric_results = {metric_name: [] for metric_name in self.metrics}
        per_class_metrics = {metric_name: {cls: [] for cls in range(self.num_classes)} for metric_name in self.metrics}

        with torch.no_grad():
            for images, masks in tqdm(data_loader, desc='Evaluation Batches', leave=False):
                _, loss, overall_metrics, batch_per_class_metrics = self.predict(images, masks)

                total_loss += loss

                # 存储总体指标
                for metric_name, result in overall_metrics.items():
                    metric_results[metric_name].append(result)

                # 存储每类别指标
                for metric_name, cls_metrics in batch_per_class_metrics.items():
                    for cls, result in cls_metrics.items():
                        per_class_metrics[metric_name][cls].append(result)

        # 汇总平均值
        avg_val_loss = total_loss / len(data_loader)
        avg_metrics = {metric_name: round(np.mean(results), 4) for metric_name, results in metric_results.items()}

        # 汇总每个类别的平均指标
        per_class_avg_metrics = {
            metric_name: {
                cls: round(np.mean(results), 4) for cls, results in cls_results.items()
            } for metric_name, cls_results in per_class_metrics.items()
        }

        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Overall Metrics: {avg_metrics}')

        return avg_val_loss, avg_metrics, per_class_avg_metrics
