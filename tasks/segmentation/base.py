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

    def fit(self, train_loader, val_loader, class_names, num_epochs=500, model_save_dir='./checkpoints/', patience=15, delta=0.01, save_interval=1):
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
            val_loss, val_metrics, _ = self.evaluate(val_loader,class_names)
            val_first_metric = val_metrics.get(first_metric)

            if val_first_metric is not None and (self.best_val is None or val_first_metric > self.best_val):
                self.best_val = val_first_metric
                self.save(model_dir=model_save_dir, mode='best')
                print(f"New best model saved with {first_metric}: {self.best_val:.4f}")

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

    def evaluate(self, data_loader, class_names=None):
        device = next(self.parameters()).device
        self.eval()
        total_loss = 0.0
        metric_results = {metric_name: [] for metric_name in self.metrics}
        per_class_metrics = {metric_name: {cls: [] for cls in range(self.num_classes)} for metric_name in self.metrics}

        with torch.no_grad():
            for images, masks in tqdm(data_loader, desc='Evaluation Batches', leave=False):
                images = images.to(device)
                masks = masks.to(device)
                outputs = self(images)

                if outputs.size(1) != self.num_classes:
                    raise ValueError(f"Expected outputs with {self.num_classes} classes, got {outputs.size(1)}")

                predicted_labels = torch.argmax(outputs, dim=1)

                if masks is not None:
                    if masks.dim() > 3 and masks.size(1) > 1:
                        masks = masks.argmax(dim=1)

                    loss = self.criterion(outputs, masks)
                    total_loss += loss.item()

                    for metric_name, metric_fn in self.metrics.items():
                        metric_results[metric_name].append(metric_fn(predicted_labels, masks, self.num_classes))

                    for metric_name, metric_fn in self.metrics.items():
                        for cls in range(self.num_classes):
                            per_class_metrics[metric_name][cls].append(metric_fn((predicted_labels == cls).float(), (masks == cls).float(), 2))
        
        avg_val_loss = total_loss / len(data_loader)
        avg_metrics = {metric_name: round(np.mean(results), 4) for metric_name, results in metric_results.items()}
        per_class_avg_metrics = {
            metric_name: {cls: round(np.mean(results), 4) for cls, results in cls_results.items()}
            for metric_name, cls_results in per_class_metrics.items()
        }

        # 打印整体损失
        print(f'Validation Loss: {avg_val_loss:.4f}')

        # 处理单个评价指标的情况
        if len(self.metrics) == 1:  # 只有单一评价指标时
            metric_name = list(self.metrics.keys())[0]  # 获取单一指标的名称
            # 打印表头（类名 + "mean"）
            print(f"{'Class':<15}", end="")
            for cls in range(self.num_classes):
                if class_names:
                    print(f"{class_names[cls]:<15}", end="")
                else:
                    print(f"{cls:<15}", end="")
            print(f"{'mean':<15}")  # 打印最后的"mean"列

            # 打印指标值
            print(f"{metric_name:<15}", end="")
            for cls in range(self.num_classes):
                print(f"{per_class_avg_metrics[metric_name].get(cls, 0):<15.4f}", end="")
            mean_value = round(np.mean(list(per_class_avg_metrics[metric_name].values())), 4)
            print(f"{mean_value:<15.4f}")  # 打印平均值并与类别数值在同一行

        else:  # 多个评价指标时
            # 打印表头（评价指标的每个类的结果）
            print(f"{'Class':<15}", end="")
            for cls in range(self.num_classes):
                if class_names:
                    print(f"{class_names[cls]:<15}", end="")
                else:
                    print(f"{cls:<15}", end="")
            print()  # 换行

            # 打印每个类的结果
            for cls in range(self.num_classes):
                class_metrics_str = f"{cls:<15}" if not class_names else f"{class_names[cls]:<15}"
                for metric_name in self.metrics:
                    class_metrics_str += f"{per_class_avg_metrics[metric_name].get(cls, 0):<15.4f}"
                print(class_metrics_str)

            # 输出mean（放在最后一列）
            print(f"{'mean':<15}", end="")
            for metric_name in self.metrics:
                mean_value = round(np.mean(list(per_class_avg_metrics[metric_name].values())), 4)
                print(f"{mean_value:<15.4f}", end="")
            print()  # 换行

        return avg_val_loss, avg_metrics, per_class_avg_metrics
