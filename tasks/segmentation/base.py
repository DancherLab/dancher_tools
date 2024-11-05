from dancher_tools.core import Core
from dancher_tools.utils import EarlyStopping
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

class SegModel(Core):
    def __init__(self, num_classes, *args, **kwargs):
        super(SegModel, self).__init__(*args, **kwargs)
        self.model_name = 'segmentation_model'
        self.num_classes = num_classes


    def fit(self, train_loader, val_loader, num_epochs=500, model_save_dir='./checkpoints/', patience=15, delta=0.01):
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
                images = images.to(device)
                
                # 根据 criterion 设置标签格式
                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    masks = masks.to(device).squeeze(1).long()  # CrossEntropyLoss 需要 long 类型标签
                elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    masks = masks.to(device).unsqueeze(1).float()  # BCEWithLogitsLoss 需要 float 类型标签
                
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

            # 验证阶段
            val_loss, val_metrics = self.evaluate(val_loader)
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

    def evaluate(self, data_loader):
        device = next(self.parameters()).device
        self.eval()
        metric_results = {metric_name: [] for metric_name in self.metrics}
        total_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(data_loader, desc='Evaluation Batches', leave=False):
                images = images.to(device)
                
                # 验证数据的标签类型设置
                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    masks = masks.to(device).squeeze(1).long()  # 使用 long 类型标签
                elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    masks = masks.to(device).unsqueeze(1).float()  # 使用 float 类型标签

                outputs = self(images)

                # 确保输出的通道数与类别数一致
                if outputs.size(1) != self.num_classes:
                    raise ValueError(f"Expected outputs with {self.num_classes} classes, got {outputs.size(1)}")

                # 计算损失
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                # 计算每个指标的结果
                binary_outputs = torch.argmax(outputs, dim=1)
                for metric_name, metric_fn in self.metrics.items():
                    result = metric_fn(binary_outputs.float(), masks.float(), self.num_classes)
                    metric_results[metric_name].append(result)

        avg_val_loss = total_loss / len(data_loader)
        avg_metrics = {metric_name: round(np.mean(results), 4) for metric_name, results in metric_results.items()}
        print(f'Validation Loss: {avg_val_loss:.4f}, Metrics: {avg_metrics}')

        return avg_val_loss, avg_metrics
