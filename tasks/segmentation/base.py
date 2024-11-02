from dancher_tools.core import Core
from dancher_tools.utils import EarlyStopping
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

class SegModel(Core):
    def __init__(self, *args, **kwargs):
        super(SegModel, self).__init__(*args, **kwargs)
        self.model_name = 'segmentation_model'

    def fit(
        self,
        train_loader,
        val_loader,
        num_classes,
        num_epochs=500,
        model_save_dir='./checkpoints/',
        patience=15,
        delta=0.01,
    ):

        early_stopping = EarlyStopping(patience=patience, delta=delta)
        device = next(self.parameters()).device  # 获取模型所在设备
        current_epoch = getattr(self, 'last_epoch', 0)
        total_epochs = current_epoch + num_epochs

        # 选择第一个评价指标作为最佳值的依据
        first_metric = list(self.metrics.keys())[0]
        best_val = None

        print(f"Starting training from epoch {current_epoch + 1} to epoch {total_epochs}")

        for epoch in range(current_epoch + 1, total_epochs + 1):
            self.last_epoch = epoch
            print(f"\nStarting epoch {epoch}/{total_epochs}")
            self.train()
            running_loss = 0.0
            for images, masks in tqdm(train_loader, desc='Training Batches', leave=False):
                images = images.to(device)
                masks = masks.to(device).squeeze(1).float()

                self.optimizer.zero_grad()
                outputs = self(images)
                outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch}/{total_epochs}, Loss: {epoch_loss:.4f}')

            self.save(model_dir=model_save_dir, mode='latest')

            # 验证阶段
            val_loss, val_metrics = self.evaluate(val_loader, num_classes)
            val_first_metric = val_metrics.get(first_metric)

            # print(f'Validation Loss: {val_loss:.4f}, Metrics: {val_metrics}')

            # 如果当前指标优于历史最佳，保存模型并更新最佳值
            if best_val is None or (val_first_metric is not None and val_first_metric > best_val):
                best_val = val_first_metric
                self.best_val = best_val
                self.save(model_dir=model_save_dir, mode='best')
                print(f"New best model saved with {first_metric}: {best_val:.4f}")

            # 检查早停条件
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # 训练结束，加载最佳模型
        self.load(model_dir=model_save_dir, mode='best')
        print(f'Training complete. Best {first_metric}: {self.best_val:.4f}')


    def evaluate(self, data_loader, save_dir='.', export=False):
        """
        在测试集或验证集上评估模型，使用 compile 设置的 metrics 计算评价指标。
        """
        device = next(self.parameters()).device  # 获取模型所在设备
        self.eval()
        metric_results = {metric_name: [] for metric_name in self.metrics}  # 初始化每个指标的结果列表
        total_loss = 0.0  # 新增 total_loss 计算验证损失

        with torch.no_grad():
            for images, masks in tqdm(data_loader, desc='Evaluation Batches', leave=False):
                images = images.to(device)
                masks = masks.to(device).squeeze(1).float()

                self.optimizer.zero_grad()
                outputs = self(images)
                outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                # 计算并记录每个指标的结果
                for metric_name, metric_fn in self.metrics.items():
                    result = metric_fn(outputs, masks)
                    metric_results[metric_name].append(result)

                # 可选：如果需要导出预测结果
                if export:
                    save_dir = os.path.join(save_dir, 'outputs')
                    os.makedirs(save_dir, exist_ok=True)
                    predicted = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                    for i in range(predicted.shape[0]):
                        output_image = (predicted[i] * 255).astype(np.uint8)
                        output_path = os.path.join(save_dir, f'output_{i}.png')
                        Image.fromarray(output_image).save(output_path)
                
        # 计算平均验证损失
        avg_val_loss = total_loss / len(data_loader)

        # 计算每个指标的平均值
        avg_metrics = {metric_name: round(np.mean(results), 4) for metric_name, results in metric_results.items()}
        print(f'Validation Loss: {avg_val_loss:.4f}, Metrics: {avg_metrics}')
        # 返回平均验证损失和指标字典
        return avg_val_loss, avg_metrics