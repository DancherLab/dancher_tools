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


    def predict(self, images):
        """
        对输入的图像进行预测，并返回彩色的预测标签图像。

        :param images: 输入的图像，张量格式。
        :return: 彩色的预测标签图像，numpy 数组格式。
        """
        device = next(self.parameters()).device
        self.eval()  # 设置模型为评估模式
        images = images.to(device)

        with torch.no_grad():
            outputs = self(images)

        # 确保输出的通道数与类别数一致
        if outputs.size(1) != self.num_classes:
            raise ValueError(f"Expected outputs with {self.num_classes} classes, got {outputs.size(1)}")

        # 预测标签
        predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()

        # 如果有 color_to_class 映射，将标签映射为彩色图像
        if hasattr(self, 'class_to_color') and isinstance(self.class_to_color, dict):
            color_to_class = self.class_to_color
            # 创建一个 RGB 图像
            color_image = np.zeros((predicted_labels.shape[0], predicted_labels.shape[1], 3), dtype=np.uint8)
            for cls, color in color_to_class.items():
                color_image[predicted_labels == cls] = color
            return color_image
        else:
            return predicted_labels

    def print_metrics(self, per_class_avg_metrics, class_names=None):
        """
        打印每个类别的评估指标结果，并在最后一行显示每个指标的mean值。
        
        :param per_class_avg_metrics: 包含每个类别指标值的字典
        :param class_names: 类别名称列表（如果有）
        """
        # 如果只有单个评价指标
        if len(self.metrics) == 1:  
            metric_name = list(self.metrics.keys())[0]  # 获取单一指标的名称

            # 打印表头（Class + classes + mean）
            header = f"{'Class':<15}"
            for cls in range(self.num_classes):
                if class_names:
                    header += f"{class_names[cls]:<15}"
                else:
                    header += f"{cls:<15}"
            header += f"{'mean':<15}"
            print(header)

            # 打印指标值和mean
            metric_values = f"{metric_name:<15}"
            for cls in range(self.num_classes):
                metric_values += f"{per_class_avg_metrics[metric_name].get(cls, 0):<15.4f}"
            mean_value = round(np.mean([per_class_avg_metrics[metric_name].get(cls, 0) for cls in range(self.num_classes)]), 4)
            metric_values += f"{mean_value:<15.4f}"
            print(metric_values)

        else:  # 多个评价指标时
            # 打印表头（Class + metric names）
            header = f"{'Class':<15}"
            for metric_name in self.metrics:
                header += f"{metric_name:<15}"
            print(header)

            # 打印每个类的结果
            for cls in range(self.num_classes):
                class_name = class_names[cls] if class_names else cls
                class_metrics_str = f"{class_name:<15}"
                for metric_name in self.metrics:
                    class_metrics_str += f"{per_class_avg_metrics[metric_name].get(cls, 0):<15.4f}"
                print(class_metrics_str)

            # 打印mean行（每个指标的整体平均值）
            mean_row = f"{'mean':<15}"
            for metric_name in self.metrics:
                mean_value = round(np.mean([per_class_avg_metrics[metric_name].get(cls, 0) for cls in range(self.num_classes)]), 4)
                mean_row += f"{mean_value:<15.4f}"
            print(mean_row)


    def evaluate(self, data_loader, class_names=None):
        device = next(self.parameters()).device
        self.eval()
        total_loss = 0.0
        all_predicted = []
        all_masks = []

        with torch.no_grad():
            for images, masks in tqdm(data_loader, desc='Evaluation Batches', leave=False):
                images = images.to(device)
                masks = masks.to(device)
                outputs = self(images)

                if outputs.size(1) != self.num_classes:
                    raise ValueError(f"Expected outputs with {self.num_classes} classes, got {outputs.size(1)}")

                predicted_labels = torch.argmax(outputs, dim=1)
                all_predicted.append(predicted_labels.cpu())

                if masks is not None:
                    # 处理掩码的形状
                    if masks.dim() > 3 and masks.size(1) > 1:
                        masks = masks.argmax(dim=1)
                    else:
                        masks = masks.squeeze(1)  # 确保掩码形状为 [batch_size, H, W]
                    all_masks.append(masks.cpu())

                    # 计算并累加损失
                    loss = self.criterion(outputs, masks)
                    total_loss += loss.item()

        avg_val_loss = total_loss / len(data_loader)

        # 拼接所有预测和真实标签
        all_predicted = torch.cat(all_predicted)
        all_masks = torch.cat(all_masks)

        # 计算每个指标的结果
        per_class_avg_metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            scores = metric_fn(all_predicted, all_masks, self.num_classes)

            # 如果 scores 是列表或 NumPy 数组，则按类别存储；否则，假设所有类别共享同一评分
            if isinstance(scores, (list, np.ndarray)):
                per_class_avg_metrics[metric_name] = {cls: scores[cls] for cls in range(self.num_classes)}
            else:
                per_class_avg_metrics[metric_name] = {cls: scores for cls in range(self.num_classes)}

        # 计算整体平均指标（可选）
        avg_metrics = {
            metric_name: np.mean(list(cls_metrics.values()))
            for metric_name, cls_metrics in per_class_avg_metrics.items()
        }

        # 打印整体损失
        print(f'Validation Loss: {avg_val_loss:.4f}')

        # 调用封装后的打印函数，显示每个类别的指标
        self.print_metrics(per_class_avg_metrics, class_names)

        return avg_val_loss, avg_metrics, per_class_avg_metrics
