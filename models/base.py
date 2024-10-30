import torch
import torch.nn as nn
import os
import glob
from datetime import datetime
import re
from tqdm import tqdm
from dancher_tools.utils import CombinedLoss 

class base(nn.Module):
    def __init__(self):
        super(base, self).__init__()
        self.model_name = 'model'  # Default model name
        self.last_epoch = 0  # Initialize last_epoch
        self.best_val = 0
        self.optimizer = None
        self.criterion = None
        self.metrics = []


    def compile(self, optimizer, criterion, metrics=None, loss_weights=None):
        """
        设置模型的优化器、损失函数和评价指标。
        :param optimizer: 优化器实例
        :param criterion: 损失函数实例或损失函数列表
        :param metrics: 指标函数列表
        :param loss_weights: 损失函数对应的权重列表（如果 criterion 是列表）
        """
        self.optimizer = optimizer

        # 如果 criterion 是列表，则创建 CombinedLoss
        if isinstance(criterion, list) and len(criterion) > 1:
            self.criterion = CombinedLoss(losses=criterion, weights=loss_weights)
        elif isinstance(criterion, list) and len(criterion) == 1:
            self.criterion = criterion[0]  # 单一损失函数
        else:
            raise ValueError("Criterion must be a list of loss functions, even if only one is provided.")

        self.metrics = metrics if metrics is not None else []

        
    def fit(
        self,
        train_loader,
        val_loader=None,
        num_epochs=10,
        patience=5,
        delta=0.01,
        model_save_dir='./checkpoints'
    ):
        """
        训练模型并在验证集上评估，支持早停。
        :param train_loader: 训练数据加载器
        :param val_loader: 验证数据加载器（可选）
        :param num_epochs: 训练周期数
        :param patience: 早停的耐心值（验证损失未改善的最大周期数）
        :param delta: 早停的最小损失改善量
        :param model_save_dir: 模型保存目录
        """
        best_val_loss = float('inf')
        patience_counter = 0
        device = next(self.parameters()).device

        for epoch in range(1, num_epochs + 1):
            # Training phase
            self.train()
            running_loss = 0.0
            for features, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training"):
                features, targets = features.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self(features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}")

            # Validation phase (optional)
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader)
                print(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Metrics: {val_metrics}")

                # Check if validation loss improved
                if val_loss < best_val_loss - delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self.save(epoch, model_dir=model_save_dir, save_as='best')
                    print("Best model saved.")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break
            else:
                val_loss = None  # 如果没有验证集，不使用早停

            # 每个 epoch 末保存最新模型
            self.save(epoch, model_dir=model_save_dir, save_as='latest')

        print("Training complete.")


    def evaluate(self, data_loader):
        """
        在指定的数据集上评估模型的损失和评价指标
        :param data_loader: 数据加载器
        :return: 平均损失和各评价指标的字典
        """
        self.eval()
        total_loss = 0.0
        total_metrics = {metric.__name__: 0.0 for metric in self.metrics}
        device = next(self.parameters()).device

        with torch.no_grad():
            for features, targets in data_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = self(features)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # 计算各项指标
                for metric in self.metrics:
                    total_metrics[metric.__name__] += metric(outputs, targets).item()

        avg_loss = total_loss / len(data_loader)
        avg_metrics = {name: value / len(data_loader) for name, value in total_metrics.items()}

        print(f"Evaluation Loss: {avg_loss:.4f}, Metrics: {avg_metrics}")
        return avg_loss, avg_metrics
    
    def save(self, model_dir='./checkpoints', mode='latest'):
        """
        保存模型至指定目录。
        :param model_dir: 保存的文件夹路径
        :param mode: 保存模式，'latest'、'best' 或 'epoch'。
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 根据 mode 确定文件名
        if mode == 'latest':
            save_path = os.path.join(model_dir, f"{self.model_name}_latest.pth")
        elif mode == 'best':
            save_path = os.path.join(model_dir, f"{self.model_name}_best.pth")
        elif mode == 'epoch':
            save_path = os.path.join(model_dir, f"{self.model_name}_epoch_{self.last_epoch}.pth")
        else:
            raise ValueError("Invalid mode. Use 'latest', 'best', or 'epoch'.")

        save_dict = {
            'epoch': self.last_epoch,
            'model_state_dict': self.state_dict(),
            'best_val': self.best_val
        }
        if self.optimizer is not None:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()

        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")

    def load(self, model_dir='./checkpoints', mode='latest', optimizer=None):
        """
        从指定路径加载模型。
        :param model_dir: 模型文件的目录
        :param mode: 加载模式，'latest'、'best' 或 'epoch'。
        :param optimizer: 优化器实例（可选）
        """
        # 根据 mode 确定文件路径
        if mode == 'latest':
            load_path = os.path.join(model_dir, f"{self.model_name}_latest.pth")
        elif mode == 'best':
            load_path = os.path.join(model_dir, f"{self.model_name}_best.pth")
        elif mode == 'epoch':
            load_path = os.path.join(model_dir, f"{self.model_name}_epoch_{self.last_epoch}.pth")
        else:
            raise ValueError("Invalid mode. Use 'latest', 'best', or 'epoch'.")

        if os.path.exists(load_path):
            print(f"Loading model from {load_path}")
            checkpoint = torch.load(load_path, weights_only=True)

            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model successfully loaded from {load_path}, epoch: {checkpoint.get('epoch', 'unknown')}.")

            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded.")

            self.last_epoch = checkpoint.get('epoch', 0)
            self.best_val = checkpoint.get('best_val', 0)
        else:
            print(f"No model found at {load_path}, starting from scratch.")
            self.last_epoch = 0
            self.best_val = 0

    def transfer(self, specified_path, strict=False):
        """
        使用指定路径加载预训练模型参数，并将符合条件的参数转移到当前模型（不改变训练状态）。
        :param specified_path: 指定的权重文件路径。
        :param strict: 是否严格匹配层结构。如果为False，将跳过不匹配的参数。
        """
        if not specified_path or not os.path.exists(specified_path):
            raise FileNotFoundError(f"Specified path does not exist: {specified_path}")

        print(f"Transferring model parameters from {specified_path}")
        checkpoint = torch.load(specified_path)
        checkpoint_state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_state_dict = self.state_dict()

        new_state_dict = {}
        missing_parameters = []
        extra_parameters = []

        for name, parameter in model_state_dict.items():
            if name in checkpoint_state_dict:
                if checkpoint_state_dict[name].size() == parameter.size():
                    new_state_dict[name] = checkpoint_state_dict[name]
                else:
                    extra_parameters.append(name)
            else:
                missing_parameters.append(name)

        self.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully transferred {len(new_state_dict)} parameters from {specified_path}")

        if missing_parameters:
            print(f"Parameters not found in checkpoint (using default): {missing_parameters}")
        if extra_parameters:
            print(f"Parameters in checkpoint but not used due to size mismatch: {extra_parameters}")

        print(f"Transfer completed. Matched: {len(new_state_dict)}, Missing: {len(missing_parameters)}, Size mismatch: {len(extra_parameters)}.")