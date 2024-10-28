import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from models.Base import BaseModel

class RegressionModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(RegressionModel, self).__init__(*args, **kwargs)
        self.model_name = 'regression_model'

    def compute_metrics(self, outputs, targets):
        """
        计算回归任务的评估指标：MAE和MSE。
        """
        mae = nn.L1Loss()(outputs, targets).item()
        mse = nn.MSELoss()(outputs, targets).item()
        return mae, mse

    def fit(
        self,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=500,
        model_save_dir='./checkpoints/',
        patience=15,
        delta=0.01
    ):
        """
        训练模型并保存最佳模型。
        """
        early_stopping = EarlyStopping(patience=patience, delta=delta)
        best_val_mse = float('inf')
        device = next(self.parameters()).device  # 获取模型所在设备

        print(f"Starting training for {num_epochs} epochs.")

        for epoch in range(1, num_epochs + 1):
            self.train()
            running_loss = 0.0
            for features, targets in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} Training Batches', leave=False):
                features, targets = features.to(device), targets.to(device)

                # 扩展 targets 的维度
                targets = targets.unsqueeze(-1)  # 转换为 (batch_size, 1)

                optimizer.zero_grad()
                outputs = self(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

            # 验证阶段
            self.eval()
            val_loss = 0.0
            total_mae, total_mse = [], []
            with torch.no_grad():
                for features, targets in tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} Validation Batches', leave=False):
                    features, targets = features.to(device), targets.to(device)
                    
                    # 扩展 targets 的维度
                    targets = targets.unsqueeze(-1)

                    outputs = self(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    mae, mse = self.compute_metrics(outputs, targets)
                    total_mae.append(mae)
                    total_mse.append(mse)

            val_loss /= len(val_loader)
            avg_mae = np.mean(total_mae)
            avg_mse = np.mean(total_mse)

            print(f'Validation Loss: {val_loss:.4f}, MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}')

            # 检查是否为最佳模型
            if avg_mse < best_val_mse:
                best_val_mse = avg_mse
                self.save(epoch, model_dir=model_save_dir, mode=3)  # 保存最佳模型
                print(f'Best model updated at epoch {epoch} with best MSE: {best_val_mse:.4f}')

            # Early Stopping 检查
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        print(f'Training complete. Best validation MSE: {best_val_mse:.4f}')

    def test(self, data_loader):
        """
        测试模型并计算MAE和MSE。
        """
        device = next(self.parameters()).device  # 获取模型所在设备
        self.eval()
        total_mae, total_mse = [], []

        with torch.no_grad():
            for features, targets in tqdm(data_loader, desc='Testing Batches', leave=False):
                features, targets = features.to(device), targets.to(device)
                outputs = self(features)

                mae, mse = self.compute_metrics(outputs, targets)
                total_mae.append(mae)
                total_mse.append(mse)

        avg_mae = np.mean(total_mae)
        avg_mse = np.mean(total_mse)

        print(f'Test Results - MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}')
        return avg_mae, avg_mse

class EarlyStopping:
    def __init__(self, patience=15, delta=0):
        """
        Early stopping utility to stop training when validation loss doesn't improve.
        """
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
