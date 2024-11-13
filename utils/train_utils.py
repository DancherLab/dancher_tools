# dancher_tools/utils/train_utils.py

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
            
import torch
import numpy as np
from tqdm import tqdm

class ConfidentLearning:
    def __init__(self, model, threshold=0.6):
        self.model = model
        self.threshold = threshold

    def identify_noisy_labels(self, data_loader, device):
        self.model.eval()
        noisy_indices = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Evaluating Labels")):
                images = images.to(device)
                labels = labels.to(device)  # 转到设备上
                outputs = torch.softmax(self.model(images), dim=1)

                # 获取每个图像的最大概率和预测类别
                max_probs, preds = torch.max(outputs, dim=1)
                max_probs = max_probs.view(max_probs.size(0), -1).max(dim=1)[0]  # 在 height 和 width 上取最大值

                # 记录低置信度的样本索引
                for idx in range(len(max_probs)):
                    prob = max_probs[idx].item()
                    
                    # 比较整个标签图像与预测图像的相等性
                    label_image = labels[idx]
                    pred_image = preds[idx]
                    
                    if prob < self.threshold and not torch.equal(label_image, pred_image):
                        global_index = batch_idx * data_loader.batch_size + idx
                        noisy_indices.append(global_index)

        return noisy_indices

    def clean_data(self, dataset, noisy_indices):
        # 只保留干净的样本
        cleaned_data = torch.utils.data.Subset(dataset, [i for i in range(len(dataset)) if i not in noisy_indices])
        return cleaned_data
