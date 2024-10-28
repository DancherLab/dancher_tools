# Dancher-Tools

**Dancher-Tools** 是一个用于深度学习任务的工具库，支持回归、分割等多个模型的加载、训练、测试以及数据管理，旨在简化机器学习项目中的模型开发与实验过程。

## 主要功能

1. **模型加载与管理**：支持加载预训练模型权重，保存最佳模型权重，并且可以自定义加载模式（按最新、最佳、固定名称或时间戳保存模型）。
2. **数据加载器**：通过 `utils/data_loader.py`，可以方便地加载数据并提供给模型训练、验证和测试。
3. **损失函数选择**：提供灵活的损失函数选择工具，通过配置文件指定使用不同的损失函数。
4. **训练与验证**：支持回归和分割任务的训练过程，集成早停（Early Stopping）机制，有效防止过拟合。
5. **测试与评估**：支持对测试数据集进行评估并提供多种评估指标，如IoU、精确率、召回率和F1分数。

## 快速开始

### 1. 克隆项目

``bash
git clone https://github.com/your-repo/dancher-tools.git
cd dancher-tools
``

### 2. 设置配置文件

在 `configs/` 目录下提供了模型训练所需的配置文件，您可以在 `configs/config.yaml` 中设置训练参数，如学习率、batch size、模型名称等。

### 3. 运行主程序

主程序位于 `main.py` 中，包含模型的训练、验证和测试流程：

``bash
python main.py --configs/{config_file}.yaml
``

`main.py` 中包含以下流程：
- **数据加载**：使用 `get_dataloaders` 加载训练集和验证集的数据。
- **模型初始化**：通过 `get_model` 函数初始化模型，并加载预训练权重。
- **损失函数与优化器定义**：根据配置文件选择损失函数，并初始化优化器（默认使用 Adam）。
- **训练过程**：调用 `model.fit` 开始训练，自动保存最佳模型。
- **测试模型**：使用 `model.test` 进行模型评估。

### 4. 训练参数说明

在配置文件 `configs/config.yaml` 中，可以调整以下参数：
- `learning_rate`：学习率
- `num_epochs`：训练轮数
- `model_name`：模型名称（用于选择和保存模型）
- `weight`：预训练权重路径
- `patience`：Early Stopping 的耐心参数
- `delta`：Early Stopping 的最小改进值

### 5. 模型权重加载与保存

Dancher-Tools 支持灵活的模型权重加载方式，具体如下：
- `mode=0`：加载最新 epoch 的模型。
- `mode=1`：加载指定名称的模型文件。
- `mode=2`：加载最近修改的模型文件。
- `mode=3`：加载验证集性能最佳的模型。

### 6. 主要代码模块说明

- **BaseModel**：基础模型类，封装了模型保存、加载、迁移等方法。
- **SegModel**：继承自 `BaseModel` 的分割模型，包含训练、验证、早停和测试等核心方法。
- **compute_metrics**：用于评估模型的各种指标，包括 IoU、精确率、召回率和 F1 分数。
- **EarlyStopping**：早停机制，防止过拟合。
- **transfer**：支持参数迁移和部分层参数的加载功能。

## 贡献指南

如果您有兴趣为 Dancher-Tools 项目贡献代码或报告问题，请遵循以下步骤：
1. Fork 本仓库并创建您的分支。
2. 提交您的修改并发起 Pull Request。
3. 确保您的代码符合项目规范并通过单元测试。

## 许可证

本项目遵循 MIT 许可证，详细信息请参考 LICENSE 文件。
