# tasks/regression/models/__init__.py

from .LSTM import LSTM

# 预设模型字典
PRESET_MODELS = {
    'lstm': LSTM,
    # 其他回归模型可以在这里继续添加
}
