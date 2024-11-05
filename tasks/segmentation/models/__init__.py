# tasks/regression/models/__init__.py

from .UNet import UNet

# 预设模型字典
PRESET_MODELS = {
    'UNet': UNet,
}
