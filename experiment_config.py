"""
自链接实验配置
"""
from dataclasses import dataclass
from minimind.model.LMConfig import LMConfig

@dataclass
class SelfLinkConfig(LMConfig):
    """自链接实验专用配置"""
    self_link_enabled: bool = True  # 是否启用自链接
    self_link_init_coeff: float = 0.5  # 初始自链接系数
    self_link_threshold: float = 0.1  # 自触发阈值
    self_link_decay: float = 0.9  # 衰减系数
    compare_baseline: bool = True  # 是否与基线模型比较
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# 默认实验配置
DEFAULT_SELF_LINK_CONFIG = SelfLinkConfig(
    dim=768,
    n_heads=8,
    n_layers=6,
    vocab_size=32000,
    dropout=0.1,
    max_seq_len=2048
)

def get_config(config_dict=None):
    """获取实验配置"""
    if config_dict:
        return SelfLinkConfig(**config_dict)
    return DEFAULT_SELF_LINK_CONFIG