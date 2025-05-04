"""
自链接实验包初始化文件
"""
from .self_link_experiment import SelfLinkLM, SelfLinkBlock, SelfLinkAttention
from .experiment_config import SelfLinkConfig, get_config, DEFAULT_SELF_LINK_CONFIG
from .train import train_model, compare_with_baseline
from .eval import ExperimentLogger, evaluate_model, run_evaluation

__all__ = [
    'SelfLinkLM',
    'SelfLinkBlock', 
    'SelfLinkAttention',
    'SelfLinkConfig',
    'get_config',
    'DEFAULT_SELF_LINK_CONFIG',
    'train_model',
    'compare_with_baseline',
    'ExperimentLogger',
    'evaluate_model',
    'run_evaluation'
]