"""
神经元自链接实验实现
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from minimind.model.model import MiniMindLM, MiniMindBlock, Attention, apply_rotary_emb, repeat_kv
from learn_test.experiment_config import get_config

class SelfLinkAttention(Attention):
    """带自链接的注意力机制"""
    def __init__(self, args):
        super().__init__(args)
        # 自链接参数
        self.self_link_dim = args.dim // args.n_heads  # 与head_dim相同
        self.self_link_coeff = nn.Parameter(torch.ones(self.self_link_dim) * 0.5)  # 初始系数0.5
        self.self_link_threshold = 0.1  # 自触发阈值
        self.self_link_decay = 0.9  # 衰减系数
        
    def _self_link_activation(self, x: Tensor) -> Tensor:
        """自链接激活函数"""
        # 计算激活值
        activation = torch.sigmoid(x * self.self_link_coeff)
        # 应用衰减
        self.self_link_coeff.data *= self.self_link_decay
        return activation
        
    def forward(self, x: Tensor, pos_cis: Tensor, 
                past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
                use_cache: bool = False):
        # 原始注意力计算
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # 调整形状 [batch, seq_len, heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        
        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        
        # 自链接计算
        self_link_act = self._self_link_activation(xq)
        xq = xq * self_link_act  # 应用自链接激活
        
        # 剩余部分保持原样
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv
            
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv

class SelfLinkBlock(MiniMindBlock):
    """带自链接的Transformer块"""
    def __init__(self, layer_id: int, config):
        super().__init__(layer_id, config)
        # 替换原始注意力层
        self.attention = SelfLinkAttention(config)

class SelfLinkLM(MiniMindLM):
    """带自链接的完整模型"""
    def __init__(self, params=None):
        super().__init__(params)
        # 替换所有Transformer块
        self.layers = nn.ModuleList([
            SelfLinkBlock(l, params) for l in range(self.n_layers)
        ])

def test_self_link():
    """测试自链接模型"""
    from minimind.model.LMConfig import LMConfig
    
    config = LMConfig(
        dim=512,
        n_heads=8,
        n_layers=6,
        vocab_size=32000
    )
    
    model = SelfLinkLM(config)
    print("自链接模型创建成功")
    print("模型结构:", model)
    
    # 测试前向传播
    input_ids = torch.randint(0, 32000, (1, 32))
    output = model(input_ids)
    print("前向传播测试通过, 输出形状:", output.logits.shape)

if __name__ == "__main__":
    test_self_link()