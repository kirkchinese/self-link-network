import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import PreTrainedModel, PretrainedConfig

class SelfLinkConfig(PretrainedConfig):
    """自链接模型配置"""
    
    def __init__(
        self,
        vocab_size=50257,
        n_embd=768,
        n_layer=12,
        n_head=12,
        self_link_ratio=0.05,
        max_position_embeddings=1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.self_link_ratio = self_link_ratio  # 自链接比例
        self.max_position_embeddings = max_position_embeddings  # 最大位置编码长度

class EnhancedSelfLinkLayer(nn.Module):
    """增强版自链接层"""
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        # 多头自链接设计
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_embd, self.head_dim),
                nn.GELU(),
                nn.Linear(self.head_dim, self.head_dim)
            ) for _ in range(config.n_head)
        ])
        
        # 投影层
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """前向传播"""
        residual = x
        
        # 多头处理
        head_outputs = [head(x) for head in self.heads]
        x = torch.cat(head_outputs, dim=-1)
        
        # 投影和残差连接
        x = self.proj(x)
        x = self.dropout(x)
        return self.norm(residual + x)

def apply_rope(q, k, seq_len):
    """旋转位置编码实现"""
    dim = q.size(-1)
    position = torch.arange(seq_len, device=q.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=q.device) * (-math.log(10000.0) / dim))
    sin = torch.sin(position * div_term)
    cos = torch.cos(position * div_term)
    
    q_rot = torch.zeros_like(q)
    q_rot[..., 0::2] = q[..., 0::2] * cos - q[..., 1::2] * sin
    q_rot[..., 1::2] = q[..., 0::2] * sin + q[..., 1::2] * cos
    
    k_rot = torch.zeros_like(k)
    k_rot[..., 0::2] = k[..., 0::2] * cos - k[..., 1::2] * sin
    k_rot[..., 1::2] = k[..., 0::2] * sin + k[..., 1::2] * cos
    
    return q_rot, k_rot

class HybridAttentionLayer(nn.Module):
    """混合注意力层"""
    
    def __init__(self, config):
        super().__init__()
        # 自注意力机制
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=0.1
        )
        # 自链接机制
        self.self_link = EnhancedSelfLinkLayer(config)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # 混合注意力与自链接
        attn_out, _ = self.attn(x, x, x)
        link_out = self.self_link(attn_out)
        ffn_out = self.ffn(link_out)
        return self.norm(ffn_out + link_out)

class SelfLinkLM(PreTrainedModel):
    """增强版自链接语言模型"""
    
    config_class = SelfLinkConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # 词嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # 旋转位置编码参数
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)))
        
        # 混合层堆叠
        self.layers = nn.ModuleList([
            HybridAttentionLayer(config) for _ in range(config.n_layer)
        ])
        
        # 输出层
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """优化后的权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        模型前向传播
        参数:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签token ids [batch_size, seq_len]
        返回:
            loss (可选), logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        
        # 获取位置id
        position_ids = self.position_ids[:, :seq_len]
        
        # 词嵌入
        tok_emb = self.wte(input_ids)
        
        # 应用旋转位置编码
        x = tok_emb
        
        # 自链接层堆叠
        for layer in self.layers:
            x = layer(x)
        
        # 输出层
        x = self.ln_f(x)
        logits = self.head(x)
        
        # 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {'loss': loss, 'logits': logits}