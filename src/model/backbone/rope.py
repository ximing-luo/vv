import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 预计算频率 inv_freq
        # theta_i = base^(-2i/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).float().to(device) / self.dim))
        # 设置 persistent=False，防止加载预训练权重时覆盖掉微调时的新配置
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 缓存 cos 和 sin 表
        self._set_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        # freqs: (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)
        
        # emb: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 缓存 cos 和 sin
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [batch, seq_len, n_head, head_dim]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class NTKAwareRotaryEmbedding(RotaryEmbedding):
    """
    NTK-Aware Scaled RoPE
    通过调整 base 来实现非线性插值，从而在微调时支持更长的上下文，且保持高频信息的精度。
    """
    def __init__(self, dim, max_position_embeddings=32768, base=10000, alpha=1.0, device=None):
        # alpha 是扩展倍数，例如想扩展到 4倍长度，alpha=4
        base = base * alpha ** (dim / (dim - 2))
        super().__init__(dim, max_position_embeddings, base, device)

class YaRNScaledRotaryEmbedding(RotaryEmbedding):
    """
    YaRN (Yet another RoPE extension)
    结合了 NTK-Aware 和 Attention Scaling，是目前效果最好的长文本扩展方案之一。
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1.0, original_max_position_embeddings=2048, device=None):
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        super().__init__(dim, max_position_embeddings, base, device)
        
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        
        if seq_len > self.original_max_position_embeddings:
            t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
            t = t / self.scale # 线性缩放时间步
        else:
            t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    应用 RoPE 到 Query 和 Key
    q, k: [batch, n_head, seq_len, head_dim]  (注意：这里假设已经转置为 BHSD 格式)
    cos, sin: [seq_len, head_dim] or [1, seq_len, 1, head_dim]
    """
    if position_ids is not None:
        # position_ids: [batch, seq_len]
        # 从缓存中根据 position_ids 提取对应的 cos, sin
        cos = cos[position_ids].unsqueeze(1) # [batch, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1) # [batch, 1, seq_len, dim]
    else:
        # 默认假设是连续的 0:seq_len
        cos = cos.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
