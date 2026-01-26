import torch
import torch.nn as nn
import math

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    应用 RoPE 到 Query 和 Key
    """
    if position_ids is not None:
        # 动态选取对应的 cos/sin: [batch, 1, seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        # 默认切片: [1, 1, seq_len, dim]
        cos = cos[:q.size(2)].unsqueeze(0).unsqueeze(0)
        sin = sin[:q.size(2)].unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    """
    [基础] 旋转位置编码 (RoPE)
    Standard RoPE implementation
    """
    def __init__(self, dim, max_position_embeddings=32768, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        # 预计算频率: theta_i = base^(-2i/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)

class NTKRotaryEmbedding(RotaryEmbedding):
    """
    [演进阶段 2] NTK-Aware RoPE
    通过非线性插值 (调整 Base) 扩展上下文窗口，保持高频信息精度
    """
    def __init__(self, dim, max_position_embeddings=32768, base=10000, scale=1.0, device=None):
        # scale > 1 时扩展上下文，base 随之增大
        base = base * scale ** (dim / (dim - 2))
        super().__init__(dim, max_position_embeddings, base, device)

class YaRNRotaryEmbedding(RotaryEmbedding):
    """
    [演进阶段 3] YaRN (Yet another RoPE extension)
    结合 NTK-Aware 与 Attention Scaling，实现更鲁棒的长文本外推
    核心: 针对不同频率分段进行不同策略的插值与外推
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1.0, original_max_position_embeddings=2048, device=None):
        # 初始化 YaRN 特有参数
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.yarn_ramp_beta_fast = 32
        self.yarn_ramp_beta_slow = 1
        # 计算 Attention Scaling Factor (mscale)
        self.mscale = float(0.1 * math.log(self.scale) + 1.0) if self.scale > 1.0 else 1.0
        
        # 调用父类初始化
        # 注意：父类初始化会调用 _set_cos_sin_cache，而该方法依赖 self.mscale
        # 所以必须先设置 mscale 再调用 super().__init__
        super().__init__(dim, max_position_embeddings, base, device)
        
        # 重新初始化 inv_freq 以支持 YaRN 逻辑
        self._init_yarn_inv_freq(dim, base, device)
        # 重新刷新 cache (使用新的 inv_freq 和 mscale)
        self._set_cos_sin_cache(max_position_embeddings, device, torch.get_default_dtype())

    def _init_yarn_inv_freq(self, dim, base, device):
        # YaRN 频率调整逻辑
        pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)
        
        low = math.floor(dim * math.log(self.original_max_position_embeddings / (self.yarn_ramp_beta_fast * 2 * math.pi)) / (2 * math.log(base)))
        high = math.ceil(dim * math.log(self.original_max_position_embeddings / (self.yarn_ramp_beta_slow * 2 * math.pi)) / (2 * math.log(base)))
        
        # 生成平滑混合掩码
        mask = (torch.arange(0, dim // 2).float().to(device) - low) / (high - low)
        mask = torch.clamp(mask, 0.0, 1.0)
        
        # 混合插值与外推频率
        inv_freq = (1.0 - mask) * inv_freq_interpolation + mask * inv_freq_extrapolation
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # YaRN 的 mscale 不应直接乘在 cos/sin 上 (否则会导致内容/位置部分缩放不一致)
        # 它应该作为整个 Attention Score 的缩放因子
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
