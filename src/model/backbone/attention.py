import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.backbone.rope import NTKRotaryEmbedding, YaRNRotaryEmbedding, apply_rotary_pos_emb
from src.model.backbone.rms import RMSNorm

class SingleHeadAttention(nn.Module):
    # [基础] 单头注意力机制
    def __init__(self, config):
        super().__init__()
        head_size = config.hidden_dim // config.n_head
        self.key = nn.Linear(config.hidden_dim, head_size)
        self.value = nn.Linear(config.hidden_dim, head_size)
        self.query = nn.Linear(config.hidden_dim, head_size)
        self.head_size = head_size
        self.register_buffer('attention_mask', torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        k, v, q = self.key(x), self.value(x), self.query(x)
        # Scaled Dot-Product Attention
        weight = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        weight = weight.masked_fill(self.attention_mask[:seq_len, :seq_len] == 0, float('-inf'))
        return F.softmax(weight, dim=-1) @ v

class MultiHeadAttention(nn.Module):
    # [基础] 多头注意力机制
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FlashAttention(nn.Module):
    """
    [进阶] Flash Attention (集成 QKV 投影与 SDPA)
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.hidden_dim = config.hidden_dim
        self.head_size = config.hidden_dim // config.n_head
        self.qkv_atten = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=config.bias)
        self.dropout = config.dropout
        self.att_dropout = nn.Dropout(self.dropout)
        self.c_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=config.bias)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv_atten(x).split(self.hidden_dim, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )
        return self.att_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))

class GroupedQueryAttention(nn.Module):
    """
    [进阶] GQA (Grouped Query Attention)
    平衡 MHA 的性能与 MQA 的推理速度
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_size = config.hidden_dim // config.n_head
        
        self.rotary_emb = NTKRotaryEmbedding(
            dim=self.head_size,
            max_position_embeddings=int(config.max_seq_len * config.rope_scale),
            base=config.rope_base,
            scale=config.rope_scale
        )
        self.qkv_proj = nn.Linear(
            config.hidden_dim, (self.n_head + 2 * self.n_kv_head) * self.head_size, bias=config.bias
        )
        self.dropout = config.dropout
        self.att_dropout = nn.Dropout(self.dropout)
        self.c_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)

    def forward(self, x, position_ids=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.n_head * self.head_size, self.n_kv_head * self.head_size, self.n_kv_head * self.head_size], dim=-1)
        
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)

        cos, sin = self.rotary_emb(v, seq_len=T)
        # 支持传入 position_ids
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

        if self.n_kv_head != self.n_head:
            # 修复：k, v shape 为 (B, n_kv_head, T, head_size)
            # repeat_interleave 是最稳的，expand 必须配合 reshape 才能正确广播
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        y = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
        )
        return self.att_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))

class LatentAttention(nn.Module):
    """
    [演进阶段 5] MLA (Multi-Head Latent Attention)
    DeepSeek-V2/V3 核心架构，通过低秩压缩 (Low-Rank Compression) 大幅降低 KV Cache 显存
    优化: 增加 k_pe_norm 以增强位置编码稳定性，解耦内容与位置计算
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_size = config.hidden_dim // config.n_head
        
        # 压缩参数
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.rope_head_dim = config.qk_rope_head_dim
        self.kv_head_dim = self.head_size 
        self.q_head_dim = self.head_size
        
        # Query Compression: x -> c_Q -> [q_nop, q_pe]
        self.q_down_proj = nn.Linear(config.hidden_dim, self.q_lora_rank, bias=config.bias)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.n_head * self.q_head_dim, bias=config.bias)
        self.q_pe_proj = nn.Linear(self.q_lora_rank, self.n_head * self.rope_head_dim, bias=config.bias)
        
        # KV Compression: x -> c_KV -> [k_nop, v] + k_pe
        self.kv_down_proj = nn.Linear(config.hidden_dim, self.kv_lora_rank, bias=config.bias)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, self.n_kv_head * (self.kv_head_dim + self.kv_head_dim), bias=config.bias)
        
        # Decoupled RoPE Key: k_pe 直接从 x 投影并归一化
        self.k_pe_proj = nn.Linear(config.hidden_dim, self.rope_head_dim, bias=config.bias)
        self.k_pe_norm = RMSNorm(self.rope_head_dim) # [优化] 增加 Norm

        # [优化] 使用 YaRNRotaryEmbedding 替换旧的 NTKAwareRotaryEmbedding
        # 默认 scale=1.0 退化为标准 RoPE，如果 config.rope_scaling_type == 'yarn' 则启用
        self.rotary_emb = YaRNRotaryEmbedding(
            dim=self.rope_head_dim,
            max_position_embeddings=int(config.max_seq_len * config.rope_scale),
            base=config.rope_base,
            scale=config.rope_scale,
            original_max_position_embeddings=config.max_seq_len
        )
        self.dropout = config.dropout # 修复: 初始化 dropout 参数
        self.att_dropout = nn.Dropout(config.dropout) 
        self.c_proj = nn.Linear(self.n_head * self.kv_head_dim, config.hidden_dim, bias=config.bias)
        
        # 缩放因子: 用于平衡内容(nop)和位置(pe)
        # 融合 YaRN 的 mscale 因子 (如果存在)
        mscale = getattr(self.rotary_emb, "mscale", 1.0)
        self.softmax_scale = ((self.q_head_dim + self.rope_head_dim) ** -0.5) * mscale
    
    def forward(self, x, position_ids=None):
        B, T, C = x.shape
        
        # 1. Query Generation
        c_Q = self.q_norm(self.q_down_proj(x))
        q_nop = self.q_up_proj(c_Q).view(B, T, self.n_head, self.q_head_dim)
        q_pe = self.q_pe_proj(c_Q).view(B, T, self.n_head, self.rope_head_dim)
        
        # 2. KV Generation
        c_KV = self.kv_norm(self.kv_down_proj(x))
        kv_up = self.kv_up_proj(c_KV)
        k_nop, v = kv_up.split([self.n_kv_head * self.kv_head_dim, self.n_kv_head * self.kv_head_dim], dim=-1)
        
        k_nop = k_nop.view(B, T, self.n_kv_head, self.kv_head_dim)
        v = v.view(B, T, self.n_kv_head, self.kv_head_dim)
        
        # Grouped KV Expansion
        if self.n_kv_head != self.n_head:
            group_size = self.n_head // self.n_kv_head
            # 使用 expand 优化显存
            k_nop = k_nop.unsqueeze(3).expand(-1, -1, -1, group_size, -1).reshape(B, T, self.n_head, self.kv_head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, group_size, -1).reshape(B, T, self.n_head, self.kv_head_dim)
        
        # k_pe Generation (Shared & Normalized)
        k_pe = self.k_pe_norm(self.k_pe_proj(x)).view(B, T, 1, self.rope_head_dim)
        
        # 3. RoPE Application (Only to PE parts)
        # Transpose to (B, H, T, D)
        q_nop, q_pe = q_nop.transpose(1, 2), q_pe.transpose(1, 2)
        k_nop, k_pe = k_nop.transpose(1, 2), k_pe.transpose(1, 2) # k_pe: (B, 1, T, D)
        v = v.transpose(1, 2)
        
        cos, sin = self.rotary_emb(v, seq_len=T)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids=position_ids)
        
        # Broadcast k_pe to all heads
        k_pe = k_pe.expand(-1, self.n_head, -1, -1)
        
        # 4. Attention Calculation
        # 拼接 content 和 pe 部分
        q = torch.cat([q_nop, q_pe], dim=-1)
        k = torch.cat([k_nop, k_pe], dim=-1)
        
        y = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True,
            scale=self.softmax_scale
        )
        
        return self.att_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.kv_head_dim)))
