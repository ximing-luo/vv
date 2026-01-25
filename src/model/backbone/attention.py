import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.backbone.rope import NTKAwareRotaryEmbedding, apply_rotary_pos_emb
from src.model.backbone.rms import RMSNorm

class SingleHeadAttention(nn.Module):
    # 单头注意力机制
    def __init__(self, config):
        super().__init__()
        head_size = config.hidden_dim // config.n_head
        self.key = nn.Linear(config.hidden_dim, head_size)
        self.value = nn.Linear(config.hidden_dim, head_size)
        self.query = nn.Linear(config.hidden_dim, head_size)
        self.head_size = head_size

        # 尝试学习新的写法，attention_mask 通过 register_buffer 注册
        # 因为不用计算 梯度，所以节约内存和显存，速度也更快
        self.register_buffer(
            'attention_mask', 
            torch.tril(
                torch.ones(config.max_seq_len, config.max_seq_len)
            ))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1)   # @ 就是 torch.matmul 的简化写法
        # 一定要在 softmax 前除以 sqrt(head_size)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0, 
            float('-inf')
        ) / math.sqrt(self.head_size)  # 这里的 hidden_size 其实是 head_size，因为是单头
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads], 
            dim=-1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output

class FlashAttention(nn.Module):
    """
    Flash Attention 模块 (参考 NanoGPT)
    集成 QKV 投影和缩放点积注意力。
    """
    def __init__(self, args):
        super().__init__()
        self.n_head = args.n_head
        self.hidden_dim = args.hidden_dim
        assert args.hidden_dim % args.n_head == 0
        self.head_size = args.hidden_dim // args.n_head
        
        # Q, K, V 联合投影
        self.qkv_atten = nn.Linear(args.hidden_dim, 3 * args.hidden_dim, bias=args.bias)
        
        self.dropout = args.dropout
        self.att_dropout = nn.Dropout(self.dropout)
        
        # 输出投影
        self.c_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=args.bias)

    def forward(self, x):
        B, T, C = x.shape
        
        # 计算 Q, K, V
        q, k, v = self.qkv_atten(x).split(self.hidden_dim, dim=2)
        
        # 变换形状: (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Flash Attention
        # 训练时使用 dropout，推理时 dropout_p=0
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0, 
            is_causal=True
        )
        
        # 恢复形状: (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.att_dropout(self.c_proj(y))


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) 模块
    支持 Flash Attention (通过 F.scaled_dot_product_attention 实现)
    
    GQA 介于 Multi-Head Attention (MHA) 和 Multi-Query Attention (MQA) 之间：
    - MHA: n_head == n_kv_head
    - MQA: n_kv_head == 1
    - GQA: n_head % n_kv_head == 0
    """
    def __init__(self, args):
        super().__init__()
        self.n_head = args.n_head
        # 如果 args 中没有 n_kv_head，默认退化为 MHA
        self.n_kv_head = getattr(args, "n_kv_head", args.n_head)
        self.hidden_dim = args.hidden_dim
        assert self.n_head % self.n_kv_head == 0, "n_head 必须是 n_kv_head 的整数倍"
        self.head_size = args.hidden_dim // args.n_head
        
        self.rotary_emb = NTKAwareRotaryEmbedding(
            dim=self.head_size,
            max_position_embeddings=args.max_seq_len,
            base=args.rope_base,
            alpha=args.rope_ntk_alpha
        )
        
        # Q, K, V 投影
        # Q 的输出维度是 n_head * head_size (即 hidden_dim)
        # K, V 的输出维度是 n_kv_head * head_size
        self.qkv_proj = nn.Linear(
            args.hidden_dim, 
            (self.n_head + 2 * self.n_kv_head) * self.head_size, 
            bias=args.bias
        )
        
        self.dropout = args.dropout
        self.att_dropout = nn.Dropout(self.dropout)
        
        # 输出投影
        self.c_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=args.bias)

    def forward(self, x):
        B, T, C = x.shape
        
        # 计算 QKV 并分割
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([
            self.n_head * self.head_size, 
            self.n_kv_head * self.head_size, 
            self.n_kv_head * self.head_size
        ], dim=-1)
        
        # 变换形状: (B, T, H, HS) -> (B, H, T, HS)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)

        # 应用 RoPE
        cos, sin = self.rotary_emb(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 如果是 GQA/MQA，需要对 K, V 进行扩展以匹配 Q 的头数
        if self.n_kv_head != self.n_head:
            # 使用 expand 替代 repeat_interleave，利用广播机制节省显存
            k = k.expand(-1, self.n_head, -1, -1)
            v = v.expand(-1, self.n_head, -1, -1)

        # 使用 Flash Attention (SDPA)
        # 训练时使用 dropout，推理时 dropout_p=0
        # 增加 .contiguous() 以提升在 Windows/BF16 环境下的内核执行稳定性
        y = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), 
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0, 
            is_causal=True
        )
        
        # 恢复形状: (B, n_head, T, HS) -> (B, T, n_head, HS) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.att_dropout(self.c_proj(y))

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA)
    DeepSeek-V2/V3 的核心 Attention 架构。
    通过对 KV (和 Query) 进行低秩压缩 (Low-Rank Compression)，大幅降低 KV Cache 显存占用，
    同时保持甚至超过标准 MHA 的性能。
    
    架构特点:
    1. KV Compression: x -> c_KV (Latent) -> [k_nop, k_pe, v]
    2. Query Compression: x -> c_Q (Latent) -> [q_nop, q_pe]
    3. Decoupled RoPE: 只有 k_pe 和 q_pe 参与 RoPE 计算，k_nop 和 q_nop 不参与。
    """
    def __init__(self, args):
        super().__init__()
        self.n_head = args.n_head
        # 如果指定了 n_kv_head 且小于 n_head，则开启投影分组
        self.n_kv_head = getattr(args, "n_kv_head", args.n_head)
        assert self.n_head % self.n_kv_head == 0, "n_head 必须是 n_kv_head 的整数倍"
        self.hidden_dim = args.hidden_dim
        self.head_size = args.hidden_dim // args.n_head
        
        # MLA 特有参数，如果没有定义则使用默认比例
        # kv_lora_rank: KV 压缩后的潜变量维度
        self.kv_lora_rank = getattr(args, "kv_lora_rank", 128)
        # q_lora_rank: Query 压缩后的潜变量维度 (DeepSeek-V2 使用，可选)
        self.q_lora_rank = getattr(args, "q_lora_rank", 96)
        # rope_head_dim: RoPE 部分的维度 (通常较小，如 64)
        self.rope_head_dim = getattr(args, "rope_head_dim", 64)
        # kv_head_dim: Key/Value 的头维度 (通常等于 head_size)
        self.kv_head_dim = self.head_size 
        # q_head_dim: Query (nop) 的头维度
        self.q_head_dim = self.head_size
        
        # --- Query Compression ---
        # Down Projection: x -> c_Q
        self.q_down_proj = nn.Linear(self.hidden_dim, self.q_lora_rank, bias=args.bias)
        self.q_norm = RMSNorm(self.q_lora_rank)
        # Up Projection: c_Q -> q_nop (n_head * q_head_dim)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.n_head * self.q_head_dim, bias=args.bias)
        # Up Projection Pe: c_Q -> q_pe (n_head * rope_head_dim)
        self.q_pe_proj = nn.Linear(self.q_lora_rank, self.n_head * self.rope_head_dim, bias=args.bias)
        
        # --- KV Compression ---
        # Down Projection: x -> c_KV
        self.kv_down_proj = nn.Linear(self.hidden_dim, self.kv_lora_rank, bias=args.bias)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # Up Projection: c_KV -> [k_nop, v] (支持分组)
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, self.n_kv_head * (self.kv_head_dim + self.kv_head_dim), bias=args.bias)
        # k_pe_proj: Key 的 RoPE 部分 (共享，直接 from 原始输入 x 投影)
        self.k_pe_proj = nn.Linear(self.hidden_dim, self.rope_head_dim, bias=args.bias)
        # self.k_pe_norm = RMSNorm(self.rope_head_dim) # [已注释] 通常在投影后也会加个 norm

        # RoPE
        self.rotary_emb = NTKAwareRotaryEmbedding(
            dim=self.rope_head_dim,
            max_position_embeddings=args.max_seq_len,
            base=args.rope_base,
            alpha=args.rope_ntk_alpha
        )

        self.dropout = args.dropout
        self.att_dropout = nn.Dropout(args.dropout) 

        # Output Projection
        self.c_proj = nn.Linear(self.n_head * self.kv_head_dim, self.hidden_dim, bias=args.bias)
        
        # MLA 缩放因子: 用于平衡内容(nop)和位置(pe)点积后的数值量级
        # 参考 DeepSeek 实现，通常是一个可学习或固定的缩放系数
        self.softmax_scale = (self.q_head_dim + self.rope_head_dim) ** -0.5
        # 如果需要更精细的控制，可以分别为 nop 和 pe 设置不同的 scale
        # 但在 SDPA 中，我们通常统一缩放拼接后的 q, k
    
    def forward(self, x, position_ids=None):
        B, T, C = x.shape
        
        # --- Query Generation ---
        c_Q = self.q_down_proj(x)
        c_Q = self.q_norm(c_Q)
        q_nop = self.q_up_proj(c_Q).view(B, T, self.n_head, self.q_head_dim)
        q_pe = self.q_pe_proj(c_Q).view(B, T, self.n_head, self.rope_head_dim)
        
        # --- KV Generation ---
        c_KV = self.kv_down_proj(x)
        c_KV = self.kv_norm(c_KV)
        
        # 生成 K_nop, V, K_pe
        kv_up = self.kv_up_proj(c_KV)
        k_nop, v = kv_up.split([self.n_kv_head * self.kv_head_dim, self.n_kv_head * self.kv_head_dim], dim=-1)
        
        k_nop = k_nop.view(B, T, self.n_kv_head, self.kv_head_dim)
        v = v.view(B, T, self.n_kv_head, self.kv_head_dim)
        
        # 如果开启了分组，则需要扩展到 n_head
        if self.n_kv_head != self.n_head:
            # 使用 expand 而不是 repeat_interleave 以节省内存
            # 在维度不为 1 时是不允许直接 expand 的，需要先 unsqueeze 一个 1 维
            group_size = self.n_head // self.n_kv_head
            k_nop = k_nop.unsqueeze(3).expand(-1, -1, -1, group_size, -1).reshape(B, T, self.n_head, self.kv_head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, group_size, -1).reshape(B, T, self.n_head, self.kv_head_dim)
        
        # k_pe 是共享的，直接从 x 投影
        k_pe = self.k_pe_proj(x)
        # k_pe = self.k_pe_norm(k_pe) # [已注释] 移除额外的 Norm 以增强位置信号
        k_pe = k_pe.view(B, T, 1, self.rope_head_dim)
        
        # --- Transpose for Attention ---
        # (B, n_head, T, dim)
        q_nop = q_nop.transpose(1, 2)
        q_pe = q_pe.transpose(1, 2)
        k_nop = k_nop.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2) # (B, 1, T, rope_head_dim)
        v = v.transpose(1, 2)
        
        # --- RoPE ---
        # 仅对 pe 部分应用 RoPE
        cos, sin = self.rotary_emb(v, seq_len=T) # v 只是用来拿 seq_len，不参与 RoPE
        # 注意：apply_rotary_pos_emb 期望的是 (B, H, T, D)
        # 这里 k_pe 是 (B, 1, T, D)，apply_rotary_pos_emb 会通过广播处理
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids=position_ids)
        
        # k_pe 需要扩展到 n_head
        k_pe = k_pe.expand(-1, self.n_head, -1, -1)
        
        q = torch.cat([q_nop, q_pe], dim=-1) # (B, H, T, q_head_dim + rope_head_dim)
        k = torch.cat([k_nop, k_pe], dim=-1) # (B, H, T, kv_head_dim + rope_head_dim)
        
        # 使用 Flash Attention
        # 增加 .contiguous() 以提升在 Windows/BF16 环境下的内核执行稳定性
        # 使用自定义的 softmax_scale 替代默认的 1/sqrt(dk)
        y = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
            scale=self.softmax_scale
        )
        
        # 恢复形状
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.kv_head_dim)
        
        return self.att_dropout(self.c_proj(y)) 