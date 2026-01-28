from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class VVConfig:
    # --- 基础架构配置 ---
    vocab_size: int = None         # 词表大小 (外部指定)
    hidden_dim: int = 576          # 隐藏层维度
    intermediate_size: int = None  # FFN 中间层维度 (None 则自动计算)
    n_layer: int = 8               # 层数
    n_head: int = 6                # 注意力头数
    n_kv_head: int = 3             # KV 头数 (GQA/MLA 使用)
    max_seq_len: int = 512         # 原始/基础最大序列长度 (训练时的长度)
    dropout: float = 0.1           # Dropout 概率
    bias: bool = False             # 是否使用偏置

    # --- MLA (Multi-Head Latent Attention) 特有配置 ---
    kv_lora_rank: int = 128        # KV 压缩秩
    q_lora_rank: int = 96          # Query 压缩秩
    qk_rope_head_dim: int = 32     # RoPE 部分的维度 (对应代码中的 rope_head_dim)

    # --- RoPE / YaRN 配置 ---
    rope_base: float = 10000.0     # RoPE 基数
    rope_scale: float = 1.0        # YaRN 插值/NTK 扩展倍数 (1.0 代表不扩展)

    # --- MoE (Mixture of Experts) 配置 ---
    num_experts: int = 4           # 总专家数
    num_experts_per_tok: int = 1   # 每个 Token 激活的专家数
    num_shared_experts: int = 1    # 共享专家数（可配 0）
    router_aux_loss_coef: float = 0.01 # 辅助损失系数
    bias_update_rate: float = 0.001 # DeepSeek-V3 动态偏置更新率

    # --- 特殊 Token ID (适配 Transformers 接口) ---
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

@dataclass
class VisualVVConfig(VVConfig):
    # --- 视觉投影配置 ---
    vision_hidden_dim: int = 768   # 视觉模型隐藏层维度
    vision_model_path: str = "./models/clip-vit-base-patch16" # 视觉模型路径
    image_special_token: str = '@' * 196 # 图像占位符文本
    image_ids: List[int] = field(default_factory=lambda: [34] * 196) # 图像占位符 ID
