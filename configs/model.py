from dataclasses import dataclass, field
from typing import List

@dataclass
class VVConfig:
    max_seq_len: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 2   # 组注意力，头数可以小于 n_head，用于减少计算量
    hidden_dim: int = 512
    dropout: float = 0.1
    bias: bool = False
    # vocab_size 默认为 None，必须在实例化时指定（通常由 tokenizer.vocab_size 决定）
    vocab_size: int = None
    # Transformers 库期望的特殊 Token ID
    bos_token_id: int = None
    eos_token_id: int = None
    pad_token_id: int = None
    # MoE 相关配置
    intermediate_size: int = None
    num_experts: int = 3
    num_experts_per_tok: int = 1
    num_shared_experts: int = 1
    bias_update_rate: float = 0.001
    # RoPE 配置
    rope_base: float = 10000.0
    rope_ntk_alpha: float = 1.0  # NTK 的扩展倍数
    
@dataclass
class VisualVVConfig(VVConfig):
    # 视觉投影配置
    vision_hidden_dim: int = 768
    vision_model_path: str = "./model/vision_model/clip-vit-base-patch16"
    image_special_token: str = '@' * 196
    image_ids: List[int] = field(default_factory=lambda: [34] * 196)

