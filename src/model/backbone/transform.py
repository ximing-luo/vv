import torch.nn as nn
from .attention import MultiHeadAttention, FlashAttention, GroupedQueryAttention, MultiHeadLatentAttention
from .rms import RMSNorm
from .moe import FeedForward, GatedMLP, MoE, SharedMoE, AuxiliaryLossMoE, DeepseekMoE

class BaseBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class BaseMoEBlock(nn.Module):
    """
    Transformer Block
    Structure: Pre-Norm -> Attention -> Residual -> Pre-Norm -> MLP -> Residual
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.attn = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.hidden_dim)
        self.mlp = SharedMoE(config)

    def forward(self, x, position_ids=None):
        x = x + self.attn(self.norm1(x), position_ids=position_ids)
        x = x + self.mlp(self.norm2(x))
        return x

class MoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.attn = MultiHeadLatentAttention(config)
        self.norm2 = RMSNorm(config.hidden_dim)
        self.mlp = AuxiliaryLossMoE(config)

    def forward(self, x, position_ids=None):
        x = x + self.attn(self.norm1(x), position_ids=position_ids)
        mlp_out, aux_loss = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x, aux_loss

class DeepseekBlock(nn.Module):
    """
    DeepSeek-V3 风格的 Block
    自适应负载均衡已在 DeepSeekMoE 内部完成，此处逻辑极简
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.attn = MultiHeadLatentAttention(config)
        self.norm2 = RMSNorm(config.hidden_dim)
        self.mlp = DeepseekMoE(config)

    def forward(self, x, position_ids=None):
        x = x + self.attn(self.norm1(x), position_ids=position_ids)
        x = x + self.mlp(self.norm2(x))
        return x

