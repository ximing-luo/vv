import torch.nn as nn
from .attention import MultiHeadAttention, GroupedQueryAttention, LatentAttention
from .rms import RMSNorm
from .moe import FeedForward, HybridMoE, SoftBalancedMoE, SelfAdaptiveMoE

class StandardBlock(nn.Module):
    """
    [演进阶段 1] 标准 Transformer Block
    结构: Pre-Norm -> MHA -> Residual -> Pre-Norm -> FFN -> Residual
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.att = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.ffn = FeedForward(config)

    def forward(self, x, **kwargs):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class AdvancedBlock(nn.Module):
    """
    [演进阶段 2] 进阶 Block (Llama Style)
    改进: RMSNorm + GQA + HybridMoE (可选)
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.attn = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.hidden_dim)
        # 默认使用 HybridMoE，如果 config 指定了 MoE 参数
        self.mlp = HybridMoE(config) if config.num_experts > 1 else FeedForward(config)

    def forward(self, x, position_ids=None):
        # 注意: GQA 支持传入 position_ids 用于 RoPE
        x = x + self.attn(self.norm1(x), position_ids=position_ids) # GQA 暂未完全适配 position_ids 接口，需注意
        x = x + self.mlp(self.norm2(x))
        return x

class DeepSeekV2Block(nn.Module):
    """
    [演进阶段 3] DeepSeek-V2 Block
    改进: MLA (Latent Attention) + SoftBalancedMoE (Aux Loss)
    [注意] 此处返回了辅助损失 (aux_loss)，但在 BaseModel 的 Sequential 遍历中会被丢失
    若使用该 Block，需在 model_llm.py 的 forward 循环中适配元组解包，否则会导致类型错误
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.attn = LatentAttention(config)
        self.norm2 = RMSNorm(config.hidden_dim)
        self.mlp = SoftBalancedMoE(config)

    def forward(self, x, position_ids=None):
        x = x + self.attn(self.norm1(x), position_ids=position_ids)
        mlp_out, aux_loss = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x, aux_loss

class DeepSeekV3Block(nn.Module):
    """
    [演进阶段 4] DeepSeek-V3 Block
    改进: MLA + SelfAdaptiveMoE (无损负载均衡)
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.attn = LatentAttention(config)
        self.norm2 = RMSNorm(config.hidden_dim)
        self.mlp = SelfAdaptiveMoE(config)

    def forward(self, x, position_ids=None):
        x = x + self.attn(self.norm1(x), position_ids=position_ids)
        x = x + self.mlp(self.norm2(x))
        return x
