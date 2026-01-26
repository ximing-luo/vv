import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.transform import StandardBlock, DeepSeekV3Block
from .backbone.rms import RMSNorm

class BaseModel(nn.Module):
    """
    基础语言模型架构 (Evolutionary Base)
    定义通用的 Causal Transformer 流程: Embedding -> Blocks -> Norm -> Head
    """
    def __init__(self, config, block_cls=StandardBlock):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.hidden_dim)
        # 演进式架构：支持注入不同等级的 Block
        self.blocks = nn.Sequential(*[block_cls(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight # Weight Tying
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def device(self): 
        return next(self.parameters()).device

    def forward(self, input_ids, labels=None, position_ids=None, **kwargs):
        B, T = input_ids.size()
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding_table(input_ids)
        for block in self.blocks:
            x = block(x, position_ids=position_ids)
        x = self.norm(x)
        logits = self.lm_head(x) # (B, T, V)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return loss, logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, **kwargs):
        """标准生成循环"""
        for _ in range(int(max_new_tokens)):
            # 裁剪上下文以适应最大长度 (考虑 RoPE/YaRN 扩展)
            # 实际最大长度 = 原始长度 * 缩放系数
            scale = getattr(self.config, 'rope_scale', 1.0)
            max_ctx = int(self.config.max_seq_len * scale)
            idx_cond = idx[:, -max_ctx:] if idx.size(1) > max_ctx else idx
            
            _, logits = self(idx_cond, **kwargs)
            logits = logits[:, -1, :] # (B, V)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None, **kwargs):
        """流式生成器"""
        for _ in range(int(max_new_tokens)):
            scale = getattr(self.config, 'rope_scale', 1.0)
            max_ctx = int(self.config.max_seq_len * scale)
            idx_cond = idx[:, -max_ctx:] if idx.size(1) > max_ctx else idx
            
            _, logits = self(idx_cond, **kwargs)
            logits = logits[:, -1, :] / temperature
            
            # 数值稳定性处理
            if torch.isnan(logits).any():
                logits = torch.where(torch.isnan(logits), torch.tensor(-float('Inf'), device=logits.device), logits)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            
            # 概率分布校验
            if torch.isnan(probs).any() or (probs.sum(dim=-1) <= 0).any():
                probs = torch.ones_like(probs) / probs.size(-1) # Fallback uniform

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next

class VV(BaseModel):
    """
    VV (DeepSeekV3-based) 实现
    继承自 BaseModel，指定使用 DeepSeekV3Block 构建深度模型
    """
    def __init__(self, config):
        super().__init__(config, block_cls=DeepSeekV3Block)
        self.apply(self._init_weights)
