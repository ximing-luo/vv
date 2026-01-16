import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.transform import DeepseekBlock
from .backbone.rms import RMSNorm

class VV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.Sequential(
            *[DeepseekBlock(config) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 这里使用的是正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, **kwargs):
        # input_ids 是输入的 token ids
        batch, seq_len = input_ids.size()

        token_emb = self.token_embedding_table(input_ids)
        x = self.blocks(token_emb)
        x = self.norm(x)
        logits = self.lm_head(x)   # shape is (batch, seq_len, vocab_size)
        
        loss = None
        if labels is not None:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = labels.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets, ignore_index=-100)

        return (loss, logits)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, **kwargs):
        # idx is (B, T) array of indices in the current context
        for _ in range(int(max_new_tokens)):
            # 如果序列太长，只取最后 max_seq_len 个token
            # 允许通过 rope_ntk_alpha 扩展推理时的上下文长度
            max_context_len = int(self.config.max_seq_len * self.config.rope_ntk_alpha)
            idx_cond = idx if idx.size(1) <= max_context_len else idx[:, -max_context_len:]
            _, logits = self(idx_cond, **kwargs)
            # 只关注最后一个时间步的预测
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 附加到序列上
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.3, top_k=75, **kwargs):
        """
        流式生成 Token。
        :param idx: 输入的 Token IDs, 形状为 (B, T)
        :param max_new_tokens: 最大生成数量
        :param temperature: 温度参数，越高越随机
        :param top_k: Top-k 采样
        :yield: 每一个生成的 Token ID (1, 1)
        """
        for _ in range(int(max_new_tokens)):
            # 裁剪上下文，确保不超过 max_seq_len# 允许通过 rope_ntk_alpha 扩展推理时的上下文长度
            max_context_len = int(self.config.max_seq_len * self.config.rope_ntk_alpha)
            idx_cond = idx if idx.size(1) <= max_context_len else idx[:, -max_context_len:]
            _, logits = self(idx_cond, **kwargs)
            # 取最后一个时间步并应用温度
            logits = logits[:, -1, :] / temperature
            
            # --- 增加数值稳定性保护 ---
            # 1. 替换 NaN 为 -Inf
            if torch.isnan(logits).any():
                logits = torch.where(torch.isnan(logits), torch.tensor(-float('Inf'), device=logits.device), logits)
            # Top-k 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 采样
            probs = F.softmax(logits, dim=-1)
            # 2. 检查概率分布有效性 (防止 multinomial 报错 device-side assert triggered)
            # 如果概率和为0 (例如所有 logits 都是 -Inf)，或者包含 NaN
            if torch.isnan(probs).any() or (probs.sum(dim=-1) <= 0).any():
                # 回退到均匀分布，避免崩溃
                probs = torch.ones_like(probs) / probs.size(-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            # 更新序列并 yield
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next






