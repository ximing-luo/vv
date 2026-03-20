import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class FeedForward(nn.Module):
    # 标准 FFN / MLP
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class GatedMLP(nn.Module):
    """
    Gated MLP (SwiGLU 变体) - 增强表达能力
    结构: Down(SiLU(Gate(x)) * Up(x))
    """
    def __init__(self, config):
        super().__init__()
        intermediate_size = config.intermediate_size
        if intermediate_size is None:
            intermediate_size = int(config.hidden_dim * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.dropout = nn.Dropout(config.dropout)
        self.up_proj = nn.Linear(config.hidden_dim, intermediate_size, bias=config.bias)
        self.down_c_proj = nn.Linear(intermediate_size, config.hidden_dim, bias=config.bias)
        self.gate = nn.Linear(config.hidden_dim, intermediate_size, bias=config.bias)
        self.act_func = F.silu

    def forward(self, x):
        return self.dropout(self.down_c_proj(self.act_func(self.gate(x)) * self.up_proj(x)))

class SparseMoE(nn.Module):
    """
    稀疏混合专家 (Sparse Mixture of Experts)
    结构: Router -> Top-K Experts Selection -> Weighted Sum
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_dim = config.hidden_dim
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([GatedMLP(config) for _ in range(self.num_experts)])

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, self.hidden_dim)
        
        # Router: 计算门控得分与 Top-K
        router_logits = self.gate(x)
        weights = F.softmax(router_logits, dim=-1)
        weights, indices = torch.topk(weights, self.num_experts_per_tok, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6) # 归一化权重

        # Dispatch & Combine: 遍历专家计算
        # 优化点: 尽管循环在 Python 层，但在专家数较少(<64)时，相比复杂的稀疏算子实现，这种方式更易读且易于调试
        final_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # 找出分配给当前专家 i 的 token 索引
            # indices: (Total_Tokens, K) -> mask: (Total_Tokens, K)
            mask = (indices == i)
            if mask.any():
                token_idx, topk_idx = torch.where(mask)
                expert_out = expert(x[token_idx])
                # 加权累加: output[token] += weight * expert_output
                final_output[token_idx] += weights[token_idx, topk_idx].unsqueeze(-1) * expert_out
        
        return final_output.view(*orig_shape)

class HybridMoE(SparseMoE):
    """
    混合专家架构 (Hybrid/Shared MoE)
    引入 Shared Experts (常驻专家) 捕获通用知识，Routed Experts 专注长尾知识
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_shared_experts = config.num_shared_experts
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([GatedMLP(config) for _ in range(self.num_shared_experts)])
        else:
            self.shared_experts = nn.ModuleList([])

    def _compute_shared(self, x_flat):
        # 辅助函数：计算常驻专家输出
        if self.num_shared_experts == 0:
            return 0.0
        shared_out = 0.0
        for expert in self.shared_experts:
            shared_out += expert(x_flat)
        return shared_out

    def forward(self, x):
        # 复用父类路由逻辑计算 Routed 部分
        routed_output = super().forward(x)
        
        # 计算 Shared 部分并叠加
        if self.num_shared_experts > 0:
            shared_output = self._compute_shared(x.view(-1, self.hidden_dim))
            return routed_output + shared_output.view_as(routed_output)
        return routed_output    

    def efficient_expert_computation(self, x_flat, weights, indices):
        """
        优化后的专家计算: Sort + Split + Concat (模拟 Grouped GEMM 的数据准备)
        复杂度: O(N * log N) 排序 vs O(E * N) 掩码
        当 E (专家数) 较大时，此方法显著更快
        """
        batch_size, dim = x_flat.shape
        num_tokens = batch_size
        topk = self.num_experts_per_tok
        
        # 1. 展平索引与权重 (N, K) -> (N*K)
        flat_indices = indices.view(-1)     # 专家ID
        flat_weights = weights.view(-1)     # 权重
        
        # 2. 生成源 Token 索引 (0,0, 1,1, ..., N-1,N-1)
        # (N, K) -> (N*K)
        src_indices = torch.arange(num_tokens, device=x_flat.device).repeat_interleave(topk)
        
        # 3. 排序 (Sort)
        # 根据专家 ID 对任务进行排序，将相同专家的任务聚在一起
        sorted_expert_ids, argsort_idx = torch.sort(flat_indices)
        
        # 对输入和权重进行相应的重排
        permuted_src_indices = src_indices[argsort_idx]
        permuted_weights = flat_weights[argsort_idx]
        
        # 提取输入: (N*K, D)
        # 这一步会复制数据，但避免了 mask 的大量无效计算
        permuted_x = x_flat[permuted_src_indices]
        
        # 4. 分组 (Split)
        # 统计每个专家的任务数量
        tokens_per_expert = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        
        # 5. 专家计算 (Compute)
        # 这里依然是 Python 循环，但只循环“活跃”专家，且无 Mask 生成开销
        results = torch.zeros_like(permuted_x)
        
        # 注意：split 需要 list，这会导致 CPU sync。在大规模训练中通常可接受
        splits = permuted_x.split(tokens_per_expert.tolist())
        
        output_chunks = []
        for i, chunk in enumerate(splits):
            if chunk.numel() > 0:
                output_chunks.append(self.experts[i](chunk))
            else:
                output_chunks.append(chunk) # empty tensor
        
        results = torch.cat(output_chunks, dim=0)
        
        # 6. 还原 (Scatter / Index Add)
        # output[src_idx] += result * weight
        final_output = torch.zeros_like(x_flat)
        
        # 加权
        results = results * permuted_weights.unsqueeze(-1)
        
        # index_add_ 处理重叠索引的累加
        final_output.index_add_(0, permuted_src_indices, results)
        
        return final_output


class SoftBalancedMoE(HybridMoE):
    """
    软负载均衡 MoE (DeepSeek-V2 Style)
    引入 Auxiliary Loss 防止路由坍缩 (Routing Collapse)
    """
    def __init__(self, config):
        super().__init__(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef

    def forward(self, x):
        # 为了返回 aux_loss，我们需要重写 forward 流程，无法简单复用 super().forward
        # 但我们可以复用 _compute_shared
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_dim)
        
        # 1. Shared Experts
        shared_output = self._compute_shared(x_flat)
        
        # 2. Routed Experts with Aux Loss
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Top-K
        weights, indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Aux Loss Calculation
        aux_loss = 0.0
        if self.training:
            # P: 每个专家的平均概率 (B*T, N) -> (N,)
            P = routing_weights.mean(dim=0)
            # f: 每个专家的选中频率 (B*T, N) -> (N,)
            mask = torch.zeros_like(routing_weights).scatter_(1, indices, 1.0)
            f = mask.mean(dim=0)
            aux_loss = self.router_aux_loss_coef * self.num_experts * torch.sum(P * f)

        # Expert Computation (Optimized)
        routed_output = self.efficient_expert_computation(x_flat, weights, indices)
        
        final_output = (shared_output + routed_output).view(*orig_shape)
        return final_output, aux_loss

class SelfAdaptiveMoE(HybridMoE):
    """
    自适应负载均衡 MoE (DeepSeek-V3 Style)
    弃用显式 Loss，改用动态 Bias 自适应调整负载，实现无损均衡
    """
    def __init__(self, config):
        super().__init__(config)
        # 注册不可训练的 buffer 存储偏置
        self.register_buffer('bias', torch.zeros(self.num_experts))
        self.bias_update_rate = config.bias_update_rate
        # [优化] 增加计数器，减少分布式同步频率
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.sync_interval = 10 # 每 10 步同步一次，显著降低通信开销

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_dim)
        
        # 1. Shared Experts
        shared_output = self._compute_shared(x_flat)
        
        # 2. Routed Experts with Dynamic Bias
        logits = self.gate(x_flat)
        # 核心机制: Logits + Bias (Bias 动态反映专家忙碌程度，越忙 Bias 越小)
        scores = logits + self.bias
        
        routing_weights = F.softmax(scores, dim=-1)
        weights, indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Expert Computation (Optimized)
        routed_output = self.efficient_expert_computation(x_flat, weights, indices)
        
        # 3. Update Bias (Training Only)
        if self.training:
            self.update_biases(indices)
            
        final_output = (shared_output + routed_output).view(*orig_shape)
        return final_output

    @torch.no_grad()
    def update_biases(self, indices):
        """
        [优化] 动态偏置更新逻辑，支持分布式环境
        采用分布式延迟同步策略 (Deferred Synchronization)，减少 90% 的网络开销
        """
        # 1. 更新本地计数器
        self.step_count += 1
        
        # 2. 计算本地负载
        flat_indices = indices.view(-1)
        counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        
        # 3. 定期分布式同步 (仅在计数器到达阈值时执行 all_reduce)
        # 这种“最终一致性”策略在深度学习负载均衡中非常有效，且能极大提升 GPU 利用率
        if dist.is_initialized() and self.step_count % self.sync_interval == 0:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            total_tokens = indices.numel() * dist.get_world_size() * self.sync_interval
        else:
            total_tokens = indices.numel()

        # 4. 计算全局/本地负载比例并更新偏置
        current_load = counts / total_tokens
        target_load = self.num_experts_per_tok / self.num_experts
        
        # 4. 负反馈调节: 负载过高 -> error > 0 -> bias 减小 -> 选中概率降低
        load_diff = current_load - target_load
        self.bias -= self.bias_update_rate * load_diff
