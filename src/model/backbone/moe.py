import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    # 实际上就是 MLP
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
    Gated MLP (SwiGLU 变体)
    结构: Down(SiLU(Gate(x)) * Up(x))
    相比标准 FFN，引入了门控机制，表达能力更强
    """
    def __init__(self, config):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_dim * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.dropout = nn.Dropout(config.dropout)
        self.up_proj = nn.Linear(config.hidden_dim, intermediate_size, bias=config.bias)
        self.down_c_proj = nn.Linear(intermediate_size, config.hidden_dim, bias=config.bias)
        self.gate = nn.Linear(config.hidden_dim, intermediate_size, bias=config.bias)
        
        self.act_func = F.silu # 现代模型通常使用 SiLU

    def forward(self, x):
        # Gated MLP logic: (Act(Gate(x)) * Up(x)) -> Down
        gate_proj = self.gate(x)
        up_proj = self.up_proj(x)
        
        x = self.act_func(gate_proj) * up_proj 
        x = self.down_c_proj(x)
        return self.dropout(x)

class MoE(nn.Module):
    """
    Mixture of Experts (混合专家模型) - 纯路由版本
    只包含路由专家 (Routed Experts)
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok # Top-k
        self.hidden_dim = config.hidden_dim
        
        # 门控网络 (Router)
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        
        # 路由专家列表
        self.experts = nn.ModuleList([GatedMLP(config) for _ in range(self.num_experts)])

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, n_embed)
        """
        orig_shape = x.shape
        batch_size, seq_len, n_embed = x.shape
        
        # 展平以便处理: (batch_size * seq_len, n_embed)
        x_flat = x.view(-1, n_embed)
        
        # 计算门控得分 (logits)
        router_logits = self.gate(x_flat) # (Token, num_experts)
        
        # 获取 top-k 专家
        weights = F.softmax(router_logits, dim=-1)
        weights, indices = torch.topk(weights, self.num_experts_per_tok, dim=-1)
        
        # 归一化 top-k 权重
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        routed_output = torch.zeros_like(x_flat)
        # 遍历路由专家进行计算
        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            if mask.any():
                token_idx, topk_idx = torch.where(mask)
                expert_out = expert(x_flat[token_idx])
                routed_output[token_idx] += weights[token_idx, topk_idx].unsqueeze(-1) * expert_out
        
        return routed_output.view(*orig_shape)

class SharedMoE(MoE):
    """
    Shared Mixture of Experts (带常驻专家的混合专家模型)
    包含：路由专家 (继承自 MoE) + 常驻专家 (Shared Experts)
    总专家数 (逻辑上) = num_experts (路由) + num_shared_experts (常驻)
    """
    def __init__(self, config):
        super().__init__(config)
        # 常驻专家数量
        self.num_shared_experts = getattr(config, "num_shared_experts", 0)
        
        # 常驻专家列表 (Shared Experts)
        # 注意：这里直接创建 GatedMLP 列表
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([GatedMLP(config) for _ in range(self.num_shared_experts)])
        else:
            self.shared_experts = nn.ModuleList([])

    def forward(self, x):
        # 1. 计算路由专家的输出 (复用父类逻辑)
        routed_output = super().forward(x)
        
        # 如果没有常驻专家，直接返回路由输出
        if self.num_shared_experts == 0:
            return routed_output
            
        # 2. 计算常驻专家的输出
        # 需要展平输入
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_dim)
        
        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output += expert(x_flat)
            
        # 还原形状
        shared_output = shared_output.view(*orig_shape)
        
        # 3. 合并输出
        return routed_output + shared_output

class AuxiliaryLossMoE(SharedMoE):
    """
    DeepSeek MoE 架构 (参考 DeepSeek-V2/V3)
    核心改进:
    1. 细粒度专家 (Fine-Grained Experts): (逻辑上模拟，或者通过 config.num_experts 设置较大值)
    2. Shared Experts: 总是激活的常驻专家 (继承自 SharedMoE)
    3. 负载均衡 (Load Balancing): 计算 Auxiliary Loss 以防止路由坍缩
    """
    def __init__(self, config):
        super().__init__(config)
        # 负载均衡系数，通常在 0.01 ~ 0.1 之间
        self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.01)
        
    def forward(self, x):
        final_output, aux_loss = self.forward_with_loss(x)
        return final_output, aux_loss

    def forward_with_loss(self, x):
        """
        DeepSeekMoE Forward
        返回: (output, aux_loss)
        注意：如果用于传统的 Block，可能需要适配返回值
        """
        # 1. Shared Experts 计算
        orig_shape = x.shape
        batch_size, seq_len, n_embed = x.shape
        x_flat = x.view(-1, n_embed)
        
        shared_output = torch.zeros_like(x_flat)
        if self.num_shared_experts > 0:
            for expert in self.shared_experts:
                shared_output += expert(x_flat)
        
        # 2. Routed Experts 计算 (带负载均衡 Loss)
        # 计算 Router Logits
        router_logits = self.gate(x_flat) # (B*T, num_experts)
        
        # Softmax 得到概率 (用于计算 loss 和路由)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # --- 计算 Auxiliary Load Balancing Loss ---
        # loss = alpha * N * sum(f_i * P_i)
        # f_i: 专家 i 被选中的频率 (fraction of tokens dispatched to expert i)
        # P_i: 专家 i 的平均路由概率 (average routing probability)
        
        # Top-k selection
        weights, indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        # 归一化权重 (用于后续加权求和)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 计算 Loss
        if self.training:
            # P_i: 每个专家的平均概率
            P = routing_weights.mean(dim=0) # (num_experts,)
            
            # f_i: 每个专家的选中频率
            # 创建 mask: (B*T, num_experts)
            mask = torch.zeros_like(router_logits)
            mask.scatter_(1, indices, 1.0)
            f = mask.mean(dim=0) # (num_experts,)
            
            # Aux Loss
            aux_loss = self.router_aux_loss_coef * self.num_experts * torch.sum(P * f)
        else:
            aux_loss = 0.0
            
        # 3. 计算专家输出
        routed_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # 找到分配给专家 i 的 token
            # indices: (B*T, k)
            # 我们需要找到 indices 中等于 i 的位置
            # 使用 mask 可能会更慢，但更直观。
            # 优化写法：使用 index select
            
            # 这里的 mask 逻辑:
            # indices == i 返回 (B*T, k) 的 bool tensor
            mask = (indices == i)
            if mask.any():
                # 获取 token index 和 top-k index
                token_idx, topk_idx = torch.where(mask)
                
                # 计算专家输出
                expert_out = expert(x_flat[token_idx])
                
                # 加权累加
                # weights[token_idx, topk_idx] 对应于该 token 对该专家的权重
                routed_output[token_idx] += weights[token_idx, topk_idx].unsqueeze(-1) * expert_out
                
        # 4. 合并结果
        final_output = shared_output + routed_output
        final_output = final_output.view(*orig_shape)
        
        return final_output, aux_loss

class DeepseekMoE(SharedMoE):
    """
    DeepSeek-V3 MoE 架构的简化实现
    核心改进:
    1. Auxiliary-Loss-Free Load Balancing: 使用动态偏置 b_i 替代显式 Loss
    2. 依然保留 Shared Experts 结构
    """
    def __init__(self, config):
        super().__init__(config)
        # 注册一个 buffer 用于存储专家偏置 b_i
        # 它不参与梯度更新，而是通过 update_biases 手动调节
        self.register_buffer('bias', torch.zeros(self.num_experts))
        
        # 偏置调节的步长 (也称为控制增益)
        self.bias_update_rate = getattr(config, "bias_update_rate", 0.001)

    def forward(self, x):
        """
        DeepSeek-V3 的 Forward 逻辑
        """
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_dim)
        
        # 1. Shared Experts 计算 (总是全量参与)
        shared_output = torch.zeros_like(x_flat)
        if self.num_shared_experts > 0:
            for expert in self.shared_experts:
                shared_output += expert(x_flat)
        
        # 2. 路由专家计算 (无损失负载均衡版本)
        # 直接计算门控 logits
        logits = self.gate(x_flat) # (N, num_experts)
        
        # --- 核心：加入负载均衡偏置 b ---
        # DeepSeek-V3 公式: scores = gate(x) + bias
        # 这个 bias 会根据专家的“忙碌程度”动态调整
        scores = logits + self.bias
        
        # 计算路由概率
        routing_weights = F.softmax(scores, dim=-1)
        
        # Top-k 选取
        weights, indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        # 权重重新归一化
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 3. 专家计算 (矢量化实现)
        routed_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            if mask.any():
                token_idx, topk_idx = torch.where(mask)
                expert_out = expert(x_flat[token_idx])
                routed_output[token_idx] += weights[token_idx, topk_idx].unsqueeze(-1) * expert_out
        
        # 4. 汇总结果
        final_output = (shared_output + routed_output).view(*orig_shape)
        
        # --- 核心改进：自适应负载均衡 ---
        # 如果是在训练模式，直接在模块内部更新偏置，不需要把索引传出去
        if self.training:
            self.update_biases(indices)
            
        return final_output

    @torch.no_grad()
    def update_biases(self, indices):
        """
        DeepSeek-V3 的动态偏置更新逻辑
        indices: 当前 batch 中被选中的专家索引 (N, K)
        """
        # 1. 计算当前负载 (每个专家被选中的频率)
        # 展平 indices
        flat_indices = indices.view(-1)
        # 统计每个专家出现的次数
        counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        # 计算比例（所有token中选择某个专家的次数/总token数）
        current_load = counts / indices.size(0) # indices(N, K)
        
        # 2. 目标负载 (top-k/专家总数)
        target_load = self.num_experts_per_tok / self.num_experts
        
        # 3. 计算偏差并更新
        load_diff = current_load - target_load
        self.bias -= self.bias_update_rate * load_diff
