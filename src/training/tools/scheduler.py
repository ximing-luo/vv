from transformers import TrainerCallback
import math

class BatchSizeSchedulerCallback(TrainerCallback):
    """
    动态 Batch Size 调度器 (Batch Size Scheduling)
    
    原理:
    在训练初期使用较小的 Batch Size (通过较小的梯度累积步数实现)，
    利用较大的梯度噪声帮助模型跳出局部最优解，增加探索能力。
    随着训练进行，逐渐增加梯度累积步数，降低噪声，使收敛更扎实。
    
    这在数学上类似于学习率衰减 (LR Decay)，但能更好地利用硬件算力。
    """
    def __init__(self, initial_grad_steps, max_grad_steps=None, strategy="linear"):
        super().__init__()
        self.initial_grad_steps = initial_grad_steps
        self.max_grad_steps = max_grad_steps or (initial_grad_steps * 4)
        self.strategy = strategy # "linear", "exp", or "milestones"
        self.milestones = [0.3, 0.6, 0.9] # 进度百分比

    def on_step_end(self, args, state, control, **kwargs):
        # 1. 监测 Epoch 进度，确保数据训练完时自动触发保存 (解决进度条不准导致不保存的问题)
        # state.epoch 是一个浮点数，代表当前消耗的数据量占总数据集的比例
        # 例如 num_train_epochs=1.0，当 epoch >= 0.999 时代表数据基本喂完
        target_epoch = args.num_train_epochs
        if state.epoch >= (target_epoch - 0.001):
            print(f"\n[Scheduler] 检测到 Epoch {state.epoch:.4f} 已接近目标 {target_epoch}，触发安全保存并停止...")
            control.should_training_stop = True
            control.should_evaluate = True   # 停止时触发最后一次评估
            control.should_save = True       # 强制保存最后一个检查点
            return # 触发停止后不再执行后续逻辑

        # 2. 动态 Batch Size 调度逻辑
        # 计算当前训练进度 (0.0 到 1.0)
        progress = state.global_step / max(1, state.max_steps)
        
        old_grad_steps = args.gradient_accumulation_steps
        new_grad_steps = old_grad_steps

        if self.strategy == "milestones":
            # 阶梯式增长
            # 例如: 初始 8 -> 30% 进度变为 16 -> 60% 变为 32 -> 90% 变为 64
            count = sum(1 for m in self.milestones if progress >= m)
            new_grad_steps = self.initial_grad_steps * (2 ** count)
        
        elif self.strategy == "linear":
            # 线性增长
            new_grad_steps = int(self.initial_grad_steps + (self.max_grad_steps - self.initial_grad_steps) * progress)
            
        elif self.strategy == "exp":
            # 指数增长
            new_grad_steps = int(self.initial_grad_steps * math.exp(math.log(self.max_grad_steps / self.initial_grad_steps) * progress))

        # 确保不超过最大值且不小于初始值
        new_grad_steps = min(max(new_grad_steps, self.initial_grad_steps), self.max_grad_steps)
        
        if new_grad_steps != old_grad_steps:
            args.gradient_accumulation_steps = new_grad_steps
            # 注意：在训练过程中打印日志不要太频繁
            if state.global_step % 100 == 0 or progress in self.milestones:
                print(f"\n[Scheduler] 进度 {progress:.1%}: 梯度累积步数调整 {old_grad_steps} -> {new_grad_steps}")
