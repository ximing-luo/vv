import os
import torch
from datetime import datetime
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

class CustomTensorBoardCallback(TrainerCallback):
    """
    自定义 TensorBoard 回调，记录权重分布、梯度分布。
    """
    def __init__(self, log_dir=None):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        self.histogram_freq = 500

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.log_dir is None:
            # 自动生成带时间戳的子目录，避免多次运行的数据混淆
            run_name = getattr(args, "run_name", None)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if run_name:
                sub_dir = f"{run_name}_{timestamp}"
            else:
                sub_dir = timestamp
            
            self.log_dir = os.path.join(args.logging_dir, sub_dir)

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"[System] Custom TensorBoard logging to {self.log_dir}")
        
        # 记录模型总参数量到 TensorBoard 文本
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_text = (
                 f"Total Parameters: {total_params:,} ({total_params / 1e8:.4f} 亿)  \n"
                 f"Trainable Parameters: {trainable_params:,} ({trainable_params / 1e8:.4f} 亿)"
             )
            self.writer.add_text("Model/Parameters", param_text, 0)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is not None and "loss" in logs and args.gradient_accumulation_steps > 1:
            # 修正梯度累积导致的训练 loss 显示异常
            # 由于自定义模型非 PreTrainedModel，Trainer 记录的是累加值，此处除以步数以校准
            logs["loss"] = logs["loss"] / args.gradient_accumulation_steps

        # 记录显存占用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            
            # 打印到控制台
            if state.global_step % (args.logging_steps * 10)== 0:
                print(f"[System] Step {state.global_step} - Memory | Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

        if self.writer is None:
            return
            
        step = state.global_step

        # 0. 记录基础指标 (Loss, LR, etc.)
        if logs:
            for k, v in logs.items():
                # 过滤掉 eval 相关的日志，不记录 eval 曲线
                if k.startswith("eval_"):
                    continue

                if isinstance(v, (int, float)):
                    tag = f"Train/{k}" if k not in ["epoch"] else k
                    self.writer.add_scalar(tag, v, step)

        if model is None:
            return
        
        # 1. 记录权重和梯度分布 (低频)
        if step % self.histogram_freq == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'Weights/{name}', param, step)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, step)
        


    def on_train_end(self, args, state, control, **kwargs):
        if self.writer:
            self.writer.close()
            self.writer = None
