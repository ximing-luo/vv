import os
import torch
from datetime import datetime
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

class CustomTensorBoardCallback(TrainerCallback):
    """
    自定义 TensorBoard 回调：记录 Loss、显存、权重/梯度分布。
    """
    def __init__(self, log_dir=None):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        self.histogram_freq = 500  # 默认值，会在 on_train_begin 中根据 args.eval_steps 更新

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # 1. 自动配置日志目录
        if self.log_dir is None:
            run_name = getattr(args, "run_name", None)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sub_dir = f"{run_name}_{timestamp}" if run_name else timestamp
            self.log_dir = os.path.join(args.logging_dir, sub_dir)

        # 2. 初始化 SummaryWriter
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"[System] TensorBoard 日志目录: {self.log_dir}")

        # 3. 同步记录频率与模型评估步数
        if hasattr(args, "eval_steps") and args.eval_steps and args.eval_steps > 0:
            self.histogram_freq = args.eval_steps

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if self.writer is None or logs is None:
            return
        step = state.global_step
        # 1. 显存监控 (每 10 次日志打印一次)
        if torch.cuda.is_available() and step % (args.logging_steps * 10) == 0:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"[System] Step {step} - Memory | Alloc: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        # 2. 修正梯度累积下的 Loss 显示 (自定义模型非 PreTrainedModel 时 Trainer 可能记录累加值)
        if "loss" in logs and args.gradient_accumulation_steps > 1:
            logs["loss"] = logs["loss"] / args.gradient_accumulation_steps
        # 3. 记录标量 (Loss, LR 等)
        ignore_keys = {"train_runtime", "train_samples_per_second", "train_steps_per_second", "total_flos", "train_loss"}
        for k, v in logs.items():
            if k in ignore_keys:continue
            if not k.startswith("eval_") and isinstance(v, (int, float)):
                tag = f"Train/{k}" if k != "epoch" else k
                self.writer.add_scalar(tag, v, step)
        # 4. 记录权重和梯度分布 (频率与评估步数一致)
        if model and step % self.histogram_freq == 0:
            self._log_histograms(model, step)

    def _log_histograms(self, model, step):
        """记录模型参数和梯度的直方图"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Weights/{name}', param, step)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer:
            self.writer.close()
            self.writer = None
