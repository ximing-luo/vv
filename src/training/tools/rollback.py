import os
import torch
from transformers import TrainerCallback

class RollbackCallback(TrainerCallback):
    """
    轻量化自动回退回调 (RollbackCallback)
    原理：
    1. 不再在内存中备份权重，节省系统内存。
    2. 监控 Loss。如果发现 NaN 或异常飙升，直接从 HF 自动保存的最优检查点 (best_model_checkpoint) 加载回档。
    """
    def __init__(self, rollback_threshold=2.5):
        self.rollback_threshold = rollback_threshold

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
            
        current_loss = logs["loss"]
        grad_norm = logs.get("grad_norm")
        
        # 使用 state 中记录的当前最优 metric (通常是 eval_loss)
        best_loss = state.best_metric if state.best_metric is not None else float('inf')
        
        # 检查是否出现异常：NaN, Grad NaN, Loss 0 (坍塌) 或 Loss 飙升
        is_nan = torch.isnan(torch.tensor(current_loss))
        is_grad_nan = grad_norm is not None and torch.isnan(torch.tensor(grad_norm))
        is_collapse = current_loss <= 1e-6 # 对于 LLM 训练，Loss 接近 0 通常意味着训练坍塌
        is_spike = best_loss != float('inf') and current_loss > best_loss * self.rollback_threshold
        
        if is_nan or is_grad_nan or is_collapse or is_spike:
            reason = ""
            if is_nan: reason = "Loss 为 NaN"
            elif is_grad_nan: reason = "梯度 (grad_norm) 为 NaN"
            elif is_collapse: reason = f"训练坍塌 (Loss={current_loss:.6f})"
            elif is_spike: reason = f"Loss 飙升 (当前 {current_loss:.2f} > 最优 {best_loss:.2f} * {self.rollback_threshold})"
            
            print(f"\n[警报] 检测到训练异常: {reason}")
            
            # 获取最优检查点路径
            best_ckpt_path = state.best_model_checkpoint
            
            # 备选方案：如果 state.best_model_checkpoint 为空，尝试寻找最新的 checkpoint
            if not best_ckpt_path:
                output_dir = args.output_dir
                if os.path.exists(output_dir):
                    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
                    if checkpoints:
                        # 按修改时间排序，取最新的一个（虽然不是最优，但总比坍塌强）
                        checkpoints.sort(key=os.path.getmtime)
                        best_ckpt_path = checkpoints[-1]
                        print(f"[Rollback] 未找到 best_model_checkpoint，尝试回退到最新检查点: {best_ckpt_path}")

            if best_ckpt_path and os.path.exists(best_ckpt_path):
                print(f"[Rollback] 正在从硬盘加载最优存档: {best_ckpt_path} ...")
                model = kwargs.get("model")
                optimizer = kwargs.get("optimizer")
                
                try:
                    # 1. 加载模型权重 (支持 bin 和 safetensors 两种格式)
                    weights_bin = os.path.join(best_ckpt_path, "pytorch_model.bin")
                    weights_safe = os.path.join(best_ckpt_path, "model.safetensors")
                    
                    # 获取模型当前设备
                    device = getattr(model, 'device', next(model.parameters()).device)
                    
                    if os.path.exists(weights_bin):
                        model.load_state_dict(torch.load(weights_bin, map_location=device, weights_only=True), strict=False)
                    elif os.path.exists(weights_safe):
                        from safetensors.torch import load_file
                        model.load_state_dict(load_file(weights_safe, device=str(device)), strict=False)
                    
                    # 2. 加载优化器状态
                    opt_path = os.path.join(best_ckpt_path, "optimizer.pt")
                    if os.path.exists(opt_path):
                        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
                    
                    print(f"[Rollback] 硬盘回档成功！已恢复至最优状态，继续训练。")
                except Exception as e:
                    print(f"[Rollback] 加载存档失败: {e}")
            else:
                print(f"[Rollback] 抱歉，硬盘上尚未生成任何有效最优检查点 (Best: {best_loss})，无法执行回退。")