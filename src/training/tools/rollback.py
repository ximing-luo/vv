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
        # 使用 state 中记录的当前最优 metric (通常是 eval_loss)
        best_loss = state.best_metric if state.best_metric is not None else float('inf')
        
        # 检查是否出现 NaN 或 Loss 飙升
        is_nan = torch.isnan(torch.tensor(current_loss))
        is_spike = best_loss != float('inf') and current_loss > best_loss * self.rollback_threshold
        
        if is_nan or is_spike:
            reason = "NaN" if is_nan else f"Loss 飙升 (当前 {current_loss:.2f} > 最优 {best_loss:.2f} * {self.rollback_threshold})"
            print(f"\n[警报] 检测到训练异常: {reason}")
            
            # 获取最优检查点路径
            best_ckpt_path = state.best_model_checkpoint
            
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                print(f"[Rollback] 正在从硬盘加载最优存档: {best_ckpt_path} ...")
                model = kwargs.get("model")
                optimizer = kwargs.get("optimizer")
                
                try:
                    # 1. 加载模型权重 (支持 bin 和 safetensors 两种格式)
                    weights_bin = os.path.join(best_ckpt_path, "pytorch_model.bin")
                    weights_safe = os.path.join(best_ckpt_path, "model.safetensors")
                    
                    if os.path.exists(weights_bin):
                        model.load_state_dict(torch.load(weights_bin, map_location=model.device, weights_only=True))
                    elif os.path.exists(weights_safe):
                        from safetensors.torch import load_file
                        model.load_state_dict(load_file(weights_safe, device=str(model.device)))
                    
                    # 2. 加载优化器状态
                    opt_path = os.path.join(best_ckpt_path, "optimizer.pt")
                    if os.path.exists(opt_path):
                        optimizer.load_state_dict(torch.load(opt_path, map_location=model.device))
                    
                    print(f"[Rollback] 硬盘回档成功！已恢复至最优状态，继续训练。")
                except Exception as e:
                    print(f"[Rollback] 加载存档失败: {e}")
            else:
                print(f"[Rollback] 抱歉，硬盘上尚未生成任何有效最优检查点 (Best: {best_loss})，无法执行回退。")