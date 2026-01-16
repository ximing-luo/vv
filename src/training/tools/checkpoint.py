import os
import json
import torch
from transformers import TrainerCallback

class CheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_dir):
            return

        model = kwargs.get('model')
        
        # 仅在模型存在时导出配置，无需再处理 tokenizer（Trainer 会自动处理）
        if model:
            print(f"[CheckpointCallback] 检测到模型保存，正在导出元数据至 {checkpoint_dir} ...")
            ModelExporter.export_metadata(model, checkpoint_dir)

class ModelExporter:
    @staticmethod
    def export_metadata(model, output_dir):
        """导出模型配置信息"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        ModelExporter._save_config(model, output_dir)

    @staticmethod
    def _save_config(model, output_dir):
        # 尝试从模型中获取配置对象
        config_obj = getattr(model, "config", getattr(model, "args", None))
        if config_obj:
            config_path = os.path.join(output_dir, "model_config.json")
            try:
                # 转换配置为字典
                if hasattr(config_obj, "to_dict"):
                    config_dict = config_obj.to_dict()
                elif hasattr(config_obj, "__dict__"):
                    config_dict = {k: v for k, v in config_obj.__dict__.items() if not k.startswith('_')}
                else:
                    config_dict = str(config_obj)
                
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=4, ensure_ascii=False)
                print(f"[Checkpoint] 模型配置已保存至: {config_path}")
            except Exception as e:
                print(f"[Checkpoint] 保存模型配置失败: {e}")
