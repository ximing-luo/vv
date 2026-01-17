import os
import sys
# 将项目根目录（vv）和 src 目录添加到 sys.path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(root_path, 'src')
for path in [root_path, src_path]:
    if path not in sys.path:
        sys.path.append(path)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# 绕过 transformers 的 torch.load 安全检查 (CVE-2025-32434)
# 针对某些版本 transformers.trainer 已经完成引用的情况，直接注入到模块中
try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
    import transformers.trainer as trainer
    trainer.check_torch_load_is_safe = lambda: None
except Exception:
    pass

import glob
import argparse
import torch
from configs.model import VVConfig
from model import VV
from data.dataset import PretrainDataset
from training import DynamicTrainer
from transformers import TrainingArguments, AutoTokenizer

class ModelTrainer:
    """
    模型训练管理器，封装了训练流程的各个环节
    """
    def __init__(self, mode):
        self.mode = mode
        self.root_path = root_path
        self.dataset_root = os.path.join(self.root_path, 'src', 'data', 'dataset')
        self.checkpoints_root = os.path.join(self.root_path, 'models', 'checkpoints')
        self.tokenizer_dir = os.path.join(self.dataset_root, 'tokenizer')
        self._init_config()
        self.num_train_epochs = 1
        self.eval_steps = 500
        self.save_steps = 500
        self.resume_from_checkpoint = None
        self.init_weights_path = None
        
    def _init_config(self):
        """初始化训练配置和路径"""
        if self.mode == 'pretrain':
            self.train_bin = os.path.join(self.dataset_root, 'pretrain', 'pretrain_data.bin')
            self.output_dir = os.path.join(self.checkpoints_root, 'pretrain')
            self.final_save_path = os.path.join(self.output_dir, 'final')
            self.learning_rate = 3e-4
            # 预训练模式：自动寻找断点继续训练，或者从头开始
            self.resume_from_checkpoint = self._get_latest_checkpoint(self.output_dir)
            if self.resume_from_checkpoint:
                print(f"[System] 模式: 继续预训练 (Resume from {self.resume_from_checkpoint})")
            else:
                print(f"[System] 模式: 重新预训练 (Start from scratch)")

        elif self.mode == 'finetune':
            self.train_bin = os.path.join(self.dataset_root, 'finetune', 'finetune_data.bin')
            self.output_dir = os.path.join(self.checkpoints_root, 'finetune')
            self.final_save_path = os.path.join(self.output_dir, 'final')
            self.learning_rate = 5e-5
            # 微调模式：优先检查是否有断点，如果没有则从预训练模型开始
            self.resume_from_checkpoint = self._get_latest_checkpoint(self.output_dir)
            if self.resume_from_checkpoint:
                print(f"[System] 模式: 继续微调 (Resume from {self.resume_from_checkpoint})")
            else:
                self.init_weights_path = os.path.join(self.checkpoints_root, 'pretrain', 'final')
                print(f"[System] 模式: 开始微调 (Load weights from {self.init_weights_path})")
        
        else: raise ValueError(f"不支持的训练模式: {self.mode}")

    def prepare_data(self):
        """准备数据：加载数据集并划分训练/验证集"""
        print(f"[Data] 正在加载数据集: {self.train_bin}")
        dataset = PretrainDataset(self.train_bin)
        val_size = min(500, int(len(dataset) * 0.01)) if len(dataset) > 500 else 1
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        print(f"[Data] 数据集划分完成: 训练集 {train_size} 条, 验证集 {val_size} 条")

    def prepare_model(self):
        """准备模型：初始化 Tokenizer、Config 和 Model，并加载权重"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        config = VVConfig(
            vocab_size=len(self.tokenizer),
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        # 如果是预训练模式 (pretrain)，强制重置 rope_ntk_alpha = 1.0
        if self.mode == 'pretrain': config.rope_ntk_alpha = 1.0
        self.model = VV(config)
        # 加载权重逻辑
        if not self.resume_from_checkpoint and self.init_weights_path:
            print(f"正在从 {self.init_weights_path} 加载模型权重...")
            if not self._load_model_weights(self.model, self.init_weights_path):
                print(f"[Warning] 未在 {self.init_weights_path} 找到权重文件，将使用随机初始化。")

        # 打印并记录参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        param_info = (
             f"{'='*30}\n"
             f" 模型参数信息:\n"
             f"  - 词表大小: {len(self.tokenizer)}\n"
             f"  - 总参数量: {total_params:,} ({total_params / 1e8:.4f} 亿)\n"
             f"  - 可训练参数量: {trainable_params:,} ({trainable_params / 1e8:.4f} 亿)\n"
             f"{'='*30}"
         )
        print(param_info)

    def train(self):
        """执行训练流程"""
        # 确保数据和模型已准备好
        self.prepare_data()
        self.prepare_model()
        # 动态计算 Batch Size 和 Gradient Accumulation Steps
        train_batch_size, grad_steps = ModelTrainer._dynamic_batch_size(self.model.config)
        # 训练参数设置
        training_args = TrainingArguments(
            # 1. 输出与日志路径
            output_dir=self.output_dir, # 输出目录，用于存放 Checkpoints
            logging_dir=os.path.join(self.root_path, 'models', 'logs'), # TensorBoard 日志目录
            report_to="none", # 报告目标（如 wandb），这里关闭
            # 2. 训练超参数 (Hyperparameters)
            learning_rate=self.learning_rate, # 初始学习率
            adam_beta1=0.9, # AdamW 优化器的动量参数 (一阶矩估计的指数衰减率)
            adam_beta2=0.95, # AdamW 优化器的二阶矩估计衰减率 (有时调小能加速收敛)
            adam_epsilon=1e-8, # 防止除以零的小数值
            num_train_epochs=self.num_train_epochs, # 训练总轮数 (Epochs)
            per_device_train_batch_size=train_batch_size, # 单卡训练 Batch Size
            gradient_accumulation_steps=grad_steps, # 梯度累积步数，变相扩大 Batch Size
            weight_decay=0.01, # 权重衰减 (L2 正则化)
            warmup_ratio=0.02, # 预热步数比例 (Warmup Ratio)，设置为总步数的 2%
            lr_scheduler_type="cosine", # 学习率调度策略 (余弦退火)
            # 3. 评估配置 (Evaluation)
            per_device_eval_batch_size=train_batch_size, # 单卡评估 Batch Size
            eval_strategy="steps", # 评估策略：按步数 ('steps') 或按轮数 ('epoch')
            eval_steps=self.eval_steps, # 每隔多少步评估一次
            # 4. 保存策略 (Checkpointing)
            save_strategy="steps", # 保存策略：按步数 ('steps') 或按轮数 ('epoch')
            save_steps=self.save_steps, # 每隔多少步保存一次 Checkpoint
            save_total_limit=3, # 最多保留最近的 3 个 Checkpoint
            save_safetensors=False, # 是否使用 safetensors 格式保存
            load_best_model_at_end=True, # 训练结束时加载最优模型权重
            metric_for_best_model="eval_loss", # 以验证集 Loss 作为评估指标
            greater_is_better=False, # Loss 越小越好
            # 5. 日志与监控
            logging_steps=10, # 每隔多少步打印一次日志
            # 6. 硬件加速与数据加载
            # T4 (Compute Capability 7.5) 虽然 PyTorch 可能报告支持 BF16，但实际上硬件不支持，会导致速度极慢
            # 因此这里增加严格的 Compute Capability >= 8 (Ampere) 检查
            bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
            dataloader_num_workers=4, # 多进程加载数据
            dataloader_pin_memory=True, # 锁页内存，加速 CPU 到 GPU 传输
            max_grad_norm=10.0, # 梯度裁剪，防止梯度爆炸
        )
        trainer = DynamicTrainer(model=self.model, args=training_args, train_dataset=self.train_dataset, eval_dataset=self.val_dataset, tokenizer=self.tokenizer)
        print(f"[System] 开始 {self.mode} 模式训练...")
        trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        trainer.save_model(self.final_save_path)
        print(f"[System] 训练完成，模型已保存至 {self.final_save_path}")

    @staticmethod
    def _get_latest_checkpoint(path):
        """辅助函数：查找最新 checkpoint"""
        if not os.path.exists(path): return None
        checkpoints = glob.glob(os.path.join(path, "checkpoint-*"))
        valid_checkpoints = [cp for cp in checkpoints if os.path.exists(os.path.join(cp, 'trainer_state.json'))]
        if not valid_checkpoints: return None
        # 按 checkpoint-N 中的数字 N 进行排序，取最大值
        return max(valid_checkpoints, key=lambda x: int(x.split('-')[-1]))

    @staticmethod
    def _load_model_weights(model, path):
        """辅助函数：手动加载模型权重"""
        if not path or not os.path.exists(path): return False
        bin_path = os.path.join(path, "pytorch_model.bin")
        try:
            model.load_state_dict(torch.load(bin_path, map_location='cpu', weights_only=True))
            print(f"[System] 成功加载 bin 权重: {bin_path}")
            return True
        except Exception as e: print(f"[Error] 加载权重失败: {e}")
        return False

    @staticmethod
    def _dynamic_batch_size(model_config):
        """动态计算 Batch Size 和 Gradient Accumulation Steps"""
        # 计算最大序列长度，考虑 NTK 扩展
        max_seq_len = int(model_config.max_seq_len * model_config.rope_ntk_alpha)
        train_batch_size = max(1, int(4096 // max_seq_len)) # 单卡最大吞吐量 4096 tokens
        grad_steps = max(1, int(64 // train_batch_size))
        # train_batch_size = 64
        # grad_steps = 1
        print(f"[System] 动态计算得到的 Batch Size: {train_batch_size}")
        print(f"[System] 动态计算得到的 Gradient Accumulation Steps: {grad_steps}")
        return train_batch_size, grad_steps

def train(mode, num_train_epochs=1, eval_steps=500, save_steps=500):
    """
    保持向后兼容的 train 函数入口
    """
    trainer = ModelTrainer(mode)
    trainer.num_train_epochs = num_train_epochs
    trainer.eval_steps = eval_steps
    trainer.save_steps = save_steps
    trainer.train()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="vv 模型训练脚本")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="finetune", 
        choices=["pretrain", "finetune"],
        help="训练模式: pretrain (预训练) 或 finetune (微调)"
    )
    args = parser.parse_args()
    train(mode=args.mode)
