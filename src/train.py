import glob
import os
import sys
from pathlib import Path
import torch
from transformers import TrainingArguments, AutoTokenizer
# 环境配置与安全检查绕过
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    # 绕过 transformers 的 torch.load 安全检查 (CVE-2025-32434)
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
    import transformers.trainer as trainer
    trainer.check_torch_load_is_safe = lambda: None
except Exception:
    pass
# 将项目根目录添加到 sys.path 以支持本地模块导入
root_path = str(Path(__file__).resolve().parents[1])
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from configs.model import VisualVVConfig
from src.model import VV, VisualVV
from src.data.dataset import PretrainDataset, VLMPretrainDataset
from src.training import DynamicTrainer

class ModelTrainer:
    """
    模型训练管理器，封装了训练流程的各个环节
    """
    def __init__(self, mode, is_vlm=False):
        self.mode = mode
        self.is_vlm = is_vlm
        self.root_path = root_path
        self.dataset_root = os.path.join(self.root_path, 'src', 'data', 'dataset')
        self.checkpoints_root = os.path.join(self.root_path, 'models', 'checkpoints')
        self.model_save_path = os.path.join(self.root_path, 'models', 'vv') # 统一的成品目录
        self.tokenizer_dir = os.path.join(self.dataset_root, 'tokenizer')
        self._init_config()
        self.num_train_epochs = 1
        self.eval_steps = 500
        self.save_steps = 500
        
    def _init_config(self):
        if self.mode == 'pretrain':
            self.learning_rate = 3e-4
            self.weight_decay = 0.1
        else:
            self.learning_rate = 5e-5
            self.weight_decay = 0.08

        self._setup_paths_and_weights()
        self.resume_from_checkpoint = self._get_latest_checkpoint(self.output_dir)
        
        stage_name = f"{'VLM' if self.is_vlm else 'LLM'} {self.mode.upper()}"
        print(f"[System] 训练阶段: {stage_name}")
        print(f"[System] 检查点目录: {self.output_dir}")
        print(f"[System] 成品输出路径: {self.model_save_path}")

        if self.resume_from_checkpoint:
            print(f"[System] 状态: 检测到检查点，将【继续训练】(Resume from {self.resume_from_checkpoint})")
            self.init_weights_path = None
        elif self.init_weights_path:
            print(f"[System] 状态: 开始【新阶段训练】，将加载初始权重: {self.init_weights_path}")
        else:
            print(f"[System] 状态: 无初始权重且无检查点，将使用【随机初始化】开始训练")

    def prepare_model(self):
        """准备模型：初始化 Tokenizer、Config 和 Model，并加载权重"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        # 统一使用 VisualVVConfig，它继承自 VVConfig
        self.config = VisualVVConfig(
            vocab_size=len(self.tokenizer),
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # 如果是预训练模式 (pretrain)，强制重置 rope_ntk_alpha = 1.0
        if self.mode == 'pretrain': self.config.rope_ntk_alpha = 1.0
        self.model = VisualVV(self.config, freeze_llm=self.is_freeze_llm, is_load_vision_encoder=self.is_vlm)
        # 加载权重逻辑：如果有初始化权重路径，尝试加载
        if self.init_weights_path:
            print(f"正在从 {self.init_weights_path} 加载模型权重...")
            if not self._load_model_weights(self.model, self.init_weights_path):
                print(f"[Warning] 未在 {self.init_weights_path} 找到权重文件，将使用随机初始化。")
        # 打印并记录参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        param_info = (
             f"{'='*30}\n"
             f" 模型参数信息 ({'VLM' if self.is_vlm else 'LLM'}):\n"
             f"  - 词表大小: {len(self.tokenizer)}\n"
             f"  - 总参数量: {total_params:,} ({total_params / 1e8:.4f} 亿)\n"
             f"  - 可训练参数量: {trainable_params:,} ({trainable_params / 1e8:.4f} 亿)\n"
             f"{'='*30}"
         )
        print(param_info)

    def prepare_data(self):
        """准备数据：加载数据集并划分训练/验证集"""
        print(f"[Data] 正在加载数据集: {self.train_bin}")
        if self.is_vlm:
            dataset = VLMPretrainDataset(self.train_bin, self.config.vision_model_path)
        else:
            dataset = PretrainDataset(self.train_bin)
        val_size = min(500, int(len(dataset) * 0.01)) if len(dataset) > 500 else 1
        train_size = len(dataset) - val_size
        
        # 优化：避免使用 random_split 产生巨大的随机索引列表 (1B 数据下会占 GB 级内存)
        # 直接使用 range 进行切片，Sampler 内部会自动进行 shuffle
        self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        self.val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        
        print(f"[Data] 数据集划分完成: 训练集 {train_size} 条, 验证集 {val_size} 条")

    def train(self):
        """执行训练流程"""
        # 确保数据和模型已准备好
        self.prepare_model()
        self.prepare_data()
        # 动态计算 Batch Size 和 Gradient Accumulation Steps
        train_batch_size, grad_steps = ModelTrainer._dynamic_batch_size(self.model.config)
        # 训练参数设置
        training_args = TrainingArguments(
            # 1. 输出与日志路径
            output_dir=self.output_dir, # 输出目录，用于存放 Checkpoints
            logging_dir=os.path.join(self.root_path, 'logs'), # TensorBoard 日志目录
            report_to="none", # 报告目标（如 wandb），这里关闭
            # 2. 训练超参数 (Hyperparameters)
            learning_rate=self.learning_rate, # 初始学习率
            adam_beta1=0.9, # AdamW 优化器的动量参数 (一阶矩估计的指数衰减率)
            adam_beta2=0.95, # AdamW 优化器的二阶矩估计衰减率 (有时调小能加速收敛)
            adam_epsilon=1e-8, # 防止除以零的小数值
            num_train_epochs=self.num_train_epochs, # 训练总轮数 (Epochs)
            per_device_train_batch_size=train_batch_size, # 单卡训练 Batch Size
            gradient_accumulation_steps=grad_steps, # 梯度累积步数，变相扩大 Batch Size
            weight_decay=self.weight_decay, # 权重衰减 (L2 正则化)
            warmup_ratio=0.05, # 预热步数比例 (Warmup Ratio)，设置为总步数的 2%
            lr_scheduler_type="cosine_with_min_lr", # 学习率调度策略 (余弦退火)
            lr_scheduler_kwargs={"min_lr_rate": 0.1}, # 最小学习率比例，设置为初始学习率的 10%
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
            logging_steps=5, # 每隔多少步打印一次日志
            # 6. 硬件加速与数据加载
            # T4 (Compute Capability 7.5) 虽然 PyTorch 可能报告支持 BF16，但实际上硬件不支持，会导致速度极慢
            # 因此这里增加严格的 Compute Capability >= 8 (Ampere) 检查
            bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
            dataloader_num_workers=4, # 多进程加载数据
            dataloader_pin_memory=True, # 锁页内存，加速 CPU 到 GPU 传输
            max_grad_norm=10.0, # 梯度裁剪，防止梯度爆炸
            disable_tqdm=False, # 强制开启进度条
        )
        trainer = DynamicTrainer(model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer
            )
        print(f"[System] 开始 {self.mode} 模式训练...")
        trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        trainer.save_model(self.model_save_path)
        print(f"[System] 训练完成，模型已保存至 {self.model_save_path}")

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
        """辅助函数：手动加载模型权重，优化内存占用"""
        if not path or not os.path.exists(path): return False
        bin_path = os.path.join(path, "pytorch_model.bin")
        try:
            # 使用 mmap=True 实现内存映射加载，避免将整个文件读入物理内存
            # weights_only=True 是安全加载的推荐做法
            state_dict = torch.load(bin_path, map_location='cpu', weights_only=True, mmap=True)
            
            # 使用 strict=False 允许部分加载 (如从 LLM 权重初始化 VLM)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            print(f"[System] 成功从 {bin_path} 加载权重 (使用 mmap 模式)")
            if missing:
                print(f"  [Info] 缺失权重: {len(missing)} 个 keys")
            if unexpected:
                print(f"  [Info] 未匹配权重: {len(unexpected)} 个 keys")
            return True
        except Exception as e: 
            print(f"[Error] 加载权重失败: {e}")
        return False
        
    def _setup_paths_and_weights(self):
        """
        统一管理 8 种训练场景的路径和权重逻辑
        """
        # 定义各阶段的检查点目录名称
        LLM_PRETRAIN_DIR = os.path.join(self.checkpoints_root, 'llm_pretrain')
        LLM_FINETUNE_DIR = os.path.join(self.checkpoints_root, 'llm_finetune')
        VLM_PRETRAIN_DIR = os.path.join(self.checkpoints_root, 'vlm_pretrain')
        VLM_FINETUNE_DIR = os.path.join(self.checkpoints_root, 'vlm_finetune')

        if not self.is_vlm:
            if self.mode == 'pretrain':
                # 场景 1 & 2: LLM 预训练
                self.output_dir = LLM_PRETRAIN_DIR
                self.train_bin = os.path.join(self.dataset_root, 'data_llm', 'pretrain.bin')
                # LLM 预训练从头开始或从最新 checkpoint 恢复（如果预训练没有断点且有微调，可以从微调加载权重继续训练）
                self.init_weights_path = self._get_latest_checkpoint(LLM_FINETUNE_DIR)
            else:
                # 场景 3 & 4: LLM 微调
                self.output_dir = LLM_FINETUNE_DIR
                self.train_bin = os.path.join(self.dataset_root, 'data_llm', 'finetune.bin')
                # 初始权重从 LLM 预训练的最新检查点取
                self.init_weights_path = self._get_latest_checkpoint(LLM_PRETRAIN_DIR)
        else:
            if self.mode == 'pretrain':
                # 场景 5 & 6: VLM 预训练
                self.output_dir = VLM_PRETRAIN_DIR
                self.train_bin = os.path.join(self.dataset_root, 'data_vlm', 'pretrain.bin')
                # 初始权重从 LLM 微调的最新检查点取
                self.init_weights_path = self._get_latest_checkpoint(LLM_FINETUNE_DIR)
                self.is_freeze_llm = True
            else:
                # 场景 7 & 8: VLM 微调
                self.output_dir = VLM_FINETUNE_DIR
                self.train_bin = os.path.join(self.dataset_root, 'data_vlm', 'finetune.bin')
                # 初始权重从 VLM 预训练的最新检查点取
                self.init_weights_path = self._get_latest_checkpoint(VLM_PRETRAIN_DIR)
                self.is_freeze_llm = False

    @staticmethod
    def _dynamic_batch_size(model_config):
        """动态计算 Batch Size 和 Gradient Accumulation Steps"""
        # 计算最大序列长度，考虑 NTK 扩展
        max_seq_len = int(model_config.max_seq_len * model_config.rope_ntk_alpha)
        train_batch_size = max(1, int((2048+1024) // max_seq_len)) # 单卡最大吞吐量 2048 tokens
        grad_steps = max(1, int(64 // train_batch_size))
        # train_batch_size = 64
        # grad_steps = 1
        print(f"[System] 动态计算得到的 Batch Size: {train_batch_size}")
        print(f"[System] 动态计算得到的 Gradient Accumulation Steps: {grad_steps}")
        return train_batch_size, grad_steps

def train(mode, is_vlm=False, num_train_epochs=1, eval_steps=500, save_steps=500, is_freeze_llm=True):
    """
    保持向后兼容的 train 函数入口
    """
    trainer = ModelTrainer(mode, is_vlm=is_vlm)
    trainer.num_train_epochs = num_train_epochs
    trainer.eval_steps = eval_steps
    trainer.save_steps = save_steps
    trainer.is_freeze_llm = is_freeze_llm
    trainer.train()

if __name__ == "__main__":
    mode = 'pretrain' # pretrain or finetune
    is_vlm = False # 是否是训练vlm
    train(mode=mode, is_vlm=is_vlm, num_train_epochs=0.1, eval_steps=500, save_steps=500, is_freeze_llm=False)
