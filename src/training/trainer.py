import torch
from transformers import Trainer, EarlyStoppingCallback
from src.data.dataset import TokenBucketSampler, dynamic_collate_fn
from .tools.system import SystemControlCallback
from .tools.log import CustomTensorBoardCallback
from .tools.checkpoint import CheckpointCallback
from .tools.inference import InferenceCallback
from .tools.rollback import RollbackCallback
from .tools.scheduler import BatchSizeSchedulerCallback

test_cases = [
    {
        'prompt': (
            "这个词语是什么意思？不堪入目\n"
        ),
        'mode': 'pretrain',
        'max_new_tokens': 100,
        'temperature': 1.3,
        'top_k': 75
    },
    {
        'prompt': (
            "文言文翻译：恬既孝行殊异，声著邦壤，敦风厉俗，弘益兹多。\n"
        ),
        'mode': 'pretrain',
        'max_new_tokens': 100,
        'temperature': 1.3,
        'top_k': 75
    },
    {
        'prompt': "请解释一下为什么日落时海水会变成绿色的。",
        'mode': 'chat',
        'max_new_tokens': 200,
        'temperature': 1.3,
        'top_k': 75
    },
    {
        'prompt': "请写一篇五言律诗，题目为“春风十里不如你”。",
        'mode': 'chat',
        'max_new_tokens': 100,
        'temperature': 1.3,
        'top_k': 75 
    },
    {
        'prompt': ("请描述这张图片。<image>"),
        'mode': 'vlm',
        'max_new_tokens': 50,
        'temperature': 1.3,
        'top_k': 75,
        'image_path': r'D:\Axon\ANN\llm\vv\src\data\database\gongjy\minimind-v_dataset\eval_images\彩虹瀑布-Rainbow-Falls .jpg'
    }
]

# 自定义训练器以支持动态批处理
class DynamicTrainer(Trainer):
    """
    支持动态批处理的训练器。
    原理：
    默认的 Trainer 使用 RandomSampler，每个 Batch 的样本数固定，但样本长度可能不一（需要 Padding 到最长）。
    DynamicTrainer 覆盖了 get_train_dataloader，使用 TokenBucketSampler。
    TokenBucketSampler 会将长度相近的样本聚在一起，使得每个 Batch 的 Padding 最少，
    且保证每个 Batch 的总 Token 数（batch_size * max_len）接近设定的 max_tokens。
    """
    def __init__(self, model=None, args=None, train_dataset=None, eval_dataset=None, 
                 tokenizer=None, data_collator=None, callbacks=None, **kwargs):
                 
        if callbacks is None: callbacks = []
        # 注入自定义回调逻辑
        callbacks.extend([
            SystemControlCallback(),     # 系统控制 (键盘监控、环境优化)
            CustomTensorBoardCallback(), # 自定义 TensorBoard 日志
            CheckpointCallback(),        # 检查点自动导出
            RollbackCallback(rollback_threshold=3.0), # 自动回退
            EarlyStoppingCallback(early_stopping_patience=10), # 早停
            InferenceCallback(tokenizer=tokenizer, test_cases=test_cases), # 推理模拟
            BatchSizeSchedulerCallback(initial_grad_steps=args.gradient_accumulation_steps, strategy="milestones") # 动态 Batch Size
        ])
        
        super().__init__(
            model=model, args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
            **kwargs
        )
        print(f"[DynamicTrainer] 已初始化，内置回调已注入")

    def get_train_dataloader(self):
        # 使用 TokenBucketSampler 实现固定 Token 量的动态批处理
        # max_tokens = max_seq_len * batch_size * rope_ntk_alpha
        max_tokens = self.model.config.max_seq_len * self.args.per_device_train_batch_size * self.model.config.rope_ntk_alpha
        sampler = TokenBucketSampler(self.train_dataset, max_tokens=max_tokens)
        # 使用 DataCollatorWrapper 替代 lambda 以支持 Windows 多进程 pickling
        pad_id = self.processing_class.pad_token_id if self.processing_class else 0
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_sampler=sampler,
            collate_fn=DataCollatorWrapper(padding_value=pad_id),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory
        )

    def get_eval_dataloader(self, eval_dataset=None):
        # 如果调用 evaluate() 时传入了特定数据集，则优先使用；否则使用初始化时的 eval_dataset
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        max_tokens = self.model.config.max_seq_len * self.args.per_device_eval_batch_size * self.model.config.rope_ntk_alpha
        sampler = TokenBucketSampler(eval_dataset, max_tokens=max_tokens)
        pad_id = self.processing_class.pad_token_id if self.processing_class else 0
        
        # Windows 优化：评估阶段通常数据量较小 (如 500 条)，
        # 使用多进程加载 (num_workers > 0) 在 Windows 上极易触发 Pipe 序列化错误 (OSError [Errno 22])。
        # 强制设置 num_workers=0 以确保稳定性，且对小规模评估性能影响忽略不计。
        return torch.utils.data.DataLoader(
            eval_dataset, batch_sampler=sampler,
            collate_fn=DataCollatorWrapper(padding_value=pad_id),
            num_workers=0, 
            pin_memory=self.args.dataloader_pin_memory
        )

class DataCollatorWrapper:
    """
    包装 dynamic_collate_fn 以支持多进程序列化 (Pickling)。
    Windows 下使用 spawn 模式启动多进程时，lambda 函数无法被序列化，
    因此需要使用顶级类或函数。
    """
    def __init__(self, padding_value):
        self.padding_value = padding_value
    
    def __call__(self, batch):
        return dynamic_collate_fn(batch, padding_value=self.padding_value)
