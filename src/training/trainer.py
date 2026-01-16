import torch
from transformers import Trainer, EarlyStoppingCallback
from data.dataset import TokenBucketSampler, dynamic_collate_fn
from tools.system import SystemControlCallback
from tools.log import CustomTensorBoardCallback
from tools.checkpoint import CheckpointCallback
from tools.inference import InferenceCallback
from tools.rollback import RollbackCallback

test_cases = [
    {
        'prompt': (
            "　　阳春三月。\n"
            "　　绵密的细雨淋湿山林，到处都是雾蒙蒙的白茫茫一片。\n"
            "　　即便如此。\n"
            "　　也仍然抵挡不住村中长舌妇们尽情地吐着沫子八卦着。"
        ),
        'mode': 'pretrain',
        'max_new_tokens': 400,
        'temperature': 1.3,
        'top_k': 75
    },
    {
        'prompt': (
            "贯彻落实国家和省市有关能源工作的法律、法规和政策，研究提出 如东县能源发展战略的建议，"
        ),
        'mode': 'pretrain',
        'max_new_tokens': 200,
        'temperature': 1.3,
        'top_k': 75
    },
    {
        'prompt': "用诗歌或散文形式，描述一个美丽的日出或日落场景。",
        'mode': 'chat',
        'max_new_tokens': 200,
        'temperature': 1.3,
        'top_k': 75 
    },
    {
        'prompt': "中国古代四大发明是什么。",
        'mode': 'chat',
        'max_new_tokens': 50,
        'temperature': 1.3,
        'top_k': 75 
    }
]

class BaseTrainer(Trainer):
    """
    基础训练器类 (BaseTrainer)
    继承自 transformers.Trainer，集成了自定义的系统优化和监控逻辑。
    """
    def __init__(self, model=None, args=None, train_dataset=None, eval_dataset=None, 
                 tokenizer=None, data_collator=None, callbacks=None, **kwargs):
                 
        if callbacks is None: callbacks = []
        # 1. 系统控制回调 (键盘监控、环境优化)
        callbacks.append(SystemControlCallback())
        # 2. 自定义 TensorBoard 回调
        callbacks.append(CustomTensorBoardCallback())
        # 3. 检查点回调 (自动导出模型配置)
        callbacks.append(CheckpointCallback())
        # 4. 自动回退回调 (发现跑飞时自动回到上一个最优状态)
        callbacks.append(RollbackCallback(rollback_threshold=3.0))
        # 5. 早停回调 (作为最后一道防线)
        # 如果模型回退多次依然无法在 5 次评估内降低 Loss，则彻底停止训练
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=10))
        # 6. 推理模拟与对话能力测试回调
        callbacks.append(InferenceCallback(tokenizer=tokenizer, test_cases=test_cases))
        
        # 调用父类初始化
        super().__init__(
            model=model, args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
            **kwargs
        )
        print(f"[BaseTrainer] 已继承 transformers.Trainer，内置回调(System/Log/Checkpoint)已注入")

# 自定义训练器以支持动态批处理
class DynamicTrainer(BaseTrainer):
    """
    支持动态批处理的训练器。
    原理：
    默认的 Trainer 使用 RandomSampler，每个 Batch 的样本数固定，但样本长度可能不一（需要 Padding 到最长）。
    DynamicTrainer 覆盖了 get_train_dataloader，使用 TokenBucketSampler。
    TokenBucketSampler 会将长度相近的样本聚在一起，使得每个 Batch 的 Padding 最少，
    且保证每个 Batch 的总 Token 数（batch_size * max_len）接近设定的 max_tokens。
    """
    def get_train_dataloader(self):
        # 使用 TokenBucketSampler 实现固定 Token 量的动态批处理
        # max_tokens = max_seq_len * batch_size * rope_ntk_alpha
        max_tokens = self.model.config.max_seq_len * self.args.per_device_train_batch_size * self.model.config.rope_ntk_alpha
        sampler = TokenBucketSampler(self.train_dataset, max_tokens=max_tokens, shuffle=True)
        # 使用 lambda 包装 collate_fn，传递 tokenizer 的 pad_token_id
        pad_id = self.processing_class.pad_token_id if self.processing_class else 0
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_sampler=sampler,
            collate_fn=lambda b: dynamic_collate_fn(b, padding_value=pad_id),
            pin_memory=True
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        max_tokens = self.model.config.max_seq_len * self.args.per_device_eval_batch_size * self.model.config.rope_ntk_alpha
        sampler = TokenBucketSampler(eval_dataset, max_tokens=max_tokens, shuffle=False)
        pad_id = self.processing_class.pad_token_id if self.processing_class else 0
        return torch.utils.data.DataLoader(
            eval_dataset, batch_sampler=sampler,
            collate_fn=lambda b: dynamic_collate_fn(b, padding_value=pad_id),
            pin_memory=True
        )
