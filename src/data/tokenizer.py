import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import List, Union
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from .tools.vocab_tool import export_readable_vocab, prepare_cache_files, analyze_vocab_distribution

def get_training_files(input_paths: Union[str, List[str]], max_gb: float = 1.0, sample_rate: float = 1.0):
    """
    获取训练文件列表，并根据 max_gb 和 sample_rate 进行筛选
    支持传入单个路径字符串或路径列表
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    
    all_files = []
    extensions = {".txt", ".jsonl"}
    
    for path_str in input_paths:
        path = Path(path_str)
        if path.is_dir():
            # 递归获取目录下所有匹配后缀的文件
            for ext in extensions:
                all_files.extend(list(path.rglob(f"*{ext}")))
        elif path.suffix in extensions:
            all_files.append(path)
    
    if not all_files:
        return []

    # 1. 采样逻辑
    if sample_rate < 1.0:
        random.shuffle(all_files)
        num_files = max(1, int(len(all_files) * sample_rate))
        all_files = all_files[:num_files]
        print(f"[Data] 采样模式：选取 {len(all_files)} 个文件 (比例: {sample_rate})")

    # 2. 容量限制逻辑
    selected_files = []
    total_bytes = 0
    max_bytes = max_gb * 1024**3
    
    for file_path in all_files:
        file_size = file_path.stat().st_size
        if total_bytes + file_size > max_bytes:
            # 如果加上这个文件就超标了，如果是第一个文件，我们还是保留它，否则跳过
            if not selected_files:
                selected_files.append(str(file_path))
                total_bytes += file_size
            break
        selected_files.append(str(file_path))
        total_bytes += file_size
        
    print(f"[Data] 最终选取 {len(selected_files)} 个文件，总计约 {total_bytes / 1024**3:.2f} GB")
    return selected_files

def train_tokenizer(input_paths: Union[str, List[str]], output_dir: str, vocab_size: int = 6400, max_gb: float = 1.0, sample_rate: float = 1.0):
    """
    训练 BPE 分词器并保存为 Hugging Face 格式
    """
    print(f"[Tokenizer] 正在从 {input_paths} 训练 BPE 分词器...")
    print(f"[Tokenizer] 限制数据量: {max_gb}GB, 采样率: {sample_rate}, 目标词表大小: {vocab_size}")
    
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=100,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    # 获取筛选后的文件列表
    files = get_training_files(input_paths, max_gb=max_gb, sample_rate=sample_rate)
    # 建立临时缓存目录，将预处理后的文本存入其中，让 Rust 引擎直接读取以获得最高性能
    cache_dir = tempfile.mkdtemp(prefix="tokenizer_train_cache_")
    try:
        cache_files = prepare_cache_files(files, cache_dir)
        tokenizer.train(cache_files, trainer=trainer)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[Tokenizer] 已清理临时缓存: {cache_dir}")
    tokenizer.decoder = decoders.ByteLevel()
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    # tokenizer.model.save(output_dir) # 保存 vocab.json 和 merges.txt
    # 导出可读的词表文件，分析词表分布
    export_readable_vocab(tokenizer_path, os.path.join(output_dir, "vocab.txt"))
    analyze_vocab_distribution(os.path.join(output_dir, "vocab.txt"))

    added_tokens_decoder = {
        "0": {
            "content": "<|endoftext|>",
            "lstrip": False,      # 是否去除左侧空白
            "normalized": False,  # 是否进行标准化
            "rstrip": False,      # 是否去除右侧空白
            "single_word": False, # 是否仅作为单次匹配
            "special": True       # 是否为特殊 token
        },
        "1": {
            "content": "<|im_start|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        },
        "2": {
            "content": "<|im_end|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        }
    }

    config = {
        # === 基础配置 (Basic Config) ===
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 32768,

        # === 特殊 Token 定义 (Special Tokens) ===
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",

        # === Token 处理标志 (Token Handling Flags) ===
        "add_bos_token": False,     # 是否在序列开头自动添加 bos_token
        "add_eos_token": False,     # 是否在序列结尾自动添加 eos_token
        "add_prefix_space": False,  # 是否在文本开头自动添加空格 (对于 GPT-2/RoBERTa 等很重要)
        "clean_up_tokenization_spaces": False, # 是否在解码时清理 tokenization 产生的多余空格
        "spaces_between_special_tokens": False,  # 特殊 token 之间是否保留空格

        # === 兼容性配置 (Compatibility) ===
        "legacy": True,  # 是否使用旧版行为 (保持兼容性)
        "sp_model_kwargs": {},  # SentencePiece 模型参数 (此处为空)

        # === 解码器配置 (Decoder Config) ===
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [], # 额外的特殊 token 列表 (除了 bos, eos, unk, pad 之外的)

        # === 聊天模板 (Chat Template) ===
        "chat_template": "{%- if messages[0]['role'] == 'system' -%}\n    {{- '<|im_start|>系统\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}\n{%- else -%}\n    {{- '<|im_start|>系统\\n你是一个有用的助手，由 Axon 开发。<|im_end|>\\n' -}}\n{%- endif -%}\n\n{%- for message in messages -%}\n    {%- if message['role'] == 'user' -%}\n        {{- '<|im_start|>用户\\n' + message['content'] + '<|im_end|>\\n' -}}\n    {%- elif message['role'] == 'assistant' -%}\n        {{- '<|im_start|>助手\\n' + message['content'] + '<|im_end|>\\n' -}}\n    {%- endif -%}\n{%- endfor -%}\n\n{%- if add_generation_prompt -%}\n    {{- '<|im_start|>助手\\n' -}}\n{%- endif -%}"
    }
    
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    print(f"[Tokenizer] 训练完成！保存在: {output_dir}")

if __name__ == "__main__":
    # 使用示例
    DATA_DIR = r'D:\Axon\ANN\llm\vv\src\data\metadata\pretrain'
    TOKENIZER_DIR = r'D:\Axon\ANN\llm\vv\src\data\dataset\tokenizer'
    
    # 参数说明:
    # sample_rate: 随机采样比例 (针对文件)
    # max_gb: 采样文件后限制参与训练的总数据量
    train_tokenizer(
        DATA_DIR, 
        TOKENIZER_DIR, 
        vocab_size=6400, 
        sample_rate=0.05, 
        max_gb=0.3
    )
