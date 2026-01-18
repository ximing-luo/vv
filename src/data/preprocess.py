import os
import json
import numpy as np
import random
import re
from tqdm import tqdm
import sys
import struct
import itertools
import multiprocessing as mp

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
if src_dir not in sys.path:
    sys.path.append(src_dir)
from configs.model import VVConfig
from transformers import AutoTokenizer

WORKER_PROCESSOR = None

def _worker_init(tokenizer_dir, max_seq_len):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.model_max_length = int(1e12)
    global WORKER_PROCESSOR
    WORKER_PROCESSOR = DataProcessor(tokenizer, max_seq_len)

def _worker_process_file(args):
    file_path, mode = args
    if mode == 'pretrain':
        gen = WORKER_PROCESSOR.process_pretrain_file(file_path)
    else:
        gen = WORKER_PROCESSOR.process_finetune_file(file_path)
    return [chunk for chunk in gen]

class DataProcessor:
    """
    数据处理类，封装了对不同类型数据集的处理逻辑。
    """
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def _yield_chunks(self, tokens_buffer):
        """
        辅助函数：处理 buffer，添加 BOS 和 EOS，转为 numpy array，并切分 yield。
        """
        if tokens_buffer:
            ids = np.array(tokens_buffer, dtype=np.uint16)
            for i in range(0, len(ids), self.max_seq_len):
                chunk = ids[i : i + self.max_seq_len]
                yield np.array(chunk, dtype=np.uint16)

    def process_novel(self, file_path):
        """
        处理小说数据：按章节切分
        """
        tokens_buffer = []
        # 章节正则：匹配常见的中文章节标题
        chapter_pattern = re.compile(r'^\s*(第[一二三四五六七八九十百千万零0-9]+[章节回].*|作品相关.*|正文.*|楔子.*|后记.*|内容简介.*|第[0-9]+[章节回].*)')
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                lines = line
                if not line.strip():
                    continue
                if chapter_pattern.match(line):
                    tokens_buffer = [self.tokenizer.bos_token] + tokens_buffer + [self.tokenizer.eos_token]
                    enc = self.tokenizer.encode("".join(tokens_buffer))
                    yield from self._yield_chunks(enc)
                    tokens_buffer = []
                tokens_buffer.append(lines)

    def process_wudao(self, file_path):
        """
        处理悟道数据：每一行作为一个独立的序列
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = self.tokenizer.bos_token + line + self.tokenizer.eos_token
                enc = self.tokenizer.encode(line)
                yield from self._yield_chunks(enc)
                
    def process_pretrain_minimind(self, file_path):
        """
        处理 Pretrain Minimind 数据：每一行作为一个独立的序列
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data.get('text', '')
                enc = self.tokenizer.encode(text)
                yield from self._yield_chunks(enc)

    def process_pretrain_file(self, file_path):
        """
        根据文件名自动选择处理方法
        """
        path_lower = file_path.lower()
        if 'novel' in path_lower:
            yield from self.process_novel(file_path)
        elif 'wudao' in path_lower:
            yield from self.process_wudao(file_path)
        elif 'minimind' in path_lower:
            yield from self.process_pretrain_minimind(file_path)
        else:
            print(f"未识别的预训练文件类型：{file_path}")


    def process_chat(self, file_path):
        """
        处理 multiturn_chat 数据
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                messages = []
                instruction = data.get('instruction', '')
                output = data.get('output', '')
                turns = re.split(r'(Human:|Assistant:)', instruction)
                for i in range(1, len(turns), 2):
                    role = "user" if turns[i] == "Human:" else "assistant"
                    messages.append({"role": role, "content": turns[i+1].strip()})
                messages[-1]['content'] += output
                # 标准函数构建对话数据
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                enc = self.tokenizer.encode(prompt)
                # QA对截断，确保长度不超过 max_seq_len
                if len(enc) > self.max_seq_len:
                    enc = enc[:self.max_seq_len]
                yield np.array(enc, dtype=np.uint16)
    
    def process_sft512(self, file_path):
        """
        处理 SFT 512 数据：每一行作为一个独立的序列
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                messages = []
                messages = data.get('conversations', [])
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                enc = self.tokenizer.encode(prompt)
                # QA对截断，确保长度不超过 max_seq_len
                if len(enc) > self.max_seq_len:
                    enc = enc[:self.max_seq_len]
                yield np.array(enc, dtype=np.uint16)

    def process_finetune_file(self, file_path):
            """
            根据文件名自动选择处理方法
            """
            path_lower = file_path.lower()
            if 'chat' in path_lower:
                yield from self.process_chat(file_path)
            elif 'sft' in path_lower:
                yield from self.process_sft512(file_path)
            else:
                print(f"未识别的微调文件类型：{file_path}")

class PreprocessPipeline:
    """
    数据预处理流水线，负责收集、处理和保存数据。
    """
    def __init__(self, tokenizer, max_seq_len, tokenizer_dir, preview_limit=500, exclude_keyword="firefly"):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.processor = DataProcessor(tokenizer, max_seq_len)
        self.tokenizer_dir = tokenizer_dir
        self.preview_sequences = []
        self.preview_limit = preview_limit
        self.exclude_keyword = exclude_keyword

    def collect_sequences(self, input_dir, mode='pretrain', sample_ratio=1.0, num_workers=1):
        """
        扫描文件夹并收集所有序列。                  
        """
        print(f"\n--- 开始收集 {mode} 数据序列 ({input_dir}) ---")
        # 1. 扫描文件
        files_to_process = []
        for root, _, files in os.walk(input_dir):
            if self.exclude_keyword and self.exclude_keyword in root.lower(): # 排除特定目录
                continue
            for f in files:
                files_to_process.append(os.path.join(root, f))

        # 2. 采样
        if sample_ratio < 1.0:
            random.shuffle(files_to_process)
            num_samples = max(1, int(len(files_to_process) * sample_ratio))
            print(f"[Preprocess] 采样比例 {sample_ratio}: 从 {len(files_to_process)} 个文件中选取 {num_samples} 个")
            files_to_process = files_to_process[:num_samples]

        # 3. 并行处理并 yield
        print(f"[Preprocess] 并行处理 {len(files_to_process)} 个文件，进程数: {num_workers}")
        tasks = [(fp, mode) for fp in files_to_process]
        with mp.Pool(processes=num_workers, initializer=_worker_init, initargs=(self.tokenizer_dir, self.max_seq_len)) as pool:
            for seq_list in tqdm(pool.imap_unordered(_worker_process_file, tasks), total=len(tasks), desc=f"Processing {mode} (mp={num_workers})"):
                for chunk in seq_list:
                    yield chunk
    
    def save_sequences(self, sequences_iter, output_bin):
        """
        保存序列列表到文件。使用流式写入，支持处理批量 yield 的数据。
        """
        os.makedirs(os.path.dirname(output_bin), exist_ok=True)
        idx_file = output_bin + ".idx"
        count = 0
        total_tokens = 0

        print(f"[Preprocess] 正在流式写入数据到 {output_bin} ...")
        with open(output_bin, 'wb') as f_bin, open(idx_file, 'wb') as f_idx:
            # 预留 count 的位置 (8字节, uint64)
            f_idx.write(struct.pack('<Q', 0))
            for seq_idx, seq in enumerate(sequences_iter):
                if seq is None or len(seq) == 0:
                    continue
                f_bin.write(seq.tobytes())
                f_idx.write(struct.pack('<H', len(seq)))
                count += 1
                total_tokens += len(seq)
                self._collect_preview(seq, count)
            # 回头写入真正的 count
            f_idx.seek(0)
            f_idx.write(struct.pack('<Q', count))
        
        print(f"[Preprocess] 已完成保存。序列数: {count / 10000:.2f} 万, 总 Token 数: {total_tokens / 1e9:.2f} B")
        preview_path = os.path.join(os.path.dirname(output_bin), "preview.txt")
        self._decode_preview(self.preview_sequences, preview_path, num_samples=self.preview_limit)

    def process_folder(self, input_dir, output_bin, mode='pretrain', sample_ratio=1.0, num_workers=1):
        """
        便捷函数：处理整个文件夹并保存
        """
        sequences = self.collect_sequences(input_dir, mode, sample_ratio=sample_ratio, num_workers=num_workers)
        self.save_sequences(sequences, output_bin)

    def _decode_preview(self, sequences, output_txt_path, num_samples=500):
        """
        解码预览：从处理后的序列中随机抽取 num_samples 条进行预览。
        """
        if not sequences:
            return

        sampled_indices = random.sample(range(len(sequences)), min(len(sequences), num_samples))
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Preview for processed data\n")
            f.write(f"Total sequences: {len(sequences)}\n")
            f.write(f"Total tokens (approx): {sum(len(s) for s in sequences)}\n\n")
            f.write(f">>> Randomly sampled {len(sampled_indices)} sequences:\n")
            f.write("-" * 50 + "\n")
            for idx in sampled_indices:
                seq = sequences[idx].tolist()
                decoded_text = self.tokenizer.decode(seq)
                f.write(f"[Sample #{idx} | Length: {len(seq)}]\n")
                f.write(decoded_text)
                f.write("\n" + "-" * 30 + "\n")
                
        print(f"[Preprocess] 随机预览内容已保存至: {output_txt_path}")

    def _collect_preview(self, seq, count):
        """
        收集预览序列，使用蓄水池算法确保随机分布。
        """
        if len(self.preview_sequences) < self.preview_limit:
            self.preview_sequences.append(seq)
        else:
            # 蓄水池算法：保证流式数据中每个元素被选中的概率相等
            # count 是当前已处理的总数 (1-based)
            k = random.randint(0, count - 1)
            if k < self.preview_limit:
                self.preview_sequences[k] = seq

def test():
    # 配置
    METADATA_ROOT = r'D:\Axon\ANN\llm\vv\src\data\metadata'
    DATASET_ROOT = r'D:\Axon\ANN\llm\vv\src\data\dataset'
    TOKENIZER_DIR = os.path.join(DATASET_ROOT, 'tokenizer')
    pretrain_input_dir = os.path.join(METADATA_ROOT, "pretrain")
    pretrain_output_bin = os.path.join(DATASET_ROOT, "pretrain", "pretrain_data.bin")
    finetune_input_dir = os.path.join(METADATA_ROOT, "finetune")
    finetune_output_bin = os.path.join(DATASET_ROOT, "finetune", "finetune_data.bin")
    max_seq_len = VVConfig.max_seq_len
    effective_max_len = int(VVConfig.max_seq_len * VVConfig.rope_ntk_alpha)
    print(f"[Preprocess] Config max_seq_len: {VVConfig.max_seq_len}, Alpha: {VVConfig.rope_ntk_alpha}")
    print(f"[Preprocess] Using effective max length: {effective_max_len}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.model_max_length = int(1e12)
    print(f"[Preprocess] Loaded {tokenizer.__class__.__name__}")
    pipeline = PreprocessPipeline(tokenizer, max_seq_len, TOKENIZER_DIR, preview_limit=500)

    # test
    test_file_path = os.path.join(finetune_input_dir, "sft512", "sft512_001.jsonl")
    test_output_bin = os.path.join(finetune_input_dir, "test", "test.bin")
    seq = pipeline.processor.process_sft512(test_file_path)
    pipeline.save_sequences(seq, test_output_bin)

def test_preprocess():
    test_file_path = os.path.join(finetune_input_dir, "sft512", "sft512_001.jsonl")
    test_output_bin = os.path.join(finetune_input_dir, "test", "test.bin")
    seq = pipeline.processor.process_sft512(test_file_path)
    pipeline.save_sequences(seq, test_output_bin)

def preprocess(num_workers = 16, pretrain_sample_ratio=1.0, finetune_sample_ratio=0.1, mixed_sample_ratio=0.1):
    # 配置
    METADATA_ROOT = os.path.join(current_dir, 'metadata')
    DATASET_ROOT = os.path.join(current_dir, 'dataset')
    TOKENIZER_DIR = os.path.join(DATASET_ROOT, 'tokenizer')
    pretrain_input_dir = os.path.join(METADATA_ROOT, "pretrain")
    pretrain_output_bin = os.path.join(DATASET_ROOT, "pretrain", "pretrain_data.bin")
    finetune_input_dir = os.path.join(METADATA_ROOT, "finetune")
    finetune_output_bin = os.path.join(DATASET_ROOT, "finetune", "finetune_data.bin")
    max_seq_len = VVConfig.max_seq_len
    effective_max_len = int(VVConfig.max_seq_len * VVConfig.rope_ntk_alpha)
    print(f"[Preprocess] Config max_seq_len: {VVConfig.max_seq_len}, Alpha: {VVConfig.rope_ntk_alpha}")
    print(f"[Preprocess] Using effective max length: {effective_max_len}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.model_max_length = int(1e12)
    print(f"[Preprocess] Loaded {tokenizer.__class__.__name__}")
    # 初始化 pipeline
    pipeline = PreprocessPipeline(tokenizer, max_seq_len, TOKENIZER_DIR, preview_limit=500)
    
    # 1. 预训练数据处理
    if pretrain_sample_ratio > 0 :
        pipeline.process_folder(pretrain_input_dir, pretrain_output_bin, mode='pretrain', sample_ratio=pretrain_sample_ratio, num_workers=num_workers)

    # 2. 微调数据处理
    pipeline.max_seq_len = effective_max_len
    # 收集微调数据 (全量)
    if finetune_sample_ratio > 0:
        finetune_iter = pipeline.collect_sequences(finetune_input_dir, mode='finetune', sample_ratio=finetune_sample_ratio, num_workers=num_workers)
    else:
        finetune_iter = []
    # 收集 1% 的预训练数据 (作为长文本补充)
    if mixed_sample_ratio > 0:
        mixed_pretrain_iter = pipeline.collect_sequences(pretrain_input_dir, mode='pretrain', sample_ratio=mixed_sample_ratio, num_workers=num_workers)
    else:
        mixed_pretrain_iter = []
    # 合并生成器
    total_sequences_iter = itertools.chain(finetune_iter, mixed_pretrain_iter)
    pipeline.save_sequences(total_sequences_iter, finetune_output_bin)

if __name__ == "__main__":
    preprocess(num_workers=24,
        pretrain_sample_ratio=1,
        finetune_sample_ratio=1,
        mixed_sample_ratio=0.1
        )
    # test()
