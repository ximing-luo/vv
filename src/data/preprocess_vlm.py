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
import pandas as pd
from PIL import Image
import io
import torch

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
if src_dir not in sys.path:
    sys.path.append(src_dir)
from configs.model import VisualVVConfig
from transformers import AutoTokenizer

# 导入基础 DataProcessor 以便复用部分逻辑
from preprocess import DataProcessor, PreprocessPipeline

WORKER_PROCESSOR_VLM = None

def _worker_init_vlm(tokenizer_dir, max_seq_len, visual_config):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.model_max_length = int(1e12)
    global WORKER_PROCESSOR_VLM
    WORKER_PROCESSOR_VLM = VLMDataProcessor(tokenizer, max_seq_len, visual_config)

def _worker_process_file_vlm(args):
    file_path, mode = args
    if mode == 'pretrain':
        gen = WORKER_PROCESSOR_VLM.process_pretrain_file(file_path)
    else:
        gen = WORKER_PROCESSOR_VLM.process_finetune_file(file_path)
    return [sample for sample in gen]

class VLMDataProcessor(DataProcessor):
    """
    专门处理多模态数据的处理器，继承自基础 DataProcessor。
    """
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len, visual_config: VisualVVConfig):
        super().__init__(tokenizer, max_seq_len)
        self.visual_config = visual_config
        # 优化：不再在预处理阶段加载 CLIPProcessor
        # 所有的图像处理（解码、Resize、Normalize）都推迟到 Dataset 读取时进行（Online Preprocessing）
        # 这样可以极大减少磁盘占用（40GB -> 0.4GB），利用 CPU 计算换取存储空间
        
    def _yield_chunks(self, tokens_buffer, image_bytes=None):
        """
        VLM 专用的 yield 逻辑：图像只绑定到第一个 chunk。
        """
        if tokens_buffer:
            ids = np.array(tokens_buffer, dtype=np.uint16)
            for i in range(0, len(ids), self.max_seq_len):
                chunk = ids[i : i + self.max_seq_len]
                current_img = image_bytes if i == 0 else None
                yield (np.array(chunk, dtype=np.uint16), current_img)

    def process_parquet_vlm(self, file_path):
        """
        处理 VLM 的 Parquet 数据（图像字节 + 对话）。
        """
        import pyarrow.parquet as pq
        try:
            parquet_file = pq.ParquetFile(file_path)
            # 使用 iter_batches 批量读取，避免一次性加载整个大文件
            for batch in parquet_file.iter_batches(batch_size=500):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    image_bytes = row.get('image_bytes')
                    conversations = row.get('conversations', [])
                    if isinstance(conversations, str):
                        try:
                            conversations = json.loads(conversations)
                        except Exception as e:
                            print(f"[Error] 解析 conversations JSON 失败: {e}")
                            conversations = []
                    
                    # 1. 预处理图像 (仅验证，不转 Tensor)
                    valid_image_bytes = None
                    if image_bytes:
                        try:
                            # 简单验证图像是否损坏，不进行任何处理
                            Image.open(io.BytesIO(image_bytes)).verify()
                            valid_image_bytes = image_bytes
                        except Exception as e:
                            print(f"[Error] 图像已损坏: {e}")

                    # 2. 处理对话
                    if conversations:
                        prompt = self.tokenizer.apply_chat_template(conversations, tokenize=False)
                        # 插入图像占位符
                        if valid_image_bytes is not None:
                            prompt = self.visual_config.image_special_token + prompt
                        
                        enc = self.tokenizer.encode(prompt)
                        if len(enc) > self.max_seq_len:
                            enc = enc[:self.max_seq_len]
                        
                        yield (np.array(enc, dtype=np.uint16), valid_image_bytes)
        except Exception as e:
            print(f"[Error] 读取 Parquet 失败 {file_path}: {e}")

    def process_pretrain_file(self, file_path):
        path_lower = file_path.lower()
        if path_lower.endswith('.parquet'):
            yield from self.process_parquet_vlm(file_path)
        else:
            # 复用父类的其他处理逻辑
            for tokens in super().process_pretrain_file(file_path):
                # DataProcessor 原本 yield 的是 numpy array，我们需要包装成 (tokens, None)
                yield (tokens, None)

    def process_finetune_file(self, file_path):
        path_lower = file_path.lower()
        if path_lower.endswith('.parquet'):
            yield from self.process_parquet_vlm(file_path)
        else:
            for tokens in super().process_finetune_file(file_path):
                yield (tokens, None)

class VLMPreprocessPipeline(PreprocessPipeline):
    """
    专门的多模态预处理流水线。
    """
    def __init__(self, tokenizer, max_seq_len, tokenizer_dir, visual_config, preview_limit=500):
        super().__init__(tokenizer, max_seq_len, tokenizer_dir, preview_limit)
        self.visual_config = visual_config
        self.processor = VLMDataProcessor(tokenizer, max_seq_len, visual_config)

    def collect_sequences(self, input_dir, mode='pretrain', sample_ratio=1.0, num_workers=1):
        print(f"\n--- 开始收集 VLM {mode} 数据序列 ({input_dir}) ---")
        files_to_process = []
        for root, _, files in os.walk(input_dir):
            # 排除预览目录
            if 'preview' in root.lower(): continue
            if self.exclude_keyword and self.exclude_keyword in root.lower(): continue
            
            for f in files:
                # 只处理 parquet 或 jsonl/json 文件
                if f.lower().endswith(('.parquet', '.jsonl', '.json')):
                    files_to_process.append(os.path.join(root, f))

        if sample_ratio < 1.0:
            random.shuffle(files_to_process)
            files_to_process = files_to_process[:max(1, int(len(files_to_process) * sample_ratio))]

        if num_workers <= 1:
            # 单进程模式：真正的流式处理，避免累积整个文件的结果
            # 注意：在单进程模式下，我们直接使用 self.processor，因为 _worker_init_vlm 仅用于多进程初始化
            # 但 self.processor 已经在 __init__ 中初始化了
            for fp in tqdm(files_to_process, desc=f"VLM Processing (Single Process)"):
                if mode == 'pretrain':
                    yield from self.processor.process_pretrain_file(fp)
                else:
                    yield from self.processor.process_finetune_file(fp)
        else:
            # 多进程模式：仍然是文件粒度的缓冲
            tasks = [(fp, mode) for fp in files_to_process]
            with mp.Pool(processes=num_workers, initializer=_worker_init_vlm, 
                         initargs=(self.tokenizer_dir, self.max_seq_len, self.visual_config)) as pool:
                for sample_list in tqdm(pool.imap_unordered(_worker_process_file_vlm, tasks), total=len(tasks), desc=f"VLM Processing"):
                    for sample in sample_list:
                        yield sample

    def _collect_preview(self, seq, count, has_image=False):
        """
        重写收集预览：同时保存是否有图的状态。
        """
        if len(self.preview_sequences) < self.preview_limit:
            self.preview_sequences.append((seq, has_image))
        else:
            k = random.randint(0, count - 1)
            if k < self.preview_limit:
                self.preview_sequences[k] = (seq, has_image)

    def _decode_preview(self, sequences, output_txt_path, num_samples=500):
        """
        重写解码预览：在文本中标记该样本是否有图像。
        """
        if not sequences: return
        sampled_indices = random.sample(range(len(sequences)), min(len(sequences), num_samples))
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"VLM Data Preview\nTotal samples in preview buffer: {len(sequences)}\n\n")
            for idx in sampled_indices:
                seq, has_image = sequences[idx]
                img_status = "[IMAGE PRESENT]" if has_image else "[NO IMAGE]"
                decoded_text = self.tokenizer.decode(seq.tolist())
                f.write(f"[Sample #{idx} | {img_status} | Length: {len(seq)}]\n")
                f.write(decoded_text)
                f.write("\n" + "-" * 30 + "\n")
        print(f"[VLM Preprocess] 预览内容已保存至: {output_txt_path}")

    def save_sequences(self, sequences_iter, output_bin):
        """
        方案 C 实现：在线处理模式 (Online Preprocessing)。
        保存原始图像 Bytes，并额外保存 .img.len 文件记录每张图的长度。
        """
        os.makedirs(os.path.dirname(output_bin), exist_ok=True)
        idx_file = output_bin + ".idx"
        img_file = output_bin + ".img"
        img_idx_file = output_bin + ".img.idx"
        img_len_file = output_bin + ".img.len" # 新增：记录每张图片的长度 (uint32)
        
        count, img_count, total_tokens = 0, 0, 0
        with open(output_bin, 'wb') as f_bin, open(idx_file, 'wb') as f_idx, \
             open(img_file, 'wb') as f_img, open(img_idx_file, 'wb') as f_img_idx, \
             open(img_len_file, 'wb') as f_img_len:
            
            f_idx.write(struct.pack('<Q', 0))
            # f_img_idx 不需要 header，它只是一个 sequence 到 image_index 的映射数组
            # 但为了保持一致性或对齐，如果之前的代码有 header，这里最好确认一下
            # 原代码 PretrainDataset idx 有 header，img.idx 原代码也写了 header (line 218)
            f_img_idx.write(struct.pack('<Q', 0)) 
            f_img_len.write(struct.pack('<Q', 0)) # 也加个 header 存总图片数

            for seq, img_bytes in sequences_iter:
                if seq is None or len(seq) == 0: continue
                
                f_bin.write(seq.tobytes())
                f_idx.write(struct.pack('<H', len(seq)))
                
                has_image = img_bytes is not None
                if has_image:
                    # 直接保存原始 Bytes
                    f_img.write(img_bytes)
                    f_img_len.write(struct.pack('<I', len(img_bytes))) # uint32 长度
                    
                    f_img_idx.write(struct.pack('<i', img_count))
                    img_count += 1
                else:
                    f_img_idx.write(struct.pack('<i', -1))

                count += 1
                total_tokens += len(seq)
                self._collect_preview(seq, count, has_image)

            f_idx.seek(0); f_idx.write(struct.pack('<Q', count))
            f_img_idx.seek(0); f_img_idx.write(struct.pack('<Q', count))
            f_img_len.seek(0); f_img_len.write(struct.pack('<Q', img_count))
        
        print(f"[VLM Preprocess] 完成。样本: {count}, 图像: {img_count}, Tokens: {total_tokens/1e9:.2f}B")
        self._decode_preview(self.preview_sequences, output_bin + ".preview.txt")

def preprocess_vlm(num_workers=1):
    # 配置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    METADATA_ROOT = os.path.join(current_dir, 'metadata')
    DATASET_ROOT = os.path.join(current_dir, 'dataset')
    TOKENIZER_DIR = os.path.join(DATASET_ROOT, 'tokenizer')
    
    visual_config = VisualVVConfig()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    
    pipeline = VLMPreprocessPipeline(tokenizer, visual_config.max_seq_len, TOKENIZER_DIR, visual_config)
    
    # 1. 处理 VLM 预训练数据
    vlm_pretrain_dir = os.path.join(METADATA_ROOT, "pretrain", "minimind-v")
    if os.path.exists(vlm_pretrain_dir):
        pipeline.save_sequences(
            pipeline.collect_sequences(vlm_pretrain_dir, mode='pretrain', num_workers=num_workers),
            os.path.join(DATASET_ROOT, "vlm", "pretrain.bin")
        )

    # 2. 处理 VLM SFT 数据
    vlm_sft_dir = os.path.join(METADATA_ROOT, "finetune", "minimind-v")
    if os.path.exists(vlm_sft_dir):
        pipeline.save_sequences(
            pipeline.collect_sequences(vlm_sft_dir, mode='finetune', num_workers=num_workers),
            os.path.join(DATASET_ROOT, "vlm", "sft.bin")
        )

if __name__ == "__main__":
    preprocess_vlm()
