import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import io
import pyarrow.parquet as pq

class ParquetFileManager:
    """
    Parquet 文件管理器：负责将输出内容自动分割成多个指定大小的小 Parquet 文件。
    """
    def __init__(self, output_dir, prefix, max_size_mb=20):
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_file_index = 1
        self.buffer = []
        self.current_buffer_size = 0
        self.total_written_bytes = 0
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def add(self, row_dict):
        """
        添加一行数据到缓冲区，如果缓冲区过大则写入文件。
        """
        # 估算 row 大小 (主要是 image_bytes)
        size = 0
        if 'image_bytes' in row_dict and isinstance(row_dict['image_bytes'], bytes):
             size += len(row_dict['image_bytes'])
        
        # 简单估算对话文本大小
        if 'conversations' in row_dict:
             size += len(json.dumps(row_dict['conversations'])) * 2 # 粗略估计

        self.buffer.append(row_dict)
        self.current_buffer_size += size
        
        # 如果缓冲区超过限制，或者积累了太多条目（防止内存过大），则刷新
        if self.current_buffer_size >= self.max_size_bytes:
            self.flush()

    def flush(self):
        """
        将缓冲区数据写入 Parquet 文件。
        """
        if not self.buffer:
            return

        try:
            df = pd.DataFrame(self.buffer)
            output_path = os.path.join(self.output_dir, f"{self.prefix}_{self.current_file_index:03d}.parquet")
            df.to_parquet(output_path, index=False)
            
            file_size = os.path.getsize(output_path)
            self.total_written_bytes += file_size
            # print(f"[VLM Sampler] Saved {output_path} ({file_size/1024/1024:.2f} MB)")
            
            self.current_file_index += 1
            self.buffer = []
            self.current_buffer_size = 0
        except Exception as e:
            print(f"[Error] 写入 Parquet 失败: {e}")

    def close(self):
        self.flush()

class VLMSampler:
    """
    VLM 数据采样与预览类：专门负责处理 Parquet 格式的多模态数据。
    """
    def __init__(self, base_database_dir, metadata_root):
        self.base_database_dir = base_database_dir
        self.metadata_root = metadata_root

    def sample_parquet(self, rel_path, output_sub_dir, prefix, target_gb=0.1, split_size_mb=20):
        """
        通用 Parquet 采样方法，支持流式读取和自动分卷。
        """
        input_path = os.path.join(self.base_database_dir, rel_path)
        output_dir = os.path.join(self.metadata_root, output_sub_dir)
        
        print(f"[VLM Sampler] 正在采样 Parquet 数据: {input_path}")
        if not os.path.exists(input_path):
            print(f"[Warning] 文件不存在: {input_path}")
            return None

        # 计算采样概率
        try:
            file_size_bytes = os.path.getsize(input_path)
            target_bytes = target_gb * 1024 * 1024 * 1024
            sampling_prob = min(1.0, (target_bytes / file_size_bytes) * 1.1) if file_size_bytes > 0 else 1.0
            print(f"[VLM Sampler] 原始大小: {file_size_bytes/1024/1024:.2f} MB, 目标: {target_bytes/1024/1024:.2f} MB, 采样率: {sampling_prob:.4f}")
        except Exception:
            sampling_prob = 1.0

        manager = ParquetFileManager(output_dir, prefix, max_size_mb=split_size_mb)
        
        try:
            parquet_file = pq.ParquetFile(input_path)
            # 使用 iter_batches 流式读取，避免内存爆炸
            # batch_size 可以适当调大，pyarrow 读取效率很高
            for batch in tqdm(parquet_file.iter_batches(batch_size=200), desc=f"Sampling {prefix}"):
                df_batch = batch.to_pandas()
                for _, row in df_batch.iterrows():
                    # 提前判断总大小是否达标（估算）
                    if manager.total_written_bytes >= target_bytes:
                        break

                    if random.random() < sampling_prob:
                        row_dict = row.to_dict()
                        # 验证图像是否有效，跳过损坏的图像
                        image_bytes = row_dict.get('image_bytes')
                        if image_bytes is not None:
                            # 处理 Parquet 读取时可能的 numpy.ndarray 类型
                            if isinstance(image_bytes, np.ndarray):
                                if image_bytes.dtype == object and len(image_bytes) > 0:
                                    image_bytes = image_bytes[0]  # 提取数组中的 bytes 对象
                                else:
                                    image_bytes = image_bytes.tobytes()  # 转换为 bytes
                            
                            try:
                                with Image.open(io.BytesIO(image_bytes)) as img:
                                    img.verify()
                            except Exception:
                                continue  # 跳过损坏的图像
                        manager.add(row_dict)
                
                if manager.total_written_bytes >= target_bytes:
                    break
            
            manager.close()
            print(f"[VLM Sampler] {prefix} 采样完成，总计写入: {manager.total_written_bytes / 1024 / 1024:.2f} MB")
            
            # 返回第一个生成的文件路径用于预览（如果存在）
            first_file = os.path.join(output_dir, f"{prefix}_001.parquet")
            return first_file if os.path.exists(first_file) else None

        except Exception as e:
            print(f"[Error] {prefix} 采样失败: {e}")
            return None

    def preview_parquet(self, file_path, output_dir, num_samples=5):
        """
        预览 Parquet 文件：提取文本并保存图像。
        """
        print(f"[VLM Preview] 正在预览文件: {file_path}")
        if not file_path or not os.path.exists(file_path):
            print(f"[Error] 预览路径无效: {file_path}")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            df = pd.read_parquet(file_path)
            print(f"[VLM Preview] 文件总行数: {len(df)}")
            
            # 获取采样数据
            samples = df.head(num_samples)
            preview_data = []
            
            for i, row in samples.iterrows():
                sample_info = {
                    "index": i,
                    "image_name": row.get('image_names', f"sample_{i}.jpg"),
                    "conversations": row.get('conversations', [])
                }
                
                # 保存图像
                image_bytes = row.get('image_bytes')
                if image_bytes:
                    try:
                        # 处理 Parquet 读取时可能的 numpy.ndarray 类型
                        if isinstance(image_bytes, np.ndarray):
                            if image_bytes.size == 0:
                                print(f"  警告: 图像 {i} 的 numpy 数组为空")
                                continue
                            if image_bytes.dtype == object and len(image_bytes) > 0:
                                image_bytes = image_bytes[0]  # 提取数组中的 bytes 对象
                            else:
                                image_bytes = image_bytes.tobytes()  # 转换为 bytes

                        # 确保 image_bytes 是 bytes 类型且非空
                        if not isinstance(image_bytes, bytes):
                            print(f"  警告: 图像 {i} 的字节类型为 {type(image_bytes)}，尝试转换为 bytes")
                            if hasattr(image_bytes, 'tobytes'):
                                image_bytes = image_bytes.tobytes()
                            else:
                                image_bytes = bytes(image_bytes)

                        if len(image_bytes) == 0:
                            print(f"  警告: 图像 {i} 的字节数据为空")
                            continue

                        image = Image.open(io.BytesIO(image_bytes))
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image_save_name = f"sample_{i}_{sample_info['image_name']}"
                        image_save_path = os.path.join(output_dir, image_save_name)
                        image.save(image_save_path)
                        sample_info["image_path"] = image_save_path
                    except Exception as e:
                        print(f"保存图像 {i} 失败: {e}")
                        print(f"  图像字节类型: {type(image_bytes)}, 长度: {len(image_bytes) if hasattr(image_bytes, '__len__') else 'N/A'}")
                        # 打印前几个字节用于调试
                        if hasattr(image_bytes, '__len__') and len(image_bytes) > 0:
                            print(f"  前20字节: {image_bytes[:20].hex() if hasattr(image_bytes, '__len__') else 'N/A'}")
                
                preview_data.append(sample_info)

            # 保存摘要 JSON
            summary_path = os.path.join(output_dir, "preview_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(preview_data, f, ensure_ascii=False, indent=2)
            print(f"\n[VLM Preview] 预览完成，摘要已保存至: {summary_path}")
                
        except Exception as e:
            print(f"[Error] 预览失败: {e}")

    def run_minimind_v_pipeline(self, target_gb=0.1, num_preview=5, split_size_mb=20):
        """
        运行完整的 MiniMind-V 采样与预览流水线
        """
        print("\n=== 开始 MiniMind-V 数据流水线 ===")
        
        # 1. Pretrain 采样与预览
        pretrain_rel = os.path.join("gongjy", "minimind-v_dataset", "pretrain_i2t.parquet")
        pretrain_out_sub = os.path.join("vlm_pretrain", "minimind-v")
        pretrain_sample_path = self.sample_parquet(
            rel_path=pretrain_rel,
            output_sub_dir=pretrain_out_sub,
            prefix="minimind-v_pretrain",
            target_gb=target_gb,
            split_size_mb=split_size_mb
        )
        if pretrain_sample_path:
            preview_dir = os.path.join(os.path.dirname(pretrain_sample_path), "preview")
            self.preview_parquet(pretrain_sample_path, preview_dir, num_samples=num_preview)

        # 2. SFT 采样与预览
        sft_rel = os.path.join("gongjy", "minimind-v_dataset", "sft_i2t.parquet")
        sft_out_sub = os.path.join("vlm_finetune", "minimind-v")
        sft_sample_path = self.sample_parquet(
            rel_path=sft_rel,
            output_sub_dir=sft_out_sub,
            prefix="minimind-v_sft",
            target_gb=target_gb,
            split_size_mb=split_size_mb
        )
        if sft_sample_path:
            preview_dir = os.path.join(os.path.dirname(sft_sample_path), "preview")
            self.preview_parquet(sft_sample_path, preview_dir, num_samples=num_preview)
        
        print("\n=== MiniMind-V 数据流水线完成 ===")

if __name__ == "__main__":
    # 配置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(current_dir)
    BASE_DATABASE_DIR = os.path.join(data_dir, 'database')
    METADATA_ROOT_DIR = os.path.join(data_dir, 'metadata')

    sampler = VLMSampler(BASE_DATABASE_DIR, METADATA_ROOT_DIR)
    
    # 运行采样和预览
    sampler.run_minimind_v_pipeline(target_gb=0.1, num_preview=5, split_size_mb=20)
