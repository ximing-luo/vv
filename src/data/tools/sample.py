import os
import json
import random
from tqdm import tqdm

class FileManager:
    """
    文件管理器：负责将输出内容自动分割成多个指定大小的小文件。
    修复了当第一条数据超过限制时会导致 001 文件为空的问题。
    """
    def __init__(self, base_dir, prefix, max_size_mb=10, ext='txt'):
        self.base_dir = base_dir
        self.prefix = prefix
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ext = ext
        self.current_file_index = 1
        self.current_file_handle = None
        self.current_file_size = 0
        self.total_written_bytes = 0
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _get_file_path(self, index=None):
        idx = index if index is not None else self.current_file_index
        return os.path.join(self.base_dir, f"{self.prefix}_{idx:03d}.{self.ext}")

    def _open_new_file(self):
        if self.current_file_handle:
            self.current_file_handle.close()
        
        file_path = self._get_file_path()
        self.current_file_handle = open(file_path, 'w', encoding='utf-8')
        self.current_file_size = 0

    def write(self, content):
        if not content:
            return
        bytes_len = len(content.encode('utf-8'))
        
        # 优化逻辑：
        # 1. 如果还没打开过文件，则打开第一个
        # 2. 如果当前文件已有内容，且加入新内容后会超限，则切换新文件
        if self.current_file_handle is None:
            self._open_new_file()
        elif self.current_file_size > 0 and self.current_file_size + bytes_len > self.max_size_bytes:
            self.current_file_index += 1
            self._open_new_file()
            
        self.current_file_handle.write(content)
        self.current_file_size += bytes_len
        self.total_written_bytes += bytes_len

    def close(self):
        if self.current_file_handle:
            self.current_file_handle.close()
            # 如果最后一个文件是空的（虽然逻辑上不太可能了），则删除它
            if self.current_file_size == 0:
                try:
                    os.remove(self._get_file_path())
                except:
                    pass

class DataSampler:
    """
    数据采样类：负责从原始数据库中抽取不同类型的数据并分门别类存放。
    支持将大文件拆分为多个小文件以便查看。
    """
    def __init__(self, base_dir, metadata_root):
        self.base_dir = base_dir
        self.metadata_root = metadata_root

    def sample_wudao(self, filename="high_data.txt", target_gb=0.1, split_size_mb=2):
        """
        采样 WuDao 数据，保持 txt 格式。
        优化：使用随机 Seek 读取，避免遍历几十 GB 的大文件。
        """
        input_path = os.path.join(self.base_dir, filename)
        output_dir = os.path.join(self.metadata_root, "pretrain", "wudao")
        
        print(f"[Sampler] 正在采样 WuDao 数据 (txt): {input_path}")
        if not os.path.exists(input_path):
            print(f"[Warning] 文件不存在: {input_path}")
            return

        manager = FileManager(output_dir, "wudao", max_size_mb=split_size_mb, ext='txt')
        target_bytes = int(target_gb * 1024 * 1024 * 1024)
        file_size = os.path.getsize(input_path)
        
        chunk_size = 1024 * 1024  # 每次读取 1MB
        num_chunks_needed = (target_bytes + chunk_size - 1) // chunk_size
        
        print(f"[Sampler] 原始文件大小: {file_size / 1024 / 1024:.2f} MB, 目标采样量: {target_bytes / 1024 / 1024:.2f} MB")
        print(f"[Sampler] 使用随机 Seek 采样，预计抽取 {num_chunks_needed} 个数据块")

        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 随机生成采样点
                # 为了防止采样点重叠，这里简单处理：在全文件范围内随机选起始位置
                # 如果文件特别大，重叠概率极低
                sampled_offsets = [random.randint(0, max(0, file_size - chunk_size)) for _ in range(num_chunks_needed)]
                # 排序偏移量可以稍微提高磁盘读取性能（减少磁头来回跳变）
                sampled_offsets.sort()

                for offset in tqdm(sampled_offsets, desc="Sampling WuDao (Random Seek)"):
                    f.seek(offset)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    manager.write(chunk)
                    if manager.total_written_bytes >= target_bytes:
                        break
            print(f"[Sampler] WuDao 采样完成，总计写入: {manager.total_written_bytes / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"[Error] WuDao 采样失败: {e}")
        finally:
            manager.close()

    def sample_novel(self, novel_dirname="novel", target_gb=0.1, split_size_mb=2):
        """
        采样小说数据，保持 txt 格式
        """
        input_dir = os.path.join(self.base_dir, novel_dirname)
        output_dir = os.path.join(self.metadata_root, "pretrain", "novel")
        
        print(f"[Sampler] 正在采样小说数据 (txt): {input_dir}")
        all_novels = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.txt'):
                    all_novels.append(os.path.join(root, file))

        if not all_novels:
            print(f"[Warning] 未找到小说文件: {input_dir}")
            return

        # 随机打乱小说顺序
        random.shuffle(all_novels)
        manager = FileManager(output_dir, "novel", max_size_mb=split_size_mb, ext='txt')
        target_bytes = target_gb * 1024 * 1024 * 1024
        
        for fpath in tqdm(all_novels, desc="Sampling Novels"):
            if manager.total_written_bytes >= target_bytes:
                break
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        # 写入文件名作为标识，方便查看
                        header = f"\n\n{'='*20} FILE: {os.path.basename(fpath)} {'='*20}\n\n"
                        manager.write(header + content)
            except:
                continue
        manager.close()
        print(f"[Sampler] 小说采样完成，总计写入: {manager.total_written_bytes / 1024 / 1024:.2f} MB")

    def _smart_sample_jsonl(self, input_path, manager, target_bytes, desc):
        """
        针对大 JSONL 文件的智能采样：
        如果文件很大，使用随机 Seek 跳读；如果文件较小，直接流式读取。
        """
        file_size = os.path.getsize(input_path)
        # 如果文件大于 2GB，启用 Seek 采样以提高速度
        if file_size > 2 * 1024 * 1024 * 1024:
            print(f"[Sampler] {desc}: 文件较大 ({file_size / 1024 / 1024:.2f} MB), 启用随机 Seek 采样")
            # 预估行数：先读 100 行算平均长度
            avg_line_len = 500 # 默认值
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    sample_lines = [len(f.readline()) for _ in range(10)]
                    avg_line_len = sum(sample_lines) / len(sample_lines) if sample_lines else 500
            except: pass
            
            num_samples_needed = int((target_bytes / avg_line_len) * 1.2) # 20% 冗余
            sampled_offsets = [random.randint(0, max(0, file_size - 1024)) for _ in range(num_samples_needed)]
            sampled_offsets.sort()

            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                for offset in tqdm(sampled_offsets, desc=f"{desc} (Seek)"):
                    if manager.total_written_bytes >= target_bytes:
                        break
                    f.seek(offset)
                    f.readline() # 跳过当前行（可能是残缺的）
                    line = f.readline() # 读取下一行完整的 JSON
                    if not line: continue
                    yield line
        else:
            # 文件较小，使用原来的流式采样，随机性更好
            sampling_prob = min(1.0, (target_bytes / file_size) * 1.1) if file_size > 0 else 1.0
            print(f"[Sampler] {desc}: 文件较小, 采样概率: {sampling_prob:.4f}")
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in tqdm(f, desc=desc):
                    if manager.total_written_bytes >= target_bytes:
                        break
                    if random.random() < sampling_prob:
                        yield line

    def sample_firefly(self, filename="firefly-train-1.1M.jsonl", target_gb=0.1, split_size_mb=2):
        """
        采样 Firefly 数据，保持 jsonl 格式
        """
        input_path = os.path.join(self.base_dir, filename)
        output_dir = os.path.join(self.metadata_root, "finetune", "firefly")
        
        print(f"[Sampler] 正在采样 Firefly 数据 (jsonl): {input_path}")
        if not os.path.exists(input_path):
            print(f"[Warning] 文件不存在: {input_path}")
            return

        manager = FileManager(output_dir, "firefly", max_size_mb=split_size_mb, ext='jsonl')
        target_bytes = target_gb * 1024 * 1024 * 1024
        
        try:
            for line in self._smart_sample_jsonl(input_path, manager, target_bytes, "Sampling Firefly"):
                try:
                    data = json.loads(line)
                    out_item = {
                        "instruction": data.get("input", ""),
                        "output": data.get("target", ""),
                        "source": "firefly"
                    }
                    manager.write(json.dumps(out_item, ensure_ascii=False) + '\n')
                except:
                    continue
            print(f"[Sampler] Firefly 采样完成，总计写入: {manager.total_written_bytes / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"[Error] Firefly 采样失败: {e}")
        finally:
            manager.close()

    def sample_chat(self, filename="multiturn_chat_0.8M.json", target_gb=0.1, split_size_mb=2):
        """
        采样 Chat 数据，保持 jsonl 格式
        """
        input_path = os.path.join(self.base_dir, filename)
        output_dir = os.path.join(self.metadata_root, "finetune", "multiturn_chat")
        
        print(f"[Sampler] 正在采样 Chat 数据 (jsonl): {input_path}")
        if not os.path.exists(input_path):
            print(f"[Warning] 文件不存在: {input_path}")
            return

        manager = FileManager(output_dir, "chat", max_size_mb=split_size_mb, ext='jsonl')
        target_bytes = target_gb * 1024 * 1024 * 1024
        
        try:
            for line in self._smart_sample_jsonl(input_path, manager, target_bytes, "Sampling Chat"):
                try:
                    item = json.loads(line.strip())
                    out_item = {
                        "instruction": item.get("instruction", ""),
                        "output": item.get("output", ""),
                        "source": "chat"
                    }
                    manager.write(json.dumps(out_item, ensure_ascii=False) + '\n')
                except:
                    continue
            print(f"[Sampler] Chat 采样完成，总计写入: {manager.total_written_bytes / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"[Error] Chat 采样失败: {e}")
        finally:
            manager.close()
    def sample_sft(self, filename="sft_512.jsonl", target_gb=0.1, split_size_mb=2):
        """
        采样 SFT  数据，保持原始 jsonl 格式。
        """
        input_path = os.path.join(self.base_dir, filename)
        output_dir = os.path.join(self.metadata_root, "finetune", filename.split(".")[0])
        
        print(f"[Sampler] 正在采样 SFT 数据 (原始 jsonl): {input_path}")
        if not os.path.exists(input_path):
            print(f"[Warning] 文件不存在: {input_path}")
            return

        manager = FileManager(output_dir, filename.split(".")[0], max_size_mb=split_size_mb, ext='jsonl')
        target_bytes = target_gb * 1024 * 1024 * 1024
        
        try:
            for line in self._smart_sample_jsonl(input_path, manager, target_bytes, f"Sampling {filename.split('.')[0]}"):
                # 直接写入原始行，不进行任何处理
                manager.write(line)
            print(f"[Sampler] {filename.split('.')[0]} 采样完成，总计写入: {manager.total_written_bytes / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"[Error] {filename.split('.')[0]} 采样失败: {e}")
        finally:
            manager.close()

    def sample_pretrain_minimind(self, filename="pretrain_hq.jsonl", target_gb=0.1, split_size_mb=2):
        """
        采样 Pretrain Minimind 数据，保持原始 jsonl 格式。
        """
        input_path = os.path.join(self.base_dir, filename)
        output_dir = os.path.join(self.metadata_root, "pretrain", "pretrain_minimind")
        
        print(f"[Sampler] 正在采样 Pretrain Minimind 数据 (原始 jsonl): {input_path}")
        if not os.path.exists(input_path):
            print(f"[Warning] 文件不存在: {input_path}")
            return

        manager = FileManager(output_dir, "pretrain_minimind", max_size_mb=split_size_mb, ext='jsonl')
        target_bytes = target_gb * 1024 * 1024 * 1024
        
        try:
            for line in self._smart_sample_jsonl(input_path, manager, target_bytes, "Sampling Pretrain Minimind"):
                # 直接写入原始行，不进行任何处理
                manager.write(line)
            print(f"[Sampler] Pretrain Minimind 采样完成，总计写入: {manager.total_written_bytes / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"[Error] Pretrain Minimind 采样失败: {e}")
        finally:
            manager.close()

def sample():
    # 配置路径
    BASE_DATABASE_DIR = r'.\src\data\database'
    METADATA_ROOT_DIR = r'.\src\data\metadata'
    sampler = DataSampler(BASE_DATABASE_DIR, METADATA_ROOT_DIR)
    sampler.sample_wudao(target_gb=5, split_size_mb=20)
    sampler.sample_novel(target_gb=2, split_size_mb=20)
    sampler.sample_firefly(target_gb=0.1, split_size_mb=2)
    sampler.sample_chat(target_gb=0.8, split_size_mb=20)

if __name__ == "__main__":
    # 配置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(current_dir)
    BASE_DATABASE_DIR = os.path.join(data_dir, 'database')
    METADATA_ROOT_DIR = os.path.join(data_dir, 'metadata')
    sampler = DataSampler(BASE_DATABASE_DIR, METADATA_ROOT_DIR)

    # 执行各类采样 - 这里可以控制每个部分生成的总大小和分片大小
    # 1. WuDao 预训练采样
    sampler.sample_wudao(target_gb=1, split_size_mb=10)

    # 2. 小说预训练采样
    sampler.sample_novel(target_gb=1, split_size_mb=10)

    # 3. Firefly SFT 采样
    # sampler.sample_firefly(target_gb=0.1, split_size_mb=2)

    # 4. Chat SFT 采样
    # sampler.sample_chat(target_gb=0.8, split_size_mb=20)

    # 5. SFT 采样
    filename = "sft_mini_512.jsonl"
    sampler.sample_sft(filename=filename, target_gb=1, split_size_mb=10)

    # 7. Pretrain Minimind 采样
    sampler.sample_pretrain_minimind(target_gb=1, split_size_mb=10)
