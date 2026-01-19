"""
数据处理模块：实现高性能二进制数据加载与动态批处理 (Dynamic Batching) 流程。

【核心流程说明】
- train.py ：启动训练，计算目标 max_tokens 。
- DynamicTrainer ：创建 DataLoader ，指定使用 TokenBucketSampler 。
- TokenBucketSampler ：挑选出一组总 Token 数达标的索引。
- PretrainDataset ：根据索引从 .bin 文件中 memmap 读取原始 Token。
- dynamic_collate_fn ：将这组 Token 转化为 Tensor，并用 0 (Input) 和 -100 (Label) 进行 Padding。
- Model Forward ：模型接收到形状为 [Batch, Seq_Len] 的 Tensor 开始计算
"""
import torch
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from torch.utils.data import Dataset, Sampler
import struct
import io
from PIL import Image
try:
    from transformers import CLIPProcessor
except ImportError:
    CLIPProcessor = None

class PretrainDataset(Dataset):
    """
    数据集类：高性能二进制流格式 (data.bin + data.bin.idx)。
    利用内存映射 (memmap) 实现超大数据集的零加载启动。
    """
    def __init__(self, data_path):
        idx_path = data_path + ".idx"
        # 1. 加载索引信息
        with open(idx_path, 'rb') as f:
            # 读取头部 8 字节: 总序列数 (uint64)
            self.num_samples = struct.unpack('<Q', f.read(8))[0]
            # 读取剩余部分: 所有序列的长度 (uint16)
            self.lengths = np.frombuffer(f.read(), dtype=np.uint16)
            
        # 2. 计算每个序列的起始偏移量 (字节)
        # 这里的 offsets 数组比 samples 多一个，方便最后一个样本切片
        # tokens 是 uint16 (2 bytes)，所以累加后乘以 2
        self.offsets = np.zeros(len(self.lengths) + 1, dtype=np.uint64)
        self.offsets[1:] = np.cumsum(self.lengths, dtype=np.uint64) * 2
        
        # 3. 使用内存映射加载原始数据
        # mode='r' 表示只读，不占用实际物理内存，由系统内核按需置换
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r', offset=0)
        total_tokens = np.sum(self.lengths.astype(np.uint64))

        print(f"[Dataset] 已加载二进制数据集: {data_path}")
        print(f" - 总序列数: {self.num_samples / 10000:.2f} 万")
        print(f" - 总 Token 数: {total_tokens / 1e9:.2f} B")
        print(f" - 平均序列长度: {total_tokens / self.num_samples:.2f}")

    def __len__(self):
        return self.num_samples
    
    def _get_raw_data(self, idx):
        """底层读取逻辑：获取完整的原始 token 序列"""
        offset = int(self.offsets[idx] // 2)
        length = int(self.lengths[idx])
        return self.data[offset : offset + length].astype(np.int64)

    def __getitem__(self, idx):
        # 获取原始序列
        seq = self._get_raw_data(idx)
        
        # 统一返回 (X, Y) 模式，供模型 forward 使用
        # X: 0 到 n-1 (输入)
        # Y: 1 到 n   (预测目标)
        X = torch.from_numpy(seq[:-1])
        Y = torch.from_numpy(seq[1:])
        
        return X, Y

class SFTDataset(PretrainDataset):
    """
    指令微调数据集：继承自 PretrainDataset。
    代码风格参考 minimind-master/dataset/lm_dataset.py
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 预先获取 Assistant 标记的 ID，用于生成 Mask
        # ChatML 格式中助手回答开始于 <|im_start|>助手\n
        self.bos_id = tokenizer('<|im_start|>助手\n', add_special_tokens=False).input_ids
        # Assistant 回答结束于 <|im_end|>\n
        self.eos_id = tokenizer('<|im_end|>\n', add_special_tokens=False).input_ids

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                actual_end = min(end + len(self.eos_id), len(input_ids))
                for j in range(start, actual_end):
                    loss_mask[j] = 1
                i = actual_end
            else:
                i += 1
        return loss_mask

    def __getitem__(self, idx):
        # 1. 获取完整的原始序列
        input_ids = self._get_raw_data(idx).tolist()
        
        # 2. 生成 Loss Mask
        loss_mask = self.generate_loss_mask(input_ids)
        
        # 3. 构建训练对 (移位对齐)
        # X: [T1, T2, T3]
        # Y: [T2, T3, T4]
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y_raw = torch.tensor(input_ids[1:], dtype=torch.long)
        mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        
        # 4. 关键点：将 mask 为 0 的地方设为 -100
        # 这样模型内部的 F.cross_entropy(..., ignore_index=-100) 会自动跳过这些位置
        Y = torch.where(mask == 1, Y_raw, torch.tensor(-100, dtype=torch.long))
        
        return X, Y

class VLMDatasetMixin:
    """
    VLM 数据加载混入类：负责加载 .img, .img.idx, .img.len 文件并进行在线预处理
    """
    def init_vlm_data(self, data_path, vision_model_path):
        self.img_idx_path = data_path + ".img.idx"
        self.img_path = data_path + ".img"
        self.img_len_path = data_path + ".img.len"
        
        # 1. 加载 Sequence -> Image Index 映射
        # .img.idx: [Header(Q), i32, i32, ...]
        with open(self.img_idx_path, 'rb') as f:
            _ = struct.unpack('<Q', f.read(8))[0] # Skip header
            self.seq_to_img_idx = np.frombuffer(f.read(), dtype=np.int32)

        # 2. 加载 Image Lengths 并计算 Offsets
        # .img.len: [Header(Q), u32, u32, ...]
        with open(self.img_len_path, 'rb') as f:
            self.num_images = struct.unpack('<Q', f.read(8))[0]
            self.img_lengths = np.frombuffer(f.read(), dtype=np.uint32)
        
        self.img_offsets = np.zeros(len(self.img_lengths) + 1, dtype=np.uint64)
        self.img_offsets[1:] = np.cumsum(self.img_lengths, dtype=np.uint64)

        # 3. 内存映射原始图像数据 (.img)
        self.img_data = np.memmap(self.img_path, mode='r')
        
        # 4. 初始化 Processor
        if CLIPProcessor is None:
            raise ImportError("transformers not installed or CLIPProcessor import failed")
        # 训练时必须开启 do_normalize=True
        # Explicitly set use_fast=True to avoid warning and use Rust implementation if available
        self.processor = CLIPProcessor.from_pretrained(vision_model_path, use_fast=True)

        print(f"[VLMDataset] 已加载图像索引: {self.num_images} 张图片")

    def get_image(self, seq_idx):
        """
        根据序列索引获取预处理后的图像 Tensor
        """
        if seq_idx >= len(self.seq_to_img_idx):
            return None
            
        img_idx = self.seq_to_img_idx[seq_idx]
        if img_idx == -1:
            # 如果没有对应的图片，返回全黑图占位 (必须与正常图片维度一致)
            # 假设 CLIP 输入为 224x224
            return torch.zeros((3, 224, 224), dtype=torch.float32)
            
        # 获取原始 Bytes
        offset = self.img_offsets[img_idx]
        length = self.img_lengths[img_idx]
        img_bytes = self.img_data[offset : offset + length]
        
        # 在线处理：Bytes -> PIL -> Tensor
        try:
            image = Image.open(io.BytesIO(img_bytes))
            # 调用 Model 中的静态方法处理
            from src.model.model_vlm import VisualVV
            pixel_values = VisualVV.image2tensor(image, self.processor).squeeze(0)
            return pixel_values
        except Exception as e:
            print(f"[Error] Image decode failed for seq_idx {seq_idx}, img_idx {img_idx}: {e}")
            # 返回全黑图防止 Crash
            return torch.zeros((3, 224, 224))

class VLMPretrainDataset(PretrainDataset, VLMDatasetMixin):
    def __init__(self, data_path, vision_model_path):
        super().__init__(data_path)
        self.init_vlm_data(data_path, vision_model_path)
        
        # 获取图像占位符 ID 以便在 Label 中进行 Mask
        try:
            from configs.model import VisualVVConfig
            # 假设所有 image_ids 都是相同的 (e.g. 34)
            self.image_token_id = VisualVVConfig().image_ids[0]
        except ImportError:
            print("[Warning] Could not import VisualVVConfig. Defaulting image_token_id to 34.")
            self.image_token_id = 34

    def __getitem__(self, idx):
        X, Y = super().__getitem__(idx)
        pixel_values = self.get_image(idx)
        
        # Mask 掉 Label 中的图像占位符
        # 防止模型学习预测图像占位符 (e.g. '@')
        if self.image_token_id is not None:
            # IMPORTANT: clone Y before modification because X and Y share memory (numpy view)
            # Modifying Y in-place would also modify X (input_ids), causing invalid tokens (-100) in input
            Y = Y.clone()
            Y[Y == self.image_token_id] = -100
            
        return X, Y, pixel_values

class VLMSFTDataset(SFTDataset, VLMDatasetMixin):
    def __init__(self, data_path, tokenizer, vision_model_path, max_length=512):
        super().__init__(data_path, tokenizer, max_length)
        self.init_vlm_data(data_path, vision_model_path)

    def __getitem__(self, idx):
        X, Y = super().__getitem__(idx)
        pixel_values = self.get_image(idx)
        return X, Y, pixel_values

class TokenBucketSampler(Sampler):
    """
    基于 Token 数量的 Batch Sampler。
    目标是让每个 Batch 的总 Token 数尽可能接近 max_tokens，从而实现动态 Batch Size。
    """
    def __init__(self, dataset, max_tokens):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.lengths = self._get_lengths(dataset)

    def _get_lengths(self, dataset):
        """递归获取数据集长度，支持嵌套的 Subset"""
        if hasattr(dataset, 'lengths'): return dataset.lengths
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            base = self._get_lengths(dataset.dataset)
            return base[dataset.indices]
        return None

    def __iter__(self):
        # 1. 智能排序：按长度聚类以减少 Padding
        indices = np.argsort(self.lengths)
        # 2. 贪婪分桶：动态构建 Batch
        batches, batch, max_len = [], [], 0
        for idx in indices:
            curr_len = self.lengths[idx]
            if batch and (len(batch) + 1) * max(max_len, curr_len) > self.max_tokens:
                batches.append(batch)
                batch, max_len = [], 0
            batch.append(idx)
            max_len = max(max_len, curr_len)
        # 3. 随机打乱 Batch 顺序并输出 (丢弃最后一个不满的 Batch)
        np.random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        # 注意：这里的长度只是近似的，因为动态 Batch Size 每次迭代可能不同
        # 但为了 DataLoader 的兼容性，我们需要返回一个值
        # 这里我们预估一个平均 Batch Size
        avg_len = np.mean(self.lengths)
        # 平均每个 Batch 包含的样本数
        avg_batch_size = max(1, self.max_tokens // avg_len)
        return int(np.ceil(len(self.dataset) / avg_batch_size))

def dynamic_collate_fn(batch, padding_value=0):
    """
    动态整理函数：处理 Dataset 返回的 (X, Y) 或 (X, Y, pixel_values) 元组列表。
    优化后使用 pad_sequence 进行高效填充。
    """
    # batch 是一个列表: [(X1, Y1), (X2, Y2), ...] 或 [(X1, Y1, P1), (X2, Y2, P2), ...]
    # 分离 X 和 Y
    X_list = [item[0] for item in batch]
    Y_list = [item[1] for item in batch]
    
    # 检查是否包含 pixel_values
    pixel_values_list = None
    if len(batch[0]) > 2:
        pixel_values_list = [item[2] for item in batch]

    # 使用 pad_sequence 自动处理填充
    # batch_first=True 使得输出形状为 [batch_size, max_len]
    input_ids = rnn_utils.pad_sequence(X_list, batch_first=True, padding_value=padding_value)
    labels = rnn_utils.pad_sequence(Y_list, batch_first=True, padding_value=-100)
    
    batch_dict = {
        "input_ids": input_ids,
        "labels": labels
    }
    
    if pixel_values_list is not None:
        # 安全检查：过滤 None 或 异常值
        safe_pixel_values = []
        for i, pv in enumerate(pixel_values_list):
            if pv is None:
                # 理论上 get_image 不会返回 None，但为了防御性编程
                # print(f"[Warning] pixel_values at index {i} is None. Replacing with zeros.")
                safe_pixel_values.append(torch.zeros((3, 224, 224), dtype=torch.float32))
            elif torch.isnan(pv).any() or torch.isinf(pv).any():
                print(f"[Warning] pixel_values at index {i} contains NaN/Inf. Replacing with zeros.")
                safe_pixel_values.append(torch.zeros((3, 224, 224), dtype=torch.float32))
            else:
                # 强制转换为 float32 并确保内存连续
                safe_pixel_values.append(pv.to(dtype=torch.float32).contiguous())

        # Stack 变成 [Batch, 3, 224, 224]
        pixel_values = torch.stack(safe_pixel_values)
        # 增加 num_images 维度: [Batch, 1, 3, 224, 224] 以匹配 VisualVV.forward 的期望
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
        batch_dict["pixel_values"] = pixel_values

    return batch_dict
