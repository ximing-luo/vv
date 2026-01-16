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
import numpy as np
from torch.utils.data import Dataset, Sampler
import struct

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

class TokenBucketSampler(Sampler):
    """
    基于 Token 数量的 Batch Sampler。
    目标是让每个 Batch 的总 Token 数尽可能接近 max_tokens，从而实现动态 Batch Size。
    """
    def __init__(self, dataset, max_tokens, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 预先计算每个样本的长度
        # 优化：优先从数据集获取预存的长度，避免重复迭代
        self.lengths = self._get_lengths(dataset)
        if self.lengths is None:
            print(f"[TokenBucketSampler] 警告: 数据集未预存长度，正在实时计算 (这可能比较慢)...")
            self.lengths = [len(dataset[i]) for i in range(len(dataset))]

    def _get_lengths(self, dataset):
        """递归获取数据集长度，支持嵌套的 Subset"""
        if hasattr(dataset, 'lengths'):
            return dataset.lengths
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            base_lengths = self._get_lengths(dataset.dataset)
            if base_lengths is not None:
                return [base_lengths[i] for i in dataset.indices]
        return None

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        
        if self.shuffle:
            # 优化：为了减少 Padding 浪费，我们先按长度进行“粗略排序”
            # 但为了保持一定的随机性，我们可以给长度加上一点随机扰动，或者分块排序
            # 这里采用“排序后分桶”的策略
            lengths = np.array(self.lengths)
            # 获取排序后的索引
            indices = indices[np.argsort(lengths)]
            
            # 为了防止模型总是先学短句再学长句，我们将排序后的序列切成小块进行局部打乱
            # 或者更简单的：直接分组后打乱 Batch 顺序
            
        batches = []
        current_batch = []
        max_len_in_batch = 0
        
        for idx in indices:
            seq_len = self.lengths[idx]
            # 预估加入该样本后的 Batch Token 数 (以当前最大长度为准)
            temp_max_len = max(max_len_in_batch, seq_len)
            estimated_tokens = (len(current_batch) + 1) * temp_max_len
            
            if not current_batch:
                current_batch.append(idx)
                max_len_in_batch = seq_len
                continue
            
            if estimated_tokens <= self.max_tokens:
                current_batch.append(idx)
                max_len_in_batch = temp_max_len
            else:
                batches.append(current_batch)
                current_batch = [idx]
                max_len_in_batch = seq_len
                
        if current_batch and not self.drop_last:
            batches.append(current_batch)
            
        if self.shuffle:
            # 关键：打乱 Batch 的顺序，确保训练的随机性
            # 此时每个 Batch 内部的样本长度是接近的，但 Batch 出现的顺序是随机的
            np.random.shuffle(batches)
            
        for batch in batches:
            yield batch

    def __len__(self):
        # 注意：这里的长度只是近似的，因为动态 Batch Size 每次迭代可能不同
        # 但为了 DataLoader 的兼容性，我们需要返回一个值
        # 这里我们预估一个平均 Batch Size
        avg_len = np.mean(self.lengths)
        avg_batch_size = max(1, self.max_tokens // avg_len)
        return int(np.ceil(len(self.dataset) / avg_batch_size))

def dynamic_collate_fn(batch, padding_value=0):
    """
    动态整理函数：处理 Dataset 返回的 (X, Y) 元组列表。
    """
    # batch 是一个列表: [(X1, Y1), (X2, Y2), ...]
    # 1. 分离 X 和 Y
    X_list = [item[0] for item in batch]
    Y_list = [item[1] for item in batch]
    
    # 2. 获取当前 batch 的最大长度
    max_len = max(len(x) for x in X_list)
    batch_size = len(batch)
    
    # 3. 创建填充后的 Tensor
    # 注意：Y 的填充值应该是 -100 (CrossEntropy 忽略索引)
    input_ids = torch.full((batch_size, max_len), padding_value, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    
    # 4. 填充数据
    for i in range(batch_size):
        length = len(X_list[i])
        input_ids[i, :length] = X_list[i]
        labels[i, :length] = Y_list[i]
    
    return {
        "input_ids": input_ids,
        "labels": labels
    }
