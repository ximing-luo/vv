# QWEN.md - 项目上下文指南

## 项目概述

**从 0 到 VLM** - 一个手写实现 LLM 到 VLM 演进的教育型深度学习项目。

本项目通过迭代演进的方式，从最简单的 LLM（多头自注意力 + 前馈神经网络）开始，逐步实现：
- **注意力机制演进**：MHA → Flash Attention → GQA → DeepSeek MLA
- **前馈网络演进**：MLP → SwiGLU → MoE → 共享专家 MoE → DeepSeek 隐式辅助损失 MoE
- **训练流程演进**：手写训练循环 → Hugging Face Transformers Trainer
- **分词器演进**：字符级分词器 → BPE 分词器（Hugging Face Tokenizers）
- **多模态扩展**：LLM 基础 + 视觉投影层 → VLM

## 技术栈

- **核心框架**: PyTorch >= 2.6.0, Transformers >= 4.57.0
- **加速库**: Accelerate, Flash Attention (可选)
- **数据处理**: Datasets, Tokenizers
- **可视化**: TensorBoard
- **语言**: Python

## 目录结构

```
vv/
├── configs/
│   └── model.py          # 模型配置类 (VVConfig, VisualVVConfig)
├── scripts/
│   └── train_from_scratch.py  # 完整训练流水线脚本
├── src/
│   ├── data/
│   │   ├── database/     # 原始数据集 (git-ignored)
│   │   ├── dataset/      # 预处理后的数据 (git-ignored)
│   │   ├── metadata/     # 采样清洗后的数据 (git-ignored)
│   │   ├── tools/        # 数据采样、清洗、分词器工具
│   │   ├── dataset.py    # 数据集类定义
│   │   ├── preprocess.py # LLM 数据预处理
│   │   ├── preprocess_vlm.py # VLM 数据预处理
│   │   └── tokenizer.py  # 分词器训练
│   ├── model/
│   │   ├── backbone/     # 核心组件实现
│   │   │   ├── attention.py  # 注意力机制 (MHA/GQA/MLA)
│   │   │   ├── moe.py        # MoE 实现 (SparseMoE/HybridMoE 等)
│   │   │   ├── rope.py       # RoPE/YaRN 位置编码
│   │   │   ├── transform.py  # Transformer 块
│   │   │   └── vision.py     # 视觉编码器
│   │   ├── model_llm.py  # LLM 模型定义
│   │   └── model_vlm.py  # VLM 模型定义
│   ├── training/         # 自定义 Trainer (DynamicTrainer)
│   ├── utils/            # 推理工具 (inference.py)
│   └── train.py          # 主训练入口
├── tests/                # 单元测试 (git-ignored)
├── models/               # 模型检查点和输出 (git-ignored)
│   ├── checkpoints/      # 各阶段检查点
│   └── vv/               # 最终模型输出
├── logs/                 # TensorBoard 日志 (git-ignored)
├── requirements.txt      # 依赖列表
└── readme.md             # 详细文档
```

## 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

**核心依赖版本要求**:
- torch >= 2.6.0
- transformers >= 4.57.0
- numpy >= 2.0.0

### 数据准备

数据集需下载到 `src/data/database/` 目录：

```bash
# 使用 modelscope 下载 Minimind 数据集
pip install modelscope
modelscope download --dataset gongjy/minimind_dataset pretrain_hq.jsonl --local_dir ./src/data/database/
modelscope download --dataset gongjy/minimind_dataset sft_512.jsonl --local_dir ./src/data/database/
```

或使用项目脚本一键下载：
```bash
python src/data/tools/download_dataset.py
```

### 训练流程

#### 完整流水线（从 0 到 VLM）

```bash
python scripts/train_from_scratch.py
```

**警告**: 此脚本会删除 `logs/`、`models/` 和已处理的数据。如需保留数据，注释掉 `delete_data()` 调用。

流水线包含：
1. 数据采样 (`sample()`)
2. 分词器训练 (`train_token()`)
3. LLM 数据预处理 (`preprocess()`)
4. VLM 数据预处理 (`preprocess_vlm()`)
5. 四阶段训练：
   - LLM 预训练
   - LLM 微调
   - VLM 预训练（冻结 LLM）
   - VLM 微调（解冻）

#### 单阶段训练

```python
from src.train import train

# LLM 预训练
train(mode='pretrain', is_vlm=False, num_train_epochs=1, eval_steps=500, save_steps=500)

# LLM 微调
train(mode='finetune', is_vlm=False, ...)

# VLM 预训练
train(mode='pretrain', is_vlm=True, ...)

# VLM 微调
train(mode='finetune', is_vlm=True, ...)
```

或直接运行：
```bash
python src/train.py  # 默认 LLM 预训练 0.1 epoch
```

### 推理测试

```python
from src.utils.inference import load_model, stream_inference

model, tokenizer, device = load_model('models/vv')
# 修改 inference.py 的 __main__ 调用 main() 进行交互式推理
```

### 监控训练

```bash
tensorboard --logdir logs
```

## 开发指南

### 代码约定

- **路径处理**: 使用 `pathlib.Path` 和绝对路径
- **配置管理**: 使用 `dataclass` 定义模型配置
- **日志**: 使用 `print` 输出训练状态，TensorBoard 记录指标
- **检查点**: 自动保存最近 3 个 checkpoint，训练结束加载最优模型

### Git 忽略路径

以下路径已被 `.gitignore` 忽略，**不要提交**:
- `logs/` - 训练日志
- `models/` - 模型检查点
- `src/data/database/` - 原始数据
- `src/data/dataset/` - 处理后的数据
- `src/data/metadata/` - 中间数据
- `tests/` - 测试输出
- `inference_output.txt` - 推理输出

### 测试

```bash
# 运行所有测试
python -m unittest discover tests

# 运行特定测试
python -m unittest tests.test_backbone
python -m unittest tests.test_models
```

### 常见问题排查

**CUDA 错误**:
```bash
# 调试模式（同步执行，便于定位错误）
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
```

**显存碎片化**: 已配置 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

**BF16 支持**: 自动检测，仅 Ampere+(compute capability >= 8) 启用 BF16，否则使用 FP16

## 模型配置说明

### VVConfig (LLM 配置)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_dim` | 576 | 隐藏层维度 |
| `n_layer` | 12 | Transformer 层数 |
| `n_head` | 6 | 注意力头数 |
| `n_kv_head` | 3 | KV 头数 (GQA) |
| `max_seq_len` | 512 | 最大序列长度 |
| `kv_lora_rank` | 128 | MLA KV 压缩秩 |
| `q_lora_rank` | 96 | MLA Query 压缩秩 |
| `num_experts` | 4 | MoE 总专家数 |
| `num_shared_experts` | 1 | 共享专家数 |
| `rope_scale` | 1.0 | RoPE/YaRN 扩展倍数 |

### VisualVVConfig (VLM 配置)

继承 `VVConfig`，额外包含：
- `vision_hidden_dim`: 视觉模型隐藏层维度 (768)
- `vision_model_path`: CLIP 模型路径
- `image_special_token`: 图像占位符文本
- `image_ids`: 图像占位符 token ID

## 相关项目

- [LLMs-Zero-to-Hero](https://github.com/bbruceyuan/LLMs-Zero-to-Hero)
- [Minimind](https://github.com/jingyaogong/minimind)
- [My_LLM](https://github.com/REXWindW/my_llm)
