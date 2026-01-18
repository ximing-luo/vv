# Axon LLM Exploration (VV Project)

本项目旨在构建一个完整且能在个人电脑（PC）上实际训练的大语言模型（LLM）框架。项目的核心理念是**平衡实践、学习与探索**。

我们不仅关注模型的效果，更关注“它是如何被构建出来的”。通过自主实现核心组件（如分词器、注意力机制、MoE 架构等），深入理解大模型底层原理；同时，在训练工程化方面，我们积极拥抱 Hugging Face `transformers` 等成熟生态，避免重复造轮子，从而将精力集中在数据质量和模型结构的创新探索上。

---

## 🚀 项目核心亮点

- **自主实现的工业级组件**：手写了从数据采样、清洗到分词器（Tokenizer）训练的完整链路。
- **模型架构的演进式探索**：
  - **Attention 演进**：从基础的 [SingleHeadAttention](file:///d:/Axon/ANN/llm/vv/src/model/backbone/attention.py#L8) 到 [GQA](file:///d:/Axon/ANN/llm/vv/src/model/backbone/attention.py#L112) (Grouped Query Attention)，再到前沿的 [MLA](file:///d:/Axon/ANN/llm/vv/src/model/backbone/attention.py#L193) (Multi-Head Latent Attention)。
  - **MoE 演进**：从简单的 [MoE](file:///d:/Axon/ANN/llm/vv/src/model/backbone/moe.py#L46) 到带常驻专家的 [SharedMoE](file:///d:/Axon/ANN/llm/vv/src/model/backbone/moe.py#L94)，以及 DeepSeek 风格的 [AuxiliaryLossMoE](file:///d:/Axon/ANN/llm/vv/src/model/backbone/moe.py#L135)。
- **高效的训练工程化**：
  - 基于 HF `Trainer` 封装了 [DynamicTrainer](file:///d:/Axon/ANN/llm/vv/src/training/trainer.py#L49)，实现了 **Token 桶采样 (TokenBucketSampler)**，支持动态批处理，极大提升了 PC 端有限算力下的训练效率。
  - 内置多种回调机制，包括自动回退 [RollbackCallback](file:///d:/Axon/ANN/llm/vv/src/training/trainer.py#L67)、训练中实时推理模拟 [InferenceCallback](file:///d:/Axon/ANN/llm/vv/src/training/trainer.py#L69) 等。

---

## 🛠️ 模块介绍

### 1. 数据处理与分词 (Data & Tokenization)
项目在 [src/data](file:///d:/Axon/ANN/llm/vv/src/data) 下实现了完整的数据管道：
- **[tokenizer.py](file:///d:/Axon/ANN/llm/vv/src/data/tokenizer.py)**: 自主训练 BPE 分词器，支持 Byte-level 映射，并能导出人类可读的词表文件供调试。
- **数据清洗与采样**: 在 `tools` 目录下提供了 [clean_data.py](file:///d:/Axon/ANN/llm/vv/src/data/tools/clean_data.py) 和 [sample.py](file:///d:/Axon/ANN/llm/vv/src/data/tools/sample.py)，用于处理原始语料，平衡数据分布。

### 2. 模型骨架 (Model Backbone)
在 [src/model/backbone](file:///d:/Axon/ANN/llm/vv/src/model/backbone) 中，你可以看到模型核心组件的迭代过程：
- **Attention**: 实现了包括 RoPE (旋转位置编码) 结合的 [GroupedQueryAttention](file:///d:/Axon/ANN/llm/vv/src/model/backbone/attention.py#L112) 和低秩压缩的 [MLA](file:///d:/Axon/ANN/llm/vv/src/model/backbone/attention.py#L193)。
- **MoE**: 实现了细粒度专家路由和负载均衡损失（Auxiliary Loss），旨在提高模型参数效率。
- **基础组件**: 包含 [RMSNorm](file:///d:/Axon/ANN/llm/vv/src/model/backbone/rms.py) 和 [SwiGLU](file:///d:/Axon/ANN/llm/vv/src/model/backbone/moe.py#L19) 等现代 LLM 的标配。

### 3. 训练与优化 (Training)
训练逻辑位于 [src/training](file:///d:/Axon/ANN/llm/vv/src/training)：
- **[trainer.py](file:///d:/Axon/ANN/llm/vv/src/training/trainer.py)**: 继承并扩展了 Hugging Face 的训练器，通过自定义 `get_train_dataloader` 实现动态长度样本的高效堆叠，减少 Padding 浪费。
- **系统工具**: [system.py](file:///d:/Axon/ANN/llm/vv/src/training/tools/system.py) 等工具提供了环境优化和键盘中断监控，让 PC 端长时间训练更可控。

---

## 📁 项目目录结构

```text
.
├── configs/            # 模型架构与训练超参数配置 (Recipe)
├── models/             # 存放训练好的模型权重
├── scripts/            # 训练启动脚本
├── src/
│   ├── data/           # 数据加载、清洗、采样及分词器实现
│   ├── model/          # 模型架构，含 backbone 核心组件
│   ├── training/       # 训练器实现及各种 Callback 工具
│   └── utils/          # 通用工具函数
├── tests/              # 测试代码
└── readme.md           # 本文件
```

---

## 🎯 愿景
我们相信，学习大模型的最好方式是“手写一遍”，而不仅仅是调用 API。本项目将持续探索数据与模型架构的边界，同时利用工业界成熟的训练工具，让每个人都能在自己的电脑上拥有一台“大模型实验室”。
