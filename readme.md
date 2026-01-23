# 从0到VLM

手写了llm到vlm的模型和数据处理，一种演进迭代的思路，从一个最简单的llm开始（多头自注意力+前馈神经网络），数据就几本小说，训练流程手写，先跑起来再说。

## 演进迭代

- **自注意力层**：从单头/多头注意力(MHA)加入Flash Attention加速，替换为分组注意力GQA，最后演进为DeepSeek的低秩潜在注意力MLA。
- **前馈网络**：从全连接层(MLP)替换到SwiGLU，再到普通MoE、共享专家MoE、带辅助损失MoE，最后实现DeepSeek的非显式辅助损失MoE。
- **训练流程**：从纯手写训练循环迭到调用Transformers库的Trainer实现（早期手写部分已移除）。
- **分词器**：从自实现的字符级分词器迭代为基于Hugging Face Tokenizers库的BPE分词器。
- **多模态**：最后在LLM基础上增加视觉投影层，实现了VLM。

## 项目目录

本项目代码主要集中在模型搭建和数据处理方面：
```text
├── configs/
├── logs/
├── models/
├── scripts/
│   ├── train_from_scratch.py # 从0到vlm脚本（包含删除日志模型、训练分词器）
├── src/
│   ├── data/
│   │   ├── database/ # 下载的数据集、小说
│   │   ├── dataset/ # 预处理后的数据
│   │   ├── metadata/ # 采样清洗后的数据
│   │   ├── tools/ # 采样、清洗、词表处理
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   ├── tokenizer.py
│   │   └── ...
│   ├── model/
│   │   ├── backbone/ # 自注意力层和前馈网络演进实现
│   │   ├── model_llm.py
│   │   └── ...
│   ├── training/ # transformer库的Trainer实现
│   ├── utils/ # 调用模型推理、测试
│   └── train.py # 主训练入口
├── .gitignore
├── inference_output.txt # 调用模型测试的输出
├── readme.md
└── requirements.txt
```

## 关于复现

如果想要复现，主要面临两个门槛：

1.  **数据对齐**：数据集下载需要和我保持一致，或者你需要自己手写一套采样和预处理流程（毕竟数据是模型的粮食）。
2.  **环境依赖**：由于采用了 `transformers` 库的 Trainer 进行训练，我使用的版本较新，顺带着 `numpy` 和 `torch` 的版本要求也水涨船高。所以不建议在 `torch < 2.6.0` 或 `numpy < 2.0.0` 的环境下强行复现。

> 后面看情况，我可能会出一个在服务器上复现的教程，直接选用 `pytorch >= 2.6.0` 的镜像，复现起来基本没有什么依赖冲突。

如果你对手写模型训练代码感兴趣，强烈建议看看以下优秀项目：
- [LLMs-Zero-to-Hero](https://github.com/bbruceyuan/LLMs-Zero-to-Hero)
- [Minimind](https://github.com/jingyaogong/minimind)
- [My_LLM](https://github.com/REXWindW/my_llm)

本项目还在更新阶段，欢迎大家一起研究。

## 数据集下载

如果你看到这里还是对本项目感兴趣，欢迎下载我用到的数据集到 `src/data/database/` 目录下，亲自体验从 0 训练 VLM 的感觉。

### 项目数据结构

```text
data
├── database
│   ├── gongjy                    # minimind-v 的数据集
│   ├── novel                     # 小说数据集（这个可以自己随便找几本）
│   ├── firefly-train-1.1M.jsonl  # 流萤指令微调数据集
│   ├── high_data.txt             # wudao 数据集
│   ├── multiturn_chat_0.8M.json  # 多轮对话数据集
│   ├── pretrain_hq.jsonl         # minimind 的预训练数据集
│   └── sft_512.jsonl             # minimind 的 SFT 数据集
├── dataset
│   ├── data_llm                  # 预处理后的 LLM 数据集
│   ├── data_vlm                  # 预处理后的 VLM 数据集
│   └── tokenizer                 # 分词器
└── metadata                      # 从 database 采样清洗后的数据
    ├── finetune
    ├── pretrain
    ├── vlm_finetune
    └── vlm_pretrain
```

### 下载指引

建议直接下载 **Minimind** 的数据集，数据量适中：
[Minimind Dataset Files](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)

您可以使用 `modelscope` SDK 轻松下载：

```bash
pip install modelscope
# 下载预训练数据
modelscope download --dataset gongjy/minimind_dataset pretrain_hq.jsonl --local_dir ./src/data/database/
# 下载 SFT 数据
modelscope download --dataset gongjy/minimind_dataset sft_512.jsonl --local_dir ./src/data/database/
```

关于 **Minimind-V** 的数据集（VLM 部分）：
[Minimind-V Dataset Files](https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset/files)

我也正在尝试用 SDK 标准化一键下载流程。目前您可以使用脚本 `src\data\tools\download_dataset.py` 将其一键下载到对应目录。后续我会继续优化，争取让所有数据集都能通过 SDK 一键搞定。

### 其他数据集

- **流萤指令微调数据集**：[Firefly Train 1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
- **多轮对话数据集**：[Multiturn Chat 0.8M](https://huggingface.co/datasets/erhwenkuo/multiturn_chat_0.8m-chinese-zhtw)
- **WuDao 数据集**：[High Data](https://huggingface.co/datasets/mdokl/WuDaoCorpora2.0-RefinedEdition60GTXT)
