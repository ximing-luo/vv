# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese-language project that implements training of LLMs (Large Language Models) from scratch, evolving through different attention mechanisms, feed-forward networks, and tokenizers, ultimately adding vision projection layers to create a VLM (Vision Language Model). The project emphasizes iterative evolution from simple components to advanced architectures.

Key evolutionary steps:
- **Attention**: Multi-head attention → Flash Attention → Grouped Query Attention (GQA) → DeepSeek's low-rank latent attention (MLA)
- **Feed-forward networks**: MLP → SwiGLU → standard MoE → shared-expert MoE → auxiliary-loss MoE → DeepSeek's implicit auxiliary loss MoE
- **Training pipeline**: Handwritten training loops → Hugging Face Transformers Trainer
- **Tokenizer**: Character-level → BPE via Hugging Face Tokenizers
- **Multimodal**: Added visual projection layers to LLM base for VLM

## Directory Structure

- `configs/` – Model configuration dataclasses (`VVConfig`, `VisualVVConfig`)
- `src/`
  - `data/` – Data pipeline: sampling, cleaning, tokenizer training, preprocessing
    - `database/` – Raw datasets (git-ignored, download separately)
    - `dataset/` – Processed data (git-ignored)
    - `metadata/` – Sampled/cleaned data (git-ignored)
    - `tools/` – Data sampling, cleaning, vocabulary processing utilities
  - `model/` – Core model implementations
    - `backbone/` – Evolutionary implementations of attention, MoE, transformer blocks
    - `model_llm.py` – LLM model definition
    - (Other model files)
  - `training/` – Custom Trainer (`DynamicTrainer`) and training utilities
  - `utils/` – Inference utilities (`inference.py`)
  - `train.py` – Main training entry point
- `scripts/`
  - `train_from_scratch.py` – Full pipeline script (deletes logs/models/data, runs sampling, tokenizer training, preprocessing, and all four training stages)
- `tests/` – Unit tests for backbone modules and models
- `models/` – Checkpoints and final models (git-ignored)
  - `checkpoints/` – Training checkpoints per stage (llm_pretrain, llm_finetune, vlm_pretrain, vlm_finetune)
  - `vv/` – Final trained model output
- `logs/` – TensorBoard logs (git-ignored)

## Common Development Tasks

### Environment Setup
```bash
pip install -r requirements.txt
```
Requirements: torch>=2.6.0, transformers>=4.57.0, datasets>=4.4.0, accelerate>=1.12.0, tokenizers>=0.22.0, numpy>=2.2.0, tqdm, tensorboard.

### Data Preparation
Download datasets to `src/data/database/` (see README.md for details). Use `modelscope` SDK or `src/data/tools/download_dataset.py`.

### Full Training Pipeline
```bash
python scripts/train_from_scratch.py
```
**Warning**: This script deletes logs, models, and processed data before running. Comment out `delete_data()` calls as needed.

The pipeline includes:
1. Data sampling (`sample()`)
2. Tokenizer training (`train_token()`)
3. Preprocessing for LLM and VLM (`preprocess()`, `preprocess_vlm()`)
4. Four training stages:
   - LLM pretrain
   - LLM finetune
   - VLM pretrain (with LLM frozen)
   - VLM finetune (unfrozen)

### Individual Training Stages
Use `src/train.py`'s `train()` function:
```python
from src.train import train
train(mode='pretrain', is_vlm=False, num_train_epochs=1, eval_steps=500, save_steps=500)
train(mode='finetune', is_vlm=False, ...)
train(mode='pretrain', is_vlm=True, ...)
train(mode='finetune', is_vlm=True, ...)
```

Alternatively, run `python src/train.py` (defaults to LLM pretrain with 0.1 epochs).

### Testing
Run all unit tests:
```bash
python -m unittest discover tests
```
Run specific test file:
```bash
python -m unittest tests.test_backbone
python -m unittest tests.test_models
```

### Inference
The `src/utils/inference.py` module provides streaming inference. By default, its `__main__` calls `test()` which runs a suite of chat, continuation, and VLM tests. For interactive inference, modify `__main__` to call `main()` instead.

Load a trained model:
```python
from src.utils.inference import load_model, stream_inference
model, tokenizer, device = load_model('models/vv')
```

### Monitoring
TensorBoard logs are written to `logs/`. Run:
```bash
tensorboard --logdir logs
```

## Architecture Notes

### Model Configuration
- `VVConfig` (in `configs/model.py`) defines base LLM architecture.
- `VisualVVConfig` adds vision projection settings (CLIP path, image placeholder tokens).
- The model uses RoPE/YaRN scaling with `rope_scale` parameter.

### Model Implementation
- `src/model/` contains `VV` (LLM) and `VisualVV` (VLM) classes.
- Backbone modules in `src/model/backbone/` implement evolutionary variants:
  - Attention: `MultiHeadAttention`, `GroupedQueryAttention`, `LatentAttention`
  - MoE: `SparseMoE`, `HybridMoE`, `SoftBalancedMoE`, `SelfAdaptiveMoE`
  - Transformer blocks: `StandardBlock`, `AdvancedBlock`, `DeepSeekV2Block`, `DeepSeekV3Block`

### Training
- Uses Hugging Face `Trainer` with custom `DynamicTrainer` (`src/training/`).
- CUDA memory optimization: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.
- Automatic batch size and gradient accumulation step calculation.
- Four distinct stages with checkpoint chaining:
  - LLM pretrain → LLM finetune → VLM pretrain → VLM finetune
- Each stage can resume from the latest checkpoint of the previous stage.

### Data Pipeline
1. **Sampling**: `DataSampler` and `VLMSampler` sample raw datasets into `metadata/`.
2. **Tokenizer training**: BPE tokenizer trained on sampled data.
3. **Preprocessing**: Convert sampled text into tokenized binaries (`data_llm/`, `data_vlm/`).

## Important Conventions

- **Git-ignored paths**: `logs/`, `models/`, `src/data/database/`, `src/data/dataset/`, `src/data/metadata/`, `tests/`, `inference_output.txt`. Do not commit these.
- **Model checkpoints**: Staged in `models/checkpoints/{llm,vlm}_{pretrain,finetune}`.
- **Final models**: Saved to `models/vv` (overwrites previous).
- **Tokenizer**: Saved to `src/data/dataset/tokenizer`.
- **Code style**: No explicit linter configuration; follow existing patterns.
- **Testing**: Uses standard `unittest`; test files mirror module structure.

## Troubleshooting

- **CUDA errors**: Enable `CUDA_LAUNCH_BLOCKING` and `TORCH_USE_CUDA_DSA` environment variables for debugging (see comments in `train.py`).
- **Memory fragmentation**: Already configured with `max_split_size_mb:128`.
- **Transformers safety check**: Bypassed due to CVE-2025-32434 (see `train.py`).
- **BF16 support**: Automatically enabled only for compute capability >= 8 (Ampere+), otherwise FP16.