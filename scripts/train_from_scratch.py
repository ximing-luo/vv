import os
import shutil
import sys
from pathlib import Path
# 将项目根目录添加到 sys.path 以支持本地模块导入
root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
from src.data import DataSampler, VLMSampler, clean_data, preprocess, preprocess_vlm, train_tokenizer
from src.train import train

def delete_data(paths_to_delete):
    """
    删除训练过程中生成的日志、检查点以及处理后的数据集
    """
    for path in paths_to_delete:
        if not os.path.exists(path): continue
        try:
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            print(f"成功删除: {path}")
        except Exception as e:
            print(f"删除 {path} 时出错: {e}")
        print("删除完成")

data_path = root_path / 'src' / 'data'
paths_to_delete = [
    os.path.join(root_path, 'logs'),
    os.path.join(root_path, 'models', 'checkpoints'),
    os.path.join(root_path, 'models', 'vv'),
    os.path.join(data_path, 'dataset', 'data_vlm'),
    os.path.join(data_path, 'dataset', 'data_llm'),
    os.path.join(data_path, 'metadata')
]

def sample():
    BASE_DATABASE_DIR = os.path.join(data_path, 'database')
    METADATA_ROOT_DIR = os.path.join(data_path, 'metadata')
    sampler = DataSampler(BASE_DATABASE_DIR, METADATA_ROOT_DIR)
    vlm_sampler = VLMSampler(BASE_DATABASE_DIR, METADATA_ROOT_DIR)
    sampler.sample_wudao(target_gb=5, split_size_mb=20)
    sampler.sample_novel(target_gb=0.5, split_size_mb=20)
    sampler.sample_pretrain_minimind(target_gb=1.5, split_size_mb=20)

    sampler.sample_sft(filename="sft_mini_512.jsonl", target_gb=7, split_size_mb=20)
    sampler.sample_firefly(target_gb=1, split_size_mb=20)
    sampler.sample_chat(target_gb=1, split_size_mb=20)

    vlm_sampler.run_minimind_v_pipeline(target_gb=2, num_preview=5, split_size_mb=20)

def train_token():
    DATA_DIR = [os.path.join(data_path, 'metadata', 'pretrain'),
        os.path.join(data_path, 'metadata', 'finetune')
        ]
    TOKENIZER_DIR = os.path.join(data_path, 'dataset', 'tokenizer')
    # sample_rate: 随机采样比例 (针对文件)
    # max_gb: 采样文件后限制参与训练的总数据量
    train_tokenizer(
        DATA_DIR, TOKENIZER_DIR, 
        vocab_size=6400, 
        sample_rate=0.5, 
        max_gb=0.2
    )

def train_from_scratch():
    delete_data(paths_to_delete) # 如果需要清空数据，取消此行注释
    sample()
    train_token()
    preprocess(num_workers=4,
        pretrain_sample_ratio=1,
        mixed_sample_ratio=0.1,
        finetune_sample_ratio=1
        )
    preprocess_vlm(num_workers=4)
    train(mode='pretrain', is_vlm=False, num_train_epochs=1, eval_steps=500, save_steps=500)
    train(mode='finetune', is_vlm=False, num_train_epochs=1, eval_steps=500, save_steps=500)
    train(mode='pretrain', is_vlm=True, num_train_epochs=1, eval_steps=500, save_steps=500)
    train(mode='finetune', is_vlm=True, num_train_epochs=1, eval_steps=500, save_steps=500)


if __name__ == "__main__":
    # delete_data(paths_to_delete) # 如果需要清空数据，取消此行注释
    # sample()
    # train_token()
    # preprocess(num_workers=8,
    #     pretrain_sample_ratio=1,
    #     mixed_sample_ratio=0.1,
    #     finetune_sample_ratio=1
    #     )
    # preprocess_vlm(num_workers=8)

    train(mode='pretrain', is_vlm=False, num_train_epochs=1, eval_steps=500, save_steps=500, is_freeze_llm= False)
    train(mode='finetune', is_vlm=False, num_train_epochs=1, eval_steps=500, save_steps=500, is_freeze_llm= False)
    train(mode='pretrain', is_vlm=True, num_train_epochs=1, eval_steps=500, save_steps=500)
    train(mode='finetune', is_vlm=True, num_train_epochs=1, eval_steps=500, save_steps=500)
