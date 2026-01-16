import os
import sys
import shutil
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(root_path, 'src')
for path in [root_path, src_path]:
    if path not in sys.path:
        sys.path.append(path)
from src.data import DataSampler, clean_data, preprocess
from src.train import train

def sample():
    BASE_DATABASE_DIR = r'D:\Axon\ANN\llm\AGI\src\data\database'
    METADATA_ROOT_DIR = r'D:\Axon\ANN\llm\AGI\src\data\metadata'
    sampler = DataSampler(BASE_DATABASE_DIR, METADATA_ROOT_DIR)
    sampler.sample_wudao(target_gb=0.5, split_size_mb=20)
    sampler.sample_novel(target_gb=0.5, split_size_mb=20)
    sampler.sample_pretrain_minimind(target_gb=0.5, split_size_mb=20)

    sampler.sample_sft512(target_gb=0.2, split_size_mb=20)
    # sampler.sample_firefly(target_gb=0.1, split_size_mb=20)
    sampler.sample_chat(target_gb=0.2, split_size_mb=20)

def delete_data(paths_to_delete):
    """
    删除训练过程中生成的日志、检查点以及处理后的数据集
    """
    for path in paths_to_delete:
        if os.path.exists(path):
            print(f"正在删除: {path}")
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f"成功删除: {path}")
            except Exception as e:
                print(f"删除 {path} 时出错: {e}")
        else:
            print(f"路径不存在，跳过: {path}")


if __name__ == "__main__":
    paths_to_delete = [
        r'D:\Axon\ANN\llm\AGI\src\training\logs',
        r'D:\Axon\ANN\llm\AGI\src\training\checkpoints',
        r'D:\Axon\ANN\llm\AGI\src\data\dataset\pretrain',
        r'D:\Axon\ANN\llm\AGI\src\data\dataset\finetune',
        r'D:\Axon\ANN\llm\AGI\src\data\metadata'
    ]
    delete_data(paths_to_delete) # 如果需要清空数据，取消此行注释
    # clean_data()
    sample()
    preprocess(num_workers=4,
        pretrain_sample_ratio=1,
        mixed_sample_ratio=0.1,
        finetune_sample_ratio=1
        )
    train(mode='pretrain', num_train_epochs=1.5)
    train(mode='finetune', num_train_epochs=1)
