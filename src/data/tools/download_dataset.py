import os
from modelscope.hub.snapshot_download import snapshot_download

# 使用动态路径获取数据库目录
# __file__ 是当前脚本的路径：src/data/tools/download_dataset.py
# os.path.dirname(__file__) 是 src/data/tools
# 目标路径是 src/data/database
current_dir = os.path.dirname(os.path.abspath(__file__))
database_dir = os.path.abspath(os.path.join(current_dir, '..', 'database'))

def download_minimind_v():
    """
    下载 minimind-v 数据集到指定的数据库目录
    """
    dataset_id = 'gongjy/minimind-v_dataset'
    
    print(f"正在准备下载数据集: {dataset_id}")
    print(f"目标存储目录: {database_dir}")
    
    # 确保目录存在
    if not os.path.exists(database_dir):
        os.makedirs(database_dir)
        print(f"创建目录: {database_dir}")

    try:
        # 执行下载
        # cache_dir 指定下载位置
        # repo_type='dataset' 指定下载的是数据集
        snapshot_download(
            dataset_id, 
            cache_dir=database_dir, 
            repo_type='dataset'
        )
        print("\n[成功] 数据集已成功下载并保存到本地。")
        print(f"文件位置: {os.path.join(database_dir, dataset_id.replace('/', os.sep))}")
        
    except Exception as e:
        print(f"\n[错误] 下载过程中出现问题: {e}")

if __name__ == "__main__":
    download_minimind_v()
