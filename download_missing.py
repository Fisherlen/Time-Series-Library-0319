"""
下载缺失的数据集
"""
import os
import numpy as np
from datasets import load_dataset

HUGGINGFACE_REPO = "thuml/Time-Series-Library"
DATASET_DIR = r"l:\AI\Trae\Time-Series-Library-0319-Doubao\dataset"

def download_anomaly_dataset(name):
    """下载异常检测数据集 (MSL, SMAP, SMD)"""
    print(f"正在下载 {name}...")
    try:
        save_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用load_dataset方式下载
        ds_data = load_dataset(HUGGINGFACE_REPO, name=f"{name}-data")
        ds_label = load_dataset(HUGGINGFACE_REPO, name=f"{name}-label")
        
        # 保存训练数据
        train_data = ds_data["train"].to_pandas().values[:, 1:].astype(np.float64)
        test_data = ds_data["test"].to_pandas().values[:, 1:].astype(np.float64)
        
        # 获取标签数据
        label_split = list(ds_label.keys())[0]
        test_label = ds_label[label_split].to_pandas().values[:, 1:].astype(np.float64)
        
        # 保存为npy格式
        np.save(os.path.join(save_dir, f"{name}_train.npy"), train_data)
        np.save(os.path.join(save_dir, f"{name}_test.npy"), test_data)
        np.save(os.path.join(save_dir, f"{name}_test_label.npy"), test_label)
        
        print(f"已保存 {name} 数据集到: {save_dir}")
        print(f"  - train形状: {train_data.shape}, test形状: {test_data.shape}, label形状: {test_label.shape}")
    except Exception as e:
        print(f"下载 {name} 失败: {e}")

def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"数据集将保存到: {DATASET_DIR}")
    print("=" * 60)
    
    # 下载异常检测数据集
    for name in ['MSL', 'SMAP', 'SMD']:
        download_anomaly_dataset(name)
    
    print("\n" + "=" * 60)
    print("数据集下载完成!")

if __name__ == "__main__":
    main()
