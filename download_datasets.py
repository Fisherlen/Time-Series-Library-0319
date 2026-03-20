"""
数据集下载脚本 - 从Hugging Face下载Time-Series-Library数据集到本地
"""
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

HUGGINGFACE_REPO = "thuml/Time-Series-Library"
DATASET_DIR = r"l:\AI\Trae\Time-Series-Library-0319-Doubao\dataset"

def download_ett_datasets():
    """下载ETT系列数据集 (ETTh1, ETTh2, ETTm1, ETTm2)"""
    ett_datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
    
    for ds_name in ett_datasets:
        print(f"正在下载 {ds_name}...")
        try:
            ds = load_dataset(HUGGINGFACE_REPO, name=ds_name)
            df = ds["train"].to_pandas()
            save_path = os.path.join(DATASET_DIR, f"{ds_name}.csv")
            df.to_csv(save_path, index=False)
            print(f"已保存: {save_path} (行数: {len(df)})")
        except Exception as e:
            print(f"下载 {ds_name} 失败: {e}")

def download_custom_datasets():
    """下载其他常用数据集 (ECL, Traffic, Weather, National_Illness, Exchange_Rate)"""
    custom_datasets = ['electricity', 'traffic', 'weather', 'national_illness', 'exchange_rate']
    
    for ds_name in custom_datasets:
        print(f"正在下载 {ds_name}...")
        try:
            ds = load_dataset(HUGGINGFACE_REPO, name=ds_name)
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            df = ds[split_name].to_pandas()
            save_path = os.path.join(DATASET_DIR, f"{ds_name}.csv")
            df.to_csv(save_path, index=False)
            print(f"已保存: {save_path} (行数: {len(df)})")
        except Exception as e:
            print(f"下载 {ds_name} 失败: {e}")

def download_psm_dataset():
    """下载PSM数据集"""
    print("正在下载 PSM...")
    try:
        ds_data = load_dataset(HUGGINGFACE_REPO, name="PSM-data")
        ds_label = load_dataset(HUGGINGFACE_REPO, name="PSM-label")
        
        train_df = ds_data["train"].to_pandas()
        test_df = ds_data["test"].to_pandas()
        test_label_df = ds_label[next(iter(ds_label))].to_pandas()
        
        psm_dir = os.path.join(DATASET_DIR, "PSM")
        os.makedirs(psm_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(psm_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(psm_dir, "test.csv"), index=False)
        test_label_df.to_csv(os.path.join(psm_dir, "test_label.csv"), index=False)
        print(f"已保存 PSM 数据集到: {psm_dir}")
    except Exception as e:
        print(f"下载 PSM 失败: {e}")

def download_npy_dataset(name, files):
    """下载npy格式的数据集"""
    print(f"正在下载 {name}...")
    try:
        save_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        
        for file in files:
            # 使用load_dataset方式下载
            ds_data = load_dataset(HUGGINGFACE_REPO, name=f"{name}-data")
            ds_label = load_dataset(HUGGINGFACE_REPO, name=f"{name}-label")
            
            # 保存训练数据
            train_data = ds_data["train"].to_pandas().values[:, 1:]  # 跳过第一列（索引）
            test_data = ds_data["test"].to_pandas().values[:, 1:]
            
            # 获取标签数据
            label_split = "train" if "train" in ds_label else list(ds_label.keys())[0]
            test_label = ds_label[label_split].to_pandas().values[:, 1:]
            
            # 保存为npy格式
            np.save(os.path.join(save_dir, f"{name}_train.npy"), train_data)
            np.save(os.path.join(save_dir, f"{name}_test.npy"), test_data)
            np.save(os.path.join(save_dir, f"{name}_test_label.npy"), test_label)
            
            print(f"已保存 {name} 数据集到: {save_dir}")
            print(f"  - train形状: {train_data.shape}, test形状: {test_data.shape}, label形状: {test_label.shape}")
            break  # 只需要下载一次
    except Exception as e:
        print(f"下载 {name} 失败: {e}")

def download_swat_dataset():
    """下载SWaT数据集"""
    print("正在下载 SWaT...")
    try:
        ds = load_dataset(HUGGINGFACE_REPO, name="SWaT")
        train_df = ds["train"].to_pandas()
        test_df = ds["test"].to_pandas()
        
        swat_dir = os.path.join(DATASET_DIR, "SWAT")
        os.makedirs(swat_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(swat_dir, "swat_train2.csv"), index=False)
        test_df.to_csv(os.path.join(swat_dir, "swat2.csv"), index=False)
        print(f"已保存 SWaT 数据集到: {swat_dir}")
    except Exception as e:
        print(f"下载 SWaT 失败: {e}")

def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"数据集将保存到: {DATASET_DIR}")
    print("=" * 60)
    
    # 1. 下载ETT系列数据集
    print("\n[1/6] 下载ETT系列数据集...")
    download_ett_datasets()
    
    # 2. 下载Custom系列数据集
    print("\n[2/6] 下载常用自定义数据集...")
    download_custom_datasets()
    
    # 3. 下载PSM数据集
    print("\n[3/6] 下载PSM数据集...")
    download_psm_dataset()
    
    # 4. 下载MSL数据集
    print("\n[4/6] 下载MSL数据集...")
    download_npy_dataset("MSL", ["MSL_train.npy", "MSL_test.npy", "MSL_test_label.npy"])
    
    # 5. 下载SMAP数据集
    print("\n[5/6] 下载SMAP数据集...")
    download_npy_dataset("SMAP", ["SMAP_train.npy", "SMAP_test.npy", "SMAP_test_label.npy"])
    
    # 6. 下载SMD数据集
    print("\n[6/6] 下载SMD数据集...")
    download_npy_dataset("SMD", ["SMD_train.npy", "SMD_test.npy", "SMD_test_label.npy"])
    
    # 额外: SWaT数据集
    print("\n[额外] 下载SWaT数据集...")
    download_swat_dataset()
    
    print("\n" + "=" * 60)
    print("数据集下载完成!")
    print(f"所有数据集已保存到: {DATASET_DIR}")

if __name__ == "__main__":
    main()
