"""
测试数据集是否可以正常加载
"""
import os
import sys
import pandas as pd
import numpy as np

DATASET_DIR = r"l:\AI\Trae\Time-Series-Library-0319-Doubao\dataset"

def test_csv_dataset(filename):
    """测试CSV格式数据集"""
    try:
        filepath = os.path.join(DATASET_DIR, filename)
        df = pd.read_csv(filepath)
        print(f"✓ {filename}: 形状 {df.shape}, 列名: {list(df.columns[:5])}..." if len(df.columns) > 5 else f"✓ {filename}: 形状 {df.shape}, 列名: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"✗ {filename}: 加载失败 - {e}")
        return False

def test_npy_dataset(dirname, files):
    """测试NPY格式数据集"""
    try:
        dirpath = os.path.join(DATASET_DIR, dirname)
        for f in files:
            filepath = os.path.join(dirpath, f)
            data = np.load(filepath)
            print(f"  ✓ {f}: 形状 {data.shape}, 数据类型: {data.dtype}")
        return True
    except Exception as e:
        print(f"  ✗ 加载失败 - {e}")
        return False

def test_dir_dataset(dirname, files):
    """测试目录中的CSV数据集"""
    try:
        dirpath = os.path.join(DATASET_DIR, dirname)
        for f in files:
            filepath = os.path.join(dirpath, f)
            df = pd.read_csv(filepath)
            print(f"  ✓ {f}: 形状 {df.shape}")
        return True
    except Exception as e:
        print(f"  ✗ 加载失败 - {e}")
        return False

def main():
    print("=" * 60)
    print("数据集测试")
    print(f"数据集目录: {DATASET_DIR}")
    print("=" * 60)
    
    # 1. 测试ETT系列数据集
    print("\n1. ETT系列数据集:")
    ett_files = ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv']
    for f in ett_files:
        test_csv_dataset(f)
    
    # 2. 测试其他CSV数据集
    print("\n2. 其他CSV数据集:")
    other_files = ['exchange_rate.csv', 'national_illness.csv', 'traffic.csv', 'weather.csv']
    for f in other_files:
        test_csv_dataset(f)
    
    # 3. 测试PSM数据集
    print("\n3. PSM数据集:")
    test_dir_dataset("PSM", ["train.csv", "test.csv", "test_label.csv"])
    
    # 4. 测试SWAT数据集
    print("\n4. SWAT数据集:")
    test_dir_dataset("SWAT", ["swat_train2.csv", "swat2.csv"])
    
    # 5. 测试MSL数据集
    print("\n5. MSL数据集:")
    test_npy_dataset("MSL", ["MSL_train.npy", "MSL_test.npy", "MSL_test_label.npy"])
    
    # 6. 测试SMAP数据集
    print("\n6. SMAP数据集:")
    test_npy_dataset("SMAP", ["SMAP_train.npy", "SMAP_test.npy", "SMAP_test_label.npy"])
    
    # 7. 测试SMD数据集
    print("\n7. SMD数据集:")
    test_npy_dataset("SMD", ["SMD_train.npy", "SMD_test.npy", "SMD_test_label.npy"])
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
