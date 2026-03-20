"""
数据集下载脚本
从 Hugging Face 下载 Time-Series-Library 所需的所有数据集到本地
"""
import os
from datasets import load_dataset
import pandas as pd

HUGGINGFACE_REPO = "thuml/Time-Series-Library"

# 定义数据集配置
DATASETS_CONFIG = {
    # ETT 数据集 - 电力变压器温度数据
    "ETTh1": {"subset": "ETTh1", "path": "ETT-small/ETTh1.csv"},
    "ETTh2": {"subset": "ETTh2", "path": "ETT-small/ETTh2.csv"},
    "ETTm1": {"subset": "ETTm1", "path": "ETT-small/ETTm1.csv"},
    "ETTm2": {"subset": "ETTm2", "path": "ETT-small/ETTm2.csv"},

    # 天气数据集
    "weather": {"subset": "weather", "path": "weather/weather.csv"},

    # 电力消耗数据集 (ECL)
    "electricity": {"subset": "electricity", "path": "electricity/electricity.csv"},

    # 交通数据集
    "traffic": {"subset": "traffic", "path": "traffic/traffic.csv"},

    # 汇率数据集
    "exchange_rate": {"subset": "exchange_rate", "path": "exchange_rate/exchange_rate.csv"},

    # 太阳能数据集 (注意：HuggingFace上可能没有这个数据集，需要从其他来源获取)

    # 疾病数据集
    "illness": {"subset": "national_illness", "path": "illness/national_illness.csv"},
}


def download_dataset(subset_name, save_path):
    """下载单个数据集并保存为CSV"""
    try:
        print(f"正在下载 {subset_name} ...")
        ds = load_dataset(HUGGINGFACE_REPO, name=subset_name)

        # 获取数据分割
        if "train" in ds:
            df = ds["train"].to_pandas()
        else:
            # 使用第一个可用的分割
            split_name = list(ds.keys())[0]
            df = ds[split_name].to_pandas()

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存为CSV
        df.to_csv(save_path, index=False)
        print(f"✓ {subset_name} 已保存到: {save_path}")
        return True
    except Exception as e:
        print(f"✗ {subset_name} 下载失败: {str(e)}")
        return False


def main():
    """主函数：下载所有数据集"""
    # 设置数据根目录
    root_path = "./data"

    print("=" * 60)
    print("Time-Series-Library 数据集下载工具")
    print("=" * 60)
    print(f"数据将保存到: {os.path.abspath(root_path)}")
    print("=" * 60)

    success_count = 0
    failed_datasets = []

    for name, config in DATASETS_CONFIG.items():
        save_path = os.path.join(root_path, config["path"])

        # 如果文件已存在，跳过
        if os.path.exists(save_path):
            print(f"✓ {name} 已存在，跳过下载")
            success_count += 1
            continue

        if download_dataset(config["subset"], save_path):
            success_count += 1
        else:
            failed_datasets.append(name)

    print("=" * 60)
    print(f"下载完成: {success_count}/{len(DATASETS_CONFIG)} 个数据集")

    if failed_datasets:
        print(f"下载失败的数据集: {', '.join(failed_datasets)}")
    else:
        print("所有数据集下载成功！")

    print("=" * 60)
    print("数据集存储路径结构:")
    print(f"  {root_path}/")
    print(f"    ├── ETT-small/")
    print(f"    │   ├── ETTh1.csv")
    print(f"    │   ├── ETTh2.csv")
    print(f"    │   ├── ETTm1.csv")
    print(f"    │   └── ETTm2.csv")
    print(f"    ├── weather/")
    print(f"    │   └── weather.csv")
    print(f"    ├── electricity/")
    print(f"    │   └── electricity.csv")
    print(f"    ├── traffic/")
    print(f"    │   └── traffic.csv")
    print(f"    ├── exchange_rate/")
    print(f"    │   └── exchange_rate.csv")
    print(f"    ├── solar/")
    print(f"    │   └── solar.csv")
    print(f"    └── illness/")
    print(f"        └── national_illness.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
