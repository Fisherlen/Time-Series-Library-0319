"""
数据集加载示例代码
展示如何加载和使用下载的数据集
"""
import os
import pandas as pd
import numpy as np

# 数据集根目录
DATASET_ROOT = "./dataset"


def load_ett_dataset(dataset_name="ETTh1"):
    """
    加载 ETT 数据集

    Args:
        dataset_name: 数据集名称, 可选 ETTh1, ETTh2, ETTm1, ETTm2

    Returns:
        pandas.DataFrame: 数据集
    """
    file_path = os.path.join(DATASET_ROOT, "ETT", f"{dataset_name}.csv")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"加载 {dataset_name} 数据集:")
    print(f"  形状: {df.shape}")
    print(f"  列: {list(df.columns)}")
    print(f"  时间范围: {df['date'].min()} ~ {df['date'].max()}")
    return df


def load_ecl_dataset():
    """
    加载 Electricity Consuming Load (ECL) 数据集

    Returns:
        pandas.DataFrame: 数据集
    """
    file_path = os.path.join(DATASET_ROOT, "ECL", "electricity.csv")
    df = pd.read_csv(file_path)
    print(f"加载 ECL 数据集:")
    print(f"  形状: {df.shape}")
    print(f"  列: {list(df.columns)[:5]}... (共 {len(df.columns)} 列)")
    return df


def load_traffic_dataset():
    """
    加载 Traffic 数据集

    Returns:
        pandas.DataFrame: 数据集
    """
    file_path = os.path.join(DATASET_ROOT, "Traffic", "traffic.csv")
    df = pd.read_csv(file_path)
    print(f"加载 Traffic 数据集:")
    print(f"  形状: {df.shape}")
    print(f"  列: {list(df.columns)[:5]}... (共 {len(df.columns)} 列)")
    return df


def load_weather_dataset():
    """
    加载 Weather 数据集

    Returns:
        pandas.DataFrame: 数据集
    """
    file_path = os.path.join(DATASET_ROOT, "Weather", "weather.csv")
    df = pd.read_csv(file_path)
    print(f"加载 Weather 数据集:")
    print(f"  形状: {df.shape}")
    print(f"  列: {list(df.columns)}")
    return df


def load_exchange_dataset():
    """
    加载 Exchange Rate 数据集

    Returns:
        pandas.DataFrame: 数据集
    """
    file_path = os.path.join(DATASET_ROOT, "Exchange", "exchange_rate.csv")
    df = pd.read_csv(file_path)
    print(f"加载 Exchange 数据集:")
    print(f"  形状: {df.shape}")
    print(f"  列: {list(df.columns)}")
    return df


def load_ili_dataset():
    """
    加载 ILI (Influenza-like Illness) 数据集

    Returns:
        pandas.DataFrame: 数据集
    """
    file_path = os.path.join(DATASET_ROOT, "ILI", "national_illness.csv")
    df = pd.read_csv(file_path)
    print(f"加载 ILI 数据集:")
    print(f"  形状: {df.shape}")
    print(f"  列: {list(df.columns)}")
    return df


def load_anomaly_dataset(dataset_name="PSM"):
    """
    加载异常检测数据集

    Args:
        dataset_name: 数据集名称, 可选 PSM, MSL, SMAP, SMD, SWAT

    Returns:
        dict: 包含 train, test, label 的数据字典
    """
    dataset_path = os.path.join(DATASET_ROOT, dataset_name)

    result = {}

    if dataset_name == "PSM":
        train_file = os.path.join(dataset_path, "train.csv")
        test_file = os.path.join(dataset_path, "test.csv")
        result['train'] = pd.read_csv(train_file)
        result['test'] = pd.read_csv(test_file)

    elif dataset_name in ["MSL", "SMAP", "SMD"]:
        train_file = os.path.join(dataset_path, f"{dataset_name}_train.npy")
        test_file = os.path.join(dataset_path, f"{dataset_name}_test.npy")
        label_file = os.path.join(dataset_path, f"{dataset_name}_test_label.npy")
        result['train'] = np.load(train_file)
        result['test'] = np.load(test_file)
        result['label'] = np.load(label_file)

    elif dataset_name == "SWAT":
        train_file = os.path.join(dataset_path, "swat_train2.csv")
        test_file = os.path.join(dataset_path, "swat2.csv")
        result['train'] = pd.read_csv(train_file)
        result['test'] = pd.read_csv(test_file)

    print(f"加载 {dataset_name} 异常检测数据集:")
    for key, data in result.items():
        print(f"  {key}: {data.shape}")

    return result


def load_m4_dataset(frequency="hourly"):
    """
    加载 M4 数据集

    Args:
        frequency: 频率, 可选 yearly, quarterly, monthly, weekly, daily, hourly

    Returns:
        dict: 包含 train 和 test 的数据字典
    """
    m4_path = os.path.join(DATASET_ROOT, "m4")
    train_file = os.path.join(m4_path, f"{frequency}_train.csv")
    test_file = os.path.join(m4_path, f"{frequency}_test.csv")

    result = {
        'train': pd.read_csv(train_file),
        'test': pd.read_csv(test_file)
    }

    print(f"加载 M4 {frequency} 数据集:")
    for key, data in result.items():
        print(f"  {key}: {data.shape}")

    return result


def get_dataset_info():
    """获取所有数据集的信息"""
    info = {
        "ETT": {
            "path": "./dataset/ETT/",
            "datasets": ["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
            "task": "长期预测",
            "description": "电力变压器温度数据集"
        },
        "ECL": {
            "path": "./dataset/ECL/",
            "file": "electricity.csv",
            "task": "长期预测",
            "description": "电力消耗负载数据集"
        },
        "Traffic": {
            "path": "./dataset/Traffic/",
            "file": "traffic.csv",
            "task": "长期预测",
            "description": "道路占用率数据集"
        },
        "Weather": {
            "path": "./dataset/Weather/",
            "file": "weather.csv",
            "task": "长期预测",
            "description": "气象数据集"
        },
        "Exchange": {
            "path": "./dataset/Exchange/",
            "file": "exchange_rate.csv",
            "task": "长期预测",
            "description": "汇率数据集"
        },
        "ILI": {
            "path": "./dataset/ILI/",
            "file": "national_illness.csv",
            "task": "长期预测",
            "description": "流感样疾病数据集"
        },
        "Anomaly Detection": {
            "PSM": {
                "path": "./dataset/PSM/",
                "task": "异常检测",
                "description": "Pooled Server Metrics"
            },
            "MSL": {
                "path": "./dataset/MSL/",
                "task": "异常检测",
                "description": "Mars Science Laboratory"
            },
            "SMAP": {
                "path": "./dataset/SMAP/",
                "task": "异常检测",
                "description": "Soil Moisture Active Passive"
            },
            "SMD": {
                "path": "./dataset/SMD/",
                "task": "异常检测",
                "description": "Server Machine Dataset"
            },
            "SWAT": {
                "path": "./dataset/SWAT/",
                "task": "异常检测",
                "description": "Secure Water Treatment"
            }
        },
        "M4": {
            "path": "./dataset/m4/",
            "frequencies": ["yearly", "quarterly", "monthly", "weekly", "daily", "hourly"],
            "task": "短期预测",
            "description": "M4 竞赛数据集"
        }
    }
    return info


def print_dataset_summary():
    """打印数据集摘要"""
    print("=" * 70)
    print("Time-Series-Library 数据集摘要")
    print("=" * 70)
    print(f"数据集根目录: {os.path.abspath(DATASET_ROOT)}")
    print()

    # 长期预测数据集
    print("【长期预测数据集】")
    print("-" * 70)

    # ETT
    ett_path = os.path.join(DATASET_ROOT, "ETT")
    if os.path.exists(ett_path):
        for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
            file_path = os.path.join(ett_path, f"{name}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"  {name}: {df.shape} - 列: {list(df.columns)}")

    # 其他预测数据集
    for folder, file in [("ECL", "electricity.csv"), ("Traffic", "traffic.csv"),
                         ("Weather", "weather.csv"), ("Exchange", "exchange_rate.csv"),
                         ("ILI", "national_illness.csv")]:
        folder_path = os.path.join(DATASET_ROOT, folder)
        file_path = os.path.join(folder_path, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"  {folder}: {df.shape} - 列数: {len(df.columns)}")

    # 异常检测数据集
    print()
    print("【异常检测数据集】")
    print("-" * 70)
    for name in ["PSM", "MSL", "SMAP", "SMD", "SWAT"]:
        folder_path = os.path.join(DATASET_ROOT, name)
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            print(f"  {name}: {files}")

    # M4 数据集
    print()
    print("【M4 短期预测数据集】")
    print("-" * 70)
    m4_path = os.path.join(DATASET_ROOT, "m4")
    if os.path.exists(m4_path):
        for freq in ["yearly", "quarterly", "monthly", "weekly", "daily", "hourly"]:
            train_file = os.path.join(m4_path, f"{freq}_train.csv")
            if os.path.exists(train_file):
                df = pd.read_csv(train_file)
                print(f"  {freq}: {df.shape}")

    print()
    print("=" * 70)


def main():
    """主函数 - 演示如何加载各种数据集"""
    print("=" * 70)
    print("数据集加载示例")
    print("=" * 70)
    print()

    # 1. 加载 ETT 数据集
    print("1. 加载 ETTh1 数据集:")
    print("-" * 70)
    ett_h1 = load_ett_dataset("ETTh1")
    print()

    # 2. 加载 ECL 数据集
    print("2. 加载 ECL 数据集:")
    print("-" * 70)
    ecl = load_ecl_dataset()
    print()

    # 3. 加载 Weather 数据集
    print("3. 加载 Weather 数据集:")
    print("-" * 70)
    weather = load_weather_dataset()
    print()

    # 4. 加载异常检测数据集 (PSM)
    print("4. 加载 PSM 异常检测数据集:")
    print("-" * 70)
    psm = load_anomaly_dataset("PSM")
    print()

    # 5. 加载 M4 数据集
    print("5. 加载 M4 hourly 数据集:")
    print("-" * 70)
    m4_hourly = load_m4_dataset("hourly")
    print()

    # 打印数据集摘要
    print_dataset_summary()

    print("\n使用示例:")
    print("-" * 70)
    print("# 加载 ETT 数据集")
    print("df = load_ett_dataset('ETTh1')")
    print()
    print("# 加载 ECL 数据集")
    print("df = load_ecl_dataset()")
    print()
    print("# 加载异常检测数据集")
    print("data = load_anomaly_dataset('PSM')")
    print("train_data = data['train']")
    print("test_data = data['test']")


if __name__ == "__main__":
    main()
