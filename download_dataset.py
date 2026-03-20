import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil

HUGGINGFACE_REPO = "thuml/Time-Series-Library"

CSV_DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "electricity", "traffic", "weather",
    "exchange_rate", "national_illness",
    "PSM-data", "PSM-label",
    "SMD-data", "SMD-label",
    "MSL-data", "MSL-label",
    "SMAP-data", "SMAP-label",
    "SWaT",
]

CLASSIFICATION_DATASETS = [
    "EthanolConcentration", "FaceDetection", "Handwriting", 
    "Heartbeat", "JapaneseVowels", "PEMS-SF",
    "SelfRegulationSCP1", "SelfRegulationSCP2", 
    "SpokenArabicDigits", "UWaveGestureLibrary",
]

M4_DATASETS = [
    "m4-yearly", "m4-quarterly", "m4-monthly", 
    "m4-weekly", "m4-daily", "m4-hourly",
]

def create_dataset_dir(base_path):
    dataset_dir = os.path.join(base_path, "dataset")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created directory: {dataset_dir}")
    return dataset_dir

def download_csv_dataset(dataset_name, save_dir):
    try:
        print(f"Downloading {dataset_name}...")
        ds = load_dataset(HUGGINGFACE_REPO, name=dataset_name, trust_remote_code=True)
        
        for split_name in ds.keys():
            df = ds[split_name].to_pandas()
            file_path = os.path.join(save_dir, f"{dataset_name}.csv")
            df.to_csv(file_path, index=False)
            print(f"  Saved: {file_path} ({len(df)} rows)")
            break
        return True
    except Exception as e:
        print(f"  Error downloading {dataset_name}: {e}")
        return False

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = create_dataset_dir(base_path)
    
    print("\n" + "="*60)
    print("Downloading Time-Series-Library datasets from Hugging Face")
    print(f"Repository: {HUGGINGFACE_REPO}")
    print(f"Target directory: {dataset_dir}")
    print("="*60 + "\n")
    
    all_datasets = CSV_DATASETS + CLASSIFICATION_DATASETS + M4_DATASETS
    
    success_count = 0
    fail_count = 0
    
    print("Downloading datasets...")
    for subset in tqdm(all_datasets):
        if download_csv_dataset(subset, dataset_dir):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "="*60)
    print("Download Summary:")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"Datasets saved to: {dataset_dir}")
    print("="*60)
    
    print("\nListing downloaded files:")
    for f in sorted(os.listdir(dataset_dir)):
        filepath = os.path.join(dataset_dir, f)
        size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {f}: {size:.2f} MB")

if __name__ == "__main__":
    main()
