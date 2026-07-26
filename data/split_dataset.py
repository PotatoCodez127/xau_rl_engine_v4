import os
import joblib
import pandas as pd

def split_master_tensor(train_ratio=0.85):
    """
    Slices the MTF dictionary into a chronological Train and OOS Holdout set,
    maintaining the exact time alignment across all timeframes.
    """
    # 1. Setup Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'master_training_tensor.pkl')
    train_out = os.path.join(project_root, 'data', 'train_tensor.pkl')
    test_out = os.path.join(project_root, 'data', 'oos_holdout_tensor.pkl')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Missing {data_path}")

    print(f"📊 Loading {data_path}...")
    mtf_dict = joblib.load(data_path)

    # 2. Calculate the exact cutoff indices for each timeframe
    len_15m = len(mtf_dict["15m"])
    len_30m = len(mtf_dict["30m"])
    len_1h = len(mtf_dict["1H"])
    len_4h = len(mtf_dict["4H"])

    cut_15m = int(len_15m * train_ratio)
    cut_30m = int(len_30m * train_ratio)
    cut_1h = int(len_1h * train_ratio)
    cut_4h = int(len_4h * train_ratio)

    print(f"✂️ Slicing Data (Train: {train_ratio*100}%, Holdout: {(1-train_ratio)*100}%)...")
    
    # 3. Slice Training Set (Beginning to Cutoff)
    train_dict = {
        "15m": mtf_dict["15m"][:cut_15m],
        "30m": mtf_dict["30m"][:cut_30m],
        "1H": mtf_dict["1H"][:cut_1h],
        "4H": mtf_dict["4H"][:cut_4h]
    }

    # 4. Slice OOS Holdout Set (Cutoff to End)
    # The indices reset to 0, turning this into a completely independent dataset
    test_dict = {
        "15m": mtf_dict["15m"][cut_15m:],
        "30m": mtf_dict["30m"][cut_30m:],
        "1H": mtf_dict["1H"][cut_1h:],
        "4H": mtf_dict["4H"][cut_4h:]
    }

    # 5. Save the new files
    print("💾 Saving Training Tensor...")
    joblib.dump(train_dict, train_out)
    
    print("💾 Saving OOS Holdout Tensor...")
    joblib.dump(test_dict, test_out)

    print("\n" + "="*40)
    print("✅ SPLIT COMPLETE")
    print("="*40)
    print(f"Train 15m Steps:   {len(train_dict['15m'])}")
    print(f"Holdout 15m Steps: {len(test_dict['15m'])}")
    print("="*40)
    print("Next Steps:")
    print("1. Upload 'train_tensor.pkl' to your Google Drive replacing the old master tensor.")
    print("2. Re-run your Colab training script to train the model on the pre-2024 data.")
    print("3. Export the new ONNX files to your local deployment folder.")
    print("4. Point your 'oos_backtester.py' to 'oos_holdout_tensor.pkl' to test it!")

if __name__ == "__main__":
    split_master_tensor()