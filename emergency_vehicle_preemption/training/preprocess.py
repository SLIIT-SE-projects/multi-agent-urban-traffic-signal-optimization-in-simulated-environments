import os
import glob
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "../data/raw/")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "../data/processed/")

SEQUENCE_LENGTH = 10
TEST_SIZE = 0.2

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_data():
    """Loads all CSV files from the raw data directory."""
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, "eta_data_*.csv"))

    if not all_files:
        print(f"ERROR: No data found in {RAW_DATA_DIR}")
        return None

    print(f"Found {len(all_files)} raw data files.")

    df_list = []
    for i, filename in enumerate(all_files):
        df = pd.read_csv(filename)
        df['run_id'] = i
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)

def calculate_actual_eta(df):
    """ETA = arrival_step - current_step, per run."""
    processed_runs = []
    for run_id, group in df.groupby('run_id'):
        group = group.copy()
        arrival_step = group['step'].max()
        group['actual_eta'] = arrival_step - group['step']
        processed_runs.append(group)
    return pd.concat(processed_runs, ignore_index=True)

def main():
    print("--- STARTING PREPROCESSING ---")
    df = load_data()
    if df is None:
        return

    print("Calculating Actual ETA targets...")
    df = calculate_actual_eta(df)

    # Save intermediate target-ready CSV for transparency/debugging
    target_ready_path = os.path.join(PROCESSED_DATA_DIR, "eta_with_targets.csv")
    df.to_csv(target_ready_path, index=False)
    print(f"Saved target-ready data to {target_ready_path}")

if __name__ == "__main__":
    main()
