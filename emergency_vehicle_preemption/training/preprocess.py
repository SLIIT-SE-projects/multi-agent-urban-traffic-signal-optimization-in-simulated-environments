import os
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "../data/raw/")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "../data/processed/")
SCALER_DIR = os.path.join(BASE_DIR, "../data/scalers/")

# Hyperparameters
SEQUENCE_LENGTH = 10  # Look back at the last 10 steps
TEST_SIZE = 0.2       # 20% of data for testing

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

def load_data():
    """Loads all CSV files."""
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, "eta_data_*.csv"))
    if not all_files:
        print(f"ERROR: No data found in {RAW_DATA_DIR}")
        return None
    
    df_list = []
    for i, filename in enumerate(all_files):
        df = pd.read_csv(filename)
        # Ensure 'ev_id' exists, if not (old data), assume 'EV_1'
        if 'ev_id' not in df.columns:
            df['ev_id'] = f"run_{i}_EV_1"
        else:
            # Make ev_id unique per file to avoid collision
            df['ev_id'] = df['ev_id'].astype(str) + f"_file_{i}"
        df_list.append(df)
    
    return pd.concat(df_list, ignore_index=True)

def calculate_actual_eta(df):
    """
    Calculates Target ETA.
    Logic: Group by EV ID. Find arrival step (max step) for THAT vehicle.
    """
    processed_evs = []
    
    for ev_id, group in df.groupby('ev_id'):
        group = group.copy()
        group = group.sort_values('step')
        
        # Assume last seen step is arrival
        arrival_step = group['step'].max()
        group['actual_eta'] = arrival_step - group['step']
        
        processed_evs.append(group)
        
    return pd.concat(processed_evs, ignore_index=True)

def create_sequences(data, feature_cols, target_col, seq_length):
    """
    Converts to LSTM sequences, grouped strictly by ev_id.
    """
    sequences = []
    targets = []
    
    # GROUP BY EV_ID
    for _, group in data.groupby('ev_id'):
        group_vals = group[feature_cols].values
        target_vals = group[target_col].values
        
        if len(group) < seq_length:
            continue
            
        for i in range(len(group) - seq_length):
            seq = group_vals[i : i + seq_length]
            sequences.append(seq)
            
            # Target is the ETA at the current moment (end of sequence)
            label = target_vals[i + seq_length - 1] 
            targets.append(label)
            
    return np.array(sequences), np.array(targets)

def main():
    print("--- STARTING MULTI-EV PREPROCESSING ---")
    
    # 1. Load
    df = load_data()
    if df is None: return

    # 2. Targets
    print(f"Calculating targets for {df['ev_id'].nunique()} unique EVs...")
    df = calculate_actual_eta(df)
    
    # 3. Normalize
    print("Normalizing features...")
    feature_cols = ['speed', 'acceleration', 'distance_to_signal', 
                    'queue_length', 'leader_gap', 'leader_speed']
    target_col = 'actual_eta'
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    with open(os.path.join(SCALER_DIR, "eta_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # 4. Sequences
    print("Creating LSTM sequences...")
    X, y = create_sequences(df, feature_cols, target_col, SEQUENCE_LENGTH)
    
    print(f"Final Dataset Shape: X={X.shape}, y={y.shape}")
    
    if len(X) == 0:
        print("Error: Not enough data.")
        return

    # 5. Save
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), y_test)
    
    print("SUCCESS: Data ready.")

if __name__ == "__main__":
    main()