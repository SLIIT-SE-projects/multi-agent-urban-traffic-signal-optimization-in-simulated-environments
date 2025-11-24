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
    """Loads all CSV files from the raw data directory."""
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, "eta_data_*.csv"))
    
    if not all_files:
        print(f"ERROR: No data found in {RAW_DATA_DIR}")
        return None
    
    print(f"Found {len(all_files)} raw data files.")
    
    df_list = []
    for i, filename in enumerate(all_files):
        df = pd.read_csv(filename)
        # Give each file a unique run_id so we don't mix them up
        df['run_id'] = i  
        df_list.append(df)
    
    return pd.concat(df_list, ignore_index=True)

def calculate_actual_eta(df):
    """
    Calculates the 'Target' variable.
    Logic: Find the last step of the run (arrival). ETA = Arrival_Time - Current_Time.
    """
    processed_runs = []
    
    # Process each simulation run separately
    for run_id, group in df.groupby('run_id'):
        group = group.copy()
        
        # We assume the last step in the log is the arrival at the intersection
        arrival_step = group['step'].max()
        
        # Calculate ETA
        group['actual_eta'] = arrival_step - group['step']
        
        processed_runs.append(group)
        
    return pd.concat(processed_runs, ignore_index=True)

def create_sequences(data, feature_cols, target_col, seq_length):
    """
    Converts tabular data into 3D sequences for LSTM: (Samples, TimeSteps, Features)
    [cite: 277]
    """
    sequences = []
    targets = []
    
    # Group by run_id again to ensure we don't create a sequence that jumps 
    # from the end of Run 1 to the start of Run 2
    for _, group in data.groupby('run_id'):
        group_vals = group[feature_cols].values
        target_vals = group[target_col].values
        
        if len(group) < seq_length:
            continue
            
        for i in range(len(group) - seq_length):
            # Input: The sequence of features (e.g., t=0 to t=9)
            seq = group_vals[i : i + seq_length]
            sequences.append(seq)
            
            # Target: The ETA at the last step of the sequence (e.g., ETA at t=9)
            # We want the model to predict the ETA at the current moment
            label = target_vals[i + seq_length - 1] 
            targets.append(label)
            
    return np.array(sequences), np.array(targets)

def main():
    print("--- STARTING PREPROCESSING ---")
    
    # 1. Load Data
    df = load_data()
    if df is None: return

    # 2. Calculate Targets
    print("Calculating Actual ETA targets...")
    df = calculate_actual_eta(df)
    
    # 3. Normalize Features
    print("Normalizing features...")
    feature_cols = [
        'speed', 
        'acceleration', 
        'distance_to_signal',
        'queue_length', 
        'leader_gap', 
        'leader_speed'
    ]
    target_col = 'actual_eta'
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # SAVE THE SCALER
    scaler_path = os.path.join(SCALER_DIR, "eta_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # 4. Create Sequences
    print("Creating LSTM sequences...")
    X, y = create_sequences(df, feature_cols, target_col, SEQUENCE_LENGTH)
    
    print(f"Final Dataset Shape: X={X.shape}, y={y.shape}")
    
    if len(X) == 0:
        print("Error: Not enough data. Try running the simulation longer.")
        return

    # 5. Split and Save
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), y_test)
    
    print("SUCCESS: Preprocessing complete. Data ready for training.")

if __name__ == "__main__":
    main()