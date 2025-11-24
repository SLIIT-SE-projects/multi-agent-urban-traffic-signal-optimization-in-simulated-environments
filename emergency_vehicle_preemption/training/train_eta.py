import os
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "../data/processed/")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "../models/saved/")

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 4
VALIDATION_SPLIT = 0.2

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def load_data():
    """Loads the preprocessed training data."""
    print("Loading data...")
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"))
    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = load_data()
    print(f"Training Data Shape: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Test Data Shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

    if X_train.shape[0] == 0:
        print("Error: No training data found.")
        return

    print("Data loaded successfully. Model definition will be added next.")

if __name__ == "__main__":
    main()
