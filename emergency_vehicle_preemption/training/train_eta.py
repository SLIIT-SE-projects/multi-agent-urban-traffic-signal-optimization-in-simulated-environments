import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "../data/processed/")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "../models/saved/")

# Hyperparameters
EPOCHS = 100          # How many times to loop through the data
BATCH_SIZE = 32       # Number of samples per update (Small because our current data is small)
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

def build_model(input_shape):
    """
    Defines a Deep Bidirectional LSTM architecture.
    """
    model = Sequential()
    
    # 1. Bidirectional LSTM Layer (The Heavy Lifter)
    model.add(Bidirectional(LSTM(units=128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # 2. Second LSTM Layer (Refinement)
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    
    # 3. Dense Layers for final regression
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def main():
    # 1. Load Data
    X_train, y_train, X_test, y_test = load_data()
    print(f"Training Data Shape: {X_train.shape}")
    
    if X_train.shape[0] == 0:
        print("Error: No training data found.")
        return

    # 2. Build Model
    # input_shape is (10, 3) based on your preprocessing
    input_shape = (X_train.shape[1], X_train.shape[2]) 
    model = build_model(input_shape)
    model.summary()

    # 3. Train Model
    print("Starting training...")
    
    # EarlyStopping stops training if it stops improving (prevents wasting time)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1
    )

    # 4. Evaluate
    print("\nEvaluating on Test Set...")
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {mae:.4f} seconds (Mean Absolute Error)")

    # 5. Save Model
    save_path = os.path.join(MODEL_SAVE_DIR, "eta_predictor.h5")
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()