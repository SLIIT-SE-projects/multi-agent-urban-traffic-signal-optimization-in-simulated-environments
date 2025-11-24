import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "../data/processed/")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "../models/saved/")

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

def build_model(input_shape):
    """Defines the LSTM architecture with Dropout."""
    model = Sequential()
    model.add(LSTM(units=64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="linear"))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model

def main():
    X_train, y_train, X_test, y_test = load_data()
    print(f"Training Data Shape: {X_train.shape}")

    if X_train.shape[0] == 0:
        print("Error: No training data found.")
        return

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    model.summary()

    print("Starting training...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1
    )

    print("Training complete. Evaluation + saving will be added next.")

if __name__ == "__main__":
    main()
