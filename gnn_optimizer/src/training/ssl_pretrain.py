import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training.dataset_loader import TrafficDataset
from src.models.hgat_core import RecurrentHGAT

# CONFIGURATION 
DATASET_PATH = "experiments/raw_data/traffic_data_1hr.pt"
MODEL_SAVE_PATH = "experiments/saved_models/pretrained_gnn.pth"
EPOCHS = 10
HIDDEN_DIM = 32
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8 # 80% Training, 20% Testing

def train_ssl():
    # 1. Load Data
    print(" Loading Dataset...")
    dataset = TrafficDataset(root="experiments", file_path=DATASET_PATH)
    
    # SPLIT DATASET 
    total_len = len(dataset)
    train_size = int(total_len * TRAIN_SPLIT)
    
    # slice the dataset. 
    # split by time (First 80% vs Last 20%)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    print(f" Data Split: {len(train_dataset)} Training steps | {len(test_dataset)} Testing steps")

if __name__ == "__main__":
    train_ssl()