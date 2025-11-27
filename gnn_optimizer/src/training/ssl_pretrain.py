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

# AUXILIARY HEAD
class StatePredictor(nn.Module):
    def __init__(self, hidden_dim, lane_feature_dim):
        super().__init__()
        self.decoder = nn.Linear(hidden_dim, lane_feature_dim)

    def forward(self, hidden_state):
        return self.decoder(hidden_state)

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
    
    # 2. Initialize Model
    print(" Initializing Model...")
    sample_graph = dataset[0]
    metadata = sample_graph.metadata()
    
    gnn_model = RecurrentHGAT(HIDDEN_DIM, 4, 2, metadata)
    predictor = StatePredictor(HIDDEN_DIM, 5)
    
    optimizer = optim.Adam(list(gnn_model.parameters()) + list(predictor.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(" Starting Self-Supervised Pre-training...")
    
    if not os.path.exists("experiments/saved_models"):
        os.makedirs("experiments/saved_models")

    for epoch in range(EPOCHS):
        # A. TRAINING PHASE
        gnn_model.train()
        predictor.train()
        total_train_loss = 0
        hidden_state = None 
        
        # Iterate through Training Data
        for t in tqdm(range(len(train_dataset) - 1), desc=f"Epoch {epoch+1} [Train]"):
            current_data = train_dataset[t]
            next_data = train_dataset[t+1]
            
            optimizer.zero_grad()
            _, hidden_state = gnn_model(current_data.x_dict, current_data.edge_index_dict, hidden_state)
            
            predicted_next_state = predictor(hidden_state)
            target = next_data['intersection'].x
            
            loss = criterion(predicted_next_state, target)
            loss.backward()
            optimizer.step()
            
            hidden_state = hidden_state.detach()
            total_train_loss += loss.item()

    print(" Pre-training Complete!")

if __name__ == "__main__":
    train_ssl()