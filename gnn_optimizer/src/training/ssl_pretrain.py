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
from src.utils.evaluator import Evaluator 

# CONFIGURATION 
DATASET_PATH = "experiments/raw_data/traffic_data_1hr.pt"
MODEL_SAVE_PATH = "experiments/saved_models/pretrained_gnn.pth"
PLOT_SAVE_DIR = "experiments/plots"
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
    # 1. Setup Directories
    if not os.path.exists("experiments/saved_models"):
        os.makedirs("experiments/saved_models")
    if not os.path.exists(PLOT_SAVE_DIR):
        os.makedirs(PLOT_SAVE_DIR)

    # 2. Load Data
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
    
    # Initialize Model & Evaluator
    print(" Initializing Model...")
    sample_graph = dataset[0]
    metadata = sample_graph.metadata()
    
    gnn_model = RecurrentHGAT(HIDDEN_DIM, 4, 2, metadata)
    predictor = StatePredictor(HIDDEN_DIM, 5)
    evaluator = Evaluator()
    
    optimizer = optim.Adam(list(gnn_model.parameters()) + list(predictor.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(" Starting Self-Supervised Pre-training...")

    # --- TRAINING LOOP ---
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

        avg_train_loss = total_train_loss / len(train_dataset)

        # B. TESTING (VALIDATION) PHASE
        gnn_model.eval()
        predictor.eval()
        total_test_loss = 0
        hidden_state_test = None
        
        # Store all predictions for metrics calculation
        all_preds = []
        all_targets = []
        
        with torch.no_grad(): # Disable gradient calculation for testing
            for t in range(len(test_dataset) - 1):
                current_data = test_dataset[t]
                next_data = test_dataset[t+1]
                
                _, hidden_state_test = gnn_model(current_data.x_dict, current_data.edge_index_dict, hidden_state_test)
                predicted = predictor(hidden_state_test)
                target = next_data['intersection'].x
                
                loss = criterion(predicted, target)
                total_test_loss += loss.item()
                
                # Collect for metrics
                all_preds.append(predicted)
                all_targets.append(target)
        
        avg_test_loss = total_test_loss / len(test_dataset)
        
        # METRICS & LOGGING 
        cat_preds = torch.cat(all_preds)
        cat_targets = torch.cat(all_targets)
        
        # Calculate R2, MAE, RMSE
        metrics = evaluator.calculate_metrics(cat_preds, cat_targets)
        evaluator.log_epoch(avg_train_loss, avg_test_loss)

        print(f" Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        print(f"  Validation Metrics: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

    # 4. Finalize
    print(" Saving Pre-trained Weights...")
    torch.save(gnn_model.state_dict(), MODEL_SAVE_PATH)
    
    # Generate Plots
    print(" Generating Evaluation Plots...")
    evaluator.plot_learning_curves(save_path=f"{PLOT_SAVE_DIR}/loss_curve.png")
    
    # Generate Scatter plot using the LAST epoch's validation data
    evaluator.plot_predictions_vs_truth(cat_preds, cat_targets, save_path=f"{PLOT_SAVE_DIR}/scatter.png")
    evaluator.plot_time_series_sample(cat_preds, cat_targets, save_path=f"{PLOT_SAVE_DIR}/timeseries.png")
    evaluator.plot_error_distribution(cat_preds, cat_targets, save_path=f"{PLOT_SAVE_DIR}/error_hist.png")
    
    print(" Pre-training Complete! Plots saved to {PLOT_SAVE_DIR}")

if __name__ == "__main__":
    train_ssl()