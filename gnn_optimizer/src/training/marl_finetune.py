import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm

# SYSTEM PATH FIX 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.graphBuilder.sumo_manager import SumoManager
from src.graphBuilder.graph_builder import TrafficGraphBuilder
from src.models.hgat_core import RecurrentHGAT
from src.training.reward_function import calculate_reward

# CONFIGURATION 
SUMO_CONFIG = "simulation/simulation.sumo.cfg"
SUMO_NET = "simulation/test.net.xml"
PRETRAINED_PATH = "experiments/saved_models/pretrained_gnn.pth"
FINAL_MODEL_PATH = "experiments/saved_models/final_marl_model.pth"

# Training Hyperparameters
EPISODES = 5          # Total simulation runs for fine-tuning
STEPS_PER_EPISODE = 500 # Steps per run
LEARNING_RATE = 0.0005
GAMMA = 0.99          # Discount factor for future rewards
EPSILON_START = 1.0   # Exploration rate start
EPSILON_END = 0.1     # End: 10% random
EPSILON_DECAY = 0.995 

def train_marl():
    print(" Starting MARL Fine-Tuning...")
    
    # 1. Initialize Components
    manager = SumoManager(SUMO_CONFIG, use_gui=False)
    graph_builder = TrafficGraphBuilder(SUMO_NET)
    
    # Get a dummy snapshot to init model metadata
    manager.start()
    manager.step()
    snap = manager.get_snapshot()
    data = graph_builder.create_hetero_data(snap)
    manager.close()
    
    # 2. Load Pre-Trained Model (The "Warm Start")
    print(f" Loading Pre-trained weights from {PRETRAINED_PATH}...")
    model = RecurrentHGAT(32, 4, 2, data.metadata())
    
    try:
        model.load_state_dict(torch.load(PRETRAINED_PATH, weights_only=True), strict=False)
        print(" Weights loaded successfully.")
    except Exception as e:
        print(f" Warning: Could not load weights ({e}). Training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epsilon = EPSILON_START
    

if __name__ == "__main__":
    train_marl()