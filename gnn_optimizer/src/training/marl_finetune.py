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

def select_action(logits, epsilon):
    # Random Action
    if random.random() < epsilon:
        return torch.randint(0, 4, (logits.size(0),))
    
    # Best Action
    return torch.argmax(logits, dim=1)

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
    
    # 3. Training Loop (Episodes)
    for episode in range(1, EPISODES + 1):
        manager.start()
        hidden_state = None
        total_reward = 0
        loss_sum = 0
        
        print(f"\n Episode {episode}/{EPISODES} (Epsilon: {epsilon:.2f})")
        
        for t in tqdm(range(STEPS_PER_EPISODE)):
            # A. Get State
            snapshot = manager.get_snapshot()
            data = graph_builder.create_hetero_data(snapshot)
            
            # B. Forward Pass
            action_logits, hidden_state = model(data.x_dict, data.edge_index_dict, hidden_state)
            
            # C. Select Action
            actions_indices = select_action(action_logits, epsilon)
            
            # Map indices to IDs for SUMO
            idx_to_id = {v: k for k, v in graph_builder.tls_map.items()}
            actions_dict = {idx_to_id[idx]: val.item() for idx, val in enumerate(actions_indices) if idx in idx_to_id}
            
            # D. Execute Action
            manager.apply_actions(actions_dict)
            manager.step()
            
            # E. Calculate Reward
            # Get NEW snapshot to see effect of action
            next_snapshot = manager.get_snapshot()
            reward = calculate_reward(next_snapshot)
            total_reward += reward
            
            # F. Learning Step
            # Get probability of the action we took
            probs = F.softmax(action_logits, dim=1)
            log_probs = torch.log(probs.gather(1, actions_indices.view(-1, 1)))
            
            # Loss = - (LogProb * Reward)
            scaled_reward = reward / 100.0 
            loss = -log_probs.mean() * scaled_reward
            
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Detach memory
            hidden_state = hidden_state.detach()
            loss_sum += loss.item()

        manager.close()
        
        # Update Epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        avg_loss = loss_sum / STEPS_PER_EPISODE
        print(f" Episode {episode} Done. Total Reward: {total_reward:.2f} | Avg Loss: {avg_loss:.4f}")
        
        # Save periodically
        torch.save(model.state_dict(), FINAL_MODEL_PATH)

    print(" MARL Fine-Tuning Complete!")

if __name__ == "__main__":
    train_marl()