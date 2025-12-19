import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.graphBuilder.sumo_manager import SumoManager
from src.graphBuilder.graph_builder import TrafficGraphBuilder
from src.models.hgat_core import RecurrentHGAT
from src.config import SimConfig, GraphConfig, TrainConfig, ModelConfig, FileConfig

STEPS = 1000
SEED = 42  # FIXED SEED for fair comparison
OLD_MODEL_PATH = "experiments/saved_models/reinforce_model.pth" 
NEW_MODEL_PATH = "experiments/saved_models/final_marl_model.pth"

def run_simulation(model_path, label, color):
    print(f"\n Running Simulation for: {label}")
    
    # 1. Setup with Fixed Seed
    manager = SumoManager(SimConfig.SUMO_CFG, use_gui=False) 
    graph_builder = TrafficGraphBuilder(SimConfig.NET_FILE)
    
    manager.start()
    manager.step()
    
    # 2. Load Model
    snapshot = manager.get_snapshot()
    data = graph_builder.create_hetero_data(snapshot)
    
    model = RecurrentHGAT(
        hidden_channels=TrainConfig.HIDDEN_DIM, 
        out_channels=GraphConfig.NUM_SIGNAL_PHASES, 
        num_heads=ModelConfig.NUM_HEADS, 
        metadata=data.metadata()
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        print(f" Loaded {label} weights.")
    except Exception as e:
        print(f" Failed to load {label}: {e}")
        manager.close()
        return []

    # 3. Run Loop
    queues = []
    hidden_state = None
    ACTION_INTERVAL = 15
    
    # Map index back to TLS ID
    idx_to_id = {v: k for k, v in graph_builder.tls_map.items()}

    for t in range(STEPS):
        if t % ACTION_INTERVAL == 0:
            snapshot = manager.get_snapshot()
            data = graph_builder.create_hetero_data(snapshot)
            
            with torch.no_grad():
                # PPO model returns (logits, val, h), Old might return (logits, h)
                out = model(data.x_dict, data.edge_index_dict, hidden_state)
                if len(out) == 3:
                    logits, _, hidden_state = out # PPO
                else:
                    logits, hidden_state = out    # REINFORCE (if old arch)
                
                # Deterministic Choice (Argmax) for fair comparison
                actions = torch.argmax(logits, dim=1).tolist()
                
                actions_dict = {}
                for idx, val in enumerate(actions):
                    if idx in idx_to_id:
                        tls_id = idx_to_id[idx]
                        phase = 2 if val == 1 else 0 # Simple mapping
                        actions_dict[tls_id] = phase
                        
                manager.apply_actions(actions_dict)
        
        manager.step()
        
        # Log Metrics
        snap = manager.get_snapshot()
        total_q = sum([l['queue_length'] for l in snap['lanes'].values()])
        queues.append(total_q)
        
        if t % 100 == 0:
            print(f"   Step {t}: Queue {total_q}")

    manager.close()
    return queues

def compare_models():
    # Run Old
    q_old = run_simulation(OLD_MODEL_PATH, "Old Model (REINFORCE)", "orange")
    
    # Run New
    q_new = run_simulation(NEW_MODEL_PATH, "New Model (PPO)", "green")
    
    # Plot
    plt.figure(figsize=(12, 6))
    if q_old: plt.plot(q_old, label='REINFORCE (Old)', color='orange', alpha=0.7)
    if q_new: plt.plot(q_new, label='PPO (New)', color='green', linewidth=2)
    
    plt.title(f"Live Traffic Performance Comparison (Seed {SEED})")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Total Network Queue (Vehicles)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "experiments/plots/live_comparison.png"
    plt.savefig(save_path)
    print(f"\n Comparison Saved to: {save_path}")
    # plt.show() # Uncomment if running locally with display

if __name__ == "__main__":
    compare_models()