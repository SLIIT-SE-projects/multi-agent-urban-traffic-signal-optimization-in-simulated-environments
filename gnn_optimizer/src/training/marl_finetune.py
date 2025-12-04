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
SUMO_CONFIG = "simulation/scenario.sumocfg"
SUMO_NET = "simulation/network.net.xml"
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
        num_actions = logits.size(1)
        return torch.randint(0, num_actions, (logits.size(0),))
    
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

    # METRIC HISTORY LISTS
    history_rewards = []
    history_queues = []
    history_losses = []
    
    # 3. Training Loop (Episodes)
    for episode in range(1, EPISODES + 1):
        manager.start()
        hidden_state = None
        
        ep_reward = 0
        ep_loss = 0
        ep_queue_sum = 0
        
        print(f"\n Episode {episode}/{EPISODES} (Epsilon: {epsilon:.2f})")
        
        for t in tqdm(range(STEPS_PER_EPISODE)):
            # A. Get State
            snapshot = manager.get_snapshot()
            data = graph_builder.create_hetero_data(snapshot)
            
            # Track Queue
            step_queue = sum([l['queue_length'] for l in snapshot['lanes'].values()])
            ep_queue_sum += step_queue
            
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
            ep_reward += reward
            
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
            ep_loss += loss.item()

        manager.close()
        
        # Update History
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        avg_loss = ep_loss / STEPS_PER_EPISODE
        avg_queue = ep_queue_sum / STEPS_PER_EPISODE
        
        history_rewards.append(ep_reward)
        history_queues.append(avg_queue)
        history_losses.append(avg_loss)
        
        print(f" Episode {episode} Done. Reward: {ep_reward:.2f} | Avg Queue: {avg_queue:.2f} | Avg Loss: {avg_loss:.4f}")

        # Run Testing every 5 episodes
        if episode % 5 == 0:
            test_score = evaluate_model(model, graph_builder, episode)
            
            # Save "Best Model" based on Test Score
            # if test_score > best_test_score:
            #     torch.save(model.state_dict(), "best_marl_model.pth")
        
        # Save periodically
        torch.save(model.state_dict(), FINAL_MODEL_PATH)

    print(" MARL Fine-Tuning Complete!")

def evaluate_model(model, graph_builder, episode_num):

    print(f"\n Starting Evaluation (Episode {episode_num})...")
    
    # 1. Setup Evaluation Environment
    # Use GUI=False for speed, or True if you want to watch the test
    eval_manager = SumoManager(SUMO_CONFIG, use_gui=False) 
    eval_manager.start()
    
    total_eval_reward = 0
    total_queue_len = 0
    steps = 0
    
    model.eval()
    hidden_state = None
    
    try:
        # Run a full simulation episode
        for t in range(STEPS_PER_EPISODE):
            snapshot = eval_manager.get_snapshot()
            data = graph_builder.create_hetero_data(snapshot)
            
            with torch.no_grad():
                # Inference
                action_logits, hidden_state = model(data.x_dict, data.edge_index_dict, hidden_state)
                
                # STRICTLY GREEDY Action
                actions_indices = torch.argmax(action_logits, dim=1)
                
                # Convert to Dict
                idx_to_id = {v: k for k, v in graph_builder.tls_map.items()}
                actions_dict = {idx_to_id[idx]: val.item() for idx, val in enumerate(actions_indices) if idx in idx_to_id}
                
            # Act & Step
            eval_manager.apply_actions(actions_dict)
            eval_manager.step()
            
            # Measure Performance (Testing Metric)
            next_snapshot = eval_manager.get_snapshot()
            reward = calculate_reward(next_snapshot)
            
            # Track Metrics
            total_eval_reward += reward
            current_q = sum([info['queue_length'] for info in next_snapshot['lanes'].values()])
            total_queue_len += current_q
            steps += 1
            
    except Exception as e:
        print(f" Evaluation Failed: {e}")
    finally:
        eval_manager.close()
        model.train() 
        
    avg_reward = total_eval_reward / steps
    avg_queue = total_queue_len / steps
    
    print(f" Evaluation Result: Avg Reward = {avg_reward:.2f} | Avg Queue Length = {avg_queue:.2f} vehicles")
    return avg_reward

if __name__ == "__main__":
    train_marl()