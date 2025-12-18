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

from src.config import FileConfig, TrainConfig, SimConfig, GraphConfig, ModelConfig
from src.graphBuilder.sumo_manager import SumoManager
from src.graphBuilder.graph_builder import TrafficGraphBuilder
from src.models.hgat_core import RecurrentHGAT
from src.training.reward_function import calculate_reward
from src.utils.evaluator import Evaluator

# CONFIGURATION 
SUMO_CONFIG = SimConfig.SUMO_CFG
SUMO_NET = SimConfig.NET_FILE
PRETRAINED_PATH = FileConfig.PRETRAINED_MODEL_PATH
FINAL_MODEL_PATH = FileConfig.FINAL_MARL_MODEL_PATH
PLOT_SAVE_DIR = FileConfig.PLOTS_DIR

# Training Hyperparameters
EPISODES = TrainConfig.MARL_EPISODES          # Total simulation runs for fine-tuning
STEPS_PER_EPISODE = TrainConfig.MARL_STEPS_PER_EPISODE # Steps per run
LEARNING_RATE = TrainConfig.MARL_LEARNING_RATE
GAMMA = TrainConfig.MARL_GAMMA       # Discount factor for future rewards
EPSILON_START = TrainConfig.EPSILON_START   # Exploration rate start
EPSILON_END = TrainConfig.EPSILON_END    # End: 10% random
EPSILON_DECAY = TrainConfig.EPSILON_DECAY  # Decay per episode

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
    evaluator = Evaluator()
    
    # Get a dummy snapshot to init model metadata
    manager.start()
    manager.step()
    snap = manager.get_snapshot()
    data = graph_builder.create_hetero_data(snap)
    manager.close()
    
    # 2. Load Pre-Trained Model (The "Warm Start")
    print(f" Loading Pre-trained weights from {PRETRAINED_PATH}...")
    model = RecurrentHGAT(
        hidden_channels=TrainConfig.HIDDEN_DIM, 
        out_channels=GraphConfig.NUM_SIGNAL_PHASES, 
        num_heads=ModelConfig.NUM_HEADS, 
        metadata=data.metadata()
    )
    
    # try:
    #     model.load_state_dict(torch.load(PRETRAINED_PATH, weights_only=True), strict=False)
    #     print(" Weights loaded.")
    # except:
    #     print(" Pre-trained weights not found. Training from scratch.")

    # 1. Try to load existing MARL model (Continue Training)
    if os.path.exists(FINAL_MODEL_PATH):
        print(f" Found existing MARL model at {FINAL_MODEL_PATH}. Resuming training...")
        try:
            model.load_state_dict(torch.load(FINAL_MODEL_PATH, weights_only=True))
            print(" Resumed from previous MARL checkpoint.")
        except Exception as e:
            print(f" Could not load MARL model ({e}). Trying SSL Pre-trained...")
            
            # 2. If MARL fails or doesn't exist, try SSL Pre-trained
            try:
                model.load_state_dict(torch.load(PRETRAINED_PATH, weights_only=True), strict=False)
                print(" Loaded SSL Pre-trained weights.")
            except:
                print(" No weights found. Training from SCRATCH.")
    else:
        # 3. First time run: Load SSL
        print(f" No previous MARL run found. Loading SSL weights from {PRETRAINED_PATH}...")
        try:
            model.load_state_dict(torch.load(PRETRAINED_PATH, weights_only=True), strict=False)
            print(" Loaded SSL Pre-trained weights.")
        except:
            print(" Training from SCRATCH.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epsilon = EPSILON_START

    # METRIC HISTORY LISTS
    history_rewards = []
    history_queues = []
    history_losses = []
    
    ACTION_INTERVAL = 15
    
    # Dynamic Baseline Variables
    running_reward_mean = 0.0
    running_reward_std = 1.0
    
    for episode in range(1, EPISODES + 1):
        manager.start()
        hidden_state = None
        
        ep_reward = 0
        ep_loss = 0
        ep_queue_sum = 0
        interval_reward = 0 
        
        # Reset baseline slightly each episode to adapt to new traffic flows
        running_reward_mean = running_reward_mean * 0.9 
        
        print(f"\n Episode {episode}/{EPISODES} (Epsilon: {epsilon:.2f})")
        
        for t in tqdm(range(STEPS_PER_EPISODE)):

            # A. DECISION (Every 15s)
            if t % ACTION_INTERVAL == 0:
                # 1. Get State
                snapshot = manager.get_snapshot()
                data = graph_builder.create_hetero_data(snapshot)
                
                # 2. Forward Pass
                action_logits, _, hidden_state = model(data.x_dict, data.edge_index_dict, hidden_state)
                
                # 3. Select Action
                actions_indices = select_action(action_logits, epsilon)
                
                # 4. Apply Action WITH MAPPING
                idx_to_id = {v: k for k, v in graph_builder.tls_map.items()}
                actions_dict = {}
                
                for idx, val in enumerate(actions_indices):
                    if idx in idx_to_id:
                        model_action = val.item()
                        tls_id = idx_to_id[idx]
                        
                        # 3: THE YELLOW TRAP FIX
                        # Map Model Action -> SUMO Phase
                        # 0 -> 0 (Green A)
                        # 1 -> 2 (Green B)  <-- SKIPS YELLOW (Phase 1)
                        # 2 -> 0 (Fallback)
                        
                        sumo_phase = 0
                        if model_action == 0: sumo_phase = 0
                        elif model_action == 1: sumo_phase = 2 
                        else: sumo_phase = 0
                        
                        actions_dict[tls_id] = sumo_phase

                manager.apply_actions(actions_dict)

                # B. TRAINING (On Previous Interval)
                if t > 0:
                    probs = F.softmax(action_logits, dim=1)
                    log_probs = torch.log(probs.gather(1, actions_indices.view(-1, 1)))
                    
                    # 4: Normalize Advantage
                    # This centers the reward around 0. 
                    # Even if reward is -500, if average is -600, this is GOOD (+1.0 advantage)
                    advantage = interval_reward
                    
                    running_reward_mean = 0.95 * running_reward_mean + 0.05 * advantage
                    running_reward_std = 0.95 * running_reward_std + 0.05 * abs(advantage - running_reward_mean)
                    
                    # (Value - Mean) / Std
                    scaled_advantage = (advantage - running_reward_mean) / (running_reward_std + 1e-8)
                    
                    # Clip to prevent massive gradient spikes
                    scaled_advantage = torch.clamp(torch.tensor(scaled_advantage), -2.0, 2.0).to(action_logits.device)
                    
                    # ADD ENTROPY REGULARIZATION
                    # 1. Calculate Entropy (Measure of uncertainty)
                    entropy = - (probs * torch.log(probs + 1e-9)).sum(dim=1).mean()
                    
                    # 2. Define Coefficient (Strength of exploration force)
                    entropy_coef = 0.05
                    
                    # 3. Update Loss Formula
                    # We subtract entropy because we want to MAXIMIZE it (minimize negative entropy)
                    loss = (-log_probs.mean() * scaled_advantage) - (entropy_coef * entropy)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients to prevent instability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    # Detach memory
                    hidden_state = hidden_state.detach()
                    ep_loss += loss.item()
                    
                    interval_reward = 0

            # C. SIMULATION STEP
            manager.step()
            
            # D. DATA COLLECTION 
            if t > 0:
                current_snap = manager.get_snapshot()
                r_step = calculate_reward(current_snap)
                interval_reward += r_step
                ep_reward += r_step
                ep_queue_sum += sum([l['queue_length'] for l in current_snap['lanes'].values()])

        manager.close()
        
        # Update History & Logging
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        avg_loss = ep_loss / (STEPS_PER_EPISODE / ACTION_INTERVAL)
        avg_queue = ep_queue_sum / STEPS_PER_EPISODE
        
        history_rewards.append(ep_reward)
        history_queues.append(avg_queue)
        history_losses.append(avg_loss)
        
        print(f" Episode {episode} Done. Reward: {ep_reward:.2f} | Avg Queue: {avg_queue:.2f} | Avg Loss: {avg_loss:.4f}")

        # Run Testing every 5 episodes
        if episode % TrainConfig.MARL_TESTING_EPISODES == 0:
            test_score = evaluate_model(model, graph_builder, episode)
            
            # Save "Best Model" based on Test Score
            # if test_score > best_test_score:
            #     torch.save(model.state_dict(), "best_marl_model.pth")
        
        # Save periodically
        torch.save(model.state_dict(), FINAL_MODEL_PATH)

    # 4. Plot Results
    print(" Generating MARL Plots...")
    evaluator.plot_marl_performance(history_rewards, history_queues, history_losses, save_dir=PLOT_SAVE_DIR)

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
    
    # Define Interval
    ACTION_INTERVAL = 15 
    
    try:
        # Run a full simulation episode
        for t in range(STEPS_PER_EPISODE):
            
            # Only Act Every 15 Seconds
            if t % ACTION_INTERVAL == 0:
                snapshot = eval_manager.get_snapshot()
                data = graph_builder.create_hetero_data(snapshot)
                
                with torch.no_grad():
                    # Inference
                    action_logits,_, hidden_state = model(data.x_dict, data.edge_index_dict, hidden_state)
                    
                    # STRICTLY GREEDY Action
                    actions_indices = torch.argmax(action_logits, dim=1)
                    
                    # Convert to Dict
                    idx_to_id = {v: k for k, v in graph_builder.tls_map.items()}
                    actions_dict = {}
                    
                    for idx, val in enumerate(actions_indices):
                        if idx in idx_to_id:
                            model_action = val.item()
                            tls_id = idx_to_id[idx]
                            
                            # Apply the same Phase Mapping as Training
                            sumo_phase = 0
                            if model_action == 0: sumo_phase = 0
                            elif model_action == 1: sumo_phase = 2 
                            else: sumo_phase = 0
                            
                            actions_dict[tls_id] = sumo_phase
                    
                # Act
                eval_manager.apply_actions(actions_dict)
            
            # Step Simulation
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
        import traceback
        traceback.print_exc()
    finally:
        eval_manager.close()
        model.train() 
        
    avg_reward = total_eval_reward / steps
    avg_queue = total_queue_len / steps
    
    print(f" Evaluation Result: Avg Reward = {avg_reward:.2f} | Avg Queue Length = {avg_queue:.2f} vehicles")
    return avg_reward

if __name__ == "__main__":
    train_marl()