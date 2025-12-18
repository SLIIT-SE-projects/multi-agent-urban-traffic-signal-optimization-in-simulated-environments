import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
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
GAE_LAMBDA = TrainConfig.GAE_LAMBDA
PPO_EPSILON = TrainConfig.PPO_EPSILON
PPO_EPOCHS = TrainConfig.PPO_EPOCHS
ENTROPY_COEF = TrainConfig.ENTROPY_COEF 
VALUE_LOSS_COEF = TrainConfig.VALUE_LOSS_COEF


class RolloutBuffer:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.states = []       
        self.actions = []      
        self.log_probs = []    
        self.rewards = []      
        self.values = []       
        self.dones = []        
        self.hidden_states = [] 

def compute_gae(buffer, next_value, gamma, lam):
    values = torch.tensor(np.array(buffer.values), dtype=torch.float32)
    next_value = torch.tensor(next_value, dtype=torch.float32)
    
    values = torch.cat([values, next_value.unsqueeze(0)], dim=0)
    rewards = torch.tensor(buffer.rewards, dtype=torch.float32).unsqueeze(1)
    dones = torch.tensor(buffer.dones, dtype=torch.float32).unsqueeze(1)
    
    advantages = []
    last_gae_lam = torch.zeros_like(values[0])
    
    num_steps = len(rewards)
    for t in reversed(range(num_steps)):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * non_terminal - values[t]
        last_gae_lam = delta + gamma * lam * non_terminal * last_gae_lam
        advantages.insert(0, last_gae_lam)
        
    advantages = torch.stack(advantages)
    return advantages, values[:-1] + advantages

def ppo_update(model, optimizer, buffer, advantages, returns):
    old_log_probs = torch.stack(buffer.log_probs).detach()
    old_actions = torch.stack(buffer.actions).detach()
    
    old_log_probs = old_log_probs.view(-1)
    old_actions = old_actions.view(-1)
    advantages = advantages.view(-1)
    returns = returns.view(-1)
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_loss_sum = 0
    old_hidden_states = [h.detach() if h is not None else None for h in buffer.hidden_states]
    
    for _ in range(PPO_EPOCHS):
        new_log_probs_list = []
        new_values_list = []
        entropy_list = []
        
        for i, data in enumerate(buffer.states):
            h_in = old_hidden_states[i]
            logits, val, _ = model(data.x_dict, data.edge_index_dict, h_in)
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            
            action_taken = buffer.actions[i]
            new_log_prob = dist.log_prob(action_taken)
            entropy = dist.entropy()
            
            new_log_probs_list.append(new_log_prob)
            new_values_list.append(val.squeeze())
            entropy_list.append(entropy)
            
        new_log_probs = torch.stack(new_log_probs_list).view(-1)
        new_values = torch.stack(new_values_list).view(-1)
        entropy = torch.stack(entropy_list).view(-1).mean()
        
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, returns)
        
        loss = policy_loss + (VALUE_LOSS_COEF * value_loss) - (ENTROPY_COEF * entropy)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss_sum += loss.item()

    return total_loss_sum / PPO_EPOCHS

def select_action(logits):
    probs = F.softmax(logits, dim=1)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action, m.log_prob(action)

def train_marl():
    print(" Starting PPO MARL Fine-Tuning (Corrected Logic)...")
    
    manager = SumoManager(SUMO_CONFIG, use_gui=False)
    graph_builder = TrafficGraphBuilder(SUMO_NET)
    evaluator = Evaluator()
    
    manager.start()
    manager.step()
    snap = manager.get_snapshot()
    data = graph_builder.create_hetero_data(snap)
    manager.close()
    
    model = RecurrentHGAT(
        hidden_channels=TrainConfig.HIDDEN_DIM, 
        out_channels=GraphConfig.NUM_SIGNAL_PHASES, 
        num_heads=ModelConfig.NUM_HEADS, 
        metadata=data.metadata()
    )
    
    if os.path.exists(FINAL_MODEL_PATH):
        print(f" Resuming from {FINAL_MODEL_PATH}...")
        try: model.load_state_dict(torch.load(FINAL_MODEL_PATH))
        except: pass
    else:
        print(f" Loading SSL weights from {PRETRAINED_PATH}...")
        try: model.load_state_dict(torch.load(PRETRAINED_PATH, weights_only=True), strict=False)
        except: print(" Training from SCRATCH.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = RolloutBuffer()
    
    history_rewards = []
    history_queues = []
    history_losses = []
    
    ACTION_INTERVAL = 15
    MIN_GREEN_TIME = 20 # Minimum seconds to hold a green light
    
    for episode in range(1, EPISODES + 1):
        manager.start()
        hidden_state = None 
        
        ep_reward = 0
        ep_queue_sum = 0 
        ep_loss = 0      
        
        interval_reward = 0
        step_counter = 0
        
        # Track last switch time for each intersection to enforce Min Green
        last_switch_step = {} 
        current_phases = {} # Track what the simulation actually has
        
        print(f"\n Episode {episode}/{EPISODES}")
        
        for t in tqdm(range(STEPS_PER_EPISODE)):
            
            # Action Step
            if t % ACTION_INTERVAL == 0:
                snap = manager.get_snapshot()
                data = graph_builder.create_hetero_data(snap)
                
                with torch.no_grad():
                    h_in = hidden_state.clone() if hidden_state is not None else None
                    logits, value, hidden_state = model(data.x_dict, data.edge_index_dict, hidden_state)
                    action, log_prob = select_action(logits)
                
                idx_to_id = {v: k for k, v in graph_builder.tls_map.items()}
                actions_dict = {}
                
                for idx, val in enumerate(action):
                    if idx in idx_to_id:
                        model_action = val.item()
                        tls_id = idx_to_id[idx]
                        
                        # Initialize tracking if new
                        if tls_id not in last_switch_step:
                            last_switch_step[tls_id] = -999
                            current_phases[tls_id] = 0
                        
                        # Map Model Action -> SUMO Phase
                        target_phase = 0
                        if model_action == 0: target_phase = 0
                        elif model_action == 1: target_phase = 2 
                        else: target_phase = 0
                        
                        # STABILITY FIX: MIN GREEN TIME
                        if (t - last_switch_step[tls_id]) < MIN_GREEN_TIME:
                            final_phase = current_phases[tls_id]
                        else:
                            final_phase = target_phase
                            
                            # If we actually changed, update timer
                            if final_phase != current_phases[tls_id]:
                                last_switch_step[tls_id] = t
                                current_phases[tls_id] = final_phase
                        
                        actions_dict[tls_id] = final_phase

                manager.apply_actions(actions_dict)
                
                if step_counter > 0:
                    buffer.rewards.append(interval_reward) 
                    buffer.dones.append(0) 
                
                buffer.states.append(data)
                buffer.actions.append(action)
                buffer.log_probs.append(log_prob)
                buffer.values.append(value.detach().cpu().numpy().flatten())
                buffer.hidden_states.append(h_in)
                
                interval_reward = 0
                step_counter += 1

            manager.step()
            
            # Data Collection
            if t > 0:
                snap = manager.get_snapshot()
                r = calculate_reward(snap)
                interval_reward += r
                ep_reward += r
                ep_queue_sum += sum([l['queue_length'] for l in snap['lanes'].values()])
        
        if len(buffer.states) > len(buffer.rewards):
            buffer.rewards.append(interval_reward)
            buffer.dones.append(1) 
            
        manager.close()
        
        if len(buffer.states) > 0:
            print(" Updating PPO...")
            next_value = np.zeros(logits.shape[0]) 
            advantages, returns = compute_gae(buffer, next_value, GAMMA, GAE_LAMBDA)
            loss_val = ppo_update(model, optimizer, buffer, advantages, returns)
            ep_loss = loss_val
            print(f" Update Complete. Loss: {ep_loss:.4f} | Ep Reward: {ep_reward:.2f}")
            buffer.reset()
        
        avg_queue = ep_queue_sum / STEPS_PER_EPISODE
        history_rewards.append(ep_reward)
        history_queues.append(avg_queue)
        history_losses.append(ep_loss)
        
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        evaluator.plot_marl_performance(history_rewards, history_queues, history_losses, save_dir=PLOT_SAVE_DIR)

        if episode % TrainConfig.MARL_TESTING_EPISODES == 0:
            evaluate_model(model, graph_builder, episode)

def evaluate_model(model, graph_builder, episode_num):
    print(f"\n Starting Evaluation (Episode {episode_num})...")
    eval_manager = SumoManager(SUMO_CONFIG, use_gui=False) 
    eval_manager.start()
    
    total_eval_reward = 0
    total_queue_len = 0
    steps = 0
    model.eval()
    hidden_state = None
    ACTION_INTERVAL = 15 
    
    try:
        for t in range(STEPS_PER_EPISODE):
            if t % ACTION_INTERVAL == 0:
                snapshot = eval_manager.get_snapshot()
                data = graph_builder.create_hetero_data(snapshot)
                
                with torch.no_grad():
                    action_logits, _, hidden_state = model(data.x_dict, data.edge_index_dict, hidden_state)
                    actions_indices = select_action(action_logits)[0] 
                    
                    idx_to_id = {v: k for k, v in graph_builder.tls_map.items()}
                    actions_dict = {}
                    
                    for idx, val in enumerate(actions_indices):
                        if idx in idx_to_id:
                            model_action = val.item()
                            tls_id = idx_to_id[idx]
                            sumo_phase = 0
                            if model_action == 0: sumo_phase = 0
                            elif model_action == 1: sumo_phase = 2 
                            else: sumo_phase = 0
                            actions_dict[tls_id] = sumo_phase
                    
                eval_manager.apply_actions(actions_dict)
            
            eval_manager.step()
            next_snapshot = eval_manager.get_snapshot()
            reward = calculate_reward(next_snapshot)
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