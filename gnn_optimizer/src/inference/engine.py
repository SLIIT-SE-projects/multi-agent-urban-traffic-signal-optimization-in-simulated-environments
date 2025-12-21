import os
import sys
import torch
import torch.nn.functional as F

# SYSTEM PATH FIX
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import SimConfig, GraphConfig, TrainConfig, ModelConfig, FileConfig
from src.graphBuilder.sumo_manager import SumoManager
from src.graphBuilder.graph_builder import TrafficGraphBuilder
from src.models.hgat_core import RecurrentHGAT

class RealTimeInferenceEngine:
    def __init__(self, config_path, net_path, model_path, use_gui=True):
        self.manager = SumoManager(config_path, use_gui)
        self.graph_builder = TrafficGraphBuilder(net_path)
        self.model_path = model_path
        self.model = None
        self.hidden_state = None
        
        self.ACTION_INTERVAL = 15
        
    def initialize_model(self):
        print(f"Initializing System...")
        self.manager.start()
        self.manager.step()
        
        # Build initial graph to get metadata
        snapshot = self.manager.get_snapshot()
        data = self.graph_builder.create_hetero_data(snapshot)
        
        # Init Model
        try:
            self.model = RecurrentHGAT(
                hidden_channels=TrainConfig.HIDDEN_DIM,
                out_channels=GraphConfig.NUM_SIGNAL_PHASES, # 4 Phases
                num_heads=ModelConfig.NUM_HEADS,
                metadata=data.metadata()
            )
            
            # 3. Load Trained Weights
            print(f"Loading weights from: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)
            self.model.load_state_dict(checkpoint)
            print(" Trained Model Loaded Successfully!")
            
        except RuntimeError as e:
            print(f"\n MODEL CONFIG MISMATCH ERROR: {e}")
            sys.exit(1)
        except Exception as e:
            print(f" Could not load model. {e}")
            sys.exit(1)

        self.model.eval() 
        print(" System Ready for Inference.")

    def run(self, steps=3600):
        print(f" Starting Traffic Optimization for {steps} steps...")
        
        try:
            idx_to_id = {v: k for k, v in self.graph_builder.tls_map.items()}
            
            for t in range(steps):
                # ACTION STEP (Every 15 Seconds)
                if t % self.ACTION_INTERVAL == 0:
                    
                    snapshot = self.manager.get_snapshot()
                    data = self.graph_builder.create_hetero_data(snapshot)
                    
                    with torch.no_grad():
                        action_logits, _, self.hidden_state = self.model(
                            data.x_dict, 
                            data.edge_index_dict, 
                            self.hidden_state
                        )
                    
                    # Greedy Action
                    chosen_phases = torch.argmax(action_logits, dim=1).tolist()
                    
                    actions_dict = {}
                    for idx, model_action in enumerate(chosen_phases):
                        if idx not in idx_to_id: continue
                        tls_id = idx_to_id[idx]
                        
                        manager_target_idx = 0 
                        
                        if model_action == 0:
                            manager_target_idx = 0 
                        elif model_action == 1:
                            manager_target_idx = 1
                        else:
                            manager_target_idx = 0
                        
                        actions_dict[tls_id] = manager_target_idx
                    
                    self.manager.apply_actions(actions_dict)
                    
                    if t % 30 == 0:
                        total_queue = sum([l['queue_length'] for l in snapshot['lanes'].values()])
                        print(f"Step {t}: 🚦 Optimization Active. Net Queue: {total_queue:.1f} veh")

                # SIMULATION STEP
                self.manager.step()
                
                if self.hidden_state is not None:
                    self.hidden_state = self.hidden_state.detach()
                    
        except KeyboardInterrupt:
            print("\n Stopped by user.")
        except Exception as e:
            print(f" Crash during run: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.manager.close()
            print("Simulation Closed.")

if __name__ == "__main__":
    CONFIG = SimConfig.SUMO_CFG
    NET = SimConfig.NET_FILE
    MODEL_PATH = FileConfig.FINAL_MARL_MODEL_PATH 
    
    if not os.path.exists(CONFIG):
        print(f" Error: Config file not found at {CONFIG}")
    elif not os.path.exists(MODEL_PATH):
        print(f" Error: Model file not found at {MODEL_PATH}")
    else:
        engine = RealTimeInferenceEngine(CONFIG, NET, MODEL_PATH, use_gui=True)
        engine.initialize_model()
        engine.run(steps=3600)