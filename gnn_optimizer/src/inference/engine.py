import os
import sys

# SYSTEM PATH FIX
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from src.graphBuilder.sumo_manager import SumoManager
from src.graphBuilder.graph_builder import TrafficGraphBuilder
from src.models.hgat_core import RecurrentHGAT

class RealTimeInferenceEngine:
    def __init__(self, config_path, net_path, use_gui=True):
        self.manager = SumoManager(config_path, use_gui)
        self.graph_builder = TrafficGraphBuilder(net_path)
        self.model = None
        self.hidden_state = None
        
    def initialize_model(self):
        print("Initializing System...")
        self.manager.start()
        self.manager.step()
        
        # Build initial graph to get metadata
        snapshot = self.manager.get_snapshot()
        data = self.graph_builder.create_hetero_data(snapshot)
        
        # Init Model
        self.model = RecurrentHGAT(
            hidden_channels=32,
            out_channels=4, # 4 Phases
            num_heads=2,
            metadata=data.metadata()
        )
        self.model.eval() 
        print("✅ System Initialized.")

    def run(self, steps=100):
        try:
            for t in range(steps):
                # 1. Sense (Get Data)
                snapshot = self.manager.get_snapshot()
                data = self.graph_builder.create_hetero_data(snapshot)
                
                # 2. Think (Inference)
                with torch.no_grad():
                    action_logits, self.hidden_state = self.model(
                        data.x_dict, 
                        data.edge_index_dict, 
                        self.hidden_state
                    )
                
                # 3. Decide (Argmax)
                chosen_phases = torch.argmax(action_logits, dim=1).tolist()
                
                # Map index back to TLS ID
                # Reverse map: Index -> ID
                idx_to_id = {v: k for k, v in self.graph_builder.tls_map.items()}
                
                actions_dict = {}
                for idx, phase_val in enumerate(chosen_phases):
                    if idx in idx_to_id:
                        tls_id = idx_to_id[idx]
                        actions_dict[tls_id] = phase_val
                
                # 4. Act (Apply Actions)
                self.manager.apply_actions(actions_dict)
                
                # 5. Step
                self.manager.step()
                
                if t % 10 == 0:
                    print(f"Step {t}/{steps}: Actions Applied to {len(actions_dict)} intersections.")
                    
        except Exception as e:
            print(f" Crash: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.manager.close()

# Direct Run for Testing
if __name__ == "__main__":
    CONFIG = "simulation/scenario.sumocfg"
    NET = "simulation/network.net.xml"
    
    # Check if files exist before running to avoid confusing errors
    if not os.path.exists(CONFIG):
        print(f" Error: Config file not found at {CONFIG}")
    elif not os.path.exists(NET):
        print(f" Error: Network file not found at {NET}")
    else:
        engine = RealTimeInferenceEngine(CONFIG, NET, use_gui=True)
        engine.initialize_model()
        engine.run(steps=200)