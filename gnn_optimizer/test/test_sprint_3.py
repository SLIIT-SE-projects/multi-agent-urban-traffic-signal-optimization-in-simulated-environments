import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from src.graphBuilder.sumo_manager import SumoManager
from src.graphBuilder.graph_builder import TrafficGraphBuilder
from src.models.hgat_core import RecurrentHGAT

# CONFIG
BASE_DIR = os.path.dirname(__file__)
SUMO_CONFIG = os.path.abspath(os.path.join(BASE_DIR, "..", "simulation", "scenario.sumocfg"))
SUMO_NET = os.path.abspath(os.path.join(BASE_DIR, "..", "simulation", "network.net.xml"))


def main():
    # 1. Setup Data Pipeline
    graph_builder = TrafficGraphBuilder(SUMO_NET)
    manager = SumoManager(SUMO_CONFIG, use_gui=False)
    manager.start()
    
    # 2. Get one snapshot to initialize the model
    manager.step()
    snapshot = manager.get_snapshot()
    data = graph_builder.create_hetero_data(snapshot)
    
    # 3. Initialize Model
    print("\n Initializing Recurrent HGAT Model...")
    model = RecurrentHGAT(
        hidden_channels=32, 
        out_channels=4,     # Assuming 4 Signal Phases
        num_heads=2, 
        metadata=data.metadata()
    )
    
    # 4. Run Inference Loop
    print("\n Running Inference Loop...")
    hidden_state = None # Memory starts empty
    
    try:
        for i in range(5):
            # A. Get Data
            manager.step()
            snapshot = manager.get_snapshot()
            data = graph_builder.create_hetero_data(snapshot)
            
            # B. Forward Pass
            action_logits, hidden_state = model(data.x_dict, data.edge_index_dict, hidden_state)
            
            # C. Print Result
            print(f"Step {i+1}: Output Shape {action_logits.shape} (Intersections, Phases)")
            
            # Detach hidden state to prevent memory leak in this test loop
            hidden_state = hidden_state.detach()

    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        manager.close()
        print("Done.")

if __name__ == "__main__":
    main()