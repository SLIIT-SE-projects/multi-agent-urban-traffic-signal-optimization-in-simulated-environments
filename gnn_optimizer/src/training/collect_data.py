import os
import sys
import torch
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.graphBuilder.sumo_manager import SumoManager
from src.graphBuilder.graph_builder import TrafficGraphBuilder

# CONFIGURATION
SUMO_CONFIG = "simulation/scenario.sumocfg"
SUMO_NET = "simulation/network.net.xml"
OUTPUT_FOLDER = "experiments/raw_data"
STEPS_TO_COLLECT = 3600 * 2  # Collect data for 2 hours of simulation

def collect_dataset():
    # 1. Create Output Directory if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f" Created output folder: {OUTPUT_FOLDER}")
        
    print(f" Starting Data Collection for {STEPS_TO_COLLECT} steps...")
    
    # 2. Initialize Components
    manager = SumoManager(SUMO_CONFIG, use_gui=False)
    graph_builder = TrafficGraphBuilder(SUMO_NET)
    
    manager.start()
    
    collected_snapshots = []
    
    try:
        # 3. The Collection Loop
        for t in tqdm(range(STEPS_TO_COLLECT), desc="Collecting Data"):
            
            # This teaches the AI the "physics" of normal traffic flow.
            manager.step()
            
            # Capture Data
            snapshot = manager.get_snapshot()
            
            # Convert to Graph (HeteroData)
            graph_data = graph_builder.create_hetero_data(snapshot)
            
            # if want to remove 'pos' to save disk space
            # if hasattr(graph_data['intersection'], 'pos'):
            #     del graph_data['intersection'].pos
            #     del graph_data['lane'].pos
            
            collected_snapshots.append(graph_data)
            
        # 4. Save to Disk
        save_path = os.path.join(OUTPUT_FOLDER, "traffic_data_1hr.pt")
        print(f" Saving {len(collected_snapshots)} graph snapshots to {save_path}...")
        
        # Torch.save is very efficient for saving lists of tensors
        torch.save(collected_snapshots, save_path)
        print(" Data Collection Complete!")
        
    except Exception as e:
        print(f" Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        manager.close()

if __name__ == "__main__":
    collect_dataset()