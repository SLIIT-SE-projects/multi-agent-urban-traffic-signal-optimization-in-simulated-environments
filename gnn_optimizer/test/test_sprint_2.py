import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.graphBuilder.sumo_manager import SumoManager
from src.graphBuilder.graph_builder import TrafficGraphBuilder
from src.utils.visualizer import plot_graph_topology_ver_2

# --- CONFIGURATION ---
# Update these to your actual file paths
SUMO_CONFIG_PATH = "simulation/scenario.sumocfg" # Your SUMO config file
SUMO_NET_PATH = "simulation/network.net.xml" # Your map file

def main():
    if not os.path.exists(SUMO_NET_PATH):
        print(f"Error: Network file not found at {SUMO_NET_PATH}")
        return

    # 1. Initialize Graph Builder (Parses the Map)
    print("Initializing Graph Builder...")
    graph_builder = TrafficGraphBuilder(SUMO_NET_PATH)

    # 2. Initialize Simulation
    print("Starting Simulation...")
    manager = SumoManager(SUMO_CONFIG_PATH, use_gui=False) # Headless for speed
    manager.start()

    try:
        # 3. Run Loop
        for i in range(3):
            manager.step()
            
            # A. Get Raw Numbers
            snapshot = manager.get_snapshot()
            
            # B. Convert to Tensor Graph
            graph_data = graph_builder.create_hetero_data(snapshot)
            
            print(f"\n--- Time Step {i+1} ---")
            print(f"Graph Object: {graph_data}")
            print(f"Intersection Node Features: {graph_data['intersection'].x.shape}")
            print(f"Lane Node Features: {graph_data['lane'].x.shape}")
            print(f"Edge ('lane', 'part_of', 'intersection'): {graph_data['lane', 'part_of', 'intersection'].edge_index.shape}")

        # --- AFTER THE LOOP (Visualize the last frame) ---
        print("\nGenerating Visualization for Demonstration...")
        plot_graph_topology_ver_2(graph_data)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        manager.close()

if __name__ == "__main__":
    main()