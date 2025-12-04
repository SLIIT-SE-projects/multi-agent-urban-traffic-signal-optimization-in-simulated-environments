import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.graphBuilder.sumo_manager import SumoManager
from src.config import SimConfig

# --- CONFIGURATION ---
SUMO_CONFIG_PATH = SimConfig.SUMO_CFG

def main():
    # 1. Initialize Manager
    print("Testing SumoManager...")
    
    if not os.path.exists(SUMO_CONFIG_PATH):
        print(f"ERROR: Could not find config file at {SUMO_CONFIG_PATH}")
        return

    manager = SumoManager(SUMO_CONFIG_PATH, use_gui=True)

    # 2. Start Simulation
    manager.start()

    # 3. Run for 5 steps and print data
    try:
        for i in range(5):
            manager.step()
            snapshot = manager.get_snapshot()
            
            print(f"\n--- Time Step {i+1} ---")
            # Print first 2 intersections found
            tls_ids = list(snapshot['intersections'].keys())[:2]
            for tls in tls_ids:
                print(f"Intersection {tls}: Phase={snapshot['intersections'][tls]['phase_index']}")
            
            # Print first 2 lanes found
            lane_ids = list(snapshot['lanes'].keys())[:2]
            for lane in lane_ids:
                data = snapshot['lanes'][lane]
                print(f"Lane {lane}: Queue={data['queue_length']}, Speed={data['avg_speed']:.2f}")
                
    except Exception as e:
        print(f"An error occurred during the loop: {e}")
    finally:
        manager.close()

if __name__ == "__main__":
    main()