import os
import sys
import traci
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/raw/")

class DataLogger:
    def __init__(self):
        """Initializes the logger to track MULTIPLE EVs."""
        self.eta_training_data = []
        # No 'ev_id' passed in init, we find them dynamically
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[EVPS Logger] Initialized. Watching for ALL vehicles starting with 'EV_'")

    def log_step(self, step):
        """Scans for any EV in the simulation and logs its data."""
        try:
            # Get all vehicle IDs currently in the simulation
            vehicle_ids = traci.vehicle.getIDList()
            
            # Filter for Emergency Vehicles
            active_evs = [v_id for v_id in vehicle_ids if v_id.startswith("EV_")]
            
            for ev_id in active_evs:
                self._collect_eta_data(step, ev_id)
                
        except Exception as e:
            print(f"Logger Error step {step}: {e}")

    def _collect_eta_data(self, step, ev_id):
        try:
            # Physics
            ev_speed = traci.vehicle.getSpeed(ev_id)
            ev_accel = traci.vehicle.getAcceleration(ev_id)
            ev_lane_id = traci.vehicle.getLaneID(ev_id)
            ev_pos = traci.vehicle.getLanePosition(ev_id)
            
            try:
                lane_len = traci.lane.getLength(ev_lane_id)
                dist_to_signal = lane_len - ev_pos
            except:
                dist_to_signal = 0

            # Traffic Context
            queue_len = traci.lane.getLastStepHaltingNumber(ev_lane_id)
            leader_info = traci.vehicle.getLeader(ev_id, 200)
            
            if leader_info:
                leader_gap = leader_info[1]
                try: leader_speed = traci.vehicle.getSpeed(leader_info[0])
                except: leader_speed = 30
            else:
                leader_gap = 200
                leader_speed = 30 # Max speed assumption

            self.eta_training_data.append({
                "run_id": "mega_run", # Group them all together
                "step": step,
                "ev_id": ev_id, # Track which EV this is
                "speed": ev_speed,
                "acceleration": ev_accel,
                "distance_to_signal": dist_to_signal,
                "queue_length": queue_len,
                "leader_gap": leader_gap,
                "leader_speed": leader_speed
            })
        except:
            pass

    def save_data(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.eta_training_data:
            df = pd.DataFrame(self.eta_training_data)
            path = os.path.join(OUTPUT_DIR, f"eta_data_MEGA_{timestamp}.csv")
            df.to_csv(path, index=False)
            print(f"[EVPS Logger] Saved MEGA dataset with {len(df)} rows to {path}")

# --- MAIN EXECUTION FOR MEGA RUN ---
if __name__ == "__main__":
    # Point to the new MEGA config relative to this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_FILE = os.path.join(BASE_DIR, "../simulation/config/mega_scenario.sumocfg")
    
    # Check if config exists
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Config file not found at {CONFIG_FILE}")
        print("Run 'mega_scenario_generator.py' first!")
        sys.exit(1)

    # Use 'sumo' (CLI) for speed, not GUI
    print("--- STARTING MEGA SIMULATION (50 EVs) ---")
    sumo_cmd = ["sumo", "-c", CONFIG_FILE]
    
    try:
        traci.start(sumo_cmd)
        logger = DataLogger()
        
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            logger.log_step(step)
            step += 1
            if step % 200 == 0:
                print(f"Simulating Step {step}...", end="\r")
                
        traci.close()
        logger.save_data()
        print("\n--- DONE ---")
    except Exception as e:
        print(f"Simulation failed: {e}")
        try: traci.close()
        except: pass