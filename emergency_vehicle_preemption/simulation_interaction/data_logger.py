import os
import sys
import traci
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/raw/")

class DataLogger:
    def __init__(self, ev_id="EV_1"):
        """
        Initializes the logger.
        :param ev_id: The ID of the vehicle to track (must match the Route file).
        """
        self.ev_id = ev_id
        self.eta_training_data = []
        self.safety_training_data = []
        
        # Ensure output directory exists immediately
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[EVPS Logger] Initialized. Watching for vehicle: {self.ev_id}")

    def log_step(self, step):
        """
        THIS IS THE FUNCTION YOUR TEAM MEMBER MUST CALL.
        It should be called once every simulation step.
        """
        try:
            # Check if our EV is in the simulation
            vehicle_ids = traci.vehicle.getIDList()
            if self.ev_id in vehicle_ids:
                self._collect_eta_data(step)
                self._collect_safety_data(step)
        except Exception as e:
            # Don't crash the main simulation if logging fails
            print(f"[EVPS Logger] Error at step {step}: {e}")

    def save_data(self):
        """
        MUST BE CALLED AT THE END OF SIMULATION.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.eta_training_data:
            df_eta = pd.DataFrame(self.eta_training_data)
            eta_path = os.path.join(OUTPUT_DIR, f"eta_data_{timestamp}.csv")
            df_eta.to_csv(eta_path, index=False)
            print(f"[EVPS Logger] Saved ETA data: {eta_path}")
        else:
            print("[EVPS Logger] No ETA data collected (EV not seen).")

        if self.safety_training_data:
            df_safety = pd.DataFrame(self.safety_training_data)
            safety_path = os.path.join(OUTPUT_DIR, f"safety_data_{timestamp}.csv")
            df_safety.to_csv(safety_path, index=False)
            print(f"[EVPS Logger] Saved Safety data: {safety_path}")


# --- STANDALONE TESTING BLOCK ---
# This runs ONLY if run 'python data_logger.py' directly.
# It will NOT run if other component imports this file.
if __name__ == "__main__":
    print("--- RUNNING IN STANDALONE TEST MODE ---")
    
    # 1. Configuration for Testing
    # POINT THIS TO YOUR LOCAL TEST SCENARIO
    CONFIG_FILE = "simulation/config/test_scenario.sumocfg" 
    
    # Check if config exists
    if not os.path.exists(CONFIG_FILE):
        # Try looking relative to this script if run from folder
        CONFIG_FILE = "../../simulation/config/test_scenario.sumocfg"
        
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Could not find config file at {CONFIG_FILE}")
        print("Please run scenario_generator.py first.")
        sys.exit(1)

    # 2. Start SUMO
    sumo_cmd = ["sumo-gui", "-c", CONFIG_FILE]
    traci.start(sumo_cmd)
    
    # 3. Instantiate the Logger
    logger = DataLogger(ev_id="EV_1")
    
    # 4. Run the Loop
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # --- CALL THE LOGGER ---
        logger.log_step(step)
        
        step += 1
    
    # 5. Save and Close
    traci.close()
    logger.save_data()