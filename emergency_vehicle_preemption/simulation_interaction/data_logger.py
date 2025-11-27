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

    def _collect_eta_data(self, step):
        """Internal function to collect EV physics data."""
        try:
            ev_speed = traci.vehicle.getSpeed(self.ev_id)
            ev_accel = traci.vehicle.getAcceleration(self.ev_id)
            ev_lane_id = traci.vehicle.getLaneID(self.ev_id)
            ev_pos = traci.vehicle.getLanePosition(self.ev_id)
            
            try:
                lane_len = traci.lane.getLength(ev_lane_id)
                dist_to_end_of_lane = lane_len - ev_pos
            except:
                dist_to_end_of_lane = 0

            self.eta_training_data.append({
                "step": step,
                "ev_id": self.ev_id,
                "speed": ev_speed,
                "acceleration": ev_accel,
                "distance_to_signal": dist_to_end_of_lane,
                "lane_id": ev_lane_id
            })
        except traci.TraCIException:
            pass # Vehicle might have just left

    def _collect_safety_data(self, step):
        """Internal function to collect Intersection data."""
        try:
            next_tls = traci.vehicle.getNextTLS(self.ev_id)
            if next_tls:
                tls_id, tls_index, dist_to_tls, state = next_tls[0]
                
                # Only log if approaching (e.g., < 100m)
                if dist_to_tls < 100:
                    self._snapshot_intersection(step, tls_id)
        except traci.TraCIException:
            pass

    def _snapshot_intersection(self, step, tls_id):
        """Helper to snapshot the intersection state."""
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        queues = []
        speeds = []
        
        for lane in controlled_lanes:
            queues.append(traci.lane.getLastStepHaltingNumber(lane))
            speeds.append(traci.lane.getLastStepMeanSpeed(lane))

        current_phase = traci.trafficlight.getPhase(tls_id)

        self.safety_training_data.append({
            "step": step,
            "tls_id": tls_id,
            "max_queue_length": max(queues) if queues else 0,
            "mean_intersection_speed": np.mean(speeds) if speeds else 0,
            "current_phase": current_phase
        })

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