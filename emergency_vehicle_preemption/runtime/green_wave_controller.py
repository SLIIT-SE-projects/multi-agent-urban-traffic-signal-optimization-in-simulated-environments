import os
import sys
import traci
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from collections import deque

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/saved/eta_predictor.h5")
SCALER_PATH = os.path.join(BASE_DIR, "../data/scalers/eta_scaler.pkl")
SUMO_CONFIG = os.path.join(BASE_DIR, "../simulation/config/mega_scenario.sumocfg") 

class GreenWaveController:
    def __init__(self, ev_id="EV_1"):
        self.ev_id = ev_id
        
        # 1. Load the "Brain" (Model & Scaler)
        print("Loading AI Models...")
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            print("SUCCESS: Models loaded.")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load models. {e}")
            sys.exit(1)

        # 2. Initialize Memory Buffer
        # The LSTM needs the last 10 steps to make 1 prediction
        self.sequence_length = 10
        self.state_buffer = deque(maxlen=self.sequence_length)
        
        # State tracking
        self.preemption_active = False

    def start(self):
        """Starts SUMO and the main control loop."""
        sumo_cmd = ["sumo-gui", "-c", SUMO_CONFIG]
        traci.start(sumo_cmd)
        print("Simulation Started. Waiting for EV...")
        
        step = 0
        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                self.control_step(step)
                step += 1
        except Exception as e:
            print(f"Error: {e}")
        finally:
            traci.close()

    def control_step(self, step):
        """
        Main logic executed every second.
        """
        # 1. Check if EV is in simulation
        if self.ev_id not in traci.vehicle.getIDList():
            return

        # 2. Get Live Data
        features = self._get_live_features()
        if features is None: return

        # 3. Add to buffer
        self.state_buffer.append(features)

        # 4. Predict ETA
        if len(self.state_buffer) == self.sequence_length:
            predicted_eta = self._predict_eta()
            
            # Get current distance for display
            # self.state_buffer[-1] is the latest [speed, accel, dist, queue, gap, l_speed]
            current_dist = self.state_buffer[-1][2] 
            current_speed = self.state_buffer[-1][0]

            print(f"Step {step}: Dist={current_dist:.1f}m | Speed={current_speed:.1f} | ETA={predicted_eta:.2f}s")
            
            # 5. TRIGGER ACTION
            if predicted_eta < 30 and not self.preemption_active:
                self._activate_green_wave()

    def _get_live_features(self):
        """Extracts the same 6 features used in training."""
        try:
            # Physics
            speed = traci.vehicle.getSpeed(self.ev_id)
            accel = traci.vehicle.getAcceleration(self.ev_id)
            lane_id = traci.vehicle.getLaneID(self.ev_id)
            pos = traci.vehicle.getLanePosition(self.ev_id)
            try:
                dist = traci.lane.getLength(lane_id) - pos
            except:
                dist = 0
            
            # Traffic Context
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            leader = traci.vehicle.getLeader(self.ev_id, 200)
            if leader:
                leader_gap = leader[1]
                try:
                    leader_speed = traci.vehicle.getSpeed(leader[0])
                except:
                    leader_speed = 30
            else:
                leader_gap = 200
                leader_speed = 30
            
            # Return as list [speed, accel, dist, queue, gap, l_speed]
            return [speed, accel, dist, queue, leader_gap, leader_speed]
            
        except:
            return None

    def _predict_eta(self):
        """Prepares data and runs inference."""
        # Convert buffer to numpy array
        raw_sequence = np.array(self.state_buffer) # Shape (10, 6)
        
        # FIX: Create a DataFrame with the correct column names to silence the warning
        feature_cols = ['speed', 'acceleration', 'distance_to_signal', 
                        'queue_length', 'leader_gap', 'leader_speed']
        
        # We need to scale each step in the sequence individually
        scaled_sequence = np.zeros_like(raw_sequence)
        for i in range(len(raw_sequence)):
            # Convert single step to DataFrame
            step_df = pd.DataFrame([raw_sequence[i]], columns=feature_cols)
            # Transform
            scaled_sequence[i] = self.scaler.transform(step_df)[0]
        
        # Reshape for LSTM: (1, 10, 6)
        input_data = scaled_sequence.reshape(1, self.sequence_length, 6)
        
        # Predict
        eta_scaled = self.model.predict(input_data, verbose=0)
        return eta_scaled[0][0]

    def _activate_green_wave(self):
        """
        Action: Turns traffic light Green.
        """
        try:
            next_tls = traci.vehicle.getNextTLS(self.ev_id)
            if next_tls:
                tls_id = next_tls[0][0]
                
                # 1. LOGIC: Force Phase 0 (Green)
                traci.trafficlight.setPhase(tls_id, 0)
                
                
                print(f"!!! GREEN WAVE ACTIVATED FOR {tls_id} !!!")
                self.preemption_active = True
                
                # 2. VISUAL: Track the vehicle in the GUI
                traci.gui.trackVehicle("View #0", self.ev_id)
                traci.gui.setZoom("View #0", 1000) # Zoom in
                
            else:
                print("DEBUG: Preemption triggered, but no traffic light found ahead.")
                
        except Exception as e:
            print(f"ERROR in _activate_green_wave: {e}")

if __name__ == "__main__":
    controller = GreenWaveController()
    controller.start()