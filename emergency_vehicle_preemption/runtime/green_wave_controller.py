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
    def __init__(self, ev_id="EV_0"):
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
        self.sequence_length = 10
        self.state_buffer = deque(maxlen=self.sequence_length)
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
                
                # Dynamic EV Detection: Switch to whichever EV is active
                # This ensures the demo works even if EV_0 finishes and EV_1 starts
                self._update_target_ev()
                
                if self.ev_id:
                    self.control_step(step)
                
                step += 1
        except Exception as e:
            print(f"Error: {e}")
        finally:
            traci.close()

    def _update_target_ev(self):
        """Finds the first active EV in the simulation."""
        ids = traci.vehicle.getIDList()
        # If current EV is gone, or we don't have one, look for a new one
        if self.ev_id not in ids:
            evs = [x for x in ids if x.startswith("EV_")]
            if evs:
                self.ev_id = evs[0]
                self.state_buffer.clear() # Reset buffer for new vehicle
                self.preemption_active = False
                print(f"--- Tracking new vehicle: {self.ev_id} ---")
            else:
                self.ev_id = None

    def control_step(self, step):
        # 1. Get Live Data
        features = self._get_live_features()
        if features is None: return

        # 2. Add to buffer
        self.state_buffer.append(features)

        # 3. Predict ETA
        if len(self.state_buffer) == self.sequence_length:
            predicted_eta = self._predict_eta()
            
            # Display: Dist, Speed, Traffic Light State
            current_dist = features[2] 
            current_speed = features[0]
            tls_state = "Red" if features[6] == 1.0 else ("Yellow" if features[6] == 0.5 else "Green")

            print(f"Step {step} [{self.ev_id}]: Dist={current_dist:.1f}m | Light={tls_state} | ETA={predicted_eta:.2f}s")
            
            # 4. TRIGGER ACTION
            if predicted_eta < 30 and not self.preemption_active:
                self._activate_green_wave()

    def _get_live_features(self):
        """Extracts 7 features including Traffic Light State."""
        try:
            # Physics
            speed = traci.vehicle.getSpeed(self.ev_id)
            accel = traci.vehicle.getAcceleration(self.ev_id)
            lane_id = traci.vehicle.getLaneID(self.ev_id)
            pos = traci.vehicle.getLanePosition(self.ev_id)
            try: dist = traci.lane.getLength(lane_id) - pos
            except: dist = 0
            
            # Traffic Context
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            
            leader = traci.vehicle.getLeader(self.ev_id, 200)
            if leader:
                l_gap, l_speed = leader[1], traci.vehicle.getSpeed(leader[0])
            else:
                l_gap, l_speed = 200, 30
            
            # Traffic Light State
            tls_state_val = 0.0
            next_tls = traci.vehicle.getNextTLS(self.ev_id)
            if next_tls:
                state_char = next_tls[0][3]
                if state_char in ['r', 'R', 'u']: tls_state_val = 1.0 # Red
                elif state_char in ['y', 'Y']: tls_state_val = 0.5    # Yellow
                else: tls_state_val = 0.0                             # Green
            
            # Return features
            return [speed, accel, dist, queue, l_gap, l_speed, tls_state_val]
            
        except:
            return None

    def _predict_eta(self):
        """Prepares data and runs inference."""
        raw_sequence = np.array(self.state_buffer) # Shape (10, 7)
        
        # Define 7 columns for DataFrame creation
        feature_cols = ['speed', 'acceleration', 'distance_to_signal', 
                        'queue_length', 'leader_gap', 'leader_speed', 'tls_state']
        
        # Scale
        scaled_sequence = np.zeros_like(raw_sequence)
        for i in range(len(raw_sequence)):
            step_df = pd.DataFrame([raw_sequence[i]], columns=feature_cols)
            scaled_sequence[i] = self.scaler.transform(step_df)[0]
        
        # FIX: Reshape for LSTM: (1, 10, 7)
        input_data = scaled_sequence.reshape(1, self.sequence_length, 7)
        
        # Predict
        eta_scaled = self.model.predict(input_data, verbose=0)
        return eta_scaled[0][0]

    def _activate_green_wave(self):
        try:
            next_tls = traci.vehicle.getNextTLS(self.ev_id)
            if next_tls:
                tls_id = next_tls[0][0]
                
                # Logic
                traci.trafficlight.setPhase(tls_id, 0)
                
                # Visuals
                traci.vehicle.setColor(self.ev_id, (0, 0, 255, 255))
                traci.gui.trackVehicle("View #0", self.ev_id)
                traci.gui.setZoom("View #0", 600)
                
                print(f"!!! GREEN WAVE ACTIVATED FOR {tls_id} !!!")
                self.preemption_active = True
        except Exception as e:
            print(f"ERROR in _activate_green_wave: {e}")

if __name__ == "__main__":
    controller = GreenWaveController()
    controller.start()