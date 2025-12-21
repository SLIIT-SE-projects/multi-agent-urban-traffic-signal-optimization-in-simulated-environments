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
        
        # --- ABSOLUTE CONTROL STATE ---
        self.active_override_tls = None 
        self.active_green_lane = None
        
        self.smoothed_eta = None 
        self.alpha = 0.3

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
                self._release_control() 
                self.smoothed_eta = None
                print(f"--- Tracking new vehicle: {self.ev_id} ---")
            else:
                self.ev_id = None

    def _release_control(self):
        if self.active_override_tls:
            try:
                # Reset to normal program (0)
                traci.trafficlight.setProgram(self.active_override_tls, "0")
                print(f"RELEASED control of {self.active_override_tls}")
            except:
                pass
            self.active_override_tls = None
            self.active_green_lane = None

    def control_step(self, step):
        features = self._get_live_features()
        if features is None: return

        self.state_buffer.append(features)

        if len(self.state_buffer) == self.sequence_length:
            raw_eta = self._predict_eta()
            if self.smoothed_eta is None: self.smoothed_eta = raw_eta
            else: self.smoothed_eta = (self.alpha * raw_eta) + ((1 - self.alpha) * self.smoothed_eta)
            
            current_dist = features[2] 
            tls_val = features[6]
            tls_text = "Red" if tls_val == 1.0 else ("Yellow" if tls_val == 0.5 else "Green")

            self._manage_preemption(self.smoothed_eta)

            print(f"Step {step} [{self.ev_id}]: Dist={current_dist:.1f}m | Light={tls_text} | ETA={self.smoothed_eta:.2f}s")

    def _manage_preemption(self, eta):
        try:
            next_tls_info = traci.vehicle.getNextTLS(self.ev_id)
            if not next_tls_info:
                if self.active_override_tls: self._release_control()
                return

            target_tls_id = next_tls_info[0][0]
            dist_to_light = next_tls_info[0][2]

            # If we moved past the light we controlled, release it
            if self.active_override_tls and self.active_override_tls != target_tls_id:
                self._release_control()
            
            # TRIGGER CONDITION
            if eta < 30 or dist_to_light < 100:
                self._force_green_wave(target_tls_id)

        except Exception as e:
            print(f"Error in preemption logic: {e}")

    def _force_green_wave(self, tls_id):
        """
        ABSOLUTE CONTROL: ALL RED except EV lane.
        """
        try:
            ev_lane = traci.vehicle.getLaneID(self.ev_id)
            
            # 1. Get current state definition to know length
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase_def = logic.phases[0].state 
            num_links = len(current_phase_def)
            
            # 2. Start with ALL RED ('r')
            new_state_list = ['r'] * num_links
            
            # 3. Find ALL indices controlling the EV's lane
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            # controlled_links is list of lists. Index i -> connections for that signal bit
            
            found_lane = False
            for i, links in enumerate(controlled_links):
                for link in links:
                    # link[0] is incoming lane ID
                    if link[0] == ev_lane:
                        new_state_list[i] = 'G' # Force Green
                        found_lane = True
            
            if not found_lane:
                print(f"DEBUG: EV lane {ev_lane} not found in TLS {tls_id} links.")
                return

            new_state_string = "".join(new_state_list)

            # 4. FORCE STATE
            traci.trafficlight.setRedYellowGreenState(tls_id, new_state_string)
            
            self.active_override_tls = tls_id
            self.active_green_lane = ev_lane
            
            # traci.vehicle.setColor(self.ev_id, (0, 0, 255, 255))
            # traci.gui.trackVehicle("View #0", self.ev_id)
            # traci.gui.setZoom("View #0", 600)

        except Exception as e:
            print(f"Error enforcing green wave: {e}")

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
                l_gap = leader[1]
                try:
                    l_speed = traci.vehicle.getSpeed(leader[0])
                except:
                    l_speed = 30
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
        raw_sequence = np.array(self.state_buffer) # Shape (10, 7)

        # Define 7 columns for DataFrame creation
        feature_cols = ['speed', 'acceleration', 'distance_to_signal', 
                        'queue_length', 'leader_gap', 'leader_speed', 'tls_state']
        # Scale
        scaled_sequence = np.zeros_like(raw_sequence)
        for i in range(len(raw_sequence)):
            # Convert single step to DataFrame
            step_df = pd.DataFrame([raw_sequence[i]], columns=feature_cols)
            # Transform
            scaled_sequence[i] = self.scaler.transform(step_df)[0]
        # Reshape for LSTM: (1, 10, 7)
        input_data = scaled_sequence.reshape(1, self.sequence_length, 7)
        
        # Predict
        eta_scaled = self.model.predict(input_data, verbose=0)
        return eta_scaled[0][0]

if __name__ == "__main__":
    controller = GreenWaveController()
    controller.start()