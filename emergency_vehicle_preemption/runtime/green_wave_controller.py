import os
import sys
import traci
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from collections import deque
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/saved/eta_predictor.h5")
SCALER_PATH = os.path.join(BASE_DIR, "../data/scalers/eta_scaler.pkl")
SUMO_CONFIG = os.path.join(BASE_DIR, "../simulation/config/test_scenario.sumocfg") 
class GreenWaveController:
    def __init__(self, ev_id="EV_1"):
        self.ev_id = ev_id
        print("Loading AI Models...")
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load models. {e}")
            sys.exit(1)
        self.sequence_length = 10
        self.state_buffer = deque(maxlen=self.sequence_length)
        self.preemption_active = False
    def start(self):
        sumo_cmd = ["sumo-gui", "-c", SUMO_CONFIG]
        traci.start(sumo_cmd)
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

        # 1. Check if EV is in simulation
        if self.ev_id not in traci.vehicle.getIDList():
            return
        
        # 2. Get Live Data
        features = self._get_live_features()
        if features is None:
            return

        # 3. Add to buffer
        self.state_buffer.append(features)

        # 4. Predict only when buffer full
        if len(self.state_buffer) == self.sequence_length:
            predicted_eta = self._predict_eta()
            print(f"Step {step}: Predicted ETA = {predicted_eta:.2f}s")

            if predicted_eta < 30 and not self.preemption_active:
                self._activate_green_wave()

    def _get_live_features(self):
        try:
            speed = traci.vehicle.getSpeed(self.ev_id)
            accel = traci.vehicle.getAcceleration(self.ev_id)
            lane_id = traci.vehicle.getLaneID(self.ev_id)
            pos = traci.vehicle.getLanePosition(self.ev_id)

            try:
                dist = traci.lane.getLength(lane_id) - pos
            except:
                dist = 0

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

            return [speed, accel, dist, queue, leader_gap, leader_speed]

        except:
            return None
        
    def _predict_eta(self):
        raw_sequence = np.array(self.state_buffer)

        feature_cols = [
            'speed', 'acceleration', 'distance_to_signal',
            'queue_length', 'leader_gap', 'leader_speed'
        ]

        scaled_sequence = np.zeros_like(raw_sequence)

        for i in range(len(raw_sequence)):
            step_df = pd.DataFrame([raw_sequence[i]], columns=feature_cols)
            scaled_sequence[i] = self.scaler.transform(step_df)[0]

        input_data = scaled_sequence.reshape(1, self.sequence_length, 6)
        eta_scaled = self.model.predict(input_data, verbose=0)
        return eta_scaled[0][0]

    def _activate_green_wave(self):
        try:
            next_tls = traci.vehicle.getNextTLS(self.ev_id)

            if next_tls:
                tls_id = next_tls[0][0]
                traci.trafficlight.setPhase(tls_id, 0)
                print(f"!!! GREEN WAVE ACTIVATED FOR {tls_id} !!!")
                self.preemption_active = True
            else:
                print("DEBUG: Preemption triggered but no traffic light found.")
        except Exception as e:
            print(f"ERROR in _activate_green_wave: {e}")

