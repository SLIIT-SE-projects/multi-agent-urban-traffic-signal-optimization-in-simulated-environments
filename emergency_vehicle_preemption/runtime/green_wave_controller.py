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
        if self.ev_id not in traci.vehicle.getIDList():
            return
