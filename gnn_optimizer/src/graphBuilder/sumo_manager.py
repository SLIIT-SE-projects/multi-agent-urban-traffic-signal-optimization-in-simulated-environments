import traci
import sumolib
import os
import sys

class SumoManager:
    def __init__(self, config_path, use_gui=True):

        self.config_path = config_path
        self.use_gui = use_gui
        self.connection = None
        
        # Check if SUMO_HOME is set (crucial for sumolib)
        if 'SUMO_HOME' not in os.environ:
            print("WARNING: SUMO_HOME environment variable is not set.")

    def start(self):
        # Locate the SUMO binary
        if self.use_gui:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')

        # The command to start SUMO
        cmd = [sumo_binary, "-c", self.config_path, "--start"]
        
        try:
            traci.start(cmd)
            self.connection = traci.getConnection()
            print(f"🚀 SUMO Simulation Started: {self.config_path}")
        except Exception as e:
            print(f"❌ Error starting SUMO: {e}")
            sys.exit(1)

    def close(self):
        traci.close()
        print("🛑 SUMO Simulation Closed.")