import traci
import sumolib
import os
import sys

class SumoManager:
    def __init__(self, config_path, use_gui=True):

        self.config_path = config_path
        self.use_gui = use_gui
        self.connection = None

        # Cache to store how many phases each intersection actually has
        self.phase_counts = {}
        
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
            print(f" SUMO Simulation Started: {self.config_path}")
        except Exception as e:
            print(f" Error starting SUMO: {e}")
            sys.exit(1)

    def get_snapshot(self):

        # --- 1. Intersection Data ---
        tls_ids = traci.trafficlight.getIDList()
        intersection_data = {}
        
        for tls_id in tls_ids:
            # Feature: Current Phase Index
            current_phase = traci.trafficlight.getPhase(tls_id)
            
            # Feature: Time spent in current phase
            next_switch = traci.trafficlight.getNextSwitch(tls_id)
            current_time = traci.simulation.getTime()
            time_to_switch = next_switch - current_time
            
            intersection_data[tls_id] = {
                "phase_index": current_phase,
                "time_to_switch": time_to_switch
            }

        # --- 2. Lane Data ---
        # Focus on lanes that allow vehicles
        lane_ids = traci.lane.getIDList()
        lane_data = {}

        for lane_id in lane_ids:
            # Feature: Queue Length (number of vehicles with speed < 0.1 m/s)
            queue_len = traci.lane.getLastStepHaltingNumber(lane_id)
            
            # Feature: Occupancy (0 to 1 percentage of lane filled)
            occupancy = traci.lane.getLastStepOccupancy(lane_id)
            
            # Feature: Average Speed on the lane (m/s)
            avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            
            # Feature: CO2 Emissions (Optional, good for Environmental Metrics)
            co2_emission = traci.lane.getCO2Emission(lane_id)

            lane_data[lane_id] = {
                "queue_length": queue_len,
                "occupancy": occupancy,
                "avg_speed": avg_speed,
                "co2": co2_emission
            }

        return {
            "intersections": intersection_data,
            "lanes": lane_data
        }
    
    def apply_actions(self, actions_dict):

        for tls_id, phase_index in actions_dict.items():
            try:
                # 1. Check Cache
                if tls_id not in self.phase_counts:
                    # Query SUMO for the logic definition
                    logics = traci.trafficlight.getAllProgramLogics(tls_id)
                    if len(logics) > 0:
                        num_phases = len(logics[0].phases)
                        self.phase_counts[tls_id] = num_phases
                    else:
                        self.phase_counts[tls_id] = 4 

                # 2. Modulo Arithmetic to ensure safety
                max_phases = self.phase_counts[tls_id]
                safe_phase = int(phase_index) % max_phases
                
                # 3. Send Command
                current_phase = traci.trafficlight.getPhase(tls_id)
                if current_phase != safe_phase:
                    traci.trafficlight.setPhase(tls_id, safe_phase)
                    
            except Exception as e:
                # Log the error and continue
                pass

    def step(self):
        traci.simulationStep()

    def close(self):
        traci.close()
        print(" SUMO Simulation Closed.")