import traci
import sumolib
import os
import sys
from src.config import GraphConfig

class SumoManager:
    def __init__(self, config_path, use_gui=True):
        self.config_path = config_path
        self.use_gui = use_gui
        self.connection = None
        self.green_phases = {}
        self.pending_switches = {} 
        self.yellow_timers = {}
        self.YELLOW_DURATION = 3 

        if 'SUMO_HOME' not in os.environ:
            print("WARNING: SUMO_HOME environment variable is not set.")

    def start(self):
        if self.use_gui:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')
        cmd = [sumo_binary, "-c", self.config_path, "--start"]
        try:
            traci.start(cmd)
            self.connection = traci.getConnection()
            print(f" SUMO Simulation Started: {self.config_path}")
        except Exception as e:
            print(f" Error starting SUMO: {e}")
            sys.exit(1)

    def get_snapshot(self):

        # 1. Intersection Data
        tls_ids = traci.trafficlight.getIDList()
        intersection_data = {}

        for tls_id in tls_ids:
            intersection_data[tls_id] = {
                "phase_index": traci.trafficlight.getPhase(tls_id),
                "time_to_switch": traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
            }
        lane_ids = traci.lane.getIDList()
        lane_data = {}
        for lane_id in lane_ids:
            lane_data[lane_id] = {
                "queue_length": traci.lane.getLastStepHaltingNumber(lane_id),
                "occupancy": traci.lane.getLastStepOccupancy(lane_id),
                "avg_speed": traci.lane.getLastStepMeanSpeed(lane_id),
                "co2": traci.lane.getCO2Emission(lane_id),
                "waiting_time": traci.lane.getWaitingTime(lane_id)
            }
        return {"intersections": intersection_data, "lanes": lane_data}

    def _get_yellow_phase(self, tls_id, current_phase):
        try:
            logics = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phases = logics.phases
            num_phases = len(phases)
            next_p_idx = (current_phase + 1) % num_phases
            next_state = phases[next_p_idx].state.lower()
            # If next phase is yellow ('y') or amber/red-yellow ('u'), return it
            if 'y' in next_state or 'u' in next_state:
                return next_p_idx
            return current_phase 
        except: return current_phase

    def apply_actions(self, actions_dict):
        for tls_id, action_idx in actions_dict.items():
            try:
                # 1. Discover Valid Green Phases (Once per light)
                if tls_id not in self.green_phases:
                    logics = traci.trafficlight.getAllProgramLogics(tls_id)
                    if len(logics) > 0:
                        phases = logics[0].phases
                        greens = []
                        for i, p in enumerate(phases):
                            state = p.state.lower()
                            # STRICTER CHECK: Must have 'g' AND no 'y' (yellow) AND no 'u' (red-yellow)
                            if ('g' in state) and ('y' not in state) and ('u' not in state):
                                greens.append(i)
                        
                        if len(greens) == 0: greens = [0] 
                        self.green_phases[tls_id] = greens
                    else:
                        self.green_phases[tls_id] = [0]

                # Map AI Action to Valid Green
                valid_greens = self.green_phases[tls_id]
                action_idx = int(action_idx)
                if action_idx >= len(valid_greens):
                    action_idx = len(valid_greens) - 1
                
                target_phase = valid_greens[action_idx]
                
                # Logic for Transition
                current_phase = traci.trafficlight.getPhase(tls_id)
                if current_phase == target_phase: continue
                if tls_id in self.pending_switches:
                    self.pending_switches[tls_id] = target_phase
                    continue

                yellow_phase = self._get_yellow_phase(tls_id, current_phase)
                
                # If we are already in yellow (or transition is impossible), force target
                if yellow_phase == current_phase:
                    # Just schedule it instantly
                    self.pending_switches[tls_id] = target_phase
                    self.yellow_timers[tls_id] = 1 
                else:
                    # Switch to Yellow
                    traci.trafficlight.setPhase(tls_id, yellow_phase)
                    self.pending_switches[tls_id] = target_phase
                    self.yellow_timers[tls_id] = self.YELLOW_DURATION

            except Exception as e:
                pass

    def step(self):
        completed_transitions = []
        for tls_id in list(self.yellow_timers.keys()):
            self.yellow_timers[tls_id] -= 1
            if self.yellow_timers[tls_id] <= 0:
                if tls_id in self.pending_switches:
                    final_phase = self.pending_switches[tls_id]
                    try: traci.trafficlight.setPhase(tls_id, final_phase)
                    except: pass
                    del self.pending_switches[tls_id]
                completed_transitions.append(tls_id)
        
        for tls_id in completed_transitions: del self.yellow_timers[tls_id]
        traci.simulationStep()

    def close(self):
        traci.close()
        print(" SUMO Simulation Closed.")