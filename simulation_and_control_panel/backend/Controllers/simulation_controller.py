import os
import sys
import traci
import threading
import time
import xml.etree.ElementTree as ET
from optimizers.gnn_adapter import GNNTrafficOptimizer

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME environment variable")


class SimulationController:
    def __init__(self, config_file, use_gui=True, step_delay=0.1):
        self.config_file = config_file
        self.use_gui = use_gui
        self.default_step_delay = step_delay
        self.is_running = False
        self.is_paused = False
        self.current_step = 0
        self.auto_stepping = False
        self.auto_step_thread = None
        self.step_lock = threading.Lock()  # Lock for thread-safe stepping
        self.optimizer = None
        self.optimization_enabled = False
        self.action_interval = 15
        self.last_action_step = 0
        self.green_phases = {}
        self.pending_switches = {}
        self.yellow_timers = {}
        self.YELLOW_DURATION = 3

    def _get_net_file_from_config(self):
        """
        Parses the current .sumo.cfg to find the associated .net.xml file.
        This ensures the AI uses the EXACT same map as the simulation.
        """
        try:
            tree = ET.parse(self.config_file)
            root = tree.getroot()
            
            # Look for <net-file value="..."/>
            net_file_entry = root.find(".//net-file")
            if net_file_entry is None or 'value' not in net_file_entry.attrib:
                raise ValueError("Could not find <net-file> in sumo config")
            
            relative_net_path = net_file_entry.attrib['value']
            
            # Resolve path relative to the config file location
            config_dir = os.path.dirname(self.config_file)
            net_file_path = os.path.normpath(os.path.join(config_dir, relative_net_path))
            
            if not os.path.exists(net_file_path):
                raise FileNotFoundError(f"Network file not found at: {net_file_path}")
                
            return net_file_path
            
        except Exception as e:
            print(f"❌ Error resolving network file: {e}")
            return None

    def load_optimizer(self, model_type="gnn"):
        """Load the AI model with the CURRENT simulation map"""
        print(f"🔌 Loading {model_type.upper()} Optimizer...")
        
        # 1. Dynamically find the map file
        current_net_file = self._get_net_file_from_config()
        
        if not current_net_file:
            print("❌ Cannot load Optimizer: Network file could not be determined from config.")
            return {"status": "error", "message": "Network file not found"}

        try:
            if model_type == "gnn":
                # 2. Inject the map file into the GNN Adapter
                self.optimizer = GNNTrafficOptimizer(
                    net_path=current_net_file
                )
                self.optimization_enabled = True
                print(f"✅ GNN Optimizer attached to: {os.path.basename(current_net_file)}")
                return {"status": "success", "message": "GNN Optimizer Loaded"}
                
        except Exception as e:
            print(f"❌ Failed to load optimizer: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    # =========================================================================
    # CORE SIMULATION LOGIC (Refactored)
    # =========================================================================
    
    def _advance_simulation(self):
        """
        Central method to advance the simulation by one step.
        Handles: AI Inference -> Yellow Phase Management -> Traci Step
        """
        # 1. AI OPTIMIZATION HOOK
        if self.optimization_enabled and self.optimizer:
            if (self.current_step - self.last_action_step) >= self.action_interval:
                print(f"🚦 GNN Action Step: {self.current_step} | Calculating phases...")
                try:
                    snapshot = self._capture_snapshot_for_ai()
                    actions = self.optimizer.predict(snapshot)
                    self._apply_ai_actions(actions)
                    self.last_action_step = self.current_step
                except Exception as e:
                    print(f"⚠️ AI Prediction Error: {e}")

        # 2. YELLOW PHASE MANAGEMENT (Critical for Safety)
        completed_transitions = []
        for tls_id in list(self.yellow_timers.keys()):
            self.yellow_timers[tls_id] -= 1
            if self.yellow_timers[tls_id] <= 0:
                if tls_id in self.pending_switches:
                    final_phase = self.pending_switches[tls_id]
                    try:
                        traci.trafficlight.setPhase(tls_id, final_phase)
                    except:
                        pass
                    del self.pending_switches[tls_id]
                completed_transitions.append(tls_id)
        
        for tls_id in completed_transitions:
            del self.yellow_timers[tls_id]

        # 3. ADVANCE SUMO
        traci.simulationStep()
        self.current_step += 1

    def start_auto_stepping(self, step_delay=None):
        """Start automatic stepping in background"""
        if not self.is_running:
            return {"status": "error", "message": "Simulation not running. Call /start first"}
        
        if self.is_paused:
            return {"status": "error", "message": "Simulation is paused. Resume the simulation first"}
        
        if self.auto_stepping:
            return {"status": "error", "message": "Auto-stepping already active"}
        
        # Use provided delay or default
        if step_delay is None:
            step_delay = self.default_step_delay
        
        self.auto_stepping = True

        def step_loop():
            while self.auto_stepping and self.is_running:
                if not self.is_paused:
                    try:
                        with self.step_lock:
                            self._advance_simulation()
                        time.sleep(step_delay)
                    except Exception as e:
                        print(f"Error in auto-stepping: {e}")
                        self.auto_stepping = False
                        break
                else:
                    time.sleep(0.1)

        self.auto_step_thread = threading.Thread(target=step_loop, daemon=True)
        self.auto_step_thread.start()

        return {"status": "success", "message": "Auto-stepping started", "step_delay": step_delay}
    
    def _capture_snapshot_for_ai(self):
        tls_ids = traci.trafficlight.getIDList()
        intersections = {}
        for tls in tls_ids:
            intersections[tls] = {
                "phase_index": traci.trafficlight.getPhase(tls),
                "time_to_switch": traci.trafficlight.getNextSwitch(tls) - traci.simulation.getTime()
            }
        
        lane_ids = traci.lane.getIDList()
        lanes = {}
        for lane in lane_ids:
            lanes[lane] = {
                "queue_length": traci.lane.getLastStepHaltingNumber(lane),
                "occupancy": traci.lane.getLastStepOccupancy(lane),
                "avg_speed": traci.lane.getLastStepMeanSpeed(lane),
                "co2": traci.lane.getCO2Emission(lane),
                "waiting_time": traci.lane.getWaitingTime(lane)
            }
        return {"intersections": intersections, "lanes": lanes}

    def _get_yellow_phase(self, tls_id, current_phase):
        try:
            logics = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phases = logics.phases
            num_phases = len(phases)
            next_p_idx = (current_phase + 1) % num_phases
            next_state = phases[next_p_idx].state.lower()
            if 'y' in next_state or 'u' in next_state:
                return next_p_idx
            return current_phase
        except:
            return current_phase

    def _apply_ai_actions(self, actions):
        for tls_id, action_idx in actions.items():
            try:
                # Initialize greens if unknown
                if tls_id not in self.green_phases:
                    logics = traci.trafficlight.getAllProgramLogics(tls_id)
                    if len(logics) > 0:
                        phases = logics[0].phases
                        greens = []
                        for i, p in enumerate(phases):
                            state = p.state.lower()
                            if ('g' in state) and ('y' not in state) and ('u' not in state):
                                greens.append(i)
                        self.green_phases[tls_id] = greens if greens else [0]
                    else:
                        self.green_phases[tls_id] = [0]

                valid_greens = self.green_phases[tls_id]
                action_idx = int(action_idx)
                if action_idx >= len(valid_greens):
                    action_idx = len(valid_greens) - 1
                
                target_phase = valid_greens[action_idx]
                current_phase = traci.trafficlight.getPhase(tls_id)
                
                if current_phase == target_phase: continue
                if tls_id in self.pending_switches:
                    self.pending_switches[tls_id] = target_phase
                    continue

                yellow_phase = self._get_yellow_phase(tls_id, current_phase)
                
                if yellow_phase == current_phase:
                    self.pending_switches[tls_id] = target_phase
                    self.yellow_timers[tls_id] = 1
                else:
                    traci.trafficlight.setPhase(tls_id, yellow_phase)
                    self.pending_switches[tls_id] = target_phase
                    self.yellow_timers[tls_id] = self.YELLOW_DURATION
            except Exception as e:
                print(f"Error applying action to {tls_id}: {e}")

    def stop_auto_stepping(self):
        """Stop automatic stepping"""
        if not self.auto_stepping:
            return {"status": "error", "message": "Auto-stepping not active"}
        
        self.auto_stepping = False
        if self.auto_step_thread:
            self.auto_step_thread.join(timeout=2)

        return {"status": "success", "message": "Auto-stepping stopped"}

    def pause_auto_stepping(self):
        """Pause auto-stepping (freeze simulation while keeping thread alive)"""
        if not self.auto_stepping:
            return {"status": "error", "message": "Auto-stepping not active"}
        
        self.is_paused = True
        return {"status": "success", "message": "Auto-stepping paused", "step": self.current_step}

    def resume_auto_stepping(self):
        """Resume auto-stepping"""
        if not self.auto_stepping:
            return {"status": "error", "message": "Auto-stepping not active"}
        
        if not self.is_paused:
            return {"status": "error", "message": "Auto-stepping not paused"}
        
        self.is_paused = False
        return {"status": "success", "message": "Auto-stepping resumed", "step": self.current_step}

    def start(self):
        """Start the simulation"""
        if self.is_running:
            return {"status": "error", "message": "Simulation already running"}
            
        # Choose SUMO binary based on use_gui setting
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.config_file, "--start"]
        
        try:
            traci.start(sumo_cmd)
            self.is_running = True
            self.is_paused = False
            self.current_step = 0
            print(f"Simulation started with config: {self.config_file}")
            print(f"GUI mode: {self.use_gui}")
            
            return {
                "status": "success", 
                "message": "Simulation started", 
                "step": self.current_step,
                "auto_stepping": self.auto_stepping
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def step(self):
        """Execute one simulation step"""
        if not self.is_running:
            return {"status": "error", "message": "Simulation not running"}
        
        if self.auto_stepping:
            return {"status": "error", "message": "Cannot manually step when auto-stepping is active. Stop auto-stepping first."}
        
        if self.is_paused:
            return {"status": "error", "message": "Simulation is paused"}
            
        try:
            with self.step_lock:

                self._advance_simulation()

                traci.simulationStep()
                self.current_step += 1
            return {"status": "success", "message": "Step executed", "step": self.current_step}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
    def _capture_snapshot_for_ai(self):
        """Helper to get data in the format GNN expects"""
        # Mirror the logic from GNN's sumo_manager.get_snapshot()
        tls_ids = traci.trafficlight.getIDList()
        intersections = {}
        for tls in tls_ids:
            intersections[tls] = {
                "phase_index": traci.trafficlight.getPhase(tls),
                "time_to_switch": traci.trafficlight.getNextSwitch(tls) - traci.simulation.getTime()
            }
            
        lane_ids = traci.lane.getIDList()
        lanes = {}
        for lane in lane_ids:
            lanes[lane] = {
                "queue_length": traci.lane.getLastStepHaltingNumber(lane),
                "occupancy": traci.lane.getLastStepOccupancy(lane),
                "avg_speed": traci.lane.getLastStepMeanSpeed(lane),
                "co2": traci.lane.getCO2Emission(lane),
                "waiting_time": traci.lane.getWaitingTime(lane)
            }
        return {"intersections": intersections, "lanes": lanes}

    # def _apply_ai_actions(self, actions):
    #     """Directly apply phases (Basic version - Add Yellow logic for production)"""
    #     for tls_id, phase_idx in actions.items():
    #         try:
    #             traci.trafficlight.setPhase(tls_id, int(phase_idx))
    #         except Exception as e:
    #             print(f"Failed to set phase for {tls_id}: {e}")
        
    def _apply_ai_actions(self, actions):
        for tls_id, action_idx in actions.items():
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
    
    def pause(self):
        """Pause simulation"""
        if not self.is_running:
            return {"status": "error", "message": "Simulation not running"}
            
        self.is_paused = True
        return {"status": "success", "message": "Simulation paused", "step": self.current_step}
    
    def resume(self):
        """Resume simulation"""
        if not self.is_running:
            return {"status": "error", "message": "Simulation not running"}
            
        if not self.is_paused:
            return {"status": "error", "message": "Simulation not paused"}
            
        self.is_paused = False
        return {"status": "success", "message": "Simulation resumed", "step": self.current_step}
    
    def stop(self):
        """Stop and close simulation"""
        if not self.is_running:
            return {"status": "error", "message": "Simulation not running"}
            
        try:
            self.auto_stepping = False
            if self.auto_step_thread:
                self.auto_step_thread.join(timeout=2)
            
            traci.close()
            self.is_running = False
            self.is_paused = False
            self.current_step = 0
            print("Simulation stopped")
            return {"status": "success", "message": "Simulation stopped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
