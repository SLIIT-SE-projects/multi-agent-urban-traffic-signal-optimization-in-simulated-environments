import os
import sys
import traci
import threading
import time

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
                            traci.simulationStep()
                            self.current_step += 1
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
                traci.simulationStep()
                self.current_step += 1
            return self.get_current_data()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
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
    
