import sys
import os
import threading
import time
import eventlet

# --- PATH FIX ---
# Allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import SimConfig, FileConfig, TrainConfig
from src.inference.engine import RealTimeInferenceEngine

class OptimizationService:
    def __init__(self, socketio):
        self.socketio = socketio
        self.engine = None
        self.running = False
        self.thread = None

    def start_simulation(self):
        """Starts the SUMO + GNN Engine in a background thread."""
        if self.running:
            return {"status": "Already running"}

        print("🔌 Booting up GNN Optimization Service...")
        
        # Initialize the Engine
        self.engine = RealTimeInferenceEngine(
            config_path=SimConfig.SUMO_CFG,
            net_path=SimConfig.NET_FILE,
            model_path=FileConfig.FINAL_MARL_MODEL_PATH,
            use_gui=True # Set to False if you want headless for the server
        )
        self.engine.initialize_model()
        
        self.running = True
        self.thread = self.socketio.start_background_task(target=self._run_loop)
        return {"status": "Started"}

    def stop_simulation(self):
        """Stops the loop safely."""
        self.running = False
        if self.engine:
            try:
                self.engine.manager.close()
            except:
                pass
        return {"status": "Stopped"}

    def _run_loop(self):
        """The main loop that steps the simulation and emits data."""
        print("🚀 Traffic Optimization Loop Started!")
        
        try:
            step = 0
            idx_to_id = {v: k for k, v in self.engine.graph_builder.tls_map.items()}
            
            # Action Interval logic from engine.py
            ACTION_INTERVAL = 15
            current_actions = {}

            while self.running:
                # 1. ACTION LOGIC
                if step % ACTION_INTERVAL == 0:
                    snapshot = self.engine.manager.get_snapshot()
                    data = self.engine.graph_builder.create_hetero_data(snapshot)
                    
                    # Inference
                    actions, _ = self.engine.get_uncertainty_prediction(data) if hasattr(self.engine, 'get_uncertainty_prediction') else (None, None)
                    
                    # Fallback to standard inference if get_uncertainty_prediction not available
                    if actions is None:
                        # ... (Copy standard inference logic here if needed or import)
                        # For now, let's assume we just step the engine manager for simplicity
                        # But ideally, you'd call a 'step_once' method on the engine.
                        pass 

                    # NOTE: To keep code clean, we should refactor engine.py to have a .step() method 
                    # that returns stats. For now, we will perform a basic simulation step.
                
                # 2. SIMULATION STEP
                self.engine.manager.step()
                
                # 3. DATA COLLECTION (For Dashboard)
                snapshot = self.engine.manager.get_snapshot()
                
                # Calculate metrics for the dashboard
                total_queue = sum(l['queue_length'] for l in snapshot['lanes'].values())
                avg_speed = sum(l['avg_speed'] for l in snapshot['lanes'].values()) / len(snapshot['lanes'])
                
                # Emit Data to Frontend
                # This sends a JSON packet to the React Dashboard
                self.socketio.emit('traffic_update', {
                    'step': step,
                    'total_queue': total_queue,
                    'avg_speed': avg_speed,
                    'intersections': snapshot['intersections'] # Phase info
                })
                
                step += 1
                # Sleep slightly to prevent CPU hogging and allow the frontend to render
                eventlet.sleep(0.1) 

        except Exception as e:
            print(f"❌ Service Error: {e}")
            self.running = False