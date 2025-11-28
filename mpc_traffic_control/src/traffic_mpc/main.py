"""
Main Application Entry Point.
Orchestrates SUMO, Estimation, and MPC Control Loop.
"""
import logging
import time
import hydra
import numpy as np
import traci 
from omegaconf import DictConfig, OmegaConf

# Import our components
from traffic_mpc.config.settings import AppConfig
from traffic_mpc.interface.sumo_client import SumoClient
from traffic_mpc.core.estimation import StateEstimator
from traffic_mpc.core.controller import MPCController
from traffic_mpc.utils.logging import setup_logging

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Configuration
    try:
        app_config = AppConfig(**OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        print(f"Configuration Error: {e}")
        return

    setup_logging(app_config.logging)
    logger = logging.getLogger(__name__)
    logger.info("--- Starting MPC Traffic Controller ---")

    # 2. Infrastructure
    client = SumoClient(app_config.sumo)
    client.start()

    try:
        # 3. Discovery
        client.step()
        detectors = client.get_detector_data()
        active_lanes = [d.replace("e2_", "") for d in detectors.keys()]
        
        logger.info(f"Discovered {len(active_lanes)} controlled lanes.")
        if not active_lanes:
            logger.error("No detectors found! Check grid_3x3.add.xml.")
            return

        tls_ids = traci.trafficlight.getIDList()
        logger.info(f"Found Intersections: {tls_ids}")

        # 4. Initialize Components
        estimator = StateEstimator(link_ids=active_lanes)
        controller = MPCController(
            mpc_config=app_config.mpc,
            opt_config=app_config.optimization,
            lane_ids=active_lanes,
            phases=[]
        )

        # 5. Main Loop
        step = 0
        control_interval = 5
        max_steps = 3600 

        while step < max_steps:
            client.step()
            
            # Observe
            raw_data = client.get_detector_data()
            state = estimator.update(raw_data)
            
            # Optimize (Every 5 seconds)
            if step % control_interval == 0:
                # Basic demand placeholder (Phase 8 will upgrade this to LSTM)
                demand_matrix = np.zeros((len(active_lanes), app_config.mpc.prediction_horizon))
                
                # Run Optimization
                u_opt = controller.optimize(state, demand_matrix)
                
                # --- ACTUATION LOGIC (With Deadlock Fix) ---
                if u_opt > 0.2:
                    extension = u_opt * app_config.mpc.max_green_time
                    
                    # Apply to ALL traffic lights
                    for tls_id in tls_ids:
                        current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
                        
                        # Only extend if GREEN and NOT YELLOW
                        is_green = ('G' in current_state or 'g' in current_state)
                        is_yellow = ('y' in current_state or 'Y' in current_state)
                        
                        if is_green and not is_yellow:
                            client.set_phase_duration(tls_id, extension)
                            
                    # Logging (Heartbeat or Action)
                    current_max_q = max(state.values()) if state else 0
                    logger.info(f"Step {step}: Queue={current_max_q:.1f} | MPC u={u_opt:.2f} -> Extending Green")

            step += 1
            # Optional GUI delay (Remove for speed)
            # if app_config.sumo.use_gui: time.sleep(0.01)

    except Exception as e:
        logger.error(f"Simulation Crashed: {e}", exc_info=True)
    finally:
        client.close()
        logger.info("Simulation Finished.")

if __name__ == "__main__":
    main()