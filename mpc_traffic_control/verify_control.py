"""
Verification Script: Task 1.1 Hello World
Tests the integration between SumoClient and the generated 3x3 Grid.
"""
import logging
import time
import os

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY")

from src.traffic_mpc.config.settings import AppConfig, SumoConfig
from src.traffic_mpc.interface.sumo_client import SumoClient

def run_verification():
    # 1. Load Configuration (Manually constructing for test)
    # In the real app, we use Hydra/YAML, but here we test the object directly.
    
    # Point to the file generated in Phase 2
    sumo_cfg_path = os.path.abspath("conf/network/grid_3x3.sumocfg")
    
    config = SumoConfig(
        sumo_binary="sumo-gui", # Use GUI to see it happen
        config_file=sumo_cfg_path,
        step_length=0.5,
        warmup_seconds=0,
        use_gui=True
    )

    logger.info("--- Initializing SumoClient ---")
    client = SumoClient(config)

    try:
        # 2. Start Simulation
        client.start()
        logger.info("SUMO Started Successfully.")

        # 3. Run a simple loop
        logger.info("Running simulation loop for 50 steps...")
        
        # Get ID of the central intersection (usually 'C' or similar in netgenerate)
        # Note: In netgenerate grids, IDs are usually node ids like '1/1' or 'C'. 
        # We will print available ones to be sure.
        import traci
        tls_ids = traci.trafficlight.getIDList()
        logger.info(f"Found Traffic Lights: {tls_ids}")
        
        target_tls = tls_ids[0] if tls_ids else None

        for step in range(50):
            client.step()
            
            # Read Data
            detectors = client.get_detector_data()
            if step % 10 == 0:
                logger.info(f"Step {step}: Time={client.get_time()}")
                # logger.info(f"Detector Data: {detectors}") # Likely empty as we haven't added E2 detectors yet

            # Test Control Action
            if target_tls and step == 20:
                logger.info(f"--- ACTUATING TLS {target_tls} ---")
                logger.info("Extending current phase by 30 seconds.")
                client.set_phase_duration(target_tls, 30.0)

            # Slow down so you can see it in the GUI
            time.sleep(0.05)

    except Exception as e:
        logger.error(f"Verification Failed: {e}", exc_info=True)
    finally:
        client.close()
        logger.info("Simulation Closed.")

if __name__ == "__main__":
    run_verification()