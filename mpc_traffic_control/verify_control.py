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
from src.traffic_mpc.core.estimation import StateEstimator

def run_verification():
    # 1. Load Configuration (Manually constructing for test)
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

        # --- NEW: Initialize Estimator ---
        # We need to perform one step to query the API for available detectors.
        client.step()
        
        # Get all detector IDs currently loaded in the simulation
        all_detectors = list(client.get_detector_data().keys())
        
        # Extract lane IDs from detector IDs (remove 'e2_' prefix)
        # This assumes your detectors are named "e2_edgeID_laneIndex" or similar
        all_lanes = [d.replace("e2_", "") for d in all_detectors]
        
        logger.info(f"Initialized Estimator for {len(all_lanes)} lanes.")
        estimator = StateEstimator(link_ids=all_lanes, alpha=0.9)

        # Get ID of the central intersection for testing actuation later
        import traci
        tls_ids = traci.trafficlight.getIDList()
        logger.info(f"Found Traffic Lights: {tls_ids}")
        target_tls = tls_ids[0] if tls_ids else None

        # 3. Run the loop
        logger.info("Running simulation loop for 50 steps...")
        
        for step in range(50):
            client.step()
            
            # --- NEW: Data Flow Logic ---
            # 1. Get Raw Data from SUMO
            raw_data = client.get_detector_data()
            
            # 2. Estimate State using the raw data
            state = estimator.update(raw_data)
            
            # Log progress and sample state data
            if step % 10 == 0:
                logger.info(f"Step {step}: Time={client.get_time()}")
                # Print state of first 3 lanes just to see that data is flowing
                sample = {k: state[k] for k in list(state)[:3]}
                logger.info(f"Estimated Queues (Sample): {sample}")

            # --- Existing: Test Control Action ---
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