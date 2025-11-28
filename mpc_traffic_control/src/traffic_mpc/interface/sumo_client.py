"""
SUMO Interface Adapter.
Wraps the TraCI library to provide a clean, object-oriented interface.
"""
import sys
import os
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Graceful handling of traci import
try:
    import traci
    from traci import constants as tc
except ImportError:
    traci = None

from traffic_mpc.config.settings import SumoConfig

logger = logging.getLogger(__name__)

class TrafficSimulator(ABC):
    """
    Abstract Base Class defining the contract for any traffic simulator.
    Allows swapping SUMO for Aimsun/Vissim later without changing the controller.
    """
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def get_detector_data(self) -> Dict[str, float]:
        """Returns dictionary of {detector_id: vehicle_count}"""
        pass

    @abstractmethod
    def set_phase_duration(self, tls_id: str, duration: float) -> None:
        """Sets the remaining duration of the current phase."""
        pass

    @abstractmethod
    def get_time(self) -> float:
        pass


class SumoClient(TrafficSimulator):
    """
    Concrete implementation of TrafficSimulator for SUMO using TraCI.
    """
    def __init__(self, config: SumoConfig):
        self.cfg = config
        self._connected = False
        self._step_count = 0

        if traci is None:
            raise ImportError("TraCI not found. Ensure SUMO_HOME is set and 'pip install traci' is run.")
        
        self._check_env()

    def _check_env(self):
        if 'SUMO_HOME' not in os.environ:
            logger.warning("SUMO_HOME is not set. Simulation might fail to launch.")

    def start(self) -> None:
        """
        Launches SUMO and establishes the TraCI connection.
        Implements retry logic for GUI startup stability.
        """
        if self._connected:
            return

        # Construct command line arguments
        cmd = [self.cfg.sumo_binary, "-c", str(self.cfg.config_file)]
        
        if self.cfg.sumo_binary.endswith("sumo-gui") or self.cfg.use_gui:
            cmd.append("--start") # Auto-start simulation in GUI
            cmd.append("--quit-on-end")

        # Retry logic (crucial for GUI race conditions) [cite: 807]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                logger.info(f"Launching SUMO (Attempt {attempt+1}): {' '.join(cmd)}")
                traci.start(cmd, port=self.cfg.port)
                self._connected = True
                break
            except (traci.FatalTraCIError, ConnectionRefusedError) as e:
                logger.warning(f"Connection failed: {e}. Retrying in 1s...")
                time.sleep(1)
        
        if not self._connected:
            raise ConnectionError("Failed to connect to SUMO after multiple retries.")

        # Optional Warmup [cite: 822]
        if self.cfg.warmup_seconds > 0:
            self._warmup()

    def _warmup(self):
        steps = int(self.cfg.warmup_seconds / self.cfg.step_length)
        logger.info(f"Warming up for {steps} steps...")
        for _ in range(steps):
            traci.simulationStep()
        self._step_count += steps

    def step(self) -> None:
        if not self._connected:
            raise RuntimeError("Simulation not started.")
        traci.simulationStep()
        self._step_count += 1

    def get_detector_data(self) -> Dict[str, float]:
        """
        Retrieves vehicle counts from induction loops (E1) or Lane Area Detectors (E2).
        For now, we query all Lane Area Detectors (E2) as they provide Queue Length directly.
        """
        data = {}
        try:
            # Get list of all E2 detectors in the network
            det_ids = traci.lanearea.getIDList()
            for det_id in det_ids:
                # JAM_LENGTH_VEHICLE returns the number of vehicles in the queue [cite: 847]
                queue_len = traci.lanearea.getJamLengthVehicle(det_id)
                data[det_id] = queue_len
        except traci.TraCIException as e:
            logger.error(f"Error reading detectors: {e}")
        return data

    def set_phase_duration(self, tls_id: str, duration: float) -> None:
        """
        Sets the duration for the CURRENT phase.
        Note: In SUMO, this modifies the remaining time of the active phase.
        """
        try:
            traci.trafficlight.setPhaseDuration(tls_id, duration)
        except traci.TraCIException as e:
            logger.error(f"Failed to set phase for {tls_id}: {e}")

    def get_time(self) -> float:
        return traci.simulation.getTime()

    def close(self) -> None:
        if self._connected:
            traci.close()
            self._connected = False
            logger.info("SUMO connection closed.")