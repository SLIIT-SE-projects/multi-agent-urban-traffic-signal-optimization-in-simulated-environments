import numpy as np
import pytest
from traffic_mpc.core.controller import MPCController
from traffic_mpc.config.settings import MPCConfig, OptimizationConfig

def test_mpc_reaction():
    # 1. Config
    mpc_cfg = MPCConfig(prediction_horizon=10, control_horizon=5, max_green_time=60)
    # Aggressive weights
    opt_cfg = OptimizationConfig(weight_queue=100.0, weight_switch=0.1)
    
    lanes = ["lane_1", "lane_2"]
    controller = MPCController(mpc_cfg, opt_cfg, lanes, phases=[])
    
    # 2. Scenario: Heavy Queue
    queues = {"lane_1": 15.0, "lane_2": 2.0}
    demand = np.zeros((2, 10))
    
    # 3. Solve
    u = controller.optimize(queues, demand)
    print(f"Computed Control u: {u}")
    
    # 4. Assert
    # With a queue of 15 and max_capacity of ~30, the solver should
    # output a significant green time (e.g., > 0.4)
    assert u > 0.3
    assert u <= 1.0 + 1e-4

if __name__ == "__main__":
    test_mpc_reaction()