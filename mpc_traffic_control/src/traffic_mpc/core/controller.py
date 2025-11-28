"""
MPC Controller Core.
Synthesizes state, prediction, and constraints to formulate the control decision.
Uses CasADi for symbolic optimization.
"""
import logging
import numpy as np
import casadi as ca
from typing import List, Dict, Optional, Tuple

from src.traffic_mpc.config.settings import MPCConfig, OptimizationConfig

logger = logging.getLogger(__name__)

class MPCController:
    """
    Model Predictive Controller for Traffic Signals.
    Formulates a QP/NLP to minimize queue lengths.
    """
    def __init__(self, 
                 mpc_config: MPCConfig, 
                 opt_config: OptimizationConfig,
                 lane_ids: List[str],
                 phases: List[str]): # phases are lists of active lanes for that phase
        """
        Args:
            mpc_config: Horizons and timing constraints.
            opt_config: Weights (Q, R matrices).
            lane_ids: List of all controlled lane IDs (defines State vector order).
            phases: List of "Green Strings" or phase IDs. 
                    (Note: In a full implementation, we need a mapping of Phase -> Active Lanes.
                     For this simplified grid, we assume a simple mapping or pass it in.)
        """
        self.cfg = mpc_config
        self.opt_cfg = opt_config
        self.lane_ids = lane_ids
        self.n_lanes = len(lane_ids)
        
        # Horizons
        self.Np = self.cfg.prediction_horizon
        self.Nu = self.cfg.control_horizon
        
        # Setup CasADi Solver
        self._setup_solver()

    def _setup_solver(self):
        """
        Constructs the symbolic optimization problem.
        This is done ONCE during initialization to save time.
        """
        self.opti = ca.Opti()
        
        # --- Decision Variables ---
        # X: State [n_lanes, Np+1] (Queue lengths)
        self.X = self.opti.variable(self.n_lanes, self.Np + 1)
        
        # U: Control [1, Np] (Green time fraction or extension time)
        # Simplified: We control the "Green Extension" for the active phase.
        # For a 3x3 grid, real control is complex. 
        # Here we model a SINGLE INTERSECTION approximation for the prototype.
        # Let's assume we are controlling 'Green Time' u(k) for the current phase.
        self.U = self.opti.variable(1, self.Np) 
        
        # --- Parameters (Set at runtime) ---
        # x0: Initial State [n_lanes]
        self.P_x0 = self.opti.parameter(self.n_lanes)
        
        # D: Demand Forecast [n_lanes, Np]
        self.P_demand = self.opti.parameter(self.n_lanes, self.Np)
        
        # S: Saturation Flow [n_lanes] (How fast cars leave if green)
        self.P_sat_flow = self.opti.parameter(self.n_lanes)
        
        # --- Objective Function ---
        cost = 0
        
        # Weight Matrices
        Q = self.opt_cfg.weight_queue
        R = self.opt_cfg.weight_switch # Penalize large control changes
        
        for k in range(self.Np):
            # 1. Minimize Queues: J += ||x(k+1)||_Q
            # Sum of squares of queue lengths
            cost += Q * ca.sumsqr(self.X[:, k+1])
            
            # 2. Minimize Control Effort (Smoothness): J += ||u(k)||_R
            if k > 0:
                cost += R * ca.sumsqr(self.U[:, k] - self.U[:, k-1])
        
        self.opti.minimize(cost)
        
        # --- Constraints ---
        
        # 1. Initial Condition
        self.opti.subject_to(self.X[:, 0] == self.P_x0)
        
        for k in range(self.Np):
            # 2. System Dynamics (Store-and-Forward)
            # x(k+1) = x(k) + Demand - Outflow
            # Outflow = Saturation * GreenTime (U) * B_matrix (who has green?)
            # Simplified: Assuming all lanes discharge if U > 0 (Just for prototype)
            # In reality, you multiply U by a "Phase Matrix".
            
            # Simple dynamics: x_next = x + d - s*u
            # We scale u to be "fraction of capacity used" or similar.
            # Let's use u as "Green Time Scaler" (0 to 1)
            
            outflow = self.P_sat_flow * self.U[:, k]
            
            # Dynamics equation
            x_next = self.X[:, k] + self.P_demand[:, k] - outflow
            self.opti.subject_to(self.X[:, k+1] == x_next)
            
            # 3. Physical Constraints
            # Queue cannot be negative (Soft constraint in cost is better, but hard here for now)
            self.opti.subject_to(self.X[:, k+1] >= 0)
            
            # Control limits (Min/Max Green)
            # Normalized Green: 0 (Red) to 1 (Max Capacity)
            self.opti.subject_to(self.U[:, k] >= 0.1) # Min green
            self.opti.subject_to(self.U[:, k] <= 1.0) # Max green

        # --- Solver Settings ---
        # Silence the output for real-time loops
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0,
            'ipopt.sb': 'yes'
        }
        self.opti.solver('ipopt', opts)

    def optimize(self, 
                 current_queues: Dict[str, float], 
                 predicted_demand: np.ndarray) -> float:
        """
        Solves the MPC problem for the current step.
        
        Args:
            current_queues: Dict {lane_id: queue_length}
            predicted_demand: Array [n_lanes, Np]
            
        Returns:
            optimal_u: The optimal control action for the next step (e.g. green split).
        """
        # 1. Map Dictionary to State Vector
        x0 = np.array([current_queues.get(lid, 0.0) for lid in self.lane_ids])
        
        # 2. Set Parameters
        self.opti.set_value(self.P_x0, x0)
        
        # Handle demand shape mismatch if simple constant prediction
        if predicted_demand.shape != (self.n_lanes, self.Np):
            # Broadcast if single column
            pass 
            
        self.opti.set_value(self.P_demand, predicted_demand)
        
        # Estimate Saturation Flow (e.g., 0.5 veh/sec = 1800 veh/hr)
        # In a real app, this comes from the Network model
        sat_flow = np.full(self.n_lanes, 0.5) 
        self.opti.set_value(self.P_sat_flow, sat_flow)
        
        # 3. Solve
        try:
            sol = self.opti.solve()
            # Return the first control action (Receding Horizon)
            u_opt = sol.value(self.U[:, 0])
            return float(u_opt)
        except Exception as e:
            logger.error(f"MPC Optimization Failed: {e}")
            # Fallback: Safe default
            return 0.5