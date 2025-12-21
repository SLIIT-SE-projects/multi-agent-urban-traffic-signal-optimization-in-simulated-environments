"""
MPC Controller Core.
Synthesizes state, prediction, and constraints to formulate the control decision.
Uses CasADi for symbolic optimization with Nonlinear Dynamics.
"""
import logging
import numpy as np
import casadi as ca
from typing import List, Dict

from traffic_mpc.config.settings import MPCConfig, OptimizationConfig

logger = logging.getLogger(__name__)

class MPCController:
    """
    Model Predictive Controller for Traffic Signals.
    Formulates a QP/NLP to minimize queue lengths using IPOPT.
    """
    def __init__(self, 
                 mpc_config: MPCConfig, 
                 opt_config: OptimizationConfig,
                 lane_ids: List[str],
                 phases: List[str]): 
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
        Uses nonlinear dynamics (fmin) to physically model queue discharge.
        """
        self.opti = ca.Opti()
        
        # --- Decision Variables ---
        # X: State [n_lanes, Np+1] (Queue lengths)
        self.X = self.opti.variable(self.n_lanes, self.Np + 1)
        
        # U: Control [1, Np] (Green time fraction)
        self.U = self.opti.variable(1, self.Np)
        
        # --- Parameters ---
        self.P_x0 = self.opti.parameter(self.n_lanes)         # Initial State
        self.P_demand = self.opti.parameter(self.n_lanes, self.Np) # Demand
        self.P_sat_flow = self.opti.parameter(self.n_lanes)   # Saturation Flow
        
        # --- Objective Function ---
        cost = 0
        Q = self.opt_cfg.weight_queue
        R = self.opt_cfg.weight_switch
        
        for k in range(self.Np):
            # 1. Minimize Queues (Squared error)
            cost += Q * ca.sumsqr(self.X[:, k+1])
            
            # 2. Minimize Control Effort (Smoothness)
            if k > 0:
                cost += R * ca.sumsqr(self.U[:, k] - self.U[:, k-1])
        
        self.opti.minimize(cost)
        
        # --- Constraints ---
        # 1. Initial Condition
        self.opti.subject_to(self.X[:, 0] == self.P_x0)
        
        for k in range(self.Np):
            # 2. System Dynamics (Store-and-Forward with Physical Limits)
            # Logic: You cannot discharge more cars than are currently in the queue + arriving.
            # outflow = min( SatFlow * U , CurrentQueue + Demand )
            
            potential_outflow = self.P_sat_flow * self.U[:, k]
            available_traffic = self.X[:, k] + self.P_demand[:, k]
            
            # Use CasADi's fmin for differentiable min()
            actual_outflow = ca.fmin(potential_outflow, available_traffic)
            
            x_next = self.X[:, k] + self.P_demand[:, k] - actual_outflow
            self.opti.subject_to(self.X[:, k+1] == x_next)
            
            # 3. Physical Constraints
            self.opti.subject_to(self.X[:, k+1] >= 0)
            
            # 4. Control limits
            self.opti.subject_to(self.U[:, k] >= 0.1) # Min Green
            self.opti.subject_to(self.U[:, k] <= 1.0) # Max Green

        # --- Solver Settings ---
        # Loose tolerances for speed, silence output for clean logs
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-3,
        }
        self.opti.solver('ipopt', opts)

    def optimize(self, 
                 current_queues: Dict[str, float], 
                 predicted_demand: np.ndarray) -> float:
        """
        Solves the MPC problem for the current step.
        """
        # 1. Map Dictionary to State Vector
        x0 = np.array([current_queues.get(lid, 0.0) for lid in self.lane_ids])
        
        # 2. Set Parameters
        self.opti.set_value(self.P_x0, x0)
        
        # Handle demand shape
        if predicted_demand.ndim == 1:
            predicted_demand = predicted_demand.reshape(self.n_lanes, -1)
        if predicted_demand.shape != (self.n_lanes, self.Np):
            predicted_demand = np.zeros((self.n_lanes, self.Np))

        self.opti.set_value(self.P_demand, predicted_demand)
        
        # --- CRITICAL: CORRECT CAPACITY SCALING ---
        # We tell the solver: "If U=1.0, you discharge the full cycle capacity"
        # 0.5 veh/s * 60s = 30 vehicles.
        max_capacity_per_cycle = 0.5 * self.cfg.max_green_time
        sat_flow_vector = np.full(self.n_lanes, max_capacity_per_cycle)
        self.opti.set_value(self.P_sat_flow, sat_flow_vector)
        
        # 3. Solve
        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U[:, 0])
            return float(u_opt)
        except Exception as e:
            # If solver fails, try to return the debug value or a safe fallback
            try:
                return float(self.opti.debug.value(self.U[:, 0]))
            except:
                return 0.5