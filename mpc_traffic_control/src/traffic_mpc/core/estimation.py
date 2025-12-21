"""
Traffic State Estimation Module.
Implements a conservation-based estimator to reconstruct queue lengths
from detector inputs.
"""
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LinkState(BaseModel):
    """Value Object representing the state of a single road link."""
    link_id: str
    queue_length: float = Field(default=0.0, description="Estimated vehicles in queue")
    
class StateEstimator:
    """
    Estimates traffic state variables necessary for MPC.
    Implements a recursive conservation filter.
    """
    def __init__(self, link_ids: List[str], alpha: float = 0.1):
        """
        Args:
            link_ids: List of link identifiers to track.
            alpha: Filter gain (0.0 to 1.0) for smoothing updates.
        """
        self.link_ids = link_ids
        self.alpha = alpha
        # Initialize state vector x_0 = 0
        self.estimates: Dict[str, float] = {lid: 0.0 for lid in link_ids}

    def update(self, detector_data: Dict[str, float]) -> Dict[str, float]:
        """
        Performs the state update based on new measurements.
        
        Args:
            detector_data: Dictionary mapping 'e2_laneID' -> queue_length (from SUMO)
            
        Returns:
            Updated dictionary of estimated queue lengths per lane.
        """
        updated_state = {}

        for link in self.link_ids:
            # Construct detector ID from link ID (assuming naming convention e2_linkID)
            det_id = f"e2_{link}"
            
            # Get measurement z (from E2 detector)
            # If detector missing, assume 0 or hold last value (using 0 here for safety)
            z_measured = detector_data.get(det_id, 0.0)
            
            # Simple Smoothing / Complementary Filter
            # x_new = (1 - alpha) * x_old + alpha * z_measured
            # Note: Since E2 detectors in SUMO are highly accurate "ground truth" 
            # compared to real-world induction loops, we can trust z_measured highly.
            # In a real hardware deployment, alpha would be lower (e.g., 0.1).
            # For simulation, we can set alpha ~ 0.8 or 1.0.
            
            prev_est = self.estimates.get(link, 0.0)
            n_corrected = (1 - self.alpha) * prev_est + self.alpha * z_measured
            
            # Physical Constraint: Queue cannot be negative
            n_corrected = max(0.0, n_corrected)

            self.estimates[link] = n_corrected
            updated_state[link] = n_corrected

        return updated_state

    def reset(self):
        """Resets internal state to zero."""
        self.estimates = {lid: 0.0 for lid in self.link_ids}