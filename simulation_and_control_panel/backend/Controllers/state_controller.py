import os
import json
from datetime import datetime
import traci


class StateController:
    """
    Manages simulation state operations including saving and restoring simulation states.
    """
    
    SAVED_STATES_DIR = "saved_states"
    
    def __init__(self, simulation_controller, data_controller):
        """
        Initialize StateController with references to other controllers.
        
        Args:
            simulation_controller: Reference to the main SimulationController instance
            data_controller: Reference to the DataController instance
        """
        self.sim_controller = simulation_controller
        self.data_controller = data_controller
        self.saved_states = {}
        self._load_saved_states()
    
    def _load_saved_states(self):
        """Load existing saved states from disk."""
        try:
            if os.path.exists(self.SAVED_STATES_DIR):
                for filename in os.listdir(self.SAVED_STATES_DIR):
                    if filename.endswith('.json'):
                        state_id = filename.replace('.json', '')
                        filepath = os.path.join(self.SAVED_STATES_DIR, filename)
                        with open(filepath, 'r') as f:
                            self.saved_states[state_id] = json.load(f)
        except Exception as e:
            print(f"Error loading saved states: {e}")
    
    def save_state(self, state_name="default"):
        """
        Save current simulation state to disk.
        
        Args:
            state_name: Name/identifier for this saved state
            
        Returns:
            dict: Status response with save result
        """
        if not self.sim_controller.is_running:
            return {"status": "error", "message": "Simulation not running"}
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state_id = f"{state_name}_{timestamp}"
            
            state_data = {
                "state_name": state_name,
                "step": self.sim_controller.current_step,
                "timestamp": datetime.now().isoformat(),
                "is_paused": self.sim_controller.is_paused,
                "config_file": self.sim_controller.config_file,
                "data": self.data_controller.get_current_data()
            }
            
            self.saved_states[state_id] = state_data
            
            # Save to file
            os.makedirs(self.SAVED_STATES_DIR, exist_ok=True)
            filepath = os.path.join(self.SAVED_STATES_DIR, f"{state_id}.json")
            
            with open(filepath, "w") as f:
                json.dump(state_data, f, indent=2)
            
            return {
                "status": "success",
                "message": f"State '{state_name}' saved successfully",
                "state_id": state_id,
                "step": self.sim_controller.current_step,
                "file": filepath
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to save state: {str(e)}"}
    
    def get_saved_states(self):
        """
        Get list of all saved states.
        
        Returns:
            dict: List of saved states with metadata
        """
        try:
            states_list = []
            for state_id, state_data in self.saved_states.items():
                states_list.append({
                    "state_id": state_id,
                    "state_name": state_data.get("state_name", "unknown"),
                    "step": state_data.get("step", 0),
                    "timestamp": state_data.get("timestamp", ""),
                    "config_file": state_data.get("config_file", "")
                })
            
            return {
                "status": "success",
                "saved_states": sorted(states_list, key=lambda x: x['timestamp'], reverse=True),
                "count": len(states_list)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_state_details(self, state_id):
        """
        Get detailed information about a saved state.
        
        Args:
            state_id: ID of the saved state
            
        Returns:
            dict: State details
        """
        try:
            if state_id not in self.saved_states:
                return {"status": "error", "message": f"State '{state_id}' not found"}
            
            return {
                "status": "success",
                "state_id": state_id,
                "state_data": self.saved_states[state_id]
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def delete_state(self, state_id):
        """
        Delete a saved state.
        
        Args:
            state_id: ID of the saved state to delete
            
        Returns:
            dict: Status response
        """
        try:
            if state_id not in self.saved_states:
                return {"status": "error", "message": f"State '{state_id}' not found"}
            
            # Delete from memory
            del self.saved_states[state_id]
            
            # Delete from disk
            filepath = os.path.join(self.SAVED_STATES_DIR, f"{state_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return {
                "status": "success",
                "message": f"State '{state_id}' deleted successfully"
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to delete state: {str(e)}"}
    
    def restore_state(self, state_id):
        """
        Restore to a previously saved state (restores metadata, not actual simulation).
        Note: Full state restoration would require SUMO checkpointing capability.
        
        Args:
            state_id: ID of the state to restore to
            
        Returns:
            dict: Status response with restored state information
        """
        try:
            if state_id not in self.saved_states:
                return {"status": "error", "message": f"State '{state_id}' not found"}
            
            state_data = self.saved_states[state_id]
            
            # This is a metadata restore - returns the saved state data
            # Full simulation rollback would require SUMO checkpoint functionality
            return {
                "status": "success",
                "message": f"State '{state_id}' metadata restored",
                "state_id": state_id,
                "state_name": state_data.get("state_name"),
                "step": state_data.get("step"),
                "timestamp": state_data.get("timestamp"),
                "note": "This returns saved state data. Full simulation rollback requires SUMO checkpoint support."
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to restore state: {str(e)}"}
    
    def clear_all_states(self):
        """
        Delete all saved states.
        
        Returns:
            dict: Status response
        """
        try:
            count = len(self.saved_states)
            self.saved_states.clear()
            
            # Clear saved_states directory
            if os.path.exists(self.SAVED_STATES_DIR):
                for filename in os.listdir(self.SAVED_STATES_DIR):
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.SAVED_STATES_DIR, filename)
                        os.remove(filepath)
            
            return {
                "status": "success",
                "message": f"Cleared {count} saved state(s)"
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to clear states: {str(e)}"}
