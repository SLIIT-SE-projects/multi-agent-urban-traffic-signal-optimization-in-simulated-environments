import os
import glob
import traci


class ScenarioController:
    """
    Manages scenario operations including listing, switching, and reloading scenarios.
    """
    
    SCENARIOS_DIR = os.path.join("..", "scenarios")
    
    def __init__(self, simulation_controller):
        """
        Initialize ScenarioController with reference to SimulationController.
        
        Args:
            simulation_controller: Reference to the main SimulationController instance
        """
        self.sim_controller = simulation_controller
    
    def get_available_scenarios(self):
        """
        Get list of all available scenarios.
        
        Returns:
            dict: Status response with list of scenarios
        """
        try:
            scenarios = []
            
            # Look for .sumo.cfg files in each scenario folder
            if os.path.exists(self.SCENARIOS_DIR):
                for scenario_folder in os.listdir(self.SCENARIOS_DIR):
                    scenario_path = os.path.join(self.SCENARIOS_DIR, scenario_folder)
                    
                    if os.path.isdir(scenario_path):
                        # Find .sumo.cfg file in this folder
                        cfg_files = glob.glob(os.path.join(scenario_path, "*.sumo.cfg"))
                        
                        if cfg_files:
                            cfg_file = cfg_files[0]  # Use first .sumo.cfg found
                            scenarios.append({
                                "name": scenario_folder,
                                "config_file": cfg_file,
                                "path": scenario_path
                            })
                
                return {
                    "status": "success",
                    "scenarios": sorted(scenarios, key=lambda x: x['name']),
                    "count": len(scenarios)
                }
            else:
                return {
                    "status": "error",
                    "message": f"Scenarios directory not found at {self.SCENARIOS_DIR}"
                }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to retrieve scenarios: {str(e)}"
            }
    
    def switch_scenario(self, scenario_name):
        """
        Switch to a different scenario without restarting the entire application.
        
        Args:
            scenario_name: Name of the scenario folder to switch to
            
        Returns:
            dict: Status response with result of scenario switch
        """
        try:
            # Validate scenario exists
            scenario_path = os.path.join(self.SCENARIOS_DIR, scenario_name)
            cfg_files = glob.glob(os.path.join(scenario_path, "*.sumo.cfg"))
            
            if not cfg_files:
                return {
                    "status": "error",
                    "message": f"Scenario '{scenario_name}' not found or has no .sumo.cfg file"
                }
            
            config_file = cfg_files[0]
            
            # Stop current simulation if running
            if self.sim_controller.is_running:
                stop_result = self.sim_controller.stop()
                if stop_result.get("status") == "error":
                    return {
                        "status": "error",
                        "message": f"Failed to stop current simulation: {stop_result.get('message')}"
                    }
            
            # Update the config file and start new scenario
            self.sim_controller.config_file = config_file
            self.sim_controller.current_step = 0
            
            # Start the new scenario
            result = self.sim_controller.start()
            
            if result.get("status") == "success":
                return {
                    "status": "success",
                    "message": f"Successfully switched to scenario '{scenario_name}'",
                    "scenario": scenario_name,
                    "config_file": config_file,
                    "step": 0
                }
            else:
                return result
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to switch scenario: {str(e)}"
            }
    
    def reload_scenario(self):
        """
        Reload the current scenario from the beginning.
        
        Returns:
            dict: Status response with result of reload operation
        """
        try:
            if not self.sim_controller.is_running:
                return {
                    "status": "error",
                    "message": "No simulation running to reload"
                }
            
            current_config = self.sim_controller.config_file
            
            # Stop current simulation
            stop_result = self.sim_controller.stop()
            if stop_result.get("status") == "error":
                return {
                    "status": "error",
                    "message": f"Failed to stop current simulation: {stop_result.get('message')}"
                }
            
            # Reset step counter
            self.sim_controller.current_step = 0
            
            # Start again
            result = self.sim_controller.start()
            
            if result.get("status") == "success":
                return {
                    "status": "success",
                    "message": "Scenario reloaded successfully",
                    "scenario_config": current_config,
                    "step": 0
                }
            else:
                return result
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to reload scenario: {str(e)}"
            }
    
    def get_current_scenario_info(self):
        """
        Get information about the currently loaded scenario.
        
        Returns:
            dict: Current scenario information
        """
        try:
            if not self.sim_controller.is_running:
                return {
                    "status": "error",
                    "message": "No simulation running"
                }
            
            config_file = self.sim_controller.config_file
            scenario_name = os.path.basename(os.path.dirname(config_file))
            
            return {
                "status": "success",
                "scenario_name": scenario_name,
                "config_file": config_file,
                "is_running": self.sim_controller.is_running,
                "current_step": self.sim_controller.current_step,
                "is_paused": self.sim_controller.is_paused
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get scenario info: {str(e)}"
            }
