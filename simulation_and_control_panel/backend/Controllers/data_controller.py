import traci


class DataController:
    """
    Manages data retrieval operations including vehicle data, traffic light states, and status.
    """
    
    def __init__(self, simulation_controller):
        """
        Initialize DataController with reference to SimulationController.
        
        Args:
            simulation_controller: Reference to the main SimulationController instance
        """
        self.sim_controller = simulation_controller
    
    def get_current_data(self, vehicle_limit=50):
        """
        Get current simulation data including vehicles and traffic lights.
        
        Args:
            vehicle_limit: Maximum number of vehicles to return (default: 50)
            
        Returns:
            dict: Current simulation data with vehicle and traffic light information
        """
        if not self.sim_controller.is_running:
            return {"status": "error", "message": "Simulation not running"}
        
        try:
            vehicle_ids = traci.vehicle.getIDList()
            tls_ids = traci.trafficlight.getIDList()
            
            vehicles_data = []
            for vid in vehicle_ids[:vehicle_limit]:
                try:
                    vehicles_data.append({
                        "id": vid,
                        "speed": round(traci.vehicle.getSpeed(vid), 2),
                        "position": traci.vehicle.getPosition(vid),
                        "waiting_time": round(traci.vehicle.getWaitingTime(vid), 1)
                    })
                except:
                    continue
            
            traffic_lights_data = []
            for tls_id in tls_ids:
                try:
                    traffic_lights_data.append({
                        "id": tls_id,
                        "state": traci.trafficlight.getRedYellowGreenState(tls_id),
                        "phase": traci.trafficlight.getPhase(tls_id)
                    })
                except:
                    continue
            
            return {
                "status": "success",
                "step": self.sim_controller.current_step,
                "vehicle_count": len(vehicle_ids),
                "vehicles": vehicles_data,
                "traffic_lights": traffic_lights_data,
                "is_paused": self.sim_controller.is_paused,
                "auto_stepping": self.sim_controller.auto_stepping
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_status(self):
        """
        Get current simulation status.
        
        Returns:
            dict: Simulation status information
        """
        return {
            "is_running": self.sim_controller.is_running,
            "is_paused": self.sim_controller.is_paused,
            "current_step": self.sim_controller.current_step,
            "auto_stepping": self.sim_controller.auto_stepping
        }
    
    def get_vehicle_count(self):
        """
        Get total count of vehicles in simulation.
        
        Returns:
            dict: Vehicle count information
        """
        if not self.sim_controller.is_running:
            return {"status": "error", "message": "Simulation not running"}
        
        try:
            vehicle_ids = traci.vehicle.getIDList()
            return {
                "status": "success",
                "vehicle_count": len(vehicle_ids),
                "vehicles": vehicle_ids[:100]  # Return IDs of first 100 vehicles
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_traffic_light_states(self):
        """
        Get all traffic light states.
        
        Returns:
            dict: Traffic light states
        """
        if not self.sim_controller.is_running:
            return {"status": "error", "message": "Simulation not running"}
        
        try:
            tls_ids = traci.trafficlight.getIDList()
            traffic_lights_data = []
            
            for tls_id in tls_ids:
                try:
                    traffic_lights_data.append({
                        "id": tls_id,
                        "state": traci.trafficlight.getRedYellowGreenState(tls_id),
                        "phase": traci.trafficlight.getPhase(tls_id),
                        "phase_duration": traci.trafficlight.getPhaseDuration(tls_id)
                    })
                except:
                    continue
            
            return {
                "status": "success",
                "traffic_light_count": len(tls_ids),
                "traffic_lights": traffic_lights_data
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_vehicle_details(self, vehicle_id):
        """
        Get detailed information for a specific vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
            
        Returns:
            dict: Detailed vehicle information
        """
        if not self.sim_controller.is_running:
            return {"status": "error", "message": "Simulation not running"}
        
        try:
            return {
                "status": "success",
                "id": vehicle_id,
                "speed": round(traci.vehicle.getSpeed(vehicle_id), 2),
                "position": traci.vehicle.getPosition(vehicle_id),
                "angle": round(traci.vehicle.getAngle(vehicle_id), 2),
                "waiting_time": round(traci.vehicle.getWaitingTime(vehicle_id), 1),
                "route_id": traci.vehicle.getRouteID(vehicle_id),
                "edge_id": traci.vehicle.getRoadID(vehicle_id),
                "lane_id": traci.vehicle.getLaneID(vehicle_id),
                "lane_position": round(traci.vehicle.getLanePosition(vehicle_id), 2)
            }
        except Exception as e:
            return {"status": "error", "message": f"Vehicle '{vehicle_id}' not found or error: {str(e)}"}
        
    def get_gnn_features(self):
        """
        Extracts dynamic features from SUMO for the GNN.
        Matches the keys expected by TrafficGraphBuilder.create_hetero_data()
        """
        
        # --- 1. Lane Data ---
        lane_ids = traci.lane.getIDList()
        lanes = {}
        
        for lane_id in lane_ids:
            # traci.lane.getWaitingTime returns the sum of waiting time of all vehicles on the lane
            waiting_time = traci.lane.getWaitingTime(lane_id)
            
            # traci.lane.getLastStepMeanSpeed returns m/s
            avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            
            # traci.lane.getLastStepHaltingNumber returns count of stopped vehicles
            queue_length = traci.lane.getLastStepHaltingNumber(lane_id)

            lanes[lane_id] = {
                "queue_length": queue_length,  # Used in x_lane[0]
                "avg_speed": avg_speed,        # Used in x_lane[1]
                "waiting_time": waiting_time   # Used in x_lane[2]
            }
        
        # --- 2. Intersection (Traffic Light) Data ---
        intersections = {}
        current_sim_time = traci.simulation.getTime()

        for tls_id in traci.trafficlight.getIDList():
             # Calculate time remaining until the next phase switch
             next_switch_time = traci.trafficlight.getNextSwitch(tls_id)
             time_to_switch = next_switch_time - current_sim_time

             intersections[tls_id] = {
                 "phase_index": traci.trafficlight.getPhase(tls_id), # Used for One-Hot Encoding
                 "time_to_switch": time_to_switch                    # Used as explicit feature
             }
             
        return {"lanes": lanes, "intersections": intersections}
