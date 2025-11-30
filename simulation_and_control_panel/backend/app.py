from flask import Flask, jsonify, request
from flask_cors import CORS
from Controllers.simulation_controller import SimulationController
from Controllers.scenario_controller import ScenarioController
from Controllers.data_controller import DataController
from Controllers.state_controller import StateController
from config import config
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# CHANGE THIS to your actual config file!
# CONFIG_FILE = os.path.join("..", "scenarios", "mapishara.sumo.cfg")
CONFIG_FILE = os.path.join("..", "scenarios", "grid3x3", "grid3x3.sumo.cfg")

# Initialize controllers
sim_controller = SimulationController(
    CONFIG_FILE, 
    use_gui=config.USE_GUI,
    step_delay=config.STEP_DELAY
)

data_controller = DataController(sim_controller)
scenario_controller = ScenarioController(sim_controller)
state_controller = StateController(sim_controller, data_controller)


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})


# ============================================================================
# SIMULATION LIFECYCLE ENDPOINTS
# ============================================================================

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    result = sim_controller.start()
    return jsonify(result)


@app.route('/api/simulation/step', methods=['POST'])
def step_simulation():
    result = sim_controller.step()
    # If step succeeded, get current data
    if result.get("status") == "success":
        data = data_controller.get_current_data()
        result.update(data)
    return jsonify(result)


@app.route('/api/simulation/pause', methods=['POST'])
def pause_simulation():
    result = sim_controller.pause()
    return jsonify(result)


@app.route('/api/simulation/resume', methods=['POST'])
def resume_simulation():
    result = sim_controller.resume()
    return jsonify(result)


@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    result = sim_controller.stop()
    return jsonify(result)


# ============================================================================
# AUTO-STEPPING ENDPOINTS
# ============================================================================

@app.route('/api/simulation/auto-step/start', methods=['POST'])
def start_auto_step():
    data = request.get_json(silent=True) or {}
    step_delay = data.get('step_delay', None)  # None will use config default
    result = sim_controller.start_auto_stepping(step_delay)
    return jsonify(result)


@app.route('/api/simulation/auto-step/pause', methods=['POST'])
def pause_auto_step():
    result = sim_controller.pause_auto_stepping()
    return jsonify(result)


@app.route('/api/simulation/auto-step/resume', methods=['POST'])
def resume_auto_step():
    result = sim_controller.resume_auto_stepping()
    return jsonify(result)


@app.route('/api/simulation/auto-step/stop', methods=['POST'])
def stop_auto_step():
    result = sim_controller.stop_auto_stepping()
    return jsonify(result)


# ============================================================================
# SCENARIO MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/api/scenarios', methods=['GET'])
def get_scenarios():
    """Get list of available scenarios"""
    result = scenario_controller.get_available_scenarios()
    return jsonify(result)


@app.route('/api/simulation/switch-scenario', methods=['POST'])
def switch_scenario():
    """Switch to a different scenario"""
    data = request.json
    scenario_name = data.get('scenario_name')
    
    if not scenario_name:
        return jsonify({"status": "error", "message": "scenario_name is required"}), 400
    
    result = scenario_controller.switch_scenario(scenario_name)
    return jsonify(result)


@app.route('/api/simulation/reload', methods=['POST'])
def reload_scenario():
    """Reload the current scenario from the beginning"""
    result = scenario_controller.reload_scenario()
    return jsonify(result)


@app.route('/api/simulation/current-scenario', methods=['GET'])
def get_current_scenario():
    """Get information about the currently loaded scenario"""
    result = scenario_controller.get_current_scenario_info()
    return jsonify(result)


# ============================================================================
# DATA RETRIEVAL ENDPOINTS
# ============================================================================

@app.route('/api/simulation/data', methods=['GET'])
def get_data():
    """Get current simulation data (vehicles, traffic lights)"""
    data = data_controller.get_current_data()
    return jsonify(data)


@app.route('/api/simulation/status', methods=['GET'])
def get_status():
    """Get simulation status"""
    status = data_controller.get_status()
    return jsonify(status)


@app.route('/api/simulation/vehicles', methods=['GET'])
def get_vehicles():
    """Get vehicle count and IDs"""
    result = data_controller.get_vehicle_count()
    return jsonify(result)


@app.route('/api/simulation/vehicles/<vehicle_id>', methods=['GET'])
def get_vehicle(vehicle_id):
    """Get detailed information about a specific vehicle"""
    result = data_controller.get_vehicle_details(vehicle_id)
    return jsonify(result)


@app.route('/api/simulation/traffic-lights', methods=['GET'])
def get_traffic_lights():
    """Get all traffic light states"""
    result = data_controller.get_traffic_light_states()
    return jsonify(result)


# ============================================================================
# STATE MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/api/simulation/save-state', methods=['POST'])
def save_state():
    """Save current simulation state"""
    data = request.json
    state_name = data.get('state_name', 'default')
    result = state_controller.save_state(state_name)
    return jsonify(result)


@app.route('/api/simulation/states', methods=['GET'])
def get_saved_states():
    """Get list of all saved states"""
    result = state_controller.get_saved_states()
    return jsonify(result)


@app.route('/api/simulation/states/<state_id>', methods=['GET'])
def get_state_details(state_id):
    """Get details about a specific saved state"""
    result = state_controller.get_state_details(state_id)
    return jsonify(result)


@app.route('/api/simulation/states/<state_id>', methods=['DELETE'])
def delete_state(state_id):
    """Delete a saved state"""
    result = state_controller.delete_state(state_id)
    return jsonify(result)


@app.route('/api/simulation/restore-state/<state_id>', methods=['POST'])
def restore_state(state_id):
    """Restore to a saved state"""
    result = state_controller.restore_state(state_id)
    return jsonify(result)


@app.route('/api/simulation/states', methods=['DELETE'])
def clear_all_states():
    """Delete all saved states"""
    result = state_controller.clear_all_states()
    return jsonify(result)



if __name__ == '__main__':
    print("=" * 60)
    print("Starting Traffic Simulation API...")
    print("=" * 60)
    print(f"Config file: {CONFIG_FILE}")
    print(f"GUI Mode: {config.USE_GUI}")
    print(f"Step delay: {config.STEP_DELAY}s")
    print(f"API will be available at: http://localhost:{config.PORT}")
    print("=" * 60)
    app.run(debug=config.DEBUG, port=config.PORT, host=config.HOST)