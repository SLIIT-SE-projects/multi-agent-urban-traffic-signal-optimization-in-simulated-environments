from flask import Flask, jsonify, request
from flask_cors import CORS
from Controllers.simulation_controller import SimulationController
from config import config
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# CHANGE THIS to your actual config file!
# CONFIG_FILE = os.path.join("..", "scenarios", "mapishara.sumo.cfg")
CONFIG_FILE = os.path.join("..", "scenarios", "grid3x3", "grid3x3.sumo.cfg")

# Initialize controller with config settings
controller = SimulationController(
    CONFIG_FILE, 
    use_gui=config.USE_GUI,
    step_delay=config.STEP_DELAY
)

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    result = controller.start()
    return jsonify(result)

@app.route('/api/simulation/step', methods=['POST'])
def step_simulation():
    result = controller.step()
    return jsonify(result)

@app.route('/api/simulation/pause', methods=['POST'])
def pause_simulation():
    result = controller.pause()
    return jsonify(result)

@app.route('/api/simulation/resume', methods=['POST'])
def resume_simulation():
    result = controller.resume()
    return jsonify(result)

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    result = controller.stop()
    return jsonify(result)

@app.route('/api/simulation/data', methods=['GET'])
def get_data():
    data = controller.get_current_data()
    return jsonify(data)

@app.route('/api/simulation/status', methods=['GET'])
def get_status():
    status = controller.get_status()
    return jsonify(status)

@app.route('/api/simulation/save-state', methods=['POST'])
def save_state():
    data = request.json
    state_name = data.get('state_name', 'default')
    result = controller.save_state(state_name)
    return jsonify(result)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})

@app.route('/api/simulation/auto-step/start', methods=['POST'])
def start_auto_step():
    data = request.get_json(silent=True) or {}
    step_delay = data.get('step_delay', None)  # None will use config default
    result = controller.start_auto_stepping(step_delay)
    return jsonify(result)

@app.route('/api/simulation/auto-step/pause', methods=['POST'])
def pause_auto_step():
    result = controller.pause_auto_stepping()
    return jsonify(result)

@app.route('/api/simulation/auto-step/resume', methods=['POST'])
def resume_auto_step():
    result = controller.resume_auto_stepping()
    return jsonify(result)

@app.route('/api/simulation/auto-step/stop', methods=['POST'])
def stop_auto_step():
    result = controller.stop_auto_stepping()
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