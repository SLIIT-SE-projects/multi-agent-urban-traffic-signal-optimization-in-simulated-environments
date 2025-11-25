from flask import Flask, jsonify, request
from flask_cors import CORS
from Controllers.simulation_controller import SimulationController
from config import config

app = Flask(__name__)
CORS(app)

CONFIG_FILE = os.path.join("..", "scenarios", "grid3x3", "grid3x3.sumo.cfg")
controller = SimulationController(
    CONFIG_FILE,
    use_gui=config.USE_GUI,
    auto_start_stepping=config.AUTO_START_STEPPING,
    step_delay=config.STEP_DELAY
)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"})

if __name__ == '__main__':
    app.run(debug=config.DEBUG, port=config.PORT, host=config.HOST)