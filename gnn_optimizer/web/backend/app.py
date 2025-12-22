from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'gnn_secret'
CORS(app) # Allow React to connect

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

@socketio.on('connect')
def handle_connect():
    print('✅ Client connected to Dashboard')

@socketio.on('disconnect')
def handle_disconnect():
    print('❌ Client disconnected')

if __name__ == '__main__':
    print("🌍 Starting Web Server on port 5000...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)