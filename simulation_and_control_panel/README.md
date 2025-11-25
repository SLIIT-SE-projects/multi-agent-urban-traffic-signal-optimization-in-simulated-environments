# Traffic Simulation Project

A traffic simulation system built with SUMO (Simulation of Urban Mobility) and Flask, allowing real-time control and visualization of traffic scenarios.

## Features

- **Real-time simulation control** via REST API
- **Multiple scenarios** (grid networks, real-world maps)
- **State management** (save/load simulation states)
- **Auto-stepping** with configurable delays
- **CORS enabled** for frontend integration
- **Configurable settings** for development and production

## Project Structure

```
TrafficProject/
├── backend/                      # Flask API server
│   ├── app.py                   # Main Flask application
│   ├── config.py                # Configuration management
│   ├── Controllers/
│   │   └── simulation_controller.py
│   └── saved_states/            # Saved simulation states
├── scenarios/                   # SUMO simulation scenarios
│   ├── grid3x3/                # 3x3 grid network
│   └── RealMapRajagiriyaToKollupitiya/  # Real-world map
├── scripts/                     # Utility scripts
│   ├── create_grid_network.py
│   └── test_sumo.py
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Prerequisites

- **Python 3.12+** installed
- **SUMO (Eclipse Simulation of Urban Mobility)** installed and `SUMO_HOME` environment variable set
- **pip** package manager

### Setting up SUMO_HOME

On Windows, set the `SUMO_HOME` environment variable to your SUMO installation directory:

```powershell
# Temporary (current session only)
$env:SUMO_HOME = 'C:\Program Files (x86)\Eclipse\Sumo'

# Permanent (using setx)
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
```

After setting it permanently, open a new PowerShell window for changes to take effect.

Verify:

```powershell
echo $env:SUMO_HOME
```

## Installation

1. **Clone/navigate to the project:**

   ```powershell
   cd C:\Users\msi\Desktop\simulation\TrafficProject
   ```

2. **Activate the virtual environment:**

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   If you get an execution policy error, run:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Configuration

Edit `backend/config.py` to customize:

- `DEBUG` – Enable/disable debug mode
- `USE_GUI` – Enable/disable SUMO GUI
- `AUTO_START_STEPPING` – Auto-start simulation stepping
- `STEP_DELAY` – Delay between simulation steps (in seconds)
- `PORT` – API server port (default: 5000)

Or create a `.env` file (see `.env.example`):

```
FLASK_ENV=development
USE_GUI=True
PORT=5000
```

## Running the Application

### Start the API Server

```powershell
cd backend
python app.py
```

The server will be available at `http://localhost:5000`

### API Endpoints

| Endpoint                          | Method | Description                   |
| --------------------------------- | ------ | ----------------------------- |
| `/api/health`                     | GET    | Health check                  |
| `/api/simulation/start`           | POST   | Start simulation              |
| `/api/simulation/step`            | POST   | Step simulation one time unit |
| `/api/simulation/pause`           | POST   | Pause simulation              |
| `/api/simulation/resume`          | POST   | Resume simulation             |
| `/api/simulation/stop`            | POST   | Stop simulation               |
| `/api/simulation/status`          | GET    | Get current status            |
| `/api/simulation/data`            | GET    | Get simulation data           |
| `/api/simulation/save-state`      | POST   | Save current state            |
| `/api/simulation/auto-step/start` | POST   | Start auto-stepping           |
| `/api/simulation/auto-step/stop`  | POST   | Stop auto-stepping            |

### Test the Simulation

```powershell
python test_sumo.py
```

### Create a New Grid Network

```powershell
python scripts/create_grid_network.py
```

## Dependencies

- `flask==3.0.0` – Web framework
- `flask-cors==4.0.0` – Enable CORS for frontend
- SUMO (external) – Simulation engine

See `requirements.txt` for full list.

## Troubleshooting

### "SUMO_HOME not found" error

- Set `SUMO_HOME` environment variable (see Prerequisites section)
- Ensure SUMO is installed in a valid location

### Virtual environment not activating

- Use full path: `C:\Users\msi\Desktop\simulation\TrafficProject\venv\Scripts\Activate.ps1`
- Check execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Port 5000 already in use

- Change `PORT` in `backend/config.py`
- Or kill existing process: `Get-Process | Where-Object {$_.Port -eq 5000} | Stop-Process`

## Future Improvements

- [ ] Add frontend UI (React/Vue)
- [ ] Add WebSocket support for real-time updates
- [ ] Unit tests and CI/CD pipeline
- [ ] Docker containerization
- [ ] Database integration for state persistence
- [ ] Multi-scenario management API

## License

MIT License

## Contact

For issues or questions, refer to `isharareadme.txt` for development notes.
