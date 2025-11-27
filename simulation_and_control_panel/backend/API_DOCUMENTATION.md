# Traffic Simulation API - Complete Documentation

## Overview

This API provides comprehensive control and monitoring of the SUMO traffic simulation through RESTful endpoints. The backend has been refactored into modular controllers for better maintainability and scalability.

## Architecture

### Controllers

- **SimulationController**: Handles simulation lifecycle (start, stop, step, pause, resume, auto-stepping)
- **ScenarioController**: Manages scenario operations (switch, list, reload scenarios)
- **DataController**: Retrieves simulation data (vehicles, traffic lights, status)
- **StateController**: Manages simulation state save/restore operations

---

## API Endpoints

### Health Check

#### GET `/api/health`

Verify that the API server is running.

**Response:**

```json
{
  "status": "healthy",
  "message": "API is running"
}
```

---

## Simulation Lifecycle

### POST `/api/simulation/start`

Start the simulation with the currently configured scenario.

**Response:**

```json
{
  "status": "success",
  "message": "Simulation started",
  "step": 0,
  "auto_stepping": false
}
```

### POST `/api/simulation/step`

Execute one simulation step (only when auto-stepping is not active).

**Response:**

```json
{
  "status": "success",
  "step": 1,
  "vehicle_count": 5,
  "vehicles": [...],
  "traffic_lights": [...],
  "is_paused": false,
  "auto_stepping": false
}
```

### POST `/api/simulation/pause`

Pause the simulation.

**Response:**

```json
{
  "status": "success",
  "message": "Simulation paused",
  "step": 42
}
```

### POST `/api/simulation/resume`

Resume the paused simulation.

**Response:**

```json
{
  "status": "success",
  "message": "Simulation resumed",
  "step": 42
}
```

### POST `/api/simulation/stop`

Stop and close the simulation.

**Response:**

```json
{
  "status": "success",
  "message": "Simulation stopped"
}
```

---

## Auto-Stepping Control

### POST `/api/simulation/auto-step/start`

Start automatic stepping in the background.

**Request Body:**

```json
{
  "step_delay": 0.1 // Optional, in seconds. Uses config default if omitted
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Auto-stepping started",
  "step_delay": 0.1
}
```

### POST `/api/simulation/auto-step/pause`

Pause auto-stepping (freeze simulation while keeping thread alive).

**Response:**

```json
{
  "status": "success",
  "message": "Auto-stepping paused",
  "step": 100
}
```

### POST `/api/simulation/auto-step/resume`

Resume paused auto-stepping.

**Response:**

```json
{
  "status": "success",
  "message": "Auto-stepping resumed",
  "step": 100
}
```

### POST `/api/simulation/auto-step/stop`

Stop auto-stepping completely.

**Response:**

```json
{
  "status": "success",
  "message": "Auto-stepping stopped"
}
```

---

## Scenario Management (NEW)

### GET `/api/scenarios`

Get list of all available scenarios.

**Response:**

```json
{
  "status": "success",
  "scenarios": [
    {
      "name": "grid3x3",
      "config_file": ".../scenarios/grid3x3/grid3x3.sumo.cfg",
      "path": ".../scenarios/grid3x3"
    },
    {
      "name": "RealMapRajagiriyaToKollupitiya",
      "config_file": ".../scenarios/RealMapRajagiriyaToKollupitiya/mapishara.sumo.cfg",
      "path": ".../scenarios/RealMapRajagiriyaToKollupitiya"
    }
  ],
  "count": 2
}
```

### POST `/api/simulation/switch-scenario`

Switch to a different scenario at runtime (stops current, loads new, starts new).

**Request Body:**

```json
{
  "scenario_name": "RealMapRajagiriyaToKollupitiya"
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Successfully switched to scenario 'RealMapRajagiriyaToKollupitiya'",
  "scenario": "RealMapRajagiriyaToKollupitiya",
  "config_file": "...",
  "step": 0
}
```

### POST `/api/simulation/reload`

Reload the current scenario from the beginning.

**Response:**

```json
{
  "status": "success",
  "message": "Scenario reloaded successfully",
  "scenario_config": "...",
  "step": 0
}
```

### GET `/api/simulation/current-scenario`

Get information about the currently loaded scenario.

**Response:**

```json
{
  "status": "success",
  "scenario_name": "grid3x3",
  "config_file": "...",
  "is_running": true,
  "current_step": 150,
  "is_paused": false
}
```

---

## Data Retrieval

### GET `/api/simulation/data`

Get current simulation data including vehicles and traffic lights.

**Response:**

```json
{
  "status": "success",
  "step": 100,
  "vehicle_count": 42,
  "vehicles": [
    {
      "id": "vehicle.0",
      "speed": 12.5,
      "position": [100.5, 200.3],
      "waiting_time": 0.0
    }
  ],
  "traffic_lights": [
    {
      "id": "traffic_light_1",
      "state": "GGGgrrr",
      "phase": 0
    }
  ],
  "is_paused": false,
  "auto_stepping": true
}
```

### GET `/api/simulation/status`

Get current simulation status.

**Response:**

```json
{
  "is_running": true,
  "is_paused": false,
  "current_step": 100,
  "auto_stepping": true
}
```

### GET `/api/simulation/vehicles`

Get total count of vehicles and their IDs.

**Response:**

```json
{
  "status": "success",
  "vehicle_count": 42,
  "vehicles": ["vehicle.0", "vehicle.1", ...]
}
```

### GET `/api/simulation/vehicles/<vehicle_id>`

Get detailed information about a specific vehicle.

**Response:**

```json
{
  "status": "success",
  "id": "vehicle.0",
  "speed": 12.5,
  "position": [100.5, 200.3],
  "angle": 45.2,
  "waiting_time": 0.0,
  "route_id": "route_0",
  "edge_id": "edge_1_2",
  "lane_id": "edge_1_2_0",
  "lane_position": 50.3
}
```

### GET `/api/simulation/traffic-lights`

Get all traffic light states.

**Response:**

```json
{
  "status": "success",
  "traffic_light_count": 5,
  "traffic_lights": [
    {
      "id": "traffic_light_1",
      "state": "GGGgrrr",
      "phase": 0,
      "phase_duration": 30.0
    }
  ]
}
```

---

## State Management

### POST `/api/simulation/save-state`

Save current simulation state to disk.

**Request Body:**

```json
{
  "state_name": "checkpoint_1" // Optional, defaults to "default"
}
```

**Response:**

```json
{
  "status": "success",
  "message": "State 'checkpoint_1' saved successfully",
  "state_id": "checkpoint_1_20251126_150230",
  "step": 500,
  "file": ".../saved_states/checkpoint_1_20251126_150230.json"
}
```

### GET `/api/simulation/states`

Get list of all saved states.

**Response:**

```json
{
  "status": "success",
  "saved_states": [
    {
      "state_id": "checkpoint_1_20251126_150230",
      "state_name": "checkpoint_1",
      "step": 500,
      "timestamp": "2025-11-26T15:02:30.123456",
      "config_file": "..."
    }
  ],
  "count": 1
}
```

### GET `/api/simulation/states/<state_id>`

Get detailed information about a specific saved state.

**Response:**

```json
{
  "status": "success",
  "state_id": "checkpoint_1_20251126_150230",
  "state_data": {
    "state_name": "checkpoint_1",
    "step": 500,
    "timestamp": "2025-11-26T15:02:30.123456",
    "is_paused": false,
    "config_file": "...",
    "data": { ... }
  }
}
```

### DELETE `/api/simulation/states/<state_id>`

Delete a specific saved state.

**Response:**

```json
{
  "status": "success",
  "message": "State 'checkpoint_1_20251126_150230' deleted successfully"
}
```

### POST `/api/simulation/restore-state/<state_id>`

Restore to a saved state (returns saved state metadata).

**Response:**

```json
{
  "status": "success",
  "message": "State 'checkpoint_1_20251126_150230' metadata restored",
  "state_id": "checkpoint_1_20251126_150230",
  "state_name": "checkpoint_1",
  "step": 500,
  "timestamp": "2025-11-26T15:02:30.123456",
  "note": "This returns saved state data. Full simulation rollback requires SUMO checkpoint support."
}
```

### DELETE `/api/simulation/states`

Delete all saved states.

**Response:**

```json
{
  "status": "success",
  "message": "Cleared 5 saved state(s)"
}
```

---

## Usage Examples

### Example 1: Start Simulation and List Scenarios

```bash
# Start simulation
curl -X POST http://localhost:5000/api/simulation/start

# Get available scenarios
curl http://localhost:5000/api/scenarios

# Switch to different scenario
curl -X POST http://localhost:5000/api/simulation/switch-scenario \
  -H "Content-Type: application/json" \
  -d '{"scenario_name": "RealMapRajagiriyaToKollupitiya"}'
```

### Example 2: Auto-Stepping with Data Collection

```bash
# Start auto-stepping
curl -X POST http://localhost:5000/api/simulation/auto-step/start \
  -H "Content-Type: application/json" \
  -d '{"step_delay": 0.1}'

# Get simulation data
curl http://localhost:5000/api/simulation/data

# Stop auto-stepping
curl -X POST http://localhost:5000/api/simulation/auto-step/stop
```

### Example 3: Save and Restore State

```bash
# Save current state
curl -X POST http://localhost:5000/api/simulation/save-state \
  -H "Content-Type: application/json" \
  -d '{"state_name": "my_checkpoint"}'

# Get saved states
curl http://localhost:5000/api/simulation/states

# Restore state
curl -X POST http://localhost:5000/api/simulation/restore-state/my_checkpoint_20251126_150230
```

---

## Error Responses

All endpoints follow a consistent error response format:

```json
{
  "status": "error",
  "message": "Description of what went wrong"
}
```

Common error scenarios:

- Simulation not running
- Invalid scenario name
- Auto-stepping already active
- State file not found
- SUMO connection lost

---

## Implementation Details

### Thread Safety

- Simulation stepping uses locks (`step_lock`) to prevent race conditions
- Auto-stepping runs in background thread
- Data retrieval is thread-safe

### Scenario Switching

- Automatically detects `.sumo.cfg` files in scenario folders
- Gracefully stops current simulation before switching
- Resets step counter on switch
- Supports dynamic switching without server restart

### State Management

- States saved as JSON with metadata and snapshot data
- States persisted to disk in `saved_states/` directory
- Full simulation rollback requires SUMO checkpoint support (future enhancement)

---

## Future Enhancements

- [ ] WebSocket support for real-time data streaming
- [ ] Full simulation rollback with SUMO checkpoints
- [ ] Vehicle injection and dynamic route modification
- [ ] Traffic light signal optimization endpoints
- [ ] Metrics aggregation and analysis
- [ ] Database persistence for long-term state storage
