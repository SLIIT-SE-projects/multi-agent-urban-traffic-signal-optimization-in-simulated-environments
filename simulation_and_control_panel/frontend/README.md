# Simulation Control Panel Frontend

Modern React-based frontend for the traffic simulation control panel.

## Setup

```bash
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Architecture

### Store (Zustand)

- `src/store/simulationStore.ts` - Central state management for simulation

### Components

- **Layout**: Main app container and navigation
- **SimulationControls**: Start/Stop/Step/Pause/Resume controls
- **ScenarioManager**: Switch between scenarios
- **DataViewer**: Display vehicles and traffic lights
- **StateManager**: Save/restore simulation states

### Services

- `src/services/api.ts` - API client with axios

## Features

- ✅ Real-time simulation control (start, pause, resume, step)
- ✅ Auto-stepping with configurable delays
- ✅ Scenario switching and management
- ✅ Vehicle and traffic light visualization
- ✅ Simulation state save/restore
- ✅ Responsive UI with Tailwind CSS
- ✅ Real-time status updates

## Build

```bash
npm run build
```

## Development

```bash
npm run dev
```

Uses Vite for fast HMR (Hot Module Replacement) during development.
