import { useEffect, useState } from 'react'
import { healthCheck, getStatus } from './services/api'
import { SimulationLifecycle } from './components/SimulationLifecycle'
import { ManualControls } from './components/ManualControls'
import { AutoSteppingControls } from './components/AutoSteppingControls'

function App() {
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'error'>('checking')
  const [mode, setMode] = useState<'manual' | 'auto'>('manual')
  const [simulationStatus, setSimulationStatus] = useState({
    is_running: false,
    is_paused: false,
    current_step: 0,
    auto_stepping: false,
  })

  useEffect(() => {
    const checkBackend = async () => {
      try {
        await healthCheck()
        setBackendStatus('connected')
        fetchStatus()
      } catch (error) {
        setBackendStatus('error')
      }
    }

    checkBackend()
    const interval = setInterval(fetchStatus, 1000)
    return () => clearInterval(interval)
  }, [])

  const fetchStatus = async () => {
    try {
      const response = await getStatus()
      setSimulationStatus(response.data)
    } catch (error) {
      console.error('Failed to fetch status:', error)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-7xl mx-auto py-12 px-4">
        {/* Backend Status */}
        <div className="bg-white rounded-lg shadow-lg p-8 mb-6">
          <h1 className="text-4xl font-bold text-gray-900 mb-6">
            Traffic Simulation Control Panel
          </h1>
          
          <div className="mb-4">
            <h2 className="text-lg font-semibold text-gray-700 mb-3">Backend Status</h2>
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${
                backendStatus === 'checking' ? 'bg-yellow-500 animate-pulse' :
                backendStatus === 'connected' ? 'bg-green-500' :
                'bg-red-500'
              }`}></div>
              <span>
                {backendStatus === 'checking' && 'Checking...'}
                {backendStatus === 'connected' && '✓ Backend Connected'}
                {backendStatus === 'error' && '✗ Backend Offline'}
              </span>
            </div>
          </div>
        </div>

        {/* Simulation Status */}
        {backendStatus === 'connected' && (
          <div className="bg-white rounded-lg shadow-lg p-8 mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Simulation Status</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-50 p-4 rounded">
                <p className="text-gray-600 text-sm">Step</p>
                <p className="text-2xl font-bold text-gray-900">{simulationStatus.current_step}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <p className="text-gray-600 text-sm">Running</p>
                <p className="text-2xl font-bold text-gray-900">
                  {simulationStatus.is_running ? '✓' : '✗'}
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <p className="text-gray-600 text-sm">Paused</p>
                <p className="text-2xl font-bold text-gray-900">
                  {simulationStatus.is_paused ? '✓' : '✗'}
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <p className="text-gray-600 text-sm">Auto-Stepping</p>
                <p className="text-2xl font-bold text-gray-900">
                  {simulationStatus.auto_stepping ? '✓' : '✗'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Mode Toggle */}
        {backendStatus === 'connected' && (
          <div className="bg-white rounded-lg shadow-lg p-8 mb-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Control Mode</h2>
            <div className="flex gap-3">
              <button
                onClick={() => setMode('manual')}
                className={`flex-1 py-3 px-4 rounded font-semibold transition ${
                  mode === 'manual'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Manual Stepping
              </button>
              <button
                onClick={() => setMode('auto')}
                className={`flex-1 py-3 px-4 rounded font-semibold transition ${
                  mode === 'auto'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Auto-Stepping
              </button>
            </div>
          </div>
        )}

        {/* Controls */}
        {backendStatus === 'connected' && (
          <>
            <SimulationLifecycle
              onRefresh={fetchStatus}
              isRunning={simulationStatus.is_running}
              isPaused={simulationStatus.is_paused}
            />
            
            {mode === 'manual' && (
              <ManualControls
                onRefresh={fetchStatus}
                isRunning={simulationStatus.is_running}
                isPaused={simulationStatus.is_paused}
                isAutoStepping={simulationStatus.auto_stepping}
              />
            )}
            {mode === 'auto' && (
              <AutoSteppingControls
                onRefresh={fetchStatus}
                isRunning={simulationStatus.auto_stepping}
                isPaused={simulationStatus.is_paused}
              />
            )}
          </>
        )}

        {backendStatus === 'error' && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded text-red-700">
            Backend is not running. Please start the backend server first.
          </div>
        )}
      </div>
    </div>
  )
}

export default App
