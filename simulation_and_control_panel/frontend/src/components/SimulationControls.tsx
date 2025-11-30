import { useState } from 'react'
import {
  startSimulation,
  stopSimulation,
  stepSimulation,
  pauseSimulation,
  resumeSimulation,
  startAutoStep,
  stopAutoStep,
  pauseAutoStep,
  resumeAutoStep,
} from '../services/api'

interface SimulationControlsProps {
  onRefresh?: () => void
  autoStepping?: boolean
}

export function SimulationControls({ onRefresh, autoStepping = false }: SimulationControlsProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleAction = async (action: () => Promise<any>, actionName: string) => {
    try {
      setLoading(true)
      setError(null)
      await action()
      onRefresh?.()
    } catch (err) {
      setError(`${actionName} failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Simulation Controls</h2>
      
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <button
          onClick={() => handleAction(
            () => startSimulation().then(() => startAutoStep(0.1)),
            'Play'
          )}
          disabled={loading || autoStepping}
          className="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-semibold py-2 px-4 rounded transition"
        >
          ▶ Play
        </button>

        <button
          onClick={() => handleAction(stopAutoStep, 'Stop')}
          disabled={loading || !autoStepping}
          className="bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white font-semibold py-2 px-4 rounded transition"
        >
          ⏹ Stop
        </button>

        <button
          onClick={() => handleAction(pauseAutoStep, 'Pause')}
          disabled={loading || !autoStepping}
          className="bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-400 text-white font-semibold py-2 px-4 rounded transition"
        >
          ⏸ Pause
        </button>

        <button
          onClick={() => handleAction(resumeAutoStep, 'Resume')}
          disabled={loading || autoStepping}
          className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white font-semibold py-2 px-4 rounded transition"
        >
          ⏯ Resume
        </button>

        <button
          onClick={() => handleAction(stepSimulation, 'Step')}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold py-2 px-4 rounded transition"
        >
          ⏭ Step
        </button>
      </div>
    </div>
  )
}
