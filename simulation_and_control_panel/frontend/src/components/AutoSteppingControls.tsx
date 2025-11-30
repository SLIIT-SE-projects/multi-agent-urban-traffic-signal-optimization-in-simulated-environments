import { useState } from 'react'
import { startSimulation, startAutoStep, pauseAutoStep, resumeAutoStep, stopAutoStep } from '../services/api'

interface AutoSteppingControlsProps {
  onRefresh?: () => void
  isRunning?: boolean
  isPaused?: boolean
}

export function AutoSteppingControls({
  onRefresh,
  isRunning = false,
  isPaused = false,
}: AutoSteppingControlsProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stepDelay, setStepDelay] = useState(0.1)

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
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Auto-Stepping</h3>
      
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {isPaused && (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded mb-4">
          ⚠️ Auto-stepping is disabled while simulation is paused. Resume first.
        </div>
      )}

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Step Delay (seconds)
        </label>
        <input
          type="number"
          step="0.01"
          min="0.01"
          max="10"
          value={stepDelay}
          onChange={(e) => setStepDelay(parseFloat(e.target.value))}
          disabled={loading || isRunning}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 disabled:bg-gray-100"
        />
        <p className="text-xs text-gray-500 mt-1">
          Lower = faster simulation, Higher = slower simulation
        </p>
      </div>

      <div className="flex gap-3">
        <button
          onClick={() => handleAction(
            async () => {
              await startSimulation()
              await startAutoStep(stepDelay)
            },
            'Play'
          )}
          disabled={loading || isRunning || isPaused}
          className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-semibold py-3 px-4 rounded transition"
        >
          ▶ Play
        </button>

        <button
          onClick={() => handleAction(pauseAutoStep, 'Pause')}
          disabled={loading || !isRunning || isPaused}
          className="flex-1 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-400 text-white font-semibold py-3 px-4 rounded transition"
        >
          ⏸ Pause
        </button>

        <button
          onClick={() => handleAction(resumeAutoStep, 'Resume')}
          disabled={loading || !isRunning || !isPaused}
          className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white font-semibold py-3 px-4 rounded transition"
        >
          ⏯ Resume
        </button>

        <button
          onClick={() => handleAction(stopAutoStep, 'Stop Auto-Stepping')}
          disabled={loading || !isRunning}
          className="flex-1 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white font-semibold py-3 px-4 rounded transition"
        >
          ⏹ Stop
        </button>
      </div>
    </div>
  )
}
