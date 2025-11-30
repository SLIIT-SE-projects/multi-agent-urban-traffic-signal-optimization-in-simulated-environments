import { useState } from 'react'
import { stepSimulation } from '../services/api'

interface ManualControlsProps {
  onRefresh?: () => void
  isRunning?: boolean
  isPaused?: boolean
  isAutoStepping?: boolean
}

export function ManualControls({ onRefresh, isRunning = false, isPaused = false, isAutoStepping = false }: ManualControlsProps) {
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
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Manual Stepping</h3>
      
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {isAutoStepping && (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded mb-4">
          ⚠️ Manual stepping is disabled while auto-stepping is active. Stop auto-stepping first.
        </div>
      )}

      {isPaused && (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded mb-4">
          ⚠️ Manual stepping is disabled while simulation is paused. Resume first.
        </div>
      )}

      <button
        onClick={() => handleAction(stepSimulation, 'Step')}
        disabled={loading || !isRunning || isPaused || isAutoStepping}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold py-3 px-4 rounded transition"
      >
        ⏭ Step Once
      </button>
    </div>
  )
}
