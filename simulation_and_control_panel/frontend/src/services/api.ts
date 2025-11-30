import axios from 'axios'

const apiClient = axios.create({
  baseURL: '/api',
  timeout: 5000,
})

// Health Check
export const healthCheck = () => apiClient.get('/health')

// Simulation Control
export const startSimulation = () => apiClient.post('/simulation/start')
export const stepSimulation = () => apiClient.post('/simulation/step')
export const pauseSimulation = () => apiClient.post('/simulation/pause')
export const resumeSimulation = () => apiClient.post('/simulation/resume')
export const stopSimulation = () => apiClient.post('/simulation/stop')

// Auto-Stepping
export const startAutoStep = (stepDelay?: number) =>
  apiClient.post('/simulation/auto-step/start', { step_delay: stepDelay })
export const pauseAutoStep = () => apiClient.post('/simulation/auto-step/pause')
export const resumeAutoStep = () => apiClient.post('/simulation/auto-step/resume')
export const stopAutoStep = () => apiClient.post('/simulation/auto-step/stop')

// Status
export const getStatus = () => apiClient.get('/simulation/status')

export default apiClient
