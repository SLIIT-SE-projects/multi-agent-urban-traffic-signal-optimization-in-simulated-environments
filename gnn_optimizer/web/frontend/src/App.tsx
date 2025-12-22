import { useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, Car, Zap, Play, Square } from 'lucide-react';

// Connect to Python Backend
const socket = io('http://localhost:5000');

interface TrafficData {
  step: number;
  total_queue: number;
  avg_speed: number;
}

// FIX 1: Interface for the socket response
interface TrafficResponse {
  step: number;
  total_queue: number;
  avg_speed: number;
  intersections: Record<string, any>;
}

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [dataHistory, setDataHistory] = useState<TrafficData[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<TrafficData>({ step: 0, total_queue: 0, avg_speed: 0 });

  useEffect(() => {
    socket.on('connect', () => setIsConnected(true));
    socket.on('disconnect', () => setIsConnected(false));

    // FIX 2: Use the typed interface instead of 'any'
    socket.on('traffic_update', (data: TrafficResponse) => {
      const newData = {
        step: data.step,
        total_queue: parseFloat(data.total_queue.toFixed(2)),
        avg_speed: parseFloat(data.avg_speed.toFixed(2)),
      };

      setCurrentMetrics(newData);
      
      // Keep last 50 points for the chart
      setDataHistory(prev => {
        const newHistory = [...prev, newData];
        if (newHistory.length > 50) return newHistory.slice(newHistory.length - 50);
        return newHistory;
      });
    });

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('traffic_update');
    };
  }, []);

  const handleStart = async () => {
    await fetch('http://localhost:5000/api/start', { method: 'POST' });
    setIsRunning(true);
  };

  const handleStop = async () => {
    await fetch('http://localhost:5000/api/stop', { method: 'POST' });
    setIsRunning(false);
  };

  return (
    <div className="min-h-screen p-8 max-w-7xl mx-auto">
      {/* Header */}
      <header className="flex justify-between items-center mb-8 border-b border-slate-700 pb-4">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
            GNN Traffic Optimizer
          </h1>
          <p className="text-slate-400 mt-1">Real-time MARL Inference Engine</p>
        </div>
        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${isConnected ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-400'}`} />
            {isConnected ? 'Backend Connected' : 'Disconnected'}
          </div>
        </div>
      </header>

      {/* Controls & Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {/* Control Panel */}
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
          <h2 className="text-lg font-semibold mb-4 text-slate-200">Simulation Control</h2>
          <div className="flex gap-3">
            <button 
              onClick={handleStart}
              disabled={isRunning}
              className="flex-1 flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white py-2 rounded-lg transition-colors font-medium"
            >
              <Play size={18} /> Start
            </button>
            <button 
              onClick={handleStop}
              disabled={!isRunning}
              className="flex-1 flex items-center justify-center gap-2 bg-red-600 hover:bg-red-500 disabled:opacity-50 disabled:cursor-not-allowed text-white py-2 rounded-lg transition-colors font-medium"
            >
              <Square size={18} /> Stop
            </button>
          </div>
        </div>

        {/* Metric Cards */}
        <MetricCard 
          title="Total Queue Length" 
          value={currentMetrics.total_queue} 
          unit="veh" 
          icon={<Car className="text-blue-400" />} 
        />
        <MetricCard 
          title="Avg Network Speed" 
          value={currentMetrics.avg_speed} 
          unit="m/s" 
          icon={<Zap className="text-yellow-400" />} 
        />
        <MetricCard 
          title="Simulation Step" 
          value={currentMetrics.step} 
          unit="t" 
          icon={<Activity className="text-purple-400" />} 
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Network Congestion (Total Queue)">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={dataHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="step" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} 
                itemStyle={{ color: '#f8fafc' }}
              />
              <Line 
                type="monotone" 
                dataKey="total_queue" 
                stroke="#60a5fa" 
                strokeWidth={3} 
                dot={false} 
                activeDot={{ r: 6 }} 
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Traffic Flow (Avg Speed)">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={dataHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="step" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }} 
                itemStyle={{ color: '#f8fafc' }}
              />
              <Line 
                type="monotone" 
                dataKey="avg_speed" 
                stroke="#facc15" 
                strokeWidth={3} 
                dot={false} 
                activeDot={{ r: 6 }} 
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}

// FIX 3: Add explicit interfaces for component props

interface MetricCardProps {
  title: string;
  value: number | string;
  unit: string;
  icon: React.ReactNode;
}

const MetricCard = ({ title, value, unit, icon }: MetricCardProps) => (
  <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 flex items-center justify-between">
    <div>
      <p className="text-slate-400 text-sm font-medium">{title}</p>
      <p className="text-2xl font-bold mt-1">
        {value} <span className="text-slate-500 text-sm font-normal">{unit}</span>
      </p>
    </div>
    <div className="p-3 bg-slate-700/50 rounded-lg">
      {icon}
    </div>
  </div>
);

interface ChartCardProps {
  title: string;
  children: React.ReactNode;
}

const ChartCard = ({ title, children }: ChartCardProps) => (
  <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 h-96 flex flex-col">
    <h3 className="text-slate-200 font-semibold mb-6">{title}</h3>
    <div className="flex-1 min-h-0">
      {children}
    </div>
  </div>
);

export default App;