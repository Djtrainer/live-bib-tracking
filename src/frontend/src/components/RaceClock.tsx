import { useState, useEffect } from 'react';
import { 
  Play, 
  Square, 
  Edit, 
  RotateCcw, 
  Wifi, 
  WifiOff,
  Clock
} from 'lucide-react';

interface RaceClockProps {
  className?: string;
  showControls?: boolean;
}

interface ClockState {
  raceStartTime: number | null;
  status: 'stopped' | 'running' | 'paused';
  offset: number;
}

export default function RaceClock({ className = '', showControls = false }: RaceClockProps) {
  const [clockState, setClockState] = useState<ClockState>({
    raceStartTime: null,
    status: 'stopped',
    offset: 0,
  });
  const [currentTime, setCurrentTime] = useState('00:00.00');
  const [isConnected, setIsConnected] = useState(false);

  // Format time in MM:SS.cs format
  const formatTime = (milliseconds: number): string => {
    const totalSeconds = Math.abs(milliseconds) / 1000;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    const centiseconds = Math.floor((totalSeconds % 1) * 100);
    
    const sign = milliseconds < 0 ? '-' : '';
    return `${sign}${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${centiseconds.toString().padStart(2, '0')}`;
  };

  // Calculate current race time
  const calculateRaceTime = (): number => {
    if (clockState.status !== 'running' || !clockState.raceStartTime) {
      return clockState.offset;
    }
    
    const now = Date.now() / 1000; // Current time in seconds
    const elapsedMs = (now - clockState.raceStartTime) * 1000; // Convert to milliseconds
    return elapsedMs + clockState.offset;
  };

  // Update the displayed time every 10ms for smooth animation
  useEffect(() => {
    const interval = setInterval(() => {
      const raceTime = calculateRaceTime();
      setCurrentTime(formatTime(raceTime));
    }, 10);

    return () => clearInterval(interval);
  }, [clockState]);

  // WebSocket connection for clock updates
  useEffect(() => {
    const connectWebSocket = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('RaceClock WebSocket connected');
        setIsConnected(true);
        
        // Fetch initial clock status
        fetch('/api/clock/status')
          .then(response => response.json())
          .then(result => {
            if (result.success) {
              setClockState(result.data);
            }
          })
          .catch(error => console.error('Error fetching clock status:', error));
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          if (message.type === 'clock_update') {
            console.log('Clock update received:', message.data);
            setClockState(message.data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        console.log('RaceClock WebSocket disconnected');
        setIsConnected(false);
        
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error('RaceClock WebSocket error:', error);
        setIsConnected(false);
      };

      return ws;
    };

    const ws = connectWebSocket();

    return () => {
      ws.close();
    };
  }, []);

  // Clock control functions (only used if showControls is true)
  const handleStart = async () => {
    try {
      const response = await fetch('/api/clock/start', { method: 'POST' });
      const result = await response.json();
      if (!result.success) {
        console.error('Failed to start clock:', result.message);
      }
    } catch (error) {
      console.error('Error starting clock:', error);
    }
  };

  const handleStop = async () => {
    try {
      const response = await fetch('/api/clock/stop', { method: 'POST' });
      const result = await response.json();
      if (!result.success) {
        console.error('Failed to stop clock:', result.message);
      }
    } catch (error) {
      console.error('Error stopping clock:', error);
    }
  };

  const handleReset = async () => {
    try {
      const response = await fetch('/api/clock/reset', { method: 'POST' });
      const result = await response.json();
      if (!result.success) {
        console.error('Failed to reset clock:', result.message);
      }
    } catch (error) {
      console.error('Error resetting clock:', error);
    }
  };

  const handleEdit = async () => {
    const newTime = prompt('Enter new time (MM:SS.cs format):', currentTime);
    if (newTime && newTime.trim()) {
      try {
        const response = await fetch('/api/clock/edit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ time: newTime.trim() }),
        });
        const result = await response.json();
        if (!result.success) {
          alert(`Failed to edit clock: ${result.message}`);
        }
      } catch (error) {
        console.error('Error editing clock:', error);
        alert('Error editing clock. Please try again.');
      }
    }
  };

  const getStatusColor = () => {
    switch (clockState.status) {
      case 'running':
        return 'text-green-600';
      case 'paused':
        return 'text-yellow-600';
      case 'stopped':
      default:
        return 'text-red-600';
    }
  };

  const getStatusIcon = () => {
    switch (clockState.status) {
      case 'running':
        return '▶️';
      case 'paused':
        return '⏸️';
      case 'stopped':
      default:
        return '⏹️';
    }
  };

  return (
    <div className={`race-clock ${className}`}>
      <div className="flex items-center gap-3">
        {/* Connection Status Indicator */}
        <div className="flex items-center gap-1" title={isConnected ? 'Connected' : 'Disconnected'}>
          {isConnected ? (
            <Wifi className="w-4 h-4 text-success" />
          ) : (
            <WifiOff className="w-4 h-4 text-destructive" />
          )}
        </div>
        
        {/* Clock Status Icon */}
        <div className={`flex items-center justify-center w-6 h-6 rounded-full ${
          clockState.status === 'running' ? 'bg-success/20' :
          clockState.status === 'paused' ? 'bg-warning/20' :
          'bg-muted/20'
        }`}>
          {clockState.status === 'running' ? (
            <Play className="w-3 h-3 text-success fill-current" />
          ) : clockState.status === 'paused' ? (
            <Square className="w-3 h-3 text-warning" />
          ) : (
            <Square className="w-3 h-3 text-muted-foreground" />
          )}
        </div>
        
        {/* Race Time Display */}
        <div className="font-mono text-2xl font-bold">
          <span className={getStatusColor()}>{currentTime}</span>
        </div>
        
        {/* Status Text */}
        <div className="text-sm text-muted-foreground">
          <span className={getStatusColor()}>
            {clockState.status.charAt(0).toUpperCase() + clockState.status.slice(1)}
          </span>
        </div>
      </div>

      {/* Control Buttons (only shown if showControls is true) */}
      {showControls && (
        <div className="grid grid-cols-2 gap-3 mt-6">
          <button
            onClick={handleStart}
            disabled={clockState.status === 'running'}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-success text-white rounded-lg font-semibold hover:bg-success/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4" />
            Start Race
          </button>
          
          <button
            onClick={handleStop}
            disabled={clockState.status === 'stopped'}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-destructive text-white rounded-lg font-semibold hover:bg-destructive/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Square className="w-4 h-4" />
            Stop Race
          </button>
          
          <button
            onClick={handleEdit}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:bg-primary/90 transition-colors"
          >
            <Edit className="w-4 h-4" />
            Edit Time
          </button>
          
          <button
            onClick={handleReset}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-muted text-foreground rounded-lg font-semibold hover:bg-muted/80 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset Clock
          </button>
        </div>
      )}
    </div>
  );
}
