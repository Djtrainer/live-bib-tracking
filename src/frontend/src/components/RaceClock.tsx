import { useState, useEffect } from 'react';

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
        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} 
             title={isConnected ? 'Connected' : 'Disconnected'} />
        
        {/* Clock Status Icon */}
        <span className="text-lg" title={`Clock is ${clockState.status}`}>
          {getStatusIcon()}
        </span>
        
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
        <div className="flex items-center gap-2 mt-3">
          <button
            onClick={handleStart}
            disabled={clockState.status === 'running'}
            className="btn-primary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className="material-icon text-sm">play_arrow</span>
            Start
          </button>
          
          <button
            onClick={handleStop}
            disabled={clockState.status === 'stopped'}
            className="btn-secondary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className="material-icon text-sm">stop</span>
            Stop
          </button>
          
          <button
            onClick={handleEdit}
            className="btn-ghost text-sm"
          >
            <span className="material-icon text-sm">edit</span>
            Edit Time
          </button>
          
          <button
            onClick={handleReset}
            className="btn-ghost text-sm text-red-600 hover:text-red-700"
          >
            <span className="material-icon text-sm">refresh</span>
            Reset
          </button>
        </div>
      )}
    </div>
  );
}
