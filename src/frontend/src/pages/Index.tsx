import { useState, useEffect } from 'react';
import Layout from '@/components/Layout';
import StatusChip from '@/components/StatusChip';
import DataTable from '@/components/DataTable';
import { formatTime } from '../lib/utils'; 

interface Finisher {
  id: string;
  rank: number;
  bibNumber: string;
  finishTime: number;
  racerName?: string;
}

const Index = () => {
  const [finishers, setFinishers] = useState<Finisher[]>([]);
  const [totalFinishers, setTotalFinishers] = useState(0);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch initial data from backend
    const fetchFinishers = async () => {
      try {
        setLoading(true);
        // Use relative path to leverage Vite proxy
        const response = await fetch('/api/results');
        const result = await response.json();
        
        if (result.success && result.data) {
          setFinishers(result.data);
          setTotalFinishers(result.data.length);
          setLastUpdated(new Date());
        } else {
          console.error('Failed to fetch results:', result);
        }
      } catch (error) {
        console.error('Error fetching finishers:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchFinishers();

    // Set up WebSocket connection for real-time updates
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected to leaderboard');
    };
    
    ws.onmessage = async (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log('Received WebSocket message:', message);
        
        if (message.action === 'reload') {
          // Reload all data when instructed (for delete/reorder operations)
          const response = await fetch('/api/results');
          const result = await response.json();
          
          if (result.success && result.data) {
            setFinishers(result.data);
            setTotalFinishers(result.data.length);
            setLastUpdated(new Date());
          }
        } else if (message.type === 'add' || message.type === 'update') {
          setFinishers(prev => {
            const existingIndex = prev.findIndex(f => f.id === message.data.id || f.bibNumber === message.data.bibNumber);
            let updated;
            if (existingIndex > -1) {
              updated = [...prev];
              updated[existingIndex] = { ...updated[existingIndex], ...message.data };
            } else {
              updated = [...prev, message.data];
            }
            // --- THIS IS THE CORRECT NUMERICAL SORT ---
            updated.sort((a, b) => a.finishTime - b.finishTime);
            return updated;
          });
          setLastUpdated(new Date());
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, []);

  const columns = [
    { key: 'rank', title: 'Rank', width: '80px', align: 'center' as const },
    { key: 'bibNumber', title: 'Bib #', width: '100px', align: 'center' as const },
    { key: 'racerName', title: 'Racer Name', width: '200px', align: 'left' as const },
    { key: 'finishTime', title: 'Finish Time', width: '120px', align: 'center' as const },
  ];

  const renderRow = (finisher: Finisher, index: number) => (
    <tr key={finisher.id || finisher.bibNumber}>
      {/* Re-calculate rank based on the sorted index */}
      <td className="text-center font-mono" style={{ width: '80px' }}>
        {index + 1}
      </td>
      <td className="text-center font-mono font-medium" style={{ width: '100px' }}>
        {finisher.bibNumber}
      </td>
      <td className="text-left" style={{ width: '200px' }}> {/* Names look better left-aligned */}
        {finisher.racerName || 'N/A'}
      </td>
      <td className="text-center font-mono text-lg font-medium" style={{ width: '120px' }}>
        {/* Always format the time, as it comes from the DB as a number */}
        {formatTime(finisher.finishTime)}
      </td>
    </tr>
  );

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">2025 Slay Sarcoma Race Results</h1>
          <p className="text-muted-foreground">Live race results updated in real-time</p>
        </div>

        {/* Stats */}
        <div className="flex flex-wrap gap-4 mb-8">
          <StatusChip 
            label="Total Finishers" 
            value={totalFinishers} 
            variant="success"
            icon="flag"
          />
          <StatusChip 
            label="Last Updated" 
            value={lastUpdated.toLocaleTimeString()} 
            variant="live"
            icon="update"
          />
          <StatusChip 
            label="Race Status" 
            value="In Progress" 
            variant="live"
            icon="radio_button_checked"
          />
        </div>

        {/* Results Table */}
        <div className="bg-card rounded-lg border border-border p-6">
          <div className="flex items-center gap-2 mb-6">
            <span className="material-icon text-primary">leaderboard</span>
            <h2 className="text-xl font-semibold">Live Results</h2>
          </div>
          
          <DataTable
            columns={columns}
            data={finishers}
            renderRow={renderRow}
            emptyMessage="No finishers yet. Results will appear as runners cross the finish line."
          />
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-sm text-muted-foreground">
          <p>Results are updated automatically. No refresh required.</p>
        </div>
      </div>
    </Layout>
  );
};

export default Index;
