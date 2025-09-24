import { useState, useEffect } from 'react';
import Layout from '@/components/Layout';
import StatusChip from '@/components/StatusChip';
import DataTable from '@/components/DataTable';
import RaceClock from '@/components/RaceClock';
import { formatTime } from '../lib/utils';

interface Finisher {
  id: string;
  rank: number;
  bibNumber: string;
  finishTime: number;
  racerName?: string;
  gender?: string;
  team?: string;
}

interface TeamScore {
  teamName: string;
  totalTime: number;
  runners: Finisher[];
}

const Index = () => {
  const [finishers, setFinishers] = useState<Finisher[]>([]);
  const [totalFinishers, setTotalFinishers] = useState(0);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [loading, setLoading] = useState(true);

  // Category-specific leaderboards
  const [topMen, setTopMen] = useState<Finisher[]>([]);
  const [topWomen, setTopWomen] = useState<Finisher[]>([]);
  const [teamData, setTeamData] = useState<Map<string, Finisher[]>>(new Map());

  // Function to calculate category leaderboards
  const calculateCategoryLeaderboards = (finishersData: Finisher[]) => {
    // Filter and sort men (top 3)
    const men = finishersData
      .filter(f => f.gender === 'M')
      .sort((a, b) => a.finishTime - b.finishTime)
      .slice(0, 3);
    setTopMen(men);

    // Filter and sort women (top 3)
    const women = finishersData
      .filter(f => f.gender === 'W')
      .sort((a, b) => a.finishTime - b.finishTime)
      .slice(0, 3);
    setTopWomen(women);

    // Group finishers by team (NEW: Dynamic team grouping)
    const teamMap = new Map<string, Finisher[]>();
    
    finishersData.forEach(finisher => {
      if (finisher.team && finisher.team.trim()) {
        const teamName = finisher.team.trim();
        if (!teamMap.has(teamName)) {
          teamMap.set(teamName, []);
        }
        teamMap.get(teamName)!.push(finisher);
      }
    });

    // Sort each team's finishers by finish time
    teamMap.forEach((runners, teamName) => {
      runners.sort((a, b) => a.finishTime - b.finishTime);
    });

    setTeamData(teamMap);
  };

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
          calculateCategoryLeaderboards(result.data);
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
      console.log('✅ WebSocket connection opened - Index.tsx');
      console.log('🔗 WebSocket URL:', wsUrl);
      console.log('🔗 WebSocket readyState:', ws.readyState);
    };
    
    ws.onmessage = async (event) => {
      console.log('📨 Raw WebSocket message received in Index.tsx:', event.data);
      try {
        const message = JSON.parse(event.data);
        console.log('✅ Parsed WebSocket message in Index.tsx:', message);
        console.log('📋 Message type:', message.type || message.action || 'unknown');
        
        if (message.action === 'reload') {
          // Reload all data when instructed (for delete/reorder operations)
          const response = await fetch('/api/results');
          const result = await response.json();
          
          if (result.success && result.data) {
            setFinishers(result.data);
            setTotalFinishers(result.data.length);
            setLastUpdated(new Date());
            calculateCategoryLeaderboards(result.data);
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
            calculateCategoryLeaderboards(updated);
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

  const renderCategoryCard = (title: string, icon: string, data: Finisher[] | TeamScore[], type: 'individual' | 'team') => (
    <div className="bg-card rounded-lg border border-border p-6">
      <div className="flex items-center gap-2 mb-4">
        <span className="material-icon text-primary">{icon}</span>
        <h3 className="text-lg font-semibold">{title}</h3>
      </div>
      
      {data.length === 0 ? (
        <p className="text-muted-foreground text-sm">No results yet</p>
      ) : (
        <div className="space-y-3">
          {data.map((item, index) => (
            <div key={index} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-8 h-8 bg-primary text-primary-foreground rounded-full text-sm font-bold">
                  {index + 1}
                </div>
                <div>
                  {type === 'individual' ? (
                    <>
                      <div className="font-medium">{(item as Finisher).racerName || 'N/A'}</div>
                      <div className="text-sm text-muted-foreground">Bib #{(item as Finisher).bibNumber}</div>
                    </>
                  ) : (
                    <>
                      <div className="font-medium">{(item as TeamScore).teamName}</div>
                      <div className="text-sm text-muted-foreground">
                        {(item as TeamScore).runners.map(r => `#${r.bibNumber}`).join(', ')}
                      </div>
                    </>
                  )}
                </div>
              </div>
              <div className="text-right">
                <div className="font-mono font-medium">
                  {type === 'individual' 
                    ? formatTime((item as Finisher).finishTime)
                    : formatTime((item as TeamScore).totalTime)
                  }
                </div>
                {type === 'individual' ? (
                  <div className="text-xs text-muted-foreground">
                    Overall #{(item as Finisher).rank || 'N/A'}
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">
                    Combined time
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold mb-2">2025 Slay Sarcoma Race Results</h1>
              <p className="text-muted-foreground">Live race results updated in real-time</p>
            </div>
            <div className="text-right">
              <div className="mb-2">
                <span className="text-sm text-muted-foreground">Official Race Time</span>
              </div>
              <RaceClock showControls={false} />
            </div>
          </div>
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

        {/* Category Leaderboards */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-6">
            <span className="material-icon text-primary">emoji_events</span>
            <h2 className="text-xl font-semibold">Category Leaders</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {renderCategoryCard("Top 3 Men", "male", topMen, 'individual')}
            {renderCategoryCard("Top 3 Women", "female", topWomen, 'individual')}
          </div>
        </div>

        {/* Dynamic Team Leaderboards */}
        {teamData.size > 0 && (
          <div className="mb-8">
            <div className="flex items-center gap-2 mb-6">
              <span className="material-icon text-primary">groups</span>
              <h2 className="text-xl font-semibold">Team Results</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Array.from(teamData.entries()).map(([teamName, runners]) => (
                <div key={teamName} className="bg-card rounded-lg border border-border p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <span className="material-icon text-primary">group</span>
                    <h3 className="text-lg font-semibold">{teamName}</h3>
                  </div>
                  
                  {runners.length === 0 ? (
                    <p className="text-muted-foreground text-sm">No finishers yet</p>
                  ) : (
                    <div className="space-y-3">
                      {runners.slice(0, 3).map((runner, index) => (
                        <div key={runner.id} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                          <div className="flex items-center gap-3">
                            <div className="flex items-center justify-center w-8 h-8 bg-primary text-primary-foreground rounded-full text-sm font-bold">
                              {index + 1}
                            </div>
                            <div>
                              <div className="font-medium">{runner.racerName || 'N/A'}</div>
                              <div className="text-sm text-muted-foreground">Bib #{runner.bibNumber}</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-mono font-medium">
                              {formatTime(runner.finishTime)}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              Overall #{runner.rank || 'N/A'}
                            </div>
                          </div>
                        </div>
                      ))}
                      {runners.length > 3 && (
                        <div className="text-center text-sm text-muted-foreground pt-2">
                          +{runners.length - 3} more team member{runners.length - 3 !== 1 ? 's' : ''}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Live Results */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-6">
            <span className="material-icon text-primary">leaderboard</span>
            <h2 className="text-xl font-semibold">Live Results</h2>
          </div>
          
          <div className="bg-card rounded-lg border border-border p-6">
            <DataTable
              columns={columns}
              data={finishers}
              renderRow={renderRow}
              emptyMessage="No finishers yet. Results will appear as runners cross the finish line."
            />
          </div>
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
