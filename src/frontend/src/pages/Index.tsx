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
    { key: 'bibNumber', title: 'Bib Number', width: '120px', align: 'center' as const },
    { key: 'racerName', title: 'Runner Name', width: '250px', align: 'left' as const },
    { key: 'finishTime', title: 'Finish Time', width: '140px', align: 'center' as const },
  ];

  const renderRow = (finisher: Finisher, index: number) => (
    <tr key={finisher.id || finisher.bibNumber} className="group hover:bg-primary/5 transition-all duration-200">
      <td className="text-center py-4 px-6" style={{ width: '80px' }}>
        <div className={`inline-flex items-center justify-center w-8 h-8 rounded-full font-bold text-sm transition-all duration-200 ${
          index < 3 
            ? 'bg-gradient-to-r from-primary to-accent text-white shadow-lg' 
            : 'bg-muted text-muted-foreground group-hover:bg-primary group-hover:text-white'
        }`}>
          {index + 1}
        </div>
      </td>
      <td className="text-center font-mono font-bold text-lg py-4 px-6" style={{ width: '100px' }}>
        <span className="px-3 py-1 bg-primary/10 text-primary rounded-full border border-primary/20">
          #{finisher.bibNumber}
        </span>
      </td>
      <td className="text-left py-4 px-6" style={{ width: '200px' }}>
        <div className="flex flex-col">
          <span className="font-semibold text-foreground">
            {finisher.racerName || 'N/A'}
          </span>
          {finisher.gender && (
            <span className="text-xs text-muted-foreground flex items-center gap-1">
              <span className="material-icon text-xs">
                {finisher.gender === 'M' ? 'male' : 'female'}
              </span>
              {finisher.gender === 'M' ? 'Male' : 'Female'}
            </span>
          )}
        </div>
      </td>
      <td className="text-center py-4 px-6" style={{ width: '120px' }}>
        <div className="font-mono text-lg font-bold text-primary">
          {formatTime(finisher.finishTime)}
        </div>
      </td>
    </tr>
  );

  const renderCategoryCard = (title: string, icon: string, data: Finisher[] | TeamScore[], type: 'individual' | 'team') => (
    <div className="modern-card p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-r from-primary to-accent text-white">
          <span className="material-icon text-xl">{icon}</span>
        </div>
        <div>
          <h3 className="text-xl font-bold text-foreground">{title}</h3>
          <p className="text-sm text-muted-foreground">Top performers</p>
        </div>
      </div>
      
      {data.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <span className="material-icon text-6xl text-muted-foreground/30 mb-4">emoji_events</span>
          <p className="text-muted-foreground font-medium">No results yet</p>
          <p className="text-sm text-muted-foreground/60">Rankings will appear as racers finish</p>
        </div>
      ) : (
        <div className="space-y-4">
          {data.map((item, index) => (
            <div key={index} className="group relative overflow-hidden rounded-xl bg-gradient-to-r from-card to-muted/30 p-4 border border-border/50 hover:border-primary/30 transition-all duration-300">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`flex items-center justify-center w-12 h-12 rounded-xl font-bold text-lg transition-all duration-300 ${
                    index === 0 
                      ? 'bg-gradient-to-r from-yellow-500 to-yellow-400 text-black shadow-lg shadow-yellow-500/30' 
                      : index === 1 
                      ? 'bg-gradient-to-r from-gray-400 to-gray-300 text-black shadow-lg shadow-gray-400/30'
                      : index === 2 
                      ? 'bg-gradient-to-r from-orange-500 to-orange-400 text-white shadow-lg shadow-orange-500/30'
                      : 'bg-gradient-to-r from-primary/80 to-accent/80 text-white'
                  }`}>
                    {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : index + 1}
                  </div>
                  <div>
                    {type === 'individual' ? (
                      <>
                        <div className="font-bold text-lg text-foreground">{(item as Finisher).racerName || 'N/A'}</div>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <span className="material-icon text-sm">confirmation_number</span>
                          Bib #{(item as Finisher).bibNumber}
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="font-bold text-lg text-foreground">{(item as TeamScore).teamName}</div>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <span className="material-icon text-sm">group</span>
                          {(item as TeamScore).runners.map(r => `#${r.bibNumber}`).join(', ')}
                        </div>
                      </>
                    )}
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-2xl font-bold text-primary mb-1">
                    {type === 'individual' 
                      ? formatTime((item as Finisher).finishTime)
                      : formatTime((item as TeamScore).totalTime)
                    }
                  </div>
                  {type === 'individual' ? (
                    <div className="text-xs text-muted-foreground flex items-center gap-1">
                      <span className="material-icon text-xs">leaderboard</span>
                      Overall #{(item as Finisher).rank || 'N/A'}
                    </div>
                  ) : (
                    <div className="text-xs text-muted-foreground">
                      Combined time
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <Layout>
      {/* Modern Hero Section */}
      <div className="relative bg-gradient-to-br from-background via-background to-primary/5">
        <div className="absolute inset-0 bg-grid-pattern opacity-[0.02]"></div>
        <div className="container mx-auto px-6 py-16">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-3 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary font-medium text-sm mb-6">
              <span className="material-icon text-lg animate-pulse">event</span>
              2025 Official Race Event
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-foreground via-primary to-accent bg-clip-text text-transparent">
              Slay Sarcoma Race
            </h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Live race results and leaderboard updated in real-time. 
              Track your performance and celebrate every finish line crossed.
            </p>
            <div className="flex items-center justify-center gap-2">
              <RaceClock showControls={false} />
            </div>
          </div>

          {/* Enhanced Stats Grid */}
          <div className="stats-grid mb-16">
            <div className="stat-card">
              <StatusChip 
                label="Total Finishers" 
                value={totalFinishers} 
                variant="success"
                icon="flag"
              />
            </div>
            <div className="stat-card">
              <StatusChip 
                label="Last Updated" 
                value={lastUpdated.toLocaleTimeString()} 
                variant="live"
                icon="update"
              />
            </div>
            <div className="stat-card">
              <StatusChip 
                label="Men Finished" 
                value={topMen.length > 0 ? `${topMen.length}+` : 0} 
                variant="default"
                icon="male"
              />
            </div>
            <div className="stat-card">
              <StatusChip 
                label="Women Finished" 
                value={topWomen.length > 0 ? `${topWomen.length}+` : 0} 
                variant="default"
                icon="female"
              />
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-12">
        {/* Category Champions */}
        <section className="mb-16">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-3 mb-4">
              <span className="material-icon text-4xl text-primary">emoji_events</span>
              <h2 className="text-3xl font-bold text-foreground">Category Champions</h2>
            </div>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Leading the pack in each category. These are the top performers setting the pace.
            </p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {renderCategoryCard("Men's Division", "male", topMen, 'individual')}
            {renderCategoryCard("Women's Division", "female", topWomen, 'individual')}
          </div>
        </section>

        {/* Team Competition */}
        {teamData.size > 0 && (
          <section className="mb-16">
            <div className="text-center mb-12">
              <div className="inline-flex items-center gap-3 mb-4">
                <span className="material-icon text-4xl text-primary">groups</span>
                <h2 className="text-3xl font-bold text-foreground">Team Competition</h2>
              </div>
              <p className="text-muted-foreground max-w-xl mx-auto">
                Team spirit and collaboration drive these remarkable performances.
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8">
              {Array.from(teamData.entries()).map(([teamName, runners]) => (
                <div key={teamName} className="modern-card p-6">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-r from-primary to-accent text-white">
                      <span className="material-icon text-xl">group</span>
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-foreground">{teamName}</h3>
                      <p className="text-sm text-muted-foreground">{runners.length} member{runners.length !== 1 ? 's' : ''}</p>
                    </div>
                  </div>
                  
                  {runners.length === 0 ? (
                    <div className="text-center py-8">
                      <span className="material-icon text-4xl text-muted-foreground/30 mb-2">group_off</span>
                      <p className="text-muted-foreground text-sm">No finishers yet</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {runners.slice(0, 3).map((runner, index) => (
                        <div key={runner.id} className="flex items-center justify-between p-3 bg-muted/30 rounded-xl hover:bg-primary/5 transition-all duration-200">
                          <div className="flex items-center gap-3">
                            <div className="flex items-center justify-center w-8 h-8 bg-primary/20 text-primary rounded-full text-sm font-bold">
                              {index + 1}
                            </div>
                            <div>
                              <div className="font-semibold text-foreground">{runner.racerName || 'N/A'}</div>
                              <div className="text-xs text-muted-foreground">Bib #{runner.bibNumber}</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-mono font-bold text-primary">
                              {formatTime(runner.finishTime)}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              Overall #{runner.rank || 'N/A'}
                            </div>
                          </div>
                        </div>
                      ))}
                      {runners.length > 3 && (
                        <div className="text-center text-sm text-muted-foreground pt-2 border-t border-border/30">
                          +{runners.length - 3} more team member{runners.length - 3 !== 1 ? 's' : ''}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Complete Results */}
        <section>
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-3 mb-4">
              <span className="material-icon text-4xl text-primary">leaderboard</span>
              <h2 className="text-3xl font-bold text-foreground">Complete Results</h2>
            </div>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Every runner's journey matters. Complete live rankings as racers cross the finish line.
            </p>
          </div>
          
          <div className="modern-card p-8">
            <DataTable
              columns={columns}
              data={finishers}
              renderRow={renderRow}
              emptyMessage="No finishers yet. Results will appear as runners cross the finish line."
            />
          </div>
        </section>

        {/* Footer */}
        <div className="mt-16 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-card border border-border">
            <span className="material-icon text-success text-sm animate-pulse">sync</span>
            <span className="text-sm text-muted-foreground">
              Results update automatically - no refresh required
            </span>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Index;
