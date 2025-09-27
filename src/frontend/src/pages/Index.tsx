import { useState, useEffect } from 'react';
import Layout from '@/components/Layout';
import ModernLeaderboard from '@/components/ModernLeaderboard';
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
  const [clockStatus, setClockStatus] = useState<any>(null);

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
          // Normalize finish times if some are stored as epoch timestamps
          const normalizeFinishTime = (ft: number | null) => {
            if (ft === null || ft === undefined) return null;
            if (ft > 1e12 && clockStatus) {
              const { raceStartTime, offset } = clockStatus;
              if (raceStartTime) {
                const elapsed = Math.round(ft - (raceStartTime * 1000) - (offset || 0));
                return elapsed > 0 ? elapsed : 0;
              }
            }
            return ft;
          };

          const normalized = result.data.map((f: any) => ({ ...f, finishTime: normalizeFinishTime(f.finishTime) ?? f.finishTime }));
          
          // CRITICAL FIX: Filter out duplicate bib numbers, keeping only the FIRST occurrence
          const uniqueFinishers: Finisher[] = [];
          const seenBibNumbers = new Set<string>();
          
          // Sort by finish time first to ensure we keep the fastest time for each bib
          const sortedNormalized = [...normalized].sort((a, b) => (a.finishTime || 0) - (b.finishTime || 0));
          
          for (const finisher of sortedNormalized) {
            if (!seenBibNumbers.has(finisher.bibNumber)) {
              seenBibNumbers.add(finisher.bibNumber);
              uniqueFinishers.push(finisher);
            } else {
              console.log(`Duplicate bib ${finisher.bibNumber} found in initial data - keeping first occurrence for Live Leaderboard`);
            }
          }
          
          setFinishers(uniqueFinishers);
          setTotalFinishers(uniqueFinishers.length);
          setLastUpdated(new Date());
          calculateCategoryLeaderboards(uniqueFinishers);
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
      // fetch clock status too
      fetch('/api/clock/status')
        .then(r => r.json())
        .then(res => {
          if (res.success && res.data) setClockStatus(res.data);
        })
        .catch(() => {});
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
            // Apply same duplicate filtering logic as initial fetch
            const uniqueFinishers: Finisher[] = [];
            const seenBibNumbers = new Set<string>();
            
            // Sort by finish time first to ensure we keep the fastest time for each bib
            const sortedData = [...result.data].sort((a, b) => (a.finishTime || 0) - (b.finishTime || 0));
            
            for (const finisher of sortedData) {
              if (!seenBibNumbers.has(finisher.bibNumber)) {
                seenBibNumbers.add(finisher.bibNumber);
                uniqueFinishers.push(finisher);
              } else {
                console.log(`Duplicate bib ${finisher.bibNumber} found in reload data - keeping first occurrence for Live Leaderboard`);
              }
            }
            
            setFinishers(uniqueFinishers);
            setTotalFinishers(uniqueFinishers.length);
            setLastUpdated(new Date());
            calculateCategoryLeaderboards(uniqueFinishers);
          }
        } else if (message.type === 'add' || message.type === 'update') {
          setFinishers(prev => {
            const normalizeFinishTime = (ft: number | null) => {
              if (ft === null || ft === undefined) return null;
              if (ft > 1e12 && clockStatus) {
                const { raceStartTime, offset } = clockStatus;
                if (raceStartTime) {
                  const elapsed = Math.round(ft - (raceStartTime * 1000) - (offset || 0));
                  return elapsed > 0 ? elapsed : 0;
                }
              }
              return ft;
            };

            const incoming = { ...message.data, finishTime: normalizeFinishTime(message.data.finishTime) ?? message.data.finishTime };

            // CRITICAL FIX: Only match by ID to allow duplicate bib numbers
            // For Live Leaderboard, we want to show only the FIRST occurrence of each bib number
            const existingIndex = prev.findIndex(f => f.id === incoming.id);
            let updated;
            if (existingIndex > -1) {
              // Update existing entry by ID
              updated = [...prev];
              updated[existingIndex] = { ...updated[existingIndex], ...incoming };
            } else {
              // Check if this bib number already exists (for duplicate handling)
              const duplicateBibIndex = prev.findIndex(f => f.bibNumber === incoming.bibNumber);
              if (duplicateBibIndex > -1) {
                // Bib number already exists - keep the FIRST one (ignore the new one for Live Leaderboard)
                console.log(`Duplicate bib ${incoming.bibNumber} detected - keeping first occurrence for Live Leaderboard`);
                updated = [...prev]; // Don't add the duplicate
              } else {
                // New unique bib number - add it
                updated = [...prev, incoming];
              }
            }
            // --- THIS IS THE CORRECT NUMERICAL SORT ---
            updated.sort((a, b) => (a.finishTime || 0) - (b.finishTime || 0));
            calculateCategoryLeaderboards(updated);
            
            // CRITICAL FIX: Update total finishers count when finishers array changes
            setTotalFinishers(updated.length);
            
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

  return (
    <Layout totalFinishers={totalFinishers} lastUpdated={lastUpdated}>
      <ModernLeaderboard
        finishers={finishers}
        totalFinishers={totalFinishers}
        lastUpdated={lastUpdated}
        topMen={topMen}
        topWomen={topWomen}
        teamData={teamData}
      />
    </Layout>
  );
};

export default Index;
