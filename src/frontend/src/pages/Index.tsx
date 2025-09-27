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
