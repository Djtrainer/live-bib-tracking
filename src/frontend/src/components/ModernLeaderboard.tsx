import { useState, useEffect, useRef } from 'react';
import { 
  Trophy, 
  Users, 
  Timer, 
  Activity, 
  Medal, 
  Target, 
  TrendingUp,
  Clock,
  Flag,
  RefreshCw
} from 'lucide-react';
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

interface ModernLeaderboardProps {
  finishers: Finisher[];
  totalFinishers: number;
  lastUpdated: Date;
  topMen: Finisher[];
  topWomen: Finisher[];
  teamData: Map<string, Finisher[]>;
}

const ModernLeaderboard = ({
  finishers,
  totalFinishers,
  lastUpdated,
  topMen,
  topWomen,
  teamData
}: ModernLeaderboardProps) => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [clockStatus, setClockStatus] = useState<any>(null);
  const scrollViewportRef = useRef<HTMLDivElement | null>(null);
  const tableBodyRef = useRef<HTMLTableSectionElement | null>(null);
  const scrollIntervalRef = useRef<number | null>(null);
  const translateYRef = useRef<number>(0);
  const originalBodyHeightRef = useRef<number>(0);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    fetch('/api/clock/status')
      .then(r => r.json())
      .then(res => {
        if (res.success && res.data) setClockStatus(res.data);
      })
      .catch(() => {});
  }, []);

  // Auto-scroll setup/teardown and helpers
  const clearAutoScroll = () => {
    if (scrollIntervalRef.current) {
      window.clearInterval(scrollIntervalRef.current);
      scrollIntervalRef.current = null;
    }
    const tbody = tableBodyRef.current;
    if (tbody) {
      // remove cloned rows
    Array.from(tbody.querySelectorAll('[data-clone="true"]') as NodeListOf<HTMLElement>).forEach(n => n.remove());
      tbody.style.transform = '';
    }
    translateYRef.current = 0;
    originalBodyHeightRef.current = 0;
  };

  const setupAutoScroll = () => {
    clearAutoScroll();
    const viewport = scrollViewportRef.current;
    const tbody = tableBodyRef.current;
    if (!viewport || !tbody) return;

  const originalHeight = Array.from(tbody.children).reduce((acc: number, row) => acc + (row as HTMLElement).offsetHeight, 0);
    const viewportHeight = viewport.clientHeight;
    if (originalHeight <= viewportHeight) return; // no autoscroll needed

    originalBodyHeightRef.current = originalHeight;

    // clone original rows to create seamless loop
    const originalRows = Array.from(tbody.children).map(r => (r as HTMLElement).cloneNode(true) as HTMLElement);
    originalRows.forEach(r => {
      r.setAttribute('data-clone', 'true');
      tbody.appendChild(r);
    });

    translateYRef.current = 0;
    const speedPxPerSec = 30; // pixels per second, tweakable
    const intervalMs = 16; // ~60fps
    const pxPerInterval = (speedPxPerSec * intervalMs) / 1000;

    scrollIntervalRef.current = window.setInterval(() => {
      translateYRef.current -= pxPerInterval;
      // reset when we've scrolled the height of the original content
      if (Math.abs(translateYRef.current) >= originalBodyHeightRef.current) {
        translateYRef.current += originalBodyHeightRef.current;
      }
      tbody.style.transform = `translateY(${translateYRef.current}px)`;
    }, intervalMs);
  };

  // Recreate/reset autoscroll whenever finishers change (pause and include new row)
  useEffect(() => {
    // brief delay to allow DOM update
    clearAutoScroll();
    const t = setTimeout(() => setupAutoScroll(), 120);
    return () => {
      clearTimeout(t);
      clearAutoScroll();
    };
  }, [finishers.length]);

  const getPodiumClass = (position: number) => {
    switch (position) {
      case 0: return 'podium-gold';
      case 1: return 'podium-silver';
      case 2: return 'podium-bronze';
      default: return 'bg-primary text-primary-foreground';
    }
  };

  const normalizeFinishTime = (ft: number | null | undefined) => {
    if (ft === null || ft === undefined) return 0;
    if (ft > 1e12 && clockStatus) {
      const { raceStartTime, offset } = clockStatus;
      if (raceStartTime) {
        const elapsed = Math.round(ft - (raceStartTime * 1000) - (offset || 0));
        return elapsed > 0 ? elapsed : 0;
      }
    }
    return ft as number;
  };

  const renderPodiumCard = (title: string, icon: React.ReactNode, data: Finisher[]) => (
    <div className="category-card-tv">
      <div className="category-header-tv">
        {icon}
        <h3 className="category-title-tv">{title}</h3>
      </div>
      
      <div className="space-y-2">
        {data.length === 0 ? (
          <div className="text-center py-6">
            <Timer className="w-6 h-6 text-muted-foreground mx-auto mb-2" />
            <p className="text-muted-foreground text-sm">Awaiting finishers...</p>
          </div>
        ) : (
          data.map((finisher, index) => (
            <div key={finisher.id} className="podium-item-tv group">
              <div className={`podium-position-tv ${getPodiumClass(index)}`}>
                {index + 1}
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="podium-name-tv group-hover:text-primary transition-colors">
                      {finisher.racerName || 'Unknown Runner'}
                    </p>
                    <p className="podium-bib-tv">Bib #{finisher.bibNumber}</p>
                  </div>
                  
                  <div className="text-right">
                    <p className="podium-time-tv">
                      {formatTime(normalizeFinishTime(finisher.finishTime))}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Overall #{finisher.rank}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  const renderResultsTable = () => (
    <div className="section-card-tv">
      <div className="section-header-tv">
        <div className="section-icon-tv">
          <Activity className="w-5 h-5 text-white" />
        </div>
        <div className="flex-1 flex items-center justify-between">
          <h2 className="section-title-tv">Live Race Results</h2>
          <div className="live-indicator-tv">
            <div className="pulse-dot"></div>
            <span className="live-text-tv">LIVE</span>
          </div>
        </div>
      </div>
      
      <div className="overflow-hidden">
        {finishers.length === 0 ? (
          <div className="text-center py-12">
            <Target className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground text-lg">
              No finishers yet. Results will appear as runners cross the finish line.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <div ref={scrollViewportRef} className="results-scroll-viewport">
              <table className="results-table-tv">
                <thead>
                <tr>
                  <th className="w-24">Rank</th>
                  <th className="w-24">Bib</th>
                  <th>Runner Name</th>
                  <th className="w-40">Finish Time</th>
                </tr>
              </thead>
                <tbody ref={tableBodyRef}>
                {finishers.map((finisher, index) => (
                  <tr key={finisher.id || finisher.bibNumber} className="group">
                    <td>
                      <div className="flex items-center gap-2">
                        <span className="rank-cell-tv">
                          {index + 1}
                        </span>
                        {index < 3 && (
                          <Medal className={`w-5 h-5 ${
                            index === 0 ? 'medal-gold' :
                            index === 1 ? 'medal-silver' :
                            'medal-bronze'
                          }`} />
                        )}
                      </div>
                    </td>
                    <td>
                      <span className="bib-cell-tv">
                        #{finisher.bibNumber}
                      </span>
                    </td>
                    <td>
                      <span className="name-cell-tv group-hover:text-primary transition-colors">
                        {finisher.racerName || 'Unknown Runner'}
                      </span>
                    </td>
                    <td>
                      <span className="time-cell-tv">
                          {formatTime(normalizeFinishTime(finisher.finishTime))}
                        </span>
                    </td>
                  </tr>
                ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="leaderboard-container">
      <div className="max-w-[1800px] mx-auto">
        
        {/* TV Header - Left title, Right stats */}
        <div className="hero-section-tv">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <h1 className="text-4xl md:text-5xl font-black mb-2 text-left">
                <span className="bg-gradient-to-r from-foreground via-primary to-secondary bg-clip-text text-transparent">
                  2025 Slay Sarcoma Race Results
                </span>
              </h1>
              <p className="text-left text-muted-foreground text-lg">
                Live race results with real-time updates
              </p>
            </div>
            
            {/* Right-aligned stats */}
            <div className="flex flex-row gap-4">
              <div className="stat-card-tv min-w-[140px]">
                <Flag className="w-5 h-5 text-primary mb-2 mx-auto" />
                <div className="race-clock-time-tv text-center font-bold text-2xl">
                  {totalFinishers}
                </div>
                <div className="stat-label-tv">Total Finishers</div>
              </div>
              
              <div className="stat-card-tv min-w-[140px]">
                <RefreshCw className="w-5 h-5 text-success mb-2 mx-auto" />
                <div className="race-clock-time-tv text-center font-bold text-lg">
                  {lastUpdated.toLocaleTimeString()}
                </div>
                <div className="stat-label-tv">Updated</div>
              </div>
              
              <div className="stat-card-tv min-w-[140px]">
                <Clock className="w-5 h-5 text-primary mb-2 mx-auto" />
                <div className="race-clock-time-tv text-center">
                  <RaceClock showControls={false} />
                </div>
                <div className="stat-label-tv">Race Time</div>
              </div>
            </div>
          </div>
        </div>

        {/* TV Dashboard Grid - Main content and sidebar */}
        <div className="tv-dashboard-grid">
          
          {/* Main Results Table */}
          <div className="tv-main-content">
            {renderResultsTable()}
          </div>

          {/* Sidebar with Category Leaders and Teams */}
          <div className="tv-sidebar">
            
            {/* Category Leaders */}
            <div className="section-card-tv">
              <div className="section-header-tv">
                <div className="section-icon-tv">
                  <Trophy className="w-5 h-5 text-white" />
                </div>
                <h2 className="section-title-tv">Category Leaders</h2>
              </div>
              
              <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                {renderPodiumCard(
                  "Top 3 Men", 
                  <Medal className="w-5 h-5 text-blue-500" />, 
                  topMen
                )}
                {renderPodiumCard(
                  "Top 3 Women", 
                  <Medal className="w-5 h-5 text-pink-500" />, 
                  topWomen
                )}
              </div>
            </div>

            {/* Team Results */}
            {teamData.size > 0 && (
              <div className="section-card-tv">
                <div className="section-header-tv">
                  <div className="section-icon-tv">
                    <Users className="w-5 h-5 text-white" />
                  </div>
                  <h2 className="section-title-tv">Team Results</h2>
                </div>
                
                <div className="p-4 space-y-3 max-h-96 overflow-y-auto">
                  {Array.from(teamData.entries()).slice(0, 6).map(([teamName, runners]) => (
                    <div key={teamName} className="team-card-tv">
                      <div className="team-header-tv">
                        <Users className="w-4 h-4 text-primary" />
                        <h3 className="team-name-tv">{teamName}</h3>
                      </div>
                      
                      <div className="space-y-1">
                        {runners.slice(0, 2).map((runner, index) => (
                          <div key={runner.id} className="flex items-center justify-between text-sm">
                            <div className="flex items-center gap-2">
                              <div className={`podium-position-tv ${getPodiumClass(index)}`}>
                                {index + 1}
                              </div>
                              <div>
                                <span className="font-semibold">{runner.racerName || 'Unknown'}</span>
                                <span className="text-xs text-muted-foreground ml-2">#{runner.bibNumber}</span>
                              </div>
                            </div>
                            <span className="font-mono font-bold text-primary">
                              {formatTime(runner.finishTime)}
                            </span>
                          </div>
                        ))}
                        
                        {runners.length > 2 && (
                          <div className="text-center text-xs text-muted-foreground pt-1 border-t border-border/30">
                            +{runners.length - 2} more
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        </div>

      </div>
    </div>
  );
};

export default ModernLeaderboard;