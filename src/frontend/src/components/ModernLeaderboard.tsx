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
  // Team results auto-scroll refs
  const teamScrollViewportRef = useRef<HTMLDivElement | null>(null);
  const teamBodyRef = useRef<HTMLDivElement | null>(null);
  const teamScrollIntervalRef = useRef<number | null>(null);
  const teamTranslateYRef = useRef<number>(0);
  const teamOriginalBodyHeightRef = useRef<number>(0);

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

  const clearTeamAutoScroll = () => {
    if (teamScrollIntervalRef.current) {
      window.clearInterval(teamScrollIntervalRef.current);
      teamScrollIntervalRef.current = null;
    }
    const tbody = teamBodyRef.current;
    if (tbody) {
      Array.from(tbody.querySelectorAll('[data-clone="true"]') as NodeListOf<HTMLElement>).forEach(n => n.remove());
      (tbody as HTMLElement).style.transform = '';
    }
    teamTranslateYRef.current = 0;
    teamOriginalBodyHeightRef.current = 0;
  };

  const startScrollInterval = (tbody: HTMLTableSectionElement, origHeight: number) => {
    // ensure any previous interval is cleared
    if (scrollIntervalRef.current) {
      window.clearInterval(scrollIntervalRef.current);
      scrollIntervalRef.current = null;
    }

    const speedPxPerSec = 30; // pixels per second, tweakable
    const intervalMs = 16; // ~60fps
    const pxPerInterval = (speedPxPerSec * intervalMs) / 1000;

    // start interval that updates transform based on translateYRef
    scrollIntervalRef.current = window.setInterval(() => {
      translateYRef.current -= pxPerInterval;
      // reset when we've scrolled the height of the original content
      if (Math.abs(translateYRef.current) >= originalBodyHeightRef.current) {
        translateYRef.current += originalBodyHeightRef.current;
      }
      tbody.style.transform = `translateY(${translateYRef.current}px)`;
    }, intervalMs);
  };

  const startTeamScrollInterval = (body: HTMLDivElement, origHeight: number) => {
    if (teamScrollIntervalRef.current) {
      window.clearInterval(teamScrollIntervalRef.current);
      teamScrollIntervalRef.current = null;
    }

    const speedPxPerSec = 30; // same speed as main
    const intervalMs = 16;
    const pxPerInterval = (speedPxPerSec * intervalMs) / 1000;

    teamScrollIntervalRef.current = window.setInterval(() => {
      teamTranslateYRef.current -= pxPerInterval;
      if (Math.abs(teamTranslateYRef.current) >= teamOriginalBodyHeightRef.current) {
        teamTranslateYRef.current += teamOriginalBodyHeightRef.current;
      }
      body.style.transform = `translateY(${teamTranslateYRef.current}px)`;
    }, intervalMs);
  };

  const setupAutoScroll = () => {
    // full setup: clear everything then build clones and start scrolling
    clearAutoScroll();
    const viewport = scrollViewportRef.current;
    const tbody = tableBodyRef.current;
    if (!viewport || !tbody) return;

    // compute height of the rendered (non-cloned) rows
    const rows = Array.from(tbody.children).filter(c => !(c as HTMLElement).hasAttribute('data-clone')) as HTMLElement[];
    const originalHeight = rows.reduce((acc: number, row) => acc + row.offsetHeight, 0);
    const viewportHeight = viewport.clientHeight;
    if (originalHeight <= viewportHeight) return; // no autoscroll needed

    originalBodyHeightRef.current = originalHeight;

    // clone original rows to create seamless loop
    rows.forEach(r => {
      const clone = r.cloneNode(true) as HTMLElement;
      clone.setAttribute('data-clone', 'true');
      tbody.appendChild(clone);
    });

    // start from current translate (fresh setup starts at 0)
    translateYRef.current = 0;
    startScrollInterval(tbody, originalHeight);
  };

  // Called when finishers change but autoscroll is already running.
  // Refreshes clones and original height without resetting translateY or interval.
  const refreshAutoScrollPreservePosition = () => {
    const viewport = scrollViewportRef.current;
    const tbody = tableBodyRef.current;
    if (!viewport || !tbody) return;

    // remove any existing clones
    Array.from(tbody.querySelectorAll('[data-clone="true"]') as NodeListOf<HTMLElement>).forEach(n => n.remove());

    // compute height of the rendered (non-cloned) rows
    const rows = Array.from(tbody.children).filter(c => !(c as HTMLElement).hasAttribute('data-clone')) as HTMLElement[];
    const originalHeight = rows.reduce((acc: number, row) => acc + row.offsetHeight, 0);
    const viewportHeight = viewport.clientHeight;
    if (originalHeight <= viewportHeight) {
      // no autoscroll needed anymore
      clearAutoScroll();
      return;
    }

    originalBodyHeightRef.current = originalHeight;

    // append fresh clones
    rows.forEach(r => {
      const clone = r.cloneNode(true) as HTMLElement;
      clone.setAttribute('data-clone', 'true');
      tbody.appendChild(clone);
    });

    // ensure interval is running
    if (!scrollIntervalRef.current) {
      startScrollInterval(tbody, originalHeight);
    }
    // keep translateYRef.current as-is so it doesn't jump
  };

  const setupTeamAutoScroll = () => {
    clearTeamAutoScroll();
    const viewport = teamScrollViewportRef.current;
    const body = teamBodyRef.current;
    if (!viewport || !body) return;

    const rows = Array.from(body.children).filter(c => !(c as HTMLElement).hasAttribute('data-clone')) as HTMLElement[];
    const originalHeight = rows.reduce((acc: number, row) => acc + row.offsetHeight, 0);
    const viewportHeight = viewport.clientHeight;
    if (originalHeight <= viewportHeight) return;

    teamOriginalBodyHeightRef.current = originalHeight;

    rows.forEach(r => {
      const clone = r.cloneNode(true) as HTMLElement;
      clone.setAttribute('data-clone', 'true');
      body.appendChild(clone);
    });

    teamTranslateYRef.current = 0;
    startTeamScrollInterval(body, originalHeight);
  };

  const refreshTeamAutoScrollPreservePosition = () => {
    const viewport = teamScrollViewportRef.current;
    const body = teamBodyRef.current;
    if (!viewport || !body) return;

    Array.from(body.querySelectorAll('[data-clone="true"]') as NodeListOf<HTMLElement>).forEach(n => n.remove());

    const rows = Array.from(body.children).filter(c => !(c as HTMLElement).hasAttribute('data-clone')) as HTMLElement[];
    const originalHeight = rows.reduce((acc: number, row) => acc + row.offsetHeight, 0);
    const viewportHeight = viewport.clientHeight;
    if (originalHeight <= viewportHeight) {
      clearTeamAutoScroll();
      return;
    }

    teamOriginalBodyHeightRef.current = originalHeight;

    rows.forEach(r => {
      const clone = r.cloneNode(true) as HTMLElement;
      clone.setAttribute('data-clone', 'true');
      body.appendChild(clone);
    });

    if (!teamScrollIntervalRef.current) {
      startTeamScrollInterval(body, originalHeight);
    }
  };

  // Recreate/reset autoscroll whenever finishers change (pause and include new row)
  useEffect(() => {
    // brief delay to allow DOM update
    const t = setTimeout(() => {
      // If autoscroll is already running, refresh clones and sizes without
      // resetting the translate/position. Otherwise do a full setup.
      if (scrollIntervalRef.current) {
        refreshAutoScrollPreservePosition();
      } else {
        setupAutoScroll();
      }
    }, 120);
    return () => {
      clearTimeout(t);
    };
  }, [finishers.length]);

  // Team auto-scroll: start when totalFinishers >= 3 (user requested) or when teamData changes
  useEffect(() => {
    const t = setTimeout(() => {
      // If autoscroll already running for teams, refresh, otherwise setup if condition met
      if (teamScrollIntervalRef.current) {
        refreshTeamAutoScrollPreservePosition();
      } else {
        // Start team autoscroll only when at least 3 finishers have been recorded
        if ((totalFinishers || 0) >= 3) {
          setupTeamAutoScroll();
        }
      }
    }, 120);

    return () => clearTimeout(t);
  }, [totalFinishers, teamData.size]);

  const getPodiumClass = (position: number) => {
    switch (position) {
      case 0: return 'podium-gold';
      case 1: return 'podium-silver';
      case 2: return 'podium-bronze';
      default: return ''; // No special styling for 4th, 5th, etc.
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
              {index < 3 && (
                <Medal className={`w-5 h-5 ml-2 ${
                  index === 0 ? 'medal-gold' : index === 1 ? 'medal-silver' : 'medal-bronze'
                }`} />
              )}
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
                  <th className="w-28">Rank</th>
                  <th className="w-40">Bib</th>
                  <th className="min-w-0 flex-1">Runner Name</th>
                  <th className="w-44">Finish Time</th>
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
                <span style={{ color: '#4F3F52' }}>
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
                  <h2 className="section-title-tv">Age Brackets</h2>
                </div>
                
                <div ref={teamScrollViewportRef} className="p-4 space-y-4 max-h-96 overflow-y-auto">
                  <div ref={teamBodyRef}>
                    {Array.from(teamData.entries()).slice(0, 6).map(([teamName, runners]) => (
                      <div key={teamName} className="category-card-tv">
                        <div className="category-header-tv">
                          <Users className="w-4 h-4 text-primary" />
                          <h3 className="category-title-tv">{teamName}</h3>
                        </div>
                        
                        <div className="space-y-2">
                          {runners.length === 0 ? (
                            <div className="text-center py-4">
                              <Timer className="w-5 h-5 text-muted-foreground mx-auto mb-2" />
                              <p className="text-muted-foreground text-xs">No finishers yet...</p>
                            </div>
                          ) : (
                            runners.slice(0, 3).map((runner, index) => (
                                <div key={runner.id} className="podium-item-tv group">
                                  <div className={`podium-position-tv ${getPodiumClass(index)}`}>
                                    {index + 1}
                                  </div>
                                  {index < 3 && (
                                    <Medal className={`w-5 h-5 ml-2 ${
                                      index === 0 ? 'medal-gold' : index === 1 ? 'medal-silver' : 'medal-bronze'
                                    }`} />
                                  )}

                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between">
                                      <div>
                                        <p className="podium-name-tv group-hover:text-primary transition-colors">
                                          {runner.racerName || 'Unknown Runner'}
                                        </p>
                                        <p className="podium-bib-tv">Bib #{runner.bibNumber}</p>
                                      </div>
                                    
                                      <div className="text-right">
                                        <p className="podium-time-tv">
                                          {formatTime(normalizeFinishTime(runner.finishTime))}
                                        </p>
                                        <p className="text-xs text-muted-foreground">
                                          Overall #{runner.rank}
                                        </p>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              ))
                          )}
                          
                          {runners.length > 3 && (
                            <div className="text-center text-xs text-muted-foreground pt-2 border-t border-border/30">
                              +{runners.length - 3} more team members
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
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
