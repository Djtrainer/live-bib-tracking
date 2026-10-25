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

const AUTO_SCROLL_PX_PER_SEC = 30;
const AUTO_SCROLL_HOLD_MS = 3000;
const AUTO_SCROLL_RESUME_MS = 10000;

// Drives a real scroll container: creep down at a steady pace, hold at the
// bottom, jump back to the top, hold, repeat. Nothing moves while the content
// fits. Any interaction hands the viewport to the viewer, and the animation
// picks up from wherever they left it once they have been quiet for a while.
// Returns the teardown.
function attachAutoScroll(el: HTMLElement): () => void {
  let rafId = 0;
  let lastTs: number | null = null;
  let waitUntil = 0;
  let jumpToTop = false;
  // scrollTop rounds to whole pixels in some browsers, so the sub-pixel
  // progress lives here and the element is only ever written from it.
  let pos = el.scrollTop;
  // Until the first write, a moved element is the browser restoring a scroll
  // position at load, not the viewer; the creep simply starts from there.
  let started = false;

  const write = (top: number) => {
    started = true;
    el.scrollTop = top;
  };

  const step = (ts: number) => {
    rafId = requestAnimationFrame(step);
    // Capped so a backgrounded tab does not leap when it comes back.
    const dt = lastTs === null ? 0 : Math.min(ts - lastTs, 100);
    lastTs = ts;
    if (ts < waitUntil) return;

    const maxTop = el.scrollHeight - el.clientHeight;
    if (maxTop <= 0) return;

    if (jumpToTop) {
      jumpToTop = false;
      pos = 0;
      write(0);
      waitUntil = ts + AUTO_SCROLL_HOLD_MS;
      return;
    }

    // Someone else moved it (the viewer, or a layout change): carry on from there.
    if (Math.abs(el.scrollTop - pos) > 1) pos = el.scrollTop;
    pos = Math.min(pos + (dt * AUTO_SCROLL_PX_PER_SEC) / 1000, maxTop);
    write(pos);
    if (pos >= maxTop) {
      jumpToTop = true;
      waitUntil = ts + AUTO_SCROLL_HOLD_MS;
    }
  };

  const onInteract = () => {
    jumpToTop = false;
    waitUntil = performance.now() + AUTO_SCROLL_RESUME_MS;
  };
  // A scroll event is the viewer's only when the element is somewhere other
  // than where the animation put it. Our own writes land within a pixel of
  // pos, and so does the odd event the browser fires at load or after a layout
  // change without moving anything; neither may pause the creep.
  const onScroll = () => {
    if (!started) { pos = el.scrollTop; return; }
    if (Math.abs(el.scrollTop - pos) <= 1) return;
    onInteract();
  };

  const interactions = ['wheel', 'touchstart', 'touchmove', 'pointerdown', 'keydown'] as const;
  interactions.forEach(type => el.addEventListener(type, onInteract, { passive: true }));
  el.addEventListener('scroll', onScroll, { passive: true });
  rafId = requestAnimationFrame(step);

  return () => {
    cancelAnimationFrame(rafId);
    interactions.forEach(type => el.removeEventListener(type, onInteract));
    el.removeEventListener('scroll', onScroll);
  };
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
  const teamScrollViewportRef = useRef<HTMLDivElement | null>(null);

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

  // The viewports only render once there is something to show, so the
  // auto-scroll attaches when one appears and detaches when it goes. Finisher
  // updates in between leave it, and the native scroll position, alone.
  const hasResults = finishers.length > 0;
  useEffect(() => {
    const el = scrollViewportRef.current;
    if (!el || !hasResults) return;
    return attachAutoScroll(el);
  }, [hasResults]);

  // The team list waits for the third finisher before it starts moving.
  const teamsScroll = teamData.size > 0 && (totalFinishers || 0) >= 3;
  useEffect(() => {
    const el = teamScrollViewportRef.current;
    if (!el || !teamsScroll) return;
    return attachAutoScroll(el);
  }, [teamsScroll]);

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
            <div ref={scrollViewportRef} className="results-scroll-viewport scroll-viewport-tv" tabIndex={0}>
              <table className="results-table-tv">
                <thead>
                <tr>
                  <th className="w-28">Rank</th>
                  <th className="w-40">Bib</th>
                  <th className="min-w-0 flex-1">Runner Name</th>
                  <th className="w-44">Finish Time</th>
                </tr>
              </thead>
                <tbody>
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
                  2026 Slay Sarcoma Race Results
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
                  <h2 className="section-title-tv">Team Name</h2>
                </div>
                
                <div ref={teamScrollViewportRef} className="scroll-viewport-tv p-4 space-y-4 max-h-96" tabIndex={0}>
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
            )}

          </div>
        </div>

      </div>
    </div>
  );
};

export default ModernLeaderboard;
