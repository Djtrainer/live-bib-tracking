import { ReactNode } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import RaceClock from '@/components/RaceClock';
import { LogOut } from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
  showNavigation?: boolean;
  totalFinishers?: number;
  lastUpdated?: Date;
  activeTab?: 'setup' | 'live';
  onTabChange?: (tab: 'setup' | 'live') => void;
}


export default function Layout({ children, showNavigation = true, totalFinishers = 0, lastUpdated, activeTab, onTabChange }: LayoutProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const isAdmin = location.pathname.startsWith('/admin');

  const handleLogout = () => {
    navigate('/admin/login');
  };

  return (
    <div className="min-h-screen bg-background">
      {showNavigation && (
        <header className="hero-section">
          <div className="max-w-[1800px] mx-auto px-6">
            <div className="flex items-center justify-between h-20">
              <div className="flex items-center gap-8">
                <Link to="/" className="flex items-center gap-4 group">
                  <img 
                    src="/ssri-logo.jpg" 
                    alt="Slay Sarcoma Research Initiative" 
                    onError={(e) => { (e.currentTarget as HTMLImageElement).src = '/placeholder.svg'; }}
                    className="h-12 w-12 rounded-lg object-cover shadow-sm group-hover:shadow-md transition-all duration-200"
                  />
                  <div className="flex flex-col">
                    <span className="text-xl font-bold text-foreground group-hover:text-primary transition-colors">
                      Slay Sarcoma
                    </span>
                    <span className="text-sm text-muted-foreground">
                      Race Timing System
                    </span>
                  </div>
                </Link>
                <nav className="flex items-center gap-1 ml-8">
                  <Link 
                    to="/" 
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                      location.pathname === '/' 
                        ? 'bg-primary text-primary-foreground shadow-sm' 
                        : 'text-muted-foreground hover:text-primary hover:bg-primary/5'
                    }`}
                  >
                    <span className="material-icon text-lg mr-2">leaderboard</span>
                    Live Leaderboard
                  </Link>
                  <Link 
                    to="/admin" 
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                      isAdmin 
                        ? 'bg-primary text-primary-foreground shadow-sm' 
                        : 'text-muted-foreground hover:text-primary hover:bg-primary/5'
                    }`}
                  >
                    <span className="material-icon text-lg mr-2">admin_panel_settings</span>
                    Admin Dashboard
                  </Link>
                  
                  {/* Admin Sub-navigation */}
                  {isAdmin && activeTab && onTabChange && (
                    <>
                      <div className="w-px h-6 bg-border mx-2"></div>
                      <button
                        onClick={() => onTabChange('setup')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                          activeTab === 'setup'
                            ? 'bg-secondary text-secondary-foreground shadow-sm'
                            : 'text-muted-foreground hover:text-foreground hover:bg-secondary/20'
                        }`}
                      >
                        <span className="material-icon text-lg mr-2">settings</span>
                        Race Setup
                      </button>
                      <button
                        onClick={() => onTabChange('live')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                          activeTab === 'live'
                            ? 'bg-secondary text-secondary-foreground shadow-sm'
                            : 'text-muted-foreground hover:text-foreground hover:bg-secondary/20'
                        }`}
                      >
                        <span className="material-icon text-lg mr-2">monitor</span>
                        Live Management
                      </button>
                    </>
                  )}
                </nav>
              </div>
              
              <div className="flex items-center gap-6">
                {/* Stats moved to top panel */}
                <div className="flex items-center gap-6">
                  <div className="flex items-center gap-2 text-right">
                    <span className="material-icon text-primary text-lg">flag</span>
                    <div>
                      <div className="text-lg font-bold text-foreground">{totalFinishers}</div>
                      <div className="text-xs text-muted-foreground uppercase tracking-wide">Finishers</div>
                    </div>
                  </div>
                  
                  {lastUpdated && (
                    <div className="flex items-center gap-2 text-right">
                      <span className="material-icon text-success text-lg">update</span>
                      <div>
                        <div className="text-sm font-semibold text-foreground">{lastUpdated.toLocaleTimeString()}</div>
                        <div className="text-xs text-muted-foreground uppercase tracking-wide">Updated</div>
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Live Status Only */}
                <div className="flex items-center gap-3 px-4 py-2 rounded-full bg-success/10 border border-success/20">
                  <span className="w-2 h-2 bg-success rounded-full animate-pulse"></span>
                  <span className="text-sm font-semibold text-success">LIVE</span>
                </div>
                
                {/* Sign Out Button - Only show on admin pages */}
                {isAdmin && (
                  <button
                    onClick={handleLogout}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-destructive hover:text-destructive hover:bg-destructive/10 transition-all duration-200"
                    title="Sign Out"
                  >
                    <LogOut className="w-4 h-4" />
                    Sign Out
                  </button>
                )}
              </div>
            </div>
          </div>
        </header>
      )}
      <main className="bg-background">{children}</main>
    </div>
  );
}
