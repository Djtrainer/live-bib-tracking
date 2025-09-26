import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import ssriLogo from '../assets/ssri-logo.jpg';

interface LayoutProps {
  children: ReactNode;
  showNavigation?: boolean;
}

export default function Layout({ children, showNavigation = true }: LayoutProps) {
  const location = useLocation();
  const isAdmin = location.pathname.startsWith('/admin');

  return (
    <div className="min-h-screen bg-background">
      {showNavigation && (
        <header className="professional-header sticky top-0 z-50">
          <div className="container mx-auto px-6">
            <div className="flex items-center justify-between h-20">
              <div className="flex items-center gap-8">
                <Link to="/" className="flex items-center gap-4 group">
                  <div className="flex items-center justify-center w-12 h-12 rounded-xl overflow-hidden bg-white/10 transition-all duration-300 group-hover:scale-105">
                    <img 
                      src={ssriLogo} 
                      alt="Slay Sarcoma Research Initiative" 
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xl font-bold text-foreground leading-tight">
                      Slay Sarcoma
                    </span>
                    <span className="text-sm text-primary font-medium">
                      Race Timing System
                    </span>
                  </div>
                </Link>
                
                <nav className="hidden md:flex items-center gap-2">
                  <Link 
                    to="/" 
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                      location.pathname === '/' 
                        ? 'bg-primary text-primary-foreground shadow-md' 
                        : 'text-muted-foreground hover:text-primary hover:bg-primary/10'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="material-icon text-lg">leaderboard</span>
                      Live Leaderboard
                    </div>
                  </Link>
                  <Link 
                    to="/admin" 
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                      isAdmin 
                        ? 'bg-primary text-primary-foreground shadow-md' 
                        : 'text-muted-foreground hover:text-primary hover:bg-primary/10'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="material-icon text-lg">admin_panel_settings</span>
                      Admin Dashboard
                    </div>
                  </Link>
                </nav>
              </div>
              
              <div className="flex items-center gap-3">
                <div className="status-chip live">
                  <span className="material-icon text-sm">radio_button_checked</span>
                  <span className="font-semibold">Live</span>
                </div>
                <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-success/10 border border-success/20">
                  <div className="w-2 h-2 bg-success rounded-full animate-pulse"></div>
                  <span className="text-xs font-medium text-success">System Online</span>
                </div>
              </div>
            </div>
          </div>
        </header>
      )}
      <main>{children}</main>
    </div>
  );
}