import { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';

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
        <header className="sticky top-0 z-50 border-b border-border bg-card/95 backdrop-blur-md">
          <div className="container mx-auto px-6">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-8">
                <Link to="/" className="flex items-center gap-3 text-xl font-bold text-foreground">
                  <img 
                    src="/src/assets/ssri-logo.jpg" 
                    alt="Slay Sarcoma Research Initiative" 
                    className="h-10 w-auto rounded"
                  />
                  <span className="text-gradient">Slay Sarcoma Race Timing</span>
                </Link>
                <nav className="hidden md:flex items-center gap-6">
                  <Link 
                    to="/" 
                    className={`text-sm font-semibold transition-all duration-200 px-3 py-2 rounded-lg ${
                      location.pathname === '/' 
                        ? 'text-primary bg-primary/10 border border-primary/20' 
                        : 'text-muted-foreground hover:text-primary hover:bg-primary/5'
                    }`}
                  >
                    Live Leaderboard
                  </Link>
                  <Link 
                    to="/admin" 
                    className={`text-sm font-semibold transition-all duration-200 px-3 py-2 rounded-lg ${
                      isAdmin 
                        ? 'text-primary bg-primary/10 border border-primary/20' 
                        : 'text-muted-foreground hover:text-primary hover:bg-primary/5'
                    }`}
                  >
                    Admin Dashboard
                  </Link>
                </nav>
              </div>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-success/10 border border-success/20">
                  <span className="material-icon text-success text-sm pulse-live">radio_button_checked</span>
                  <span className="text-sm text-success font-semibold">LIVE</span>
                </div>
              </div>
            </div>
          </div>
        </header>
      )}
      <main className="relative">{children}</main>
    </div>
  );
}