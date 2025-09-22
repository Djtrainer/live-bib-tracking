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
        <header className="border-b border-border bg-card">
          <div className="container mx-auto px-4">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-8">
                <Link to="/" className="text-xl font-bold text-foreground">
                  Race Timing Pro
                </Link>
                <nav className="flex items-center gap-6">
                  <Link 
                    to="/" 
                    className={`text-sm font-medium transition-colors hover:text-primary ${
                      location.pathname === '/' ? 'text-primary' : 'text-muted-foreground'
                    }`}
                  >
                    Live Leaderboard
                  </Link>
                  <Link 
                    to="/admin" 
                    className={`text-sm font-medium transition-colors hover:text-primary ${
                      isAdmin ? 'text-primary' : 'text-muted-foreground'
                    }`}
                  >
                    Admin Dashboard
                  </Link>
                </nav>
              </div>
              <div className="flex items-center gap-2">
                <span className="material-icon text-success">radio_button_checked</span>
                <span className="text-sm text-success font-medium">Live</span>
              </div>
            </div>
          </div>
        </header>
      )}
      <main>{children}</main>
    </div>
  );
}