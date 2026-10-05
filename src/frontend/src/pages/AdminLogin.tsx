import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Shield, 
  LogIn, 
  RefreshCw, 
  Eye, 
  EyeOff,
  User,
  Lock
} from 'lucide-react';
import Layout from '@/components/Layout';
import { adminLogin, getAdminSession } from '@/lib/adminApi';
import type { AdminSession } from '@/lib/adminApi';

export default function AdminLogin() {
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  // Until the server answers, assume a login is required: showing "no
  // password set" for a moment on a locked server would be the wrong default.
  const [session, setSession] = useState<AdminSession>({ configured: true, authenticated: false });

  useEffect(() => {
    let cancelled = false;
    if (new URLSearchParams(window.location.search).get('expired')) {
      setError('Your session ended. Please sign in again.');
    }
    getAdminSession().then((s) => {
      if (cancelled) return;
      setSession(s);
      // Already signed in on this browser: straight to Live Management.
      if (s.configured && s.authenticated && !s.unknown) navigate('/admin/dashboard', { replace: true });
    });
    return () => { cancelled = true; };
  }, [navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    // Basic input validation
    if (!credentials.username.trim()) {
      setError('Username is required');
      setIsLoading(false);
      return;
    }
    
    if (!credentials.password.trim()) {
      setError('Password is required');
      setIsLoading(false);
      return;
    }

    // The API checks the credentials; this page never knows them.
    const result = await adminLogin(credentials.username.trim(), credentials.password);
    if (result.ok) {
      navigate('/admin/dashboard');
    } else {
      setError(result.message || 'Sign-in failed.');
    }
    setIsLoading(false);
  };

  return (
    <Layout showNavigation={false}>
      <div className="leaderboard-container">
        <div className="min-h-screen flex items-center justify-center px-6">
          <div className="w-full max-w-md">
            
            {/* Login Card - TV Optimized */}
            <div className="section-card-tv">
              <div className="section-header-tv">
                <div className="section-icon-tv">
                  <Shield className="w-5 h-5 text-white" />
                </div>
                <h1 className="section-title-tv">Admin Access</h1>
              </div>

              <div className="p-8">
                <p className="text-muted-foreground text-center text-lg mb-8">
                  Sign in to access the race timing dashboard
                </p>

                {error && (
                  <div className="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                    <p className="text-destructive text-sm font-medium">{error}</p>
                  </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* Username Field */}
                  <div>
                    <label htmlFor="username" className="block text-sm font-semibold text-foreground mb-3">
                      Username
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <User className="w-5 h-5 text-muted-foreground" />
                      </div>
                      <input
                        id="username"
                        type="text"
                        autoCapitalize="none"
                        autoCorrect="off"
                        spellCheck={false}
                        autoComplete="username"
                        value={credentials.username}
                        onChange={(e) => setCredentials(prev => ({ ...prev, username: e.target.value }))}
                        className="w-full pl-10 pr-4 py-3 rounded-lg border border-border bg-background text-foreground placeholder:text-muted-foreground focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                        placeholder="Enter your username"
                        required
                        disabled={isLoading}
                      />
                    </div>
                  </div>

                  {/* Password Field */}
                  <div>
                    <label htmlFor="password" className="block text-sm font-semibold text-foreground mb-3">
                      Password
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Lock className="w-5 h-5 text-muted-foreground" />
                      </div>
                      <input
                        id="password"
                        type={showPassword ? 'text' : 'password'}
                        autoCapitalize="none"
                        autoCorrect="off"
                        spellCheck={false}
                        autoComplete="current-password"
                        value={credentials.password}
                        onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
                        className="w-full pl-10 pr-12 py-3 rounded-lg border border-border bg-background text-foreground placeholder:text-muted-foreground focus:ring-2 focus:ring-primary focus:border-primary transition-all"
                        placeholder="Enter your password"
                        required
                        disabled={isLoading}
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute inset-y-0 right-0 pr-3 flex items-center text-muted-foreground hover:text-foreground transition-colors"
                        disabled={isLoading}
                      >
                        {showPassword ? (
                          <EyeOff className="w-5 h-5" />
                        ) : (
                          <Eye className="w-5 h-5" />
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Submit Button */}
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-primary text-primary-foreground rounded-lg font-bold text-lg hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isLoading ? (
                      <>
                        <RefreshCw className="w-5 h-5 animate-spin" />
                        Signing in...
                      </>
                    ) : (
                      <>
                        <LogIn className="w-5 h-5" />
                        Sign In to Dashboard
                      </>
                    )}
                  </button>
                </form>


                {/* Shown only when the server has no login configured at all. */}
                {!session.configured && (
                  <div className="mt-8 p-4 bg-muted/50 rounded-lg border border-border/50">
                    <div className="flex items-start gap-3">
                      <Shield className="w-5 h-5 text-primary mt-0.5" />
                      <div className="text-sm text-muted-foreground">
                        <p className="font-semibold text-foreground mb-1">
                          No admin password is set on this server
                        </p>
                        <p>
                          Anyone who can reach this page can manage the race. Set one in
                          config/admin.local.yml and restart the results API.
                        </p>
                        <button
                          type="button"
                          onClick={() => navigate('/admin/dashboard')}
                          className="mt-3 underline text-primary font-semibold"
                        >
                          Continue to Live Management
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}