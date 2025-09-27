import { useState } from 'react';
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

export default function AdminLogin() {
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

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

    // Simulate authentication with validation
    setTimeout(() => {
      if (credentials.username.trim() === 'admin' && credentials.password.trim() === 'admin') {
        sessionStorage.setItem('isAdmin', 'true');
        navigate('/admin/dashboard');
      } else {
        setError('Invalid username or password. Please try again.');
      }
      setIsLoading(false);
    }, 1000);
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

                {/* Demo Credentials Info */}
                <div className="mt-8 p-4 bg-muted/50 rounded-lg border border-border/50">
                  <div className="flex items-start gap-3">
                    <Shield className="w-5 h-5 text-primary mt-0.5" />
                    <div>
                      <p className="text-sm font-semibold text-foreground mb-2">
                        Demo Credentials
                      </p>
                      <div className="space-y-1 text-sm text-muted-foreground">
                        <p><strong>Username:</strong> admin</p>
                        <p><strong>Password:</strong> admin</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}