import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '@/components/Layout';

export default function AdminLogin() {
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    // Simulate authentication
    setTimeout(() => {
      if (credentials.username === 'admin' && credentials.password === 'admin') {
        sessionStorage.setItem('isAdmin', 'true');
        navigate('/admin/dashboard');
      } else {
        alert('Invalid credentials. Use admin/admin for demo.');
      }
      setIsLoading(false);
    }, 1000);
  };

  return (
    <Layout showNavigation={false}>
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="w-full max-w-md">
          <div className="race-card">
            <div className="text-center mb-8">
              <div className="flex items-center justify-center gap-3 mb-6">
                <div className="flex items-center justify-center w-16 h-16 rounded-xl bg-primary/10 border border-primary/20">
                  <span className="material-icon text-3xl text-primary">admin_panel_settings</span>
                </div>
                <h1 className="text-3xl font-bold text-gradient">Slay Sarcoma Admin</h1>
              </div>
              <p className="text-muted-foreground text-lg">Sign in to access the race timing dashboard</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label htmlFor="username" className="block text-sm font-medium mb-2">
                  Username
                </label>
                <input
                  id="username"
                  type="text"
                  value={credentials.username}
                  onChange={(e) => setCredentials(prev => ({ ...prev, username: e.target.value }))}
                  className="form-input"
                  placeholder="Enter username"
                  required
                />
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium mb-2">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  value={credentials.password}
                  onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
                  className="form-input"
                  placeholder="Enter password"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="btn-primary w-full"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <span className="material-icon animate-spin">refresh</span>
                    Signing in...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="material-icon">login</span>
                    Sign In
                  </div>
                )}
              </button>
            </form>

            <div className="mt-6 p-4 bg-muted rounded-md">
              <p className="text-sm text-muted-foreground">
                <strong>Demo credentials:</strong><br />
                Username: admin<br />
                Password: admin
              </p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}