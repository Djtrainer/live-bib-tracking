import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Settings, 
  Users, 
  Timer, 
  Activity, 
  Upload, 
  Flag, 
  RefreshCw,
  Clock,
  LogOut,
  Trash2,
  Plus,
  Monitor
} from 'lucide-react';
import Layout from '@/components/Layout';
import RaceClock from '@/components/RaceClock';
import { formatTime } from '../lib/utils';

interface Finisher {
  id: string;
  rank: number | null;
  bibNumber: string;
  racerName: string;
  finishTime: number | null;
  gender?: string;
  team?: string;
}

interface EditingCell {
  id: string;
  field: 'bibNumber' | 'racerName' | 'finishTime' | 'gender' | 'team';
}

export default function AdminDashboard() {
  const [finishers, setFinishers] = useState<Finisher[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingCell, setEditingCell] = useState<EditingCell | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const [activeTab, setActiveTab] = useState<'setup' | 'live'>('setup');
  const [bibNumber, setBibNumber] = useState('');
  const navigate = useNavigate();

  // Fetch finishers data from backend
  const fetchFinishers = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/results');
      const result = await response.json();
      
      if (result.success && result.data) {
        setFinishers(result.data);
      } else {
        console.error('Failed to fetch results:', result);
      }
    } catch (error) {
      console.error('Error fetching finishers:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFinishers();

    // Set up WebSocket connection for real-time updates
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('✅ WebSocket connection opened - AdminDashboard.tsx');
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.action === 'reload') {
          fetchFinishers();
        } else if (message.type === 'add' || message.type === 'update') {
          setFinishers(prev => {
            const existingIndex = prev.findIndex(f => f.id === message.data.id || f.bibNumber === message.data.bibNumber);
            let updated;
            if (existingIndex > -1) {
              updated = [...prev];
              updated[existingIndex] = { ...updated[existingIndex], ...message.data };
            } else {
              updated = [...prev, message.data];
            }
            updated.sort((a, b) => (a.finishTime || 0) - (b.finishTime || 0));
            return updated;
          });
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, []);

  const handleLogout = () => {
    navigate('/admin/login');
  };

  const handleQuickAdd = async () => {
    if (!bibNumber.trim()) return;
    
    try {
      const response = await fetch('/api/add-finisher', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          bibNumber: bibNumber.trim(),
          racerName: '',
          finishTime: Date.now(),
        }),
      });

      const result = await response.json();
      
      if (result.success) {
        setBibNumber('');
        // fetchFinishers(); // The WebSocket will handle the update
      } else {
        console.error('Failed to add finisher:', result.error);
      }
    } catch (error) {
      console.error('Error adding finisher:', error);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this finisher?')) return;
    
    try {
      const response = await fetch(`/api/delete-finisher/${id}`, {
        method: 'DELETE',
      });

      const result = await response.json();
      
      if (result.success) {
        // fetchFinishers(); // The WebSocket will handle the update
      } else {
        console.error('Failed to delete finisher:', result.error);
      }
    } catch (error) {
      console.error('Error deleting finisher:', error);
    }
  };

  const handleCellKeyPress = (e: React.KeyboardEvent, id: string, field: string, value: string) => {
    if (e.key === 'Enter') {
      handleCellBlur(id, field, value);
    }
  };

  const handleCellBlur = async (id: string, field: string, value: string) => {
    setEditingCell(null);
    
    const finisher = finishers.find(f => f.id === id);
    if (!finisher) return;

    let processedValue: any = value;
    
    if (field === 'finishTime') {
      if (value.trim() === '') {
        processedValue = null;
      } else {
        const timeParts = value.split(':');
        if (timeParts.length === 2) {
          const minutes = parseInt(timeParts[0]);
          const seconds = parseFloat(timeParts[1]);
          processedValue = (minutes * 60 + seconds) * 1000;
        } else {
          console.error('Invalid time format. Use MM:SS.ss');
          return;
        }
      }
    }

    if (finisher[field as keyof Finisher] === processedValue) return;

    try {
      const response = await fetch(`/api/update-finisher/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          [field]: processedValue,
        }),
      });

      const result = await response.json();
      
      if (!result.success) {
        console.error('Failed to update finisher:', result.error);
      }
    } catch (error) {
      console.error('Error updating finisher:', error);
    }
  };

  const handleRosterUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus('');

    const formData = new FormData();
    formData.append('roster', file);

    try {
      const response = await fetch('/api/upload-roster', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      
      if (result.success) {
        setUploadStatus(`✅ Successfully processed ${result.processed} participants\n${result.summary}`);
        fetchFinishers();
      } else {
        setUploadStatus(`❌ Upload failed: ${result.error}`);
      }
    } catch (error) {
      setUploadStatus(`❌ Upload error: ${error}`);
    } finally {
      setIsUploading(false);
      event.target.value = '';
    }
  };

  return (
    <Layout>
      <div className="leaderboard-container">
        <div className="max-w-[1800px] mx-auto">
          
          {/* Admin Header - TV Optimized */}
          <div className="hero-section-tv">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <h1 className="text-4xl md:text-5xl font-black mb-2 text-left">
                  <span className="bg-gradient-to-r from-foreground via-primary to-secondary bg-clip-text text-transparent">
                    Race Administration
                  </span>
                </h1>
                <p className="text-left text-muted-foreground text-lg">
                  Manage race results and finisher data
                </p>
              </div>
              
              {/* Right-aligned stats */}
              <div className="flex flex-row gap-4">
                <div className="stat-card-tv min-w-[140px]">
                  <Flag className="w-5 h-5 text-primary mb-2 mx-auto" />
                  <div className="stat-value-tv">{finishers.length}</div>
                  <div className="stat-label-tv">Total Finishers</div>
                </div>
                
                <div className="stat-card-tv min-w-[140px]">
                  <RefreshCw className="w-5 h-5 text-success mb-2 mx-auto" />
                  <div className="stat-value-tv text-lg">
                    {new Date().toLocaleTimeString()}
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
                
                <button onClick={handleLogout} className="stat-card-tv hover:bg-destructive/10 transition-colors min-w-[140px]">
                  <LogOut className="w-5 h-5 text-destructive mb-2 mx-auto" />
                  <div className="stat-label-tv text-destructive">Sign Out</div>
                </button>
              </div>
            </div>
          </div>

          {/* Tabbed Navigation - TV Style */}
          <div className="mb-6">
            <div className="flex bg-card rounded-xl border border-border/50 p-2">
              <button
                onClick={() => setActiveTab('setup')}
                className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-semibold transition-all ${
                  activeTab === 'setup'
                    ? 'bg-primary text-primary-foreground shadow-md'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted/30'
                }`}
              >
                <Settings className="w-4 h-4" />
                Race Setup
              </button>
              <button
                onClick={() => setActiveTab('live')}
                className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-semibold transition-all ${
                  activeTab === 'live'
                    ? 'bg-primary text-primary-foreground shadow-md'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted/30'
                }`}
              >
                <Monitor className="w-4 h-4" />
                Live Management
              </button>
            </div>
          </div>

          {/* Tab Content */}
          {activeTab === 'setup' && (
            <div className="tv-dashboard-grid">
              
              {/* Race Clock Controls */}
              <div className="col-span-6">
                <div className="section-card-tv">
                  <div className="section-header-tv">
                    <div className="section-icon-tv">
                      <Timer className="w-5 h-5 text-white" />
                    </div>
                    <h2 className="section-title-tv">Official Race Clock</h2>
                  </div>
                  
                  <div className="p-6">
                    <p className="text-muted-foreground mb-6 text-lg">
                      Control the official race clock. All finish times will be calculated relative to when you start the race clock.
                    </p>
                    
                    <div className="bg-gradient-to-br from-card to-muted/20 rounded-xl p-6 border border-border/30">
                      <RaceClock showControls={true} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Roster Management */}
              <div className="col-span-6">
                <div className="section-card-tv">
                  <div className="section-header-tv">
                    <div className="section-icon-tv">
                      <Upload className="w-5 h-5 text-white" />
                    </div>
                    <h2 className="section-title-tv">Roster Management</h2>
                  </div>
                  
                  <div className="p-6">
                    <p className="text-muted-foreground mb-6 text-lg">
                      Upload a CSV file to pre-register all race participants. The CSV should contain headers: bibNumber, racerName, gender (optional), team (optional).
                    </p>
                    
                    <div className="bg-gradient-to-br from-card to-muted/20 rounded-xl p-6 border border-border/30">
                      <div className="flex items-center gap-4 mb-4">
                        <div className="flex-1">
                          <input
                            type="file"
                            accept=".csv"
                            onChange={handleRosterUpload}
                            disabled={isUploading}
                            className="w-full p-3 rounded-lg border border-border bg-background text-foreground file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
                            id="roster-upload"
                          />
                        </div>
                        <button
                          onClick={() => document.getElementById('roster-upload')?.click()}
                          disabled={isUploading}
                          className="flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:bg-primary/90 transition-colors disabled:opacity-50"
                        >
                          {isUploading ? (
                            <RefreshCw className="w-4 h-4 animate-spin" />
                          ) : (
                            <Upload className="w-4 h-4" />
                          )}
                          {isUploading ? 'Uploading...' : 'Upload Roster'}
                        </button>
                      </div>

                      {uploadStatus && (
                        <div className="bg-muted/30 border border-border/50 rounded-lg p-4">
                          <pre className="text-sm whitespace-pre-wrap text-foreground">{uploadStatus}</pre>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'live' && (
            <div className="space-y-6">
              
              {/* Quick Add Finisher */}
              <div className="section-card-tv">
                <div className="section-header-tv">
                  <div className="section-icon-tv">
                    <Plus className="w-5 h-5 text-white" />
                  </div>
                  <h2 className="section-title-tv">Add Finisher</h2>
                </div>
                
                <div className="p-6">
                  <p className="text-muted-foreground mb-4 text-lg">
                    Quickly capture finishers as they cross the line. Just enter their bib number and press Enter or click Add.
                  </p>
                  <div className="flex gap-4">
                    <input
                      type="text"
                      placeholder="Bib Number"
                      value={bibNumber}
                      onChange={(e) => setBibNumber(e.target.value)}
                      className="px-4 py-3 rounded-lg border border-border bg-background text-foreground placeholder:text-muted-foreground font-mono text-lg"
                      onKeyPress={(e) => e.key === 'Enter' && handleQuickAdd()}
                    />
                    <button
                      onClick={handleQuickAdd}
                      disabled={!bibNumber.trim()}
                      className="flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:bg-primary/90 transition-colors disabled:opacity-50"
                    >
                      <Plus className="w-4 h-4" />
                      Add Finisher
                    </button>
                  </div>
                </div>
              </div>

              {/* Results Table */}
              <div className="section-card-tv">
                <div className="section-header-tv">
                  <div className="section-icon-tv">
                    <Users className="w-5 h-5 text-white" />
                  </div>
                  <h2 className="section-title-tv">Manage Results</h2>
                </div>
                
                <div className="overflow-hidden">
                  {loading ? (
                    <div className="text-center py-12">
                      <RefreshCw className="w-8 h-8 text-primary mx-auto mb-4 animate-spin" />
                      <p className="text-muted-foreground">Loading results...</p>
                    </div>
                  ) : finishers.length === 0 ? (
                    <div className="text-center py-12">
                      <Flag className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground text-lg">
                        No finishers added yet. Click 'Add Finisher' to capture the first racer crossing the finish line.
                      </p>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="results-table-tv">
                        <thead>
                          <tr>
                            <th className="w-20">Rank</th>
                            <th className="w-24">Bib #</th>
                            <th>Racer Name</th>
                            <th className="w-32">Finish Time</th>
                            <th className="w-20">Gender</th>
                            <th>Team</th>
                            <th className="w-24">Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {finishers.map((finisher, index) => (
                            <tr key={finisher.id} className="group">
                              <td className="text-center">
                                <span className="rank-cell-tv">{index + 1}</span>
                              </td>
                              <td className="text-center">
                                {editingCell?.id === finisher.id && editingCell?.field === 'bibNumber' ? (
                                  <input
                                    type="text"
                                    defaultValue={finisher.bibNumber}
                                    className="w-full p-2 rounded border border-border bg-background text-center font-mono text-lg"
                                    autoFocus
                                    onKeyPress={(e) => handleCellKeyPress(e, finisher.id, 'bibNumber', (e.target as HTMLInputElement).value)}
                                    onBlur={(e) => handleCellBlur(finisher.id, 'bibNumber', e.target.value)}
                                  />
                                ) : (
                                  <span 
                                    className="bib-cell-tv cursor-pointer hover:bg-muted/20 rounded px-2 py-1"
                                    onClick={() => setEditingCell({ id: finisher.id, field: 'bibNumber' })}
                                  >
                                    #{finisher.bibNumber}
                                  </span>
                                )}
                              </td>
                              <td>
                                {editingCell?.id === finisher.id && editingCell?.field === 'racerName' ? (
                                  <input
                                    type="text"
                                    defaultValue={finisher.racerName}
                                    className="w-full p-2 rounded border border-border bg-background text-lg"
                                    autoFocus
                                    onKeyPress={(e) => handleCellKeyPress(e, finisher.id, 'racerName', (e.target as HTMLInputElement).value)}
                                    onBlur={(e) => handleCellBlur(finisher.id, 'racerName', e.target.value)}
                                  />
                                ) : (
                                  <span 
                                    className="name-cell-tv cursor-pointer hover:bg-muted/20 rounded px-2 py-1"
                                    onClick={() => setEditingCell({ id: finisher.id, field: 'racerName' })}
                                  >
                                    {finisher.racerName || 'Unknown Runner'}
                                  </span>
                                )}
                              </td>
                              <td className="text-center">
                                {editingCell?.id === finisher.id && editingCell?.field === 'finishTime' ? (
                                  <input
                                    type="text"
                                    defaultValue={finisher.finishTime ? formatTime(finisher.finishTime) : ''}
                                    className="w-full p-2 rounded border border-border bg-background text-center font-mono text-lg"
                                    autoFocus
                                    onKeyPress={(e) => handleCellKeyPress(e, finisher.id, 'finishTime', (e.target as HTMLInputElement).value)}
                                    onBlur={(e) => handleCellBlur(finisher.id, 'finishTime', e.target.value)}
                                  />
                                ) : (
                                  <span 
                                    className="time-cell-tv cursor-pointer hover:bg-muted/20 rounded px-2 py-1"
                                    onClick={() => setEditingCell({ id: finisher.id, field: 'finishTime' })}
                                  >
                                    {finisher.finishTime ? formatTime(finisher.finishTime) : '-'}
                                  </span>
                                )}
                              </td>
                              <td className="text-center">
                                {editingCell?.id === finisher.id && editingCell?.field === 'gender' ? (
                                  <select
                                    defaultValue={finisher.gender || ''}
                                    className="w-full p-2 rounded border border-border bg-background text-center"
                                    autoFocus
                                    onBlur={(e) => handleCellBlur(finisher.id, 'gender', e.target.value)}
                                    onChange={(e) => handleCellBlur(finisher.id, 'gender', e.target.value)}
                                  >
                                    <option value="">-</option>
                                    <option value="M">M</option>
                                    <option value="W">W</option>
                                  </select>
                                ) : (
                                  <span 
                                    className="cursor-pointer hover:bg-muted/20 rounded px-2 py-1 font-semibold"
                                    onClick={() => setEditingCell({ id: finisher.id, field: 'gender' })}
                                  >
                                    {finisher.gender || '-'}
                                  </span>
                                )}
                              </td>
                              <td>
                                {editingCell?.id === finisher.id && editingCell?.field === 'team' ? (
                                  <input
                                    type="text"
                                    defaultValue={finisher.team || ''}
                                    className="w-full p-2 rounded border border-border bg-background"
                                    autoFocus
                                    onKeyPress={(e) => handleCellKeyPress(e, finisher.id, 'team', (e.target as HTMLInputElement).value)}
                                    onBlur={(e) => handleCellBlur(finisher.id, 'team', e.target.value)}
                                  />
                                ) : (
                                  <span 
                                    className="cursor-pointer hover:bg-muted/20 rounded px-2 py-1"
                                    onClick={() => setEditingCell({ id: finisher.id, field: 'team' })}
                                  >
                                    {finisher.team || '-'}
                                  </span>
                                )}
                              </td>
                              <td className="text-center">
                                <button
                                  onClick={() => handleDelete(finisher.id)}
                                  className="p-2 text-destructive hover:bg-destructive/10 rounded-lg transition-colors"
                                  title="Delete entry"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </Layout>
  );
}