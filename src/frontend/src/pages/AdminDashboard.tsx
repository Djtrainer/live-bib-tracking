import { useState, useEffect, useRef } from 'react';
import { 
  Settings, 
  Users, 
  Timer, 
  Activity, 
  Upload, 
  Flag, 
  RefreshCw,
  Clock,
  Trash2,
  Plus,
  Monitor,
  Download
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
  const [clockStatus, setClockStatus] = useState<any>(null);
  const resultsContainerRef = useRef<HTMLDivElement>(null);

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
      // fetch current clock status for display conversions
      fetch('/api/clock/status')
        .then(r => r.json())
        .then(res => {
          if (res.success && res.data) setClockStatus(res.data);
        })
        .catch(err => console.error('Failed to fetch clock status:', err));
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.action === 'reload') {
          fetchFinishers();
        } else if (message.type === 'add' || message.type === 'update') {
          setFinishers(prev => {
            // Only match by ID for updates, not by bib number to allow duplicates
            const existingIndex = prev.findIndex(f => f.id === message.data.id);
            let updated;
            const isNewFinisher = existingIndex === -1;
            
            if (existingIndex > -1) {
              // Update existing entry
              updated = [...prev];
              updated[existingIndex] = { ...updated[existingIndex], ...message.data };
            } else {
              // Add new entry (even if bib number already exists)
              updated = [...prev, message.data];
            }
            updated.sort((a, b) => (a.finishTime || 0) - (b.finishTime || 0));
            
            // Auto-scroll to bottom when a new finisher is added
            if (isNewFinisher && message.type === 'add') {
              setTimeout(scrollToBottom, 100);
            }
            // Update clock status on clock messages (if provided)
            if (message.type === 'clock_update' && message.data) {
              setClockStatus(message.data);
            }
            
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

  // Auto-scroll to bottom of results
  const scrollToBottom = () => {
    if (resultsContainerRef.current) {
      resultsContainerRef.current.scrollTop = resultsContainerRef.current.scrollHeight;
    }
  };

  const handleQuickAdd = async () => {
    try {
      // Read the official race clock so we store the elapsed race time (ms),
      // not the epoch timestamp which would render as a huge value in the UI.
      const clockResp = await fetch('/api/clock/status');
      const clockResult = await clockResp.json();

      if (!clockResult.success || !clockResult.data) {
        alert('Unable to read race clock status. Please ensure the race clock is running or reachable.');
        return;
      }

      const { raceStartTime, status, offset } = clockResult.data as {
        raceStartTime: number | null;
        status: string;
        offset: number;
      };

      let finishTimeMs: number;
      if (status !== 'running' || !raceStartTime) {
        // If the clock isn't running, use the stored offset (could be paused/stopped time)
        finishTimeMs = offset || 0;
      } else {
        const nowSec = Date.now() / 1000; // seconds
        finishTimeMs = Math.round((nowSec - raceStartTime) * 1000 + (offset || 0));
      }

      // Use provided bib number or generate a placeholder based on current finisher count
      const finalBibNumber = bibNumber.trim() || `No-Bib-${finishers.length + 1}`;

      const response = await fetch('/api/results', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          bibNumber: finalBibNumber,
          finishTime: finishTimeMs,
        }),
      });

      const result = await response.json();
      
      if (result.success) {
        setBibNumber('');
        // Auto-scroll to bottom after adding finisher
        setTimeout(scrollToBottom, 100);
        // fetchFinishers(); // The WebSocket will handle the update
      } else {
        console.error('Failed to add finisher:', result.error || result.message);
      }
    } catch (error) {
      console.error('Error adding finisher:', error);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this finisher?')) return;
    
    try {
      const response = await fetch(`/api/results/${id}`, {
        method: 'DELETE',
      });

      const result = await response.json();
      
      if (result.success) {
        // fetchFinishers(); // The WebSocket will handle the update
      } else {
        console.error('Failed to delete finisher:', result.error || result.message);
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
      const response = await fetch(`/api/results/${id}`, {
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
        console.error('Failed to update finisher:', result.error || result.message);
      } else {
        // Optimistically update local state for bib changes so racerName updates to
        // backend-provided value or falls back to a sensible default if unknown.
        if (field === 'bibNumber') {
          const newBib = processedValue;
          const newName = result.data && result.data.racerName ? result.data.racerName : `Racer #${newBib}`;
          setFinishers(prev => prev.map(f => f.id === id ? { ...f, bibNumber: newBib, racerName: newName } : f));
        }
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
    formData.append('file', file);

    try {
      const response = await fetch('/api/roster/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      
      if (result.success) {
        const message = result.message || 'Upload successful';
        setUploadStatus(`✅ ${message}`);
        fetchFinishers();
      } else {
        setUploadStatus(`❌ Upload failed: ${result.detail || result.error || 'Unknown error'}`);
      }
    } catch (error) {
      setUploadStatus(`❌ Upload error: ${error}`);
    } finally {
      setIsUploading(false);
      event.target.value = '';
    }
  };

  const handleDownloadCSV = () => {
    if (finishers.length === 0) {
      alert('No results to download');
      return;
    }

    // Create CSV content
    const headers = ['Rank', 'Bib Number', 'Racer Name', 'Finish Time', 'Gender', 'Team'];
    const normalizeFinishTime = (ft: number | null) => {
      if (ft === null || ft === undefined) return null;
      // If it looks like a unix epoch in ms (greater than year 3000 ~32503680000000),
      // or realistically > 1e12 (approx year 2001), assume it's an epoch and convert
      // to elapsed race time using clockStatus if available.
      if (ft > 1e12 && clockStatus) {
        const { raceStartTime, offset } = clockStatus;
        if (raceStartTime) {
          const elapsed = Math.round(ft - (raceStartTime * 1000) - (offset || 0));
          return elapsed > 0 ? elapsed : 0;
        }
      }
      // otherwise assume it's already elapsed ms
      return ft;
    };

    const csvContent = [
      headers.join(','),
      ...finishers.map((finisher, index) => [
        index + 1,
        finisher.bibNumber,
        `"${finisher.racerName || 'Unknown Runner'}"`,
        (() => {
          const normalized = normalizeFinishTime(finisher.finishTime);
          return normalized ? formatTime(normalized) : '';
        })(),
        finisher.gender || '',
        `"${finisher.team || ''}"`
      ].join(','))
    ].join('\n');

    // Create and download file
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `race_results_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const normalizeFinishTime = (ft: number | null) => {
    if (ft === null || ft === undefined) return null;
    if (ft > 1e12 && clockStatus) {
      const { raceStartTime, offset } = clockStatus;
      if (raceStartTime) {
        const elapsed = Math.round(ft - (raceStartTime * 1000) - (offset || 0));
        return elapsed > 0 ? elapsed : 0;
      }
    }
    return ft;
  };

  const getDisplayFinishTime = (ft: number | null) => {
    const normalized = normalizeFinishTime(ft);
    return normalized ? formatTime(normalized) : null;
  };

  // Function to check if a bib number is duplicated
  const isDuplicateBib = (bibNumber: string, currentId: string) => {
    return finishers.filter(f => f.bibNumber === bibNumber && f.id !== currentId).length > 0;
  };

  return (
    <Layout 
      activeTab={activeTab} 
      onTabChange={setActiveTab}
      totalFinishers={finishers.length}
      lastUpdated={new Date()}
    >
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
              </div>
            </div>
          </div>


          {/* Tab Content */}
          {activeTab === 'setup' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Race Clock Controls */}
              <div>
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
              <div>
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
            <div className="flex gap-6 h-[calc(100vh-320px)]">
              
              {/* Left Side - Manage Results Table (75% width, left-aligned) */}
              <div className="w-3/4 overflow-hidden">
                <div className="section-card-tv h-full flex flex-col">
                  <div className="section-header-tv flex-shrink-0">
                    <div className="section-icon-tv">
                      <Users className="w-5 h-5 text-white" />
                    </div>
                    <h2 className="section-title-tv">Manage Results</h2>
                  </div>
                  
                  <div 
                    ref={resultsContainerRef}
                    className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-border scrollbar-track-transparent"
                  >
                    {loading ? (
                      <div className="text-center py-12">
                        <RefreshCw className="w-8 h-8 text-primary mx-auto mb-4 animate-spin" />
                        <p className="text-muted-foreground">Loading results...</p>
                      </div>
                    ) : finishers.length === 0 ? (
                      <div className="text-center py-12">
                        <Flag className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                        <p className="text-muted-foreground text-lg">
                          No finishers added yet. Use the 'Add Finisher' panel on the right to capture the first racer crossing the finish line.
                        </p>
                      </div>
                    ) : (
                      <div className="overflow-x-auto">
                        <table className="results-table-tv">
                          <thead className="sticky top-0 bg-card z-10">
                            <tr>
                              <th className="w-20">Rank</th>
                              <th className="w-28">Bib #</th>
                              <th className="w-auto min-w-[200px]">Racer Name</th>
                              <th className="w-32">Finish Time</th>
                              <th className="w-20">Gender</th>
                              <th className="w-auto min-w-[150px]">Team</th>
                              <th className="w-24">Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {finishers.map((finisher, index) => (
                              <tr key={finisher.id} className={`group ${isDuplicateBib(finisher.bibNumber, finisher.id) ? 'duplicate-bib' : ''}`}>
                                <td>
                                  <span className="rank-cell-tv">{index + 1}</span>
                                </td>
                                <td>
                                  {editingCell?.id === finisher.id && editingCell?.field === 'bibNumber' ? (
                                    <input
                                      type="text"
                                      defaultValue={finisher.bibNumber}
                                      className="w-full p-2 rounded border border-border bg-background text-left font-mono text-lg"
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
                                <td>
                                  {editingCell?.id === finisher.id && editingCell?.field === 'finishTime' ? (
                                    <input
                                      type="text"
                                      defaultValue={getDisplayFinishTime(finisher.finishTime) || ''}
                                      className="w-full p-2 rounded border border-border bg-background text-left font-mono text-lg"
                                      autoFocus
                                      onKeyPress={(e) => handleCellKeyPress(e, finisher.id, 'finishTime', (e.target as HTMLInputElement).value)}
                                      onBlur={(e) => handleCellBlur(finisher.id, 'finishTime', e.target.value)}
                                    />
                                  ) : (
                                    <span 
                                      className="time-cell-tv cursor-pointer hover:bg-muted/20 rounded px-2 py-1"
                                      onClick={() => setEditingCell({ id: finisher.id, field: 'finishTime' })}
                                    >
                                      {getDisplayFinishTime(finisher.finishTime) || '-'}
                                    </span>
                                  )}
                                </td>
                                <td>
                                  {editingCell?.id === finisher.id && editingCell?.field === 'gender' ? (
                                    <select
                                      defaultValue={finisher.gender || ''}
                                      className="w-full p-2 rounded border border-border bg-background text-left"
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
                                <td>
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

              {/* Right Side - Panels (25% width) */}
              <div className="w-1/4 flex flex-col gap-6">
                
                {/* Add Finisher Panel */}
                <div className="section-card-tv">
                  <div className="section-header-tv">
                    <div className="section-icon-tv">
                      <Plus className="w-5 h-5 text-white" />
                    </div>
                    <h2 className="section-title-tv">Add Finisher</h2>
                  </div>
                  
                  <div className="p-6">
                    <p className="text-muted-foreground mb-4 text-sm">
                      Quickly capture finishers as they cross the line. Bib number is optional - leave blank to auto-assign.
                    </p>
                    <div className="space-y-4">
                      <input
                        type="text"
                        placeholder="Bib Number (optional)"
                        value={bibNumber}
                        onChange={(e) => setBibNumber(e.target.value)}
                        className="w-full px-4 py-3 rounded-lg border border-border bg-background text-foreground placeholder:text-muted-foreground font-mono text-lg"
                        onKeyPress={(e) => e.key === 'Enter' && handleQuickAdd()}
                      />
                      <button
                        onClick={handleQuickAdd}
                        className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:bg-primary/90 transition-colors"
                      >
                        <Plus className="w-4 h-4" />
                        Add Finisher
                      </button>
                    </div>
                  </div>
                </div>

                {/* Download CSV Panel */}
                {finishers.length > 0 && (
                  <div className="section-card-tv">
                    <div className="section-header-tv">
                      <div className="section-icon-tv">
                        <Download className="w-5 h-5 text-white" />
                      </div>
                      <h2 className="section-title-tv">Export Results</h2>
                    </div>
                    
                    <div className="p-6">
                      <p className="text-muted-foreground mb-4 text-sm">
                        Download race results as a CSV file for analysis or record keeping.
                      </p>
                      <button
                        onClick={handleDownloadCSV}
                        className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-success text-success-foreground rounded-lg font-semibold hover:bg-success/90 transition-colors"
                        title="Download results as CSV"
                      >
                        <Download className="w-4 h-4" />
                        Download CSV
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

        </div>
      </div>
    </Layout>
  );
}
