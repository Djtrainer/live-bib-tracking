import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '@/components/Layout';
import StatusChip from '@/components/StatusChip';
import DataTable from '@/components/DataTable';
import IconButton from '@/components/IconButton';
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
  // Removed newFinisher and showAddForm state - no longer needed for one-click add
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const [activeTab, setActiveTab] = useState<'setup' | 'live'>('setup');
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
    // Check authentication
    if (!sessionStorage.getItem('isAdmin')) {
      navigate('/admin');
      return;
    }

    fetchFinishers();

    // Set up WebSocket connection for real-time updates (DEBUG VERSION)
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('✅ WebSocket connection opened - AdminDashboard.tsx');
      console.log('🔗 WebSocket URL:', wsUrl);
      console.log('🔗 WebSocket readyState:', ws.readyState);
    };
    
    ws.onmessage = async (event) => {
      console.log('📨 Raw WebSocket message received in AdminDashboard.tsx:', event.data);
      try {
        const message = JSON.parse(event.data);
        console.log('✅ Parsed WebSocket message in AdminDashboard.tsx:', message);
        console.log('📋 Message type:', message.type || message.action || 'unknown');
        
        if (message.action === 'reload') {
          console.log('🔄 Reloading finishers data due to reload action');
          await fetchFinishers();
        } else if (message.type === 'add' || message.type === 'update') {
          console.log('🔄 Processing add/update message:', message.data);
          await fetchFinishers(); // Refresh all data to stay in sync
        }
      } catch (error) {
        console.error('❌ Error processing WebSocket message in AdminDashboard:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('❌ WebSocket connection closed - AdminDashboard.tsx');
    };
    
    ws.onerror = (error) => {
      console.error('❌ WebSocket error in AdminDashboard:', error);
    };

    return () => {
      ws.close();
    };
  }, [navigate]);

  const handleCellDoubleClick = (id: string, field: 'bibNumber' | 'racerName' | 'finishTime' | 'gender' | 'team') => {
    setEditingCell({ id, field });
  };

  const handleCellSave = async (id: string, field: string, value: string) => {
    try {
      const finisher = finishers.find(f => f.id === id);
      if (!finisher) return;

      // Prepare the update data
      const updateData: any = {};
      updateData[field] = value;

      const response = await fetch(`/api/results/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updateData),
      });

      const result = await response.json();
      
      if (result.success) {
        // Refresh all data from backend to get merged roster data if bib number was changed
        await fetchFinishers();
        setEditingCell(null);
      } else {
        console.error('Failed to update finisher:', result);
        alert('Failed to update finisher. Please try again.');
      }
    } catch (error) {
      console.error('Error updating finisher:', error);
      alert('Error updating finisher. Please try again.');
    }
  };

  const handleCellKeyPress = (e: React.KeyboardEvent, id: string, field: string, value: string) => {
    if (e.key === 'Enter') {
      handleCellSave(id, field, value);
    }
  };

  const handleCellBlur = (id: string, field: string, value: string) => {
    handleCellSave(id, field, value);
  };

  const handleDelete = async (id: string) => {
    if (confirm('Are you sure you want to delete this entry?')) {
      try {
        const response = await fetch(`/api/results/${id}`, {
          method: 'DELETE',
        });

        const result = await response.json();
        
        if (result.success) {
          setFinishers(prev => prev.filter(f => f.id !== id));
        } else {
          console.error('Failed to delete finisher:', result);
          alert('Failed to delete finisher. Please try again.');
        }
      } catch (error) {
        console.error('Error deleting finisher:', error);
        alert('Error deleting finisher. Please try again.');
      }
    }
  };

  const handleAddFinisher = async () => {
    try {
      // Generate unique placeholder bib number
      const existingUnknownBibs = finishers
        .map(f => f.bibNumber)
        .filter(bib => bib.startsWith('Unknown-'))
        .map(bib => parseInt(bib.split('-')[1]) || 0)
        .sort((a, b) => b - a); // Sort descending to get highest number first
      
      const nextUnknownNumber = existingUnknownBibs.length > 0 ? existingUnknownBibs[0] + 1 : 1;
      const placeholderBib = `Unknown-${nextUnknownNumber}`;
      
      // Capture current race time from the official race clock
      const currentWallTime = Date.now() / 1000; // Current time in seconds (Unix timestamp)
      
      console.log('🔍 One-click add finisher:', {
        placeholderBib,
        currentWallTime,
        timestamp: new Date().toISOString()
      });

      const finisher = {
        bibNumber: placeholderBib,
        racerName: `Racer ${placeholderBib}`,
        wallClockTime: currentWallTime, // Send wall-clock time to backend
      };

      const response = await fetch('/api/results', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(finisher),
      });

      const result = await response.json();
      
      if (result.success) {
        console.log('✅ Successfully added one-click finisher:', result.data);
        // Data will be updated via WebSocket, no need to manually update state
      } else {
        console.error('Failed to add finisher:', result);
        alert(`Failed to add finisher: ${result.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error adding finisher:', error);
      alert('Error adding finisher. Please try again.');
    }
  };

  const handleDownloadCSV = () => {
    // Create CSV headers
    const headers = 'Rank,Bib Number,Racer Name,Finish Time\n';
    
    // Convert finishers data to CSV rows
    const csvRows = finishers.map(finisher => {
      const formattedTime = typeof finisher.finishTime === 'number' 
        ? formatTime(finisher.finishTime) 
        : finisher.finishTime;
      
      return `${finisher.rank},"${finisher.bibNumber}","${finisher.racerName}","${formattedTime}"`;
    }).join('\n');
    
    // Combine headers and data
    const csvContent = headers + csvRows;
    
    // Create blob and download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    // Create filename with current date
    const today = new Date();
    const dateString = today.getFullYear() + '-' + 
      String(today.getMonth() + 1).padStart(2, '0') + '-' + 
      String(today.getDate()).padStart(2, '0');
    const filename = `race_results_${dateString}.csv`;
    
    // Set up download link
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.style.display = 'none';
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Clean up the URL object
    URL.revokeObjectURL(link.href);
  };

  const handleRosterUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      alert('Please select a CSV file');
      return;
    }

    setIsUploading(true);
    setUploadStatus('');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/roster/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        let statusMessage = `✅ ${result.message}`;
        
        if (result.uploaded_count > 0) {
          statusMessage += `\n📝 ${result.uploaded_count} new racers added`;
        }
        if (result.updated_count > 0) {
          statusMessage += `\n🔄 ${result.updated_count} existing racers updated`;
        }
        
        if (result.errors && result.errors.length > 0) {
          statusMessage += `\n\n⚠️ ${result.errors.length} errors occurred:\n${result.errors.join('\n')}`;
        }
        
        setUploadStatus(statusMessage);
        
        // Refresh the finishers list
        const refreshResponse = await fetch('/api/results');
        const refreshResult = await refreshResponse.json();
        if (refreshResult.success && refreshResult.data) {
          setFinishers(refreshResult.data);
        }
      } else {
        setUploadStatus(`❌ Upload failed: ${result.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error uploading roster:', error);
      setUploadStatus(`❌ Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
      // Clear the file input
      event.target.value = '';
    }
  };

  const handleLogout = () => {
    sessionStorage.removeItem('isAdmin');
    navigate('/admin');
  };

  const columns = [
    { key: 'rank', title: 'Rank', width: '80px', align: 'center' as const },
    { key: 'bibNumber', title: 'Bib #', width: '100px', align: 'center' as const },
    { key: 'racerName', title: 'Racer Name', width: '180px', align: 'left' as const },
    { key: 'finishTime', title: 'Finish Time', width: '120px', align: 'center' as const },
    { key: 'gender', title: 'Gender', width: '80px', align: 'center' as const },
    { key: 'team', title: 'Team', width: '150px', align: 'left' as const },
    { key: 'actions', title: 'Actions', width: '120px', align: 'center' as const },
  ];

  const renderRow = (finisher: Finisher, index: number, columns: any[]) => {
    const isEditingBib = editingCell?.id === finisher.id && editingCell?.field === 'bibNumber';
    const isEditingName = editingCell?.id === finisher.id && editingCell?.field === 'racerName';
    const isEditingTime = editingCell?.id === finisher.id && editingCell?.field === 'finishTime';
    const isEditingGender = editingCell?.id === finisher.id && editingCell?.field === 'gender';
    const isEditingTeam = editingCell?.id === finisher.id && editingCell?.field === 'team';
    
    return (
      <tr key={finisher.id}>
        <td 
          className={`text-${columns[0].align} font-mono`}
          style={{ width: columns[0].width }}
        >
          {finisher.rank}
        </td>
        <td 
          className={`text-${columns[1].align} font-mono font-medium cursor-pointer hover:bg-muted/50 transition-colors`}
          style={{ width: columns[1].width }}
          onDoubleClick={() => handleCellDoubleClick(finisher.id, 'bibNumber')}
          title="Double-click to edit"
        >
          {isEditingBib ? (
            <input
              type="text"
              defaultValue={finisher.bibNumber}
              className="form-input w-20 text-center"
              autoFocus
              onKeyPress={(e) => handleCellKeyPress(e, finisher.id, 'bibNumber', (e.target as HTMLInputElement).value)}
              onBlur={(e) => handleCellBlur(finisher.id, 'bibNumber', e.target.value)}
            />
          ) : (
            finisher.bibNumber
          )}
        </td>
        <td 
          className={`text-${columns[2].align} font-mono font-medium cursor-pointer hover:bg-muted/50 transition-colors`}
          style={{ width: columns[2].width }}
          onDoubleClick={() => handleCellDoubleClick(finisher.id, 'racerName')}
          title="Double-click to edit"
        >
          {isEditingName ? (
            <input
              type="text"
              defaultValue={finisher.racerName}
              className="form-input w-full"
              autoFocus
              onKeyPress={(e) => handleCellKeyPress(e, finisher.id, 'racerName', (e.target as HTMLInputElement).value)}
              onBlur={(e) => handleCellBlur(finisher.id, 'racerName', e.target.value)}
            />
          ) : (
            finisher.racerName
          )}
        </td>
        <td 
          className={`text-${columns[3].align} font-mono cursor-pointer hover:bg-muted/50 transition-colors`}
          style={{ width: columns[3].width }}
          onDoubleClick={() => handleCellDoubleClick(finisher.id, 'finishTime')}
          title="Double-click to edit"
        >
          {isEditingTime ? (
            <input
              type="text"
              defaultValue={typeof finisher.finishTime === 'number' ? formatTime(finisher.finishTime) : (finisher.finishTime || '')}
              className="form-input w-24 text-center"
              autoFocus
              onKeyPress={(e) => handleCellKeyPress(e, finisher.id, 'finishTime', (e.target as HTMLInputElement).value)}
              onBlur={(e) => handleCellBlur(finisher.id, 'finishTime', e.target.value)}
            />
          ) : (
            finisher.finishTime ? 
              (typeof finisher.finishTime === 'number' ? formatTime(finisher.finishTime) : finisher.finishTime) :
              <span className="text-muted-foreground">Not finished</span>
          )}
        </td>
        <td 
          className={`text-${columns[4].align} cursor-pointer hover:bg-muted/50 transition-colors`}
          style={{ width: columns[4].width }}
          onDoubleClick={() => handleCellDoubleClick(finisher.id, 'gender')}
          title="Double-click to edit"
        >
          {isEditingGender ? (
            <select
              defaultValue={finisher.gender || ''}
              className="form-input w-16 text-center"
              autoFocus
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleCellKeyPress(e, finisher.id, 'gender', (e.target as HTMLSelectElement).value);
                }
              }}
              onBlur={(e) => handleCellBlur(finisher.id, 'gender', e.target.value)}
            >
              <option value="">-</option>
              <option value="M">M</option>
              <option value="W">W</option>
            </select>
          ) : (
            <span className="font-mono">{finisher.gender || '-'}</span>
          )}
        </td>
        <td 
          className={`text-${columns[5].align} cursor-pointer hover:bg-muted/50 transition-colors`}
          style={{ width: columns[5].width }}
          onDoubleClick={() => handleCellDoubleClick(finisher.id, 'team')}
          title="Double-click to edit"
        >
          {isEditingTeam ? (
            <input
              type="text"
              defaultValue={finisher.team || ''}
              className="form-input w-full"
              autoFocus
              onKeyPress={(e) => handleCellKeyPress(e, finisher.id, 'team', (e.target as HTMLInputElement).value)}
              onBlur={(e) => handleCellBlur(finisher.id, 'team', e.target.value)}
            />
          ) : (
            <span className="font-mono">{finisher.team || '-'}</span>
          )}
        </td>
        <td 
          className={`text-${columns[6].align}`}
          style={{ width: columns[6].width }}
        >
          <div className="flex items-center justify-center gap-1">
            <IconButton
              icon="delete"
              onClick={() => handleDelete(finisher.id)}
              title="Delete entry"
              variant="destructive"
            />
          </div>
        </td>
      </tr>
    );
  };

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">Race Administration</h1>
            <p className="text-muted-foreground">Manage race results and finisher data</p>
          </div>
          <button onClick={handleLogout} className="btn-ghost">
            <span className="material-icon">logout</span>
            Sign Out
          </button>
        </div>

        {/* Stats */}
        <div className="flex flex-wrap gap-4 mb-8">
          <StatusChip 
            label="Total Finishers" 
            value={finishers.length} 
            variant="success"
            icon="flag"
          />
          <StatusChip 
            label="Status" 
            value="Live Updates" 
            variant="live"
            icon="sync"
          />
          <StatusChip 
            label="Last Action" 
            value={new Date().toLocaleTimeString()} 
            icon="schedule"
          />
        </div>

        {/* Tabbed Navigation */}
        <div className="mb-8">
          <div className="border-b border-border">
            <nav className="flex space-x-8">
              <button
                onClick={() => setActiveTab('setup')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'setup'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground'
                }`}
              >
                <span className="flex items-center gap-2">
                  <span className="material-icon text-sm">settings</span>
                  Race Setup
                </span>
              </button>
              <button
                onClick={() => setActiveTab('live')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'live'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground'
                }`}
              >
                <span className="flex items-center gap-2">
                  <span className="material-icon text-sm">live_tv</span>
                  Live Management
                </span>
              </button>
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'setup' && (
          <div className="space-y-6">
            {/* Race Clock Controls */}
            <div className="bg-card border border-border rounded-lg p-6">
              <div className="flex items-center gap-2 mb-4">
                <span className="material-icon text-primary">timer</span>
                <h2 className="text-xl font-semibold">Official Race Clock</h2>
              </div>
              <p className="text-muted-foreground mb-4">
                Control the official race clock. All finish times will be calculated relative to when you start the race clock.
              </p>
              
              <RaceClock showControls={true} className="mb-4" />
            </div>

            {/* Roster Management */}
            <div className="bg-card border border-border rounded-lg p-6">
              <div className="flex items-center gap-2 mb-4">
                <span className="material-icon text-primary">upload_file</span>
                <h2 className="text-xl font-semibold">Roster Management</h2>
              </div>
              <p className="text-muted-foreground mb-4">
                Upload a CSV file to pre-register all race participants. The CSV should contain headers: bibNumber, racerName, gender (optional), team (optional).
              </p>
              
              <div className="flex items-center gap-4 mb-4">
                <div className="flex-1">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleRosterUpload}
                    disabled={isUploading}
                    className="form-input"
                    id="roster-upload"
                  />
                </div>
                <button
                  onClick={() => document.getElementById('roster-upload')?.click()}
                  disabled={isUploading}
                  className="btn-primary"
                >
                  <span className="material-icon">
                    {isUploading ? 'hourglass_empty' : 'upload'}
                  </span>
                  {isUploading ? 'Uploading...' : 'Upload Roster'}
                </button>
              </div>

              {uploadStatus && (
                <div className="bg-muted/50 border border-border rounded-lg p-4">
                  <pre className="text-sm whitespace-pre-wrap">{uploadStatus}</pre>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'live' && (
          <div className="space-y-6">
            {/* Live Management Header with Clock and Actions */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <button
                  onClick={handleAddFinisher}
                  className="btn-primary"
                  title="Instantly add a finisher with current race time"
                >
                  <span className="material-icon">add</span>
                  Add Finisher
                </button>
                <button
                  onClick={handleDownloadCSV}
                  className="btn-primary"
                >
                  <span className="material-icon">download</span>
                  Download CSV
                </button>
              </div>
              
              <div className="text-right">
                <div className="mb-2">
                  <span className="text-sm text-muted-foreground">Official Race Time</span>
                </div>
                <RaceClock showControls={false} />
              </div>
            </div>

            {/* Results Table */}
            <div className="bg-card rounded-lg border border-border p-6">
              <div className="flex items-center gap-2 mb-6">
                <span className="material-icon text-primary">manage_accounts</span>
                <h2 className="text-xl font-semibold">Manage Results</h2>
              </div>
              
              <DataTable
                columns={columns}
                data={finishers}
                renderRow={renderRow}
                emptyMessage="No finishers added yet. Click 'Add Finisher' to capture the first racer crossing the finish line."
              />
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
