import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Layout from '@/components/Layout';
import StatusChip from '@/components/StatusChip';
import DataTable from '@/components/DataTable';
import IconButton from '@/components/IconButton';
import { formatTime } from '../lib/utils'; 

interface Finisher {
  id: string;
  rank: number | null;
  bibNumber: string;
  racerName: string;
  finishTime: number | null;
  gender?: string;
  team?: string;
  isEditing?: boolean;
}

export default function AdminDashboard() {
  const [finishers, setFinishers] = useState<Finisher[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [newFinisher, setNewFinisher] = useState({ 
    bibNumber: '', 
    racerName: '', 
    finishTime: '', 
    gender: '', 
    team: '' 
  });
  const [showAddForm, setShowAddForm] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
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
  }, [navigate]);

  const handleEdit = (id: string) => {
    setEditingId(id);
  };

  const handleSave = async (id: string, updatedData: Partial<Finisher>) => {
    try {
      const finisher = finishers.find(f => f.id === id);
      if (!finisher) return;

      const updatedFinisher = { ...finisher, ...updatedData };
      
      const response = await fetch(`/api/results/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedFinisher),
      });

      const result = await response.json();
      
      if (result.success) {
        // Instead of just updating local state, refresh all data from backend
        // This ensures we get the merged roster data if bib number was changed
        await fetchFinishers();
        setEditingId(null);
      } else {
        console.error('Failed to update finisher:', result);
        alert('Failed to update finisher. Please try again.');
      }
    } catch (error) {
      console.error('Error updating finisher:', error);
      alert('Error updating finisher. Please try again.');
    }
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
    if (!newFinisher.bibNumber || !newFinisher.racerName || !newFinisher.finishTime) {
      alert('Please fill in all required fields');
      return;
    }

    try {
      const rankedFinishers = finishers.filter(f => f.rank !== null);
      const newRank = rankedFinishers.length > 0 ? Math.max(...rankedFinishers.map(f => f.rank!)) + 1 : 1;
      const finisher: any = {
        bibNumber: newFinisher.bibNumber,
        racerName: newFinisher.racerName,
        finishTime: newFinisher.finishTime,
        rank: newRank,
      };

      // Add optional fields if provided
      if (newFinisher.gender && newFinisher.gender.trim()) {
        finisher.gender = newFinisher.gender.trim();
      }
      if (newFinisher.team && newFinisher.team.trim()) {
        finisher.team = newFinisher.team.trim();
      }

      const response = await fetch('/api/results', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(finisher),
      });

      const result = await response.json();
      
      if (result.success) {
        setFinishers(prev => [...prev, result.data]);
        setNewFinisher({ 
          bibNumber: '', 
          racerName: '', 
          finishTime: '', 
          gender: '', 
          team: '' 
        });
        setShowAddForm(false);
      } else {
        console.error('Failed to add finisher:', result);
        alert('Failed to add finisher. Please try again.');
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
    { key: 'racerName', title: 'Racer Name', width: '200px', align: 'left' as const },
    { key: 'finishTime', title: 'Finish Time', width: '120px', align: 'center' as const },
    { key: 'actions', title: 'Actions', width: '120px', align: 'center' as const },
  ];

  const renderRow = (finisher: Finisher, index: number, columns: any[]) => {
    const isEditing = editingId === finisher.id;
    
    return (
      <tr key={finisher.id}>
        <td 
          className={`text-${columns[0].align} font-mono`}
          style={{ width: columns[0].width }}
        >
          {finisher.rank}
        </td>
        <td 
          className={`text-${columns[1].align} font-mono font-medium`}
          style={{ width: columns[1].width }}
        >
          {isEditing ? (
            <input
              type="text"
              defaultValue={finisher.bibNumber}
              className="form-input w-20 text-center"
              onBlur={(e) => handleSave(finisher.id, { bibNumber: e.target.value })}
            />
          ) : (
            finisher.bibNumber
          )}
        </td>
        <td 
          className={`text-${columns[2].align} font-mono font-medium`}
          style={{ width: columns[2].width }}
        >
          {isEditing ? (
            <input
              type="text"
              defaultValue={finisher.racerName}
              className="form-input w-full"
              onBlur={(e) => handleSave(finisher.id, { racerName: e.target.value })}
            />
          ) : (
            finisher.racerName
          )}
        </td>
        <td 
          className={`text-${columns[3].align} font-mono`}
          style={{ width: columns[3].width }}
        >
          {isEditing ? (
            <input
              type="text"
              defaultValue={typeof finisher.finishTime === 'number' ? formatTime(finisher.finishTime) : (finisher.finishTime || '')}
              className="form-input w-24 text-center"
              onBlur={(e) => handleSave(finisher.id, { finishTime: e.target.value })}
            />
          ) : (
            finisher.finishTime ? 
              (typeof finisher.finishTime === 'number' ? formatTime(finisher.finishTime) : finisher.finishTime) :
              <span className="text-muted-foreground">Not finished</span>
          )}
        </td>
        <td 
          className={`text-${columns[4].align}`}
          style={{ width: columns[4].width }}
        >
          <div className="flex items-center justify-center gap-1">
            {isEditing ? (
              <IconButton
                icon="check"
                onClick={() => setEditingId(null)}
                title="Save changes"
                variant="success"
              />
            ) : (
              <IconButton
                icon="edit"
                onClick={() => handleEdit(finisher.id)}
                title="Edit entry"
              />
            )}
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

        {/* Roster Management */}
        <div className="bg-card border border-border rounded-lg p-6 mb-6">
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

        {/* Actions */}
        <div className="flex items-center gap-4 mb-6">
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="btn-primary"
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

        {/* Add Form */}
        {showAddForm && (
          <div className="bg-card border border-border rounded-lg p-6 mb-6">
            <h3 className="text-lg font-semibold mb-4">Add New Finisher</h3>
            <div className="grid grid-cols-1 md:grid-cols-6 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium mb-2">Bib Number *</label>
                <input
                  type="text"
                  value={newFinisher.bibNumber}
                  onChange={(e) => setNewFinisher(prev => ({ ...prev, bibNumber: e.target.value }))}
                  className="form-input"
                  placeholder="e.g., 123"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Racer Name *</label>
                <input
                  type="text"
                  value={newFinisher.racerName}
                  onChange={(e) => setNewFinisher(prev => ({ ...prev, racerName: e.target.value }))}
                  className="form-input"
                  placeholder="e.g., John Doe"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Finish Time *</label>
                <input
                  type="text"
                  value={newFinisher.finishTime}
                  onChange={(e) => setNewFinisher(prev => ({ ...prev, finishTime: e.target.value }))}
                  className="form-input"
                  placeholder="e.g., 2:30:45"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Gender</label>
                <select
                  value={newFinisher.gender}
                  onChange={(e) => setNewFinisher(prev => ({ ...prev, gender: e.target.value }))}
                  className="form-input"
                >
                  <option value="">Select...</option>
                  <option value="M">Male</option>
                  <option value="W">Female</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Team</label>
                <input
                  type="text"
                  value={newFinisher.team}
                  onChange={(e) => setNewFinisher(prev => ({ ...prev, team: e.target.value }))}
                  className="form-input"
                  placeholder="e.g., Team Phoenix"
                />
              </div>
              <div className="flex items-end gap-2">
                <button onClick={handleAddFinisher} className="btn-primary">
                  <span className="material-icon">save</span>
                  Save
                </button>
                <button onClick={() => setShowAddForm(false)} className="btn-ghost">
                  <span className="material-icon">close</span>
                  Cancel
                </button>
              </div>
            </div>
            <p className="text-sm text-muted-foreground">* Required fields</p>
          </div>
        )}

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
            emptyMessage="No finishers added yet. Add the first finisher to get started."
          />
        </div>
      </div>
    </Layout>
  );
}
