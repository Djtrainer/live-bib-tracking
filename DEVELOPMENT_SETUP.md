# Hybrid Development Setup

This document describes the hybrid development setup for the Live Bib Tracking application, where the frontend runs in a Docker container and the backend runs natively on macOS.

## Overview

- **Frontend**: React/Vite application running in Docker container (port 5173)
- **Backend**: Python FastAPI application running natively on macOS (port 8001)

## Why Hybrid Setup?

This hybrid approach is designed to work around camera hardware access issues:
- The frontend runs in Docker for consistent development environment
- The backend runs natively to ensure proper camera access on macOS

## Quick Start

1. **Prerequisites**:
   - Docker Desktop installed and running
   - Python 3.10+ with required dependencies (see `requirements.txt`)
   - Camera permissions granted to Terminal/VS Code

2. **Launch the application**:
   ```bash
   ./start-dev.sh
   ```

This single command will:
- Start the frontend container with Docker Compose
- Launch the backend natively using the existing `run_live_native.sh` script
- Set up proper networking between container and host

## Architecture

### Frontend Container
- **Image**: Node.js 18 Alpine
- **Port**: 5173 (mapped to host)
- **Volume**: `./src/frontend` mounted for live reloading
- **Environment Variables**:
  - `VITE_API_BASE_URL=http://host.docker.internal:8001`
  - `VITE_WS_BASE_URL=ws://host.docker.internal:8001`

### Backend (Native)
- **Runtime**: Native Python on macOS
- **Port**: 8001
- **Camera Access**: Direct hardware access
- **Script**: Uses existing `run_live_native.sh`

### Container-to-Host Communication
The frontend container communicates with the native backend using:
- `host.docker.internal:8001` for API calls
- `host.docker.internal:8001` for WebSocket connections

This is configured via the `extra_hosts` setting in `docker-compose.yml`.

## Files Created/Modified

### New Files
- `src/frontend/Dockerfile` - Frontend container definition
- `docker-compose.yml` - Frontend service configuration
- `start-dev.sh` - Main launcher script
- `src/frontend/.env` - Environment variables for frontend
- `DEVELOPMENT_SETUP.md` - This documentation

### Modified Files
- `src/frontend/vite.config.ts` - Updated to use environment variables for backend URLs

## Manual Setup (Alternative)

If you prefer to run components separately:

### 1. Start Frontend Container
```bash
docker-compose up -d --build
```

### 2. Start Backend Natively
```bash
./run_live_native.sh
```

### 3. Stop Everything
```bash
docker-compose down
# Stop backend with Ctrl+C
```

## Accessing the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8001
- **Backend WebSocket**: ws://localhost:8001/ws

## Troubleshooting

### Port Conflicts
If ports 5173 or 8001 are in use:
```bash
# Check what's using the ports
lsof -i :5173
lsof -i :8001

# Kill processes if needed
kill -9 <PID>
```

### Docker Issues
```bash
# Check Docker status
docker info

# View container logs
docker-compose logs

# Rebuild containers
docker-compose up --build --force-recreate
```

### Camera Access Issues
Ensure camera permissions are granted:
1. System Preferences > Security & Privacy > Camera
2. Grant access to Terminal or VS Code (whichever you're using)

### Environment Variables Not Working
Check that the `.env` file exists in `src/frontend/` and contains:
```
VITE_API_BASE_URL=http://host.docker.internal:8001
VITE_WS_BASE_URL=ws://host.docker.internal:8001
```

## Development Workflow

1. **Start Development**: `./start-dev.sh`
2. **Frontend Changes**: Edit files in `src/frontend/` - changes auto-reload
3. **Backend Changes**: Backend will restart automatically via the native script
4. **Stop Development**: Press `Ctrl+C` - both frontend and backend will stop

## Production Deployment

This setup is for development only. For production, use the existing full Docker setup in the `docker/` directory.
