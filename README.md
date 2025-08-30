# Live Bib Tracking - Monorepo Setup

This document explains the new monorepo structure and how to run the application in different modes.

## Project Structure

```
live-bib-tracking/
├── src/
│   ├── frontend/          # React frontend application
│   │   ├── src/
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   └── dist/          # Built frontend (created during build)
│   ├── api_backend/       # FastAPI backend server
│   │   └── local_server.py
│   └── image_processor/   # Computer vision processing
│       └── video_inference.py
├── docker/
│   └── Dockerfile         # Multi-stage build for production
├── config/                # Configuration files
├── data/                  # Video and model data
└── runs/                  # Training outputs
```

## Development Mode

For local development with hot reloading:

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or bun

### Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install frontend dependencies:
   ```bash
   cd src/frontend
   npm install
   cd ../..
   ```

### Running in Development
```bash
./dev_start.sh
```

This will start:
- **Backend**: http://localhost:8000 (FastAPI with auto-reload)
- **Frontend**: http://localhost:8080 (Vite dev server with hot reload)

The Vite dev server automatically proxies API requests to the backend, so CORS is handled seamlessly.

## Production Mode (Docker)

For production deployment with a single container:

### Build and Run Full-Stack Container
```bash
./build_and_run_fullstack.sh
```

This creates a single container that:
- Builds the React frontend into static files
- Serves the frontend and API from the same FastAPI server
- Available at: http://localhost:8000

### Multi-Stage Build Process
1. **Frontend Build Stage**: Uses Node.js to build React app
2. **Python Build Stage**: Installs Python dependencies
3. **Runtime Stage**: Combines built frontend + Python backend

## Configuration Changes Made

### 1. Vite Configuration (`src/frontend/vite.config.ts`)
- Added proxy configuration for `/api` and `/ws` routes
- Set build output to `dist` directory
- Enables seamless development with backend

### 2. FastAPI Server (`src/api_backend/local_server.py`)
- Auto-detects development vs production mode
- Serves static files from correct path in both modes
- Maintains all existing API endpoints

### 3. React Components
- Updated all API calls to use relative paths (`/api/*`)
- WebSocket connections use dynamic host detection
- Works in both development and production

### 4. Docker Configuration
- Multi-stage build for optimal image size
- Builds frontend during Docker build process
- Single container serves both frontend and backend

## Available Scripts

| Script | Purpose | URL |
|--------|---------|-----|
| `./dev_start.sh` | Development mode | Frontend: :8080, Backend: :8000 |
| `./build_and_run_fullstack.sh` | Production mode | http://localhost:8000 |
| `./build_and_run.sh` | Image processing only | N/A (no display) |
| `./build_and_run_vnc.sh` | Image processing with VNC | vnc://localhost:5900 |

## API Endpoints

All API endpoints remain the same:
- `GET /api/results` - Get race results
- `POST /api/results` - Add new finisher
- `PUT /api/results/{id}` - Update finisher
- `DELETE /api/results/{id}` - Delete finisher
- `WebSocket /ws` - Real-time updates

## Frontend Routes

- `/` - Live leaderboard (public view)
- `/admin` - Admin login
- `/admin/dashboard` - Admin management interface

## Environment Variables

- `PYTHONPATH` - Set to `/app/src` in Docker
- `YOLO_AUTOINSTALL` - Set to `false` to prevent permission issues
- `EASYOCR_MODULE_PATH` - Cache directory for EasyOCR

## Troubleshooting

### Development Issues
- **Port conflicts**: Make sure ports 8000 and 8080 are available
- **API not connecting**: Check that backend started successfully
- **Frontend not loading**: Ensure `npm install` completed in `src/frontend`

### Production Issues
- **Build failures**: Check Docker build logs for missing dependencies
- **Static files not found**: Verify frontend build completed successfully
- **API not responding**: Check container logs with `docker logs <container_id>`

### Common Commands
```bash
# Check running containers
docker ps

# View container logs
docker logs <container_name>

# Rebuild without cache
docker build --no-cache -f docker/Dockerfile -t live-bib-tracking-fullstack .

# Clean up Docker images
docker system prune -a
```

## Migration Notes

The new structure maintains backward compatibility:
- All existing API endpoints work unchanged
- Video processing functionality is preserved
- Database schema remains the same
- WebSocket connections continue to work

The main benefits:
- Single container deployment
- Unified development workflow
- Better separation of concerns
- Easier CI/CD pipeline setup
