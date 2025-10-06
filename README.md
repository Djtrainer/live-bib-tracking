# Live Bib Tracking - Monorepo Setup

This document explains the new monorepo structure and how to run the application in different modes.

# Live Bib Tracking

Lightweight monorepo for live bib / race-place tracking using a React frontend, FastAPI backend, and computer-vision image processing.

This README gives an up-to-date quick-start, repo layout, configuration notes, and troubleshooting steps so you can run the system locally or in Docker.

## Quick links

- Code: `src/`
- Frontend: `src/frontend`
- Backend: `src/api_backend`
- Image processing & training utils: `src/image_processor`, `yolo_utils`
- Docker compose: `docker-compose.yml`
- Helpful scripts: `start-dev.sh`, `run_live_native.sh`

## Minimal prerequisites

- macOS / Linux / Windows WSL
- Python 3.10+ (3.11 recommended)
- Node 16+ (Node 18+ recommended) and npm or yarn
- Docker & docker-compose (for containerized runs)

## Repo layout (top-level)

```
README.md
pyproject.toml         # Python project metadata
docker-compose.yml
src/
   ├─ frontend/         # React app (Vite)
   ├─ api_backend/      # FastAPI server
   └─ image_processor/  # CV inference & utilities
config/                # YAML config files
data/                  # Video, images, labels, trained weights
models/                # model outputs and artifacts
notebooks/             # analysis & helper notebooks
```

## Setup (local development)

1. Create & activate a Python virtual environment and install Python deps (uses pyproject.toml):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If the project uses a `requirements.txt` somewhere else, you can also install that; the package metadata is in `pyproject.toml`.

2. Install frontend deps and run the dev server:

```bash
cd src/frontend
npm install
npm run dev
```

3. Start the backend (FastAPI) in another terminal:

```bash
source .venv/bin/activate
python src/api_backend/local_server.py
```

There is a convenience script `start-dev.sh` that aims to wire up dev servers; use it if you prefer a single command.

## Quick: run native (local) inference

To run image/video inference locally (no Docker), use the provided script which loads models and processes video files:

```bash
./run_live_native.sh
```

Check `src/image_processor/video_inference.py` for configurable options (frame rate, model path, input source).

## Docker / Production

You can run the full stack with Docker Compose for a production-like environment. Example:

```bash
docker compose up --build
```

This will build images and start services defined in `docker-compose.yml`. The FastAPI server typically serves the built frontend in production mode on port 8000.

## Configuration

- YAML files live in `config/` (e.g. `custom_tracker.yaml`, `yolo_config.yaml`).

## Data & models

- The repo now tracks the following data folders (example contents):

   - `data/raw/`
      - Raw video files and camera dumps (examples):
         - `IMG_0066.MOV`

   - `data/processed/`
      - Processed images, labels and annotation sets used for training/evaluation:
         - `annotations/`,
         - `images/`
         - `labels/`, 
   - `data/results/`
      - Generated or manually-curated race result artifacts
- `models/` contains trained model artifacts and evaluation images (weights may be large; some model files are intentionally excluded by default).

## Common scripts

- `start-dev.sh` — helper to start frontend + backend in development (hot-reload). Verify it exists and adjust if paths differ.
- `run_live_native.sh` — runs local video inference using the image processor code (useful for live demos without Docker).
- `docker-compose.yml` — orchestrates services for local multi-container runs.

Inspect each script before running to confirm environment and paths.

## start-dev.sh — Hybrid development (frontend in Docker + backend native)

`start-dev.sh` is a convenience script that launches the frontend inside Docker (Vite) and runs the backend natively on your machine. This is helpful on macOS where camera access is easier from a native process while keeping the frontend containerized.

What it does:
- Builds and starts the frontend with `docker compose up -d --build` and exposes it on http://localhost:5173
- Runs `run_live_native.sh` in the background (native process) and binds the backend to http://localhost:8001
- Writes backend logs to `backend.log` and prints the backend PID when started

Prerequisites:
- Docker Desktop running
- Python (virtualenv recommended) and required Python packages installed

Usage examples (zsh):

```bash
# Use the default external camera (index 1)
./start-dev.sh

# Use built-in camera (index 0)
./start-dev.sh -c 0

# Process a local video file instead of camera
./start-dev.sh -v data/raw/race_clip.mp4

# Show help
./start-dev.sh -h
```

Management & troubleshooting:

- Check frontend container status:

```bash
docker compose ps
```

- Stop frontend container:

```bash
docker compose down
```

- Tail backend logs (written to `backend.log` by the script):

```bash
tail -f backend.log
```

- Stop the backend process (the script prints the PID when it starts):

```bash
kill <PID>
```

Notes:
- The frontend container communicates with the native backend using `host.docker.internal:8001` for API and WebSocket connections.
- If ports 5173 or 8001 are already in use, the script will exit with instructions to free them.

## run_live_native.sh — Native live inference (macOS)

`run_live_native.sh` runs the backend and the live/video inference natively. It includes pre-flight checks for Python, dependencies, camera or video readability, and the model file. This is the command the `start-dev.sh` script invokes for the native backend.

Default behavior and ports:
- Binds the backend server to port 8001 by default
- Uses `models/yolo11_white_bibs/weights/last.mlpackage` as the default model path (relative to repo root)

CLI options (copy-paste):

```bash
# Basic usage (uses camera index 1 and default model/port)
./run_live_native.sh

# Use built-in camera (index 0)
./run_live_native.sh -c 0

# Run against a local video file instead of camera
./run_live_native.sh -v data/raw/race_clip.mp4

# Use a custom model and port
./run_live_native.sh -m /path/to/model.pt -p 8002

# Show help
./run_live_native.sh -h
```

Environment variables accepted:
- `MODEL_PATH` — alternative to `-m`, absolute or relative path to a model file
- `CAMERA_INDEX` — alternative to `-c`, camera device index

If required Python packages are missing the script will list them and offer to install them via `pip` (or you can run `pip install -r requirements.txt` / `pip install -e .` beforehand).

Common errors & fixes:
- "Model file not found": ensure `MODEL_PATH` points to an existing `.pt` file (check `models/`)
- "Cannot open camera": try a different `CAMERA_INDEX` (0, 1, 2...) and confirm macOS camera permissions for your terminal app
- "Video file cannot open": verify the video file path and that OpenCV supports the container format

When running locally in a virtualenv, activate it first to ensure correct dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Then run the native server as shown above. The script will print checks and then run `src/api_backend/local_server.py` with the appropriate flags for `--inference_mode` (`live` or `test`).

## Troubleshooting

- Ports: Backend defaults to 8000 and frontend dev server to 5173 (Vite). If ports conflict, stop the conflicting services or change the port in Vite / FastAPI.
- Frontend not loading: confirm `npm install` succeeded and `npm run build` or `npm run dev` shows no errors.
- Backend API errors: check logs printed by `local_server.py` and any stack traces in the terminal.
- Docker build failures: look for missing system libraries, Python wheels, or permission issues in the build output.

Useful Docker commands:

```bash
docker compose ps
docker compose logs -f
docker compose up --build
docker compose down
```
