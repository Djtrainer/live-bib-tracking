#!/bin/bash

# Build and run the full-stack application (frontend + backend) in Docker
echo "Building full-stack Live Bib Tracking application..."

# Build the Docker image with multi-stage build
docker build -f docker/Dockerfile -t live-bib-tracking-fullstack .

echo ""
echo "Starting full-stack container..."
echo "The application will be available at: http://localhost:8000"
echo ""
echo "Services included:"
echo "- Frontend (React): Served at /"
echo "- Backend API: Available at /api/*"
echo "- WebSocket: Available at /ws"
echo ""
echo "Press Ctrl+C to stop the container."
echo ""

# Run the container with the FastAPI server serving both frontend and backend
docker run --rm \
  -p 8000:8000 \
  -v /Users/dantrainer/projects/live-bib-tracking/data:/app/data \
  -v /Users/dantrainer/projects/live-bib-tracking/runs/detect/train2/weights:/app/models \
  live-bib-tracking-fullstack
