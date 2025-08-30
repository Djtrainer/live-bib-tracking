#!/bin/bash

# Build the Docker image
docker build -f docker/Dockerfile -t live-bib-tracking .

echo "Starting container with VNC server..."
echo "The application will be accessible via VNC on port 5900"
echo ""
echo "To view the GUI:"
echo "1. Wait for the container to fully start (about 10-15 seconds)"
echo "2. Open Finder and press Cmd+K (or Go -> Connect to Server)"
echo "3. Enter: vnc://localhost:5900"
echo "4. Click Connect and use the built-in Screen Sharing app"
echo ""
echo "The video processing window will appear in the VNC viewer."
echo "Press Ctrl+C to stop the container."
echo ""

# Run the container with VNC port exposed
docker run --rm \
  -p 5900:5900 \
  -v /Users/dantrainer/projects/live-bib-tracking/data:/app/data \
  -v /Users/dantrainer/projects/live-bib-tracking/runs/detect/train2/weights:/app/models \
  live-bib-tracking
