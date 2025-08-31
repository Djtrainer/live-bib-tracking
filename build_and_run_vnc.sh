#!/bin/bash

# Build the Docker image
docker build -f docker/Dockerfile -t live-bib-tracking-fullstack .

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

# Run the container with VNC server and custom command
docker run --rm \
  -p 5900:5900 \
  -v /Users/dantrainer/projects/live-bib-tracking/data:/app/data \
  -v /Users/dantrainer/projects/live-bib-tracking/runs/detect/train2/weights:/app/models \
  live-bib-tracking-fullstack \
  sh -c "
    echo 'Starting VNC server and desktop environment...'
    Xvfb :0 -screen 0 1280x720x16 & 
    sleep 2
    fluxbox & 
    sleep 2
    x11vnc -display :0 -forever -create -nopw & 
    sleep 2
    echo 'VNC server started. Connect to vnc://localhost:5900'
    echo 'Starting video processing application...'
    DISPLAY=:0 python src/image_processor/video_inference.py \
      --video /app/data/raw/2024_race.MOV \
      --model /app/models/last.pt \
      --fps 8 \
      --conf 0.3
  "
