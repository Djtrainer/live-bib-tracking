#!/bin/bash

# Build the Docker image
docker build -f docker/Dockerfile -t live-bib-tracking-fullstack .

# Run with virtual display (headless mode)
# This creates a virtual display inside the container so OpenCV can work
# but doesn't require a physical display
docker run --rm \
  -v /Users/dantrainer/projects/live-bib-tracking/data:/app/data \
  -v /Users/dantrainer/projects/live-bib-tracking/runs/detect/train2/weights:/app/models \
  live-bib-tracking-fullstack \
  bash -c "
    # Start virtual display
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    export DISPLAY=:99
    
    # Run the application
    python src/image_processor/video_inference.py \
      --video /app/data/raw/2024_race.MOV \
      --model /app/models/last.pt
  "

# This allows OpenCV to create windows (which won't be visible)
# but prevents the Qt/X11 errors you were seeing
