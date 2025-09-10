#!/bin/bash

echo "Simple VNC Test - Minimal setup to verify VNC works"

# Build the Docker image
docker build -f docker/Dockerfile -t live-bib-tracking-fullstack .

echo ""
echo "Starting minimal VNC test..."
echo "Connect to vnc://localhost:5900"
echo "You should see a gray desktop background"
echo "Press Ctrl+C to stop the test."
echo ""

# Run a simple VNC test
docker run --rm \
  -p 5900:5900 \
  --user root \
  --shm-size=1g \
  live-bib-tracking-fullstack \
  sh -c "
    echo 'Starting minimal VNC setup...'
    
    # Create directories
    mkdir -p /tmp/.X11-unix
    chmod 1777 /tmp/.X11-unix
    
    # Start X server
    echo 'Starting X virtual framebuffer...'
    Xvfb :1 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
    XVFB_PID=\$!
    sleep 5
    
    export DISPLAY=:1
    
    # Start VNC server (foreground mode for easier debugging)
    echo 'Starting VNC server...'
    echo 'VNC server will start in foreground mode'
    echo 'Connect to vnc://localhost:5900 now'
    echo ''
    
    x11vnc -display :1 \
           -forever \
           -nopw \
           -listen 0.0.0.0 \
           -rfbport 5900 \
           -shared \
           -cursor arrow \
           -noxdamage \
           -noxfixes
  "
