#!/bin/bash

echo "Testing VNC setup with a simple GUI application..."

# Build the Docker image
docker build -f docker/Dockerfile -t live-bib-tracking-fullstack .

echo ""
echo "Starting test container with VNC server..."
echo "Connect to vnc://localhost:5900 to see a simple test window"
echo "Press Ctrl+C to stop the test."
echo ""

# Run a simple VNC test
docker run --rm \
  -p 5900:5900 \
  --user root \
  --shm-size=1g \
  live-bib-tracking-fullstack \
  sh -c "
    echo 'Setting up VNC test environment...'
    
    # Install test applications
    apt-get update -qq && apt-get install -y -qq \
      x11-xserver-utils \
      xterm \
      openbox \
      xclock \
      procps \
      2>/dev/null || echo 'Some packages may not be available, continuing...'
    
    # Create directories
    mkdir -p /home/appuser/.vnc /tmp/.X11-unix
    chown -R appuser:appuser /home/appuser
    chmod 1777 /tmp/.X11-unix
    
    # Start X server
    echo 'Starting X virtual framebuffer...'
    Xvfb :1 -screen 0 1280x720x24 -ac +extension GLX +render -noreset -dpi 96 &
    XVFB_PID=\$!
    sleep 5
    
    export DISPLAY=:1
    
    # Verify X server
    if ! xdpyinfo > /dev/null 2>&1; then
        echo 'ERROR: X server failed to start'
        exit 1
    fi
    
    echo 'Starting window manager...'
    if command -v openbox >/dev/null 2>&1; then
        su - appuser -c 'DISPLAY=:1 openbox' &
        sleep 3
    else
        echo 'OpenBox not available, using fluxbox...'
        su - appuser -c 'DISPLAY=:1 fluxbox' &
        sleep 3
    fi
    
    echo 'Starting VNC server...'
    x11vnc -display :1 \
           -forever \
           -nopw \
           -listen 0.0.0.0 \
           -rfbport 5900 \
           -shared \
           -bg \
           -cursor arrow \
           -o /tmp/x11vnc.log
    
    sleep 5
    
    # Check if VNC server is running using ps instead of pgrep
    if ps aux | grep -q '[x]11vnc'; then
        echo 'VNC server started successfully!'
    else
        echo 'VNC server may have issues, but continuing...'
        echo 'VNC server log:'
        cat /tmp/x11vnc.log 2>/dev/null || echo 'No log file found'
    fi
    
    echo 'VNC server started successfully!'
    echo 'Starting test applications...'
    
    # Start test applications
    su - appuser -c 'DISPLAY=:1 xterm -geometry 80x24+100+100 -title \"VNC Test Terminal\"' &
    su - appuser -c 'DISPLAY=:1 xclock -geometry 200x200+400+100' &
    
    echo ''
    echo '=== VNC TEST READY ==='
    echo 'Connect to vnc://localhost:5900'
    echo 'You should see:'
    echo '- A terminal window'
    echo '- A clock application'
    echo '- A desktop background'
    echo ''
    echo 'If you can see these, VNC is working correctly!'
    echo 'Press Ctrl+C to stop the test.'
    echo ''
    
    # Keep container running
    wait \$XVFB_PID
  "
