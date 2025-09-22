#!/bin/bash

# Live Bib Tracking - Hybrid Development Launcher
# This script launches the frontend in Docker and the backend natively on macOS

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default camera index (can be overridden)
CAMERA_INDEX=${CAMERA_INDEX:-1}
VIDEO_PATH=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--camera)
            CAMERA_INDEX="$2"
            shift 2
            ;;
        -v|--video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --camera     Camera index (0=built-in, 1=external/iPhone)"
            echo "  -v, --video      Path to video file for testing (overrides camera)"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0               # Use external camera (default)"
            echo "  $0 -c 0          # Use built-in camera"
            echo "  $0 -c 1          # Use external camera/iPhone"
            echo "  $0 -v test_race.mp4  # Process video file"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🚀 Live Bib Tracking - Hybrid Development Setup${NC}"
echo "=============================================================="
echo -e "${YELLOW}Frontend: Docker Container (port 5173)${NC}"
echo -e "${YELLOW}Backend:  Native macOS (port 8001)${NC}"

if [[ -n "$VIDEO_PATH" ]]; then
    echo -e "${YELLOW}Input:    Video file${NC}"
    echo -e "${YELLOW}          $VIDEO_PATH${NC}"
else
    echo -e "${YELLOW}Input:    Live camera (Index $CAMERA_INDEX)${NC}"
    if [[ "$CAMERA_INDEX" == "0" ]]; then
        echo -e "${YELLOW}          (Built-in MacBook camera)${NC}"
    elif [[ "$CAMERA_INDEX" == "1" ]]; then
        echo -e "${YELLOW}          (External camera/iPhone)${NC}"
    fi
fi
echo ""

# Function to cleanup manually (not automatic)
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up...${NC}"
    echo -e "${BLUE}Stopping frontend container...${NC}"
    docker compose down
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Note: No automatic cleanup trap - containers will remain running

# Function to check if Docker is running
check_docker() {
    echo -e "${YELLOW}🐳 Checking Docker...${NC}"
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running${NC}"
        echo -e "${BLUE}💡 Please start Docker Desktop and try again${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker is running${NC}"
}

# Function to check if ports are available
check_ports() {
    echo -e "${YELLOW}🔍 Checking port availability...${NC}"
    
    # Check port 5173 (frontend)
    if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}❌ Port 5173 is already in use${NC}"
        echo -e "${BLUE}💡 Please stop any services using port 5173${NC}"
        exit 1
    fi
    
    # Check port 8001 (backend)
    if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}❌ Port 8001 is already in use${NC}"
        echo -e "${BLUE}💡 Please stop any services using port 8001${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Ports 5173 and 8001 are available${NC}"
}

# Function to start frontend container
start_frontend() {
    echo -e "${YELLOW}🎨 Starting frontend container...${NC}"
    echo -e "${BLUE}Building and starting frontend with Docker Compose...${NC}"
    
    # Build and start the frontend container in detached mode
    docker compose up -d --build
    
    # Wait a moment for the container to start
    sleep 3
    
    # Check if container is running
    if docker compose ps | grep -q "Up"; then
        echo -e "${GREEN}✅ Frontend container started successfully${NC}"
        echo -e "${BLUE}🌐 Frontend available at: http://localhost:5173${NC}"
    else
        echo -e "${RED}❌ Failed to start frontend container${NC}"
        docker compose logs
        exit 1
    fi
}

# Function to start backend natively
start_backend() {
    echo -e "${YELLOW}🐍 Starting backend natively...${NC}"
    echo -e "${BLUE}Running native backend script in background...${NC}"
    
    # Check if the native backend script exists
    if [[ ! -f "run_live_native.sh" ]]; then
        echo -e "${RED}❌ Backend script 'run_live_native.sh' not found${NC}"
        exit 1
    fi
    
    # Make sure the script is executable
    chmod +x run_live_native.sh
    
    # Run the native backend script in background
    echo -e "${BLUE}🌐 Backend will be available at: http://localhost:8001${NC}"
    echo -e "${YELLOW}💡 Backend will run in the background${NC}"
    echo ""
    
    # Execute the backend script with appropriate arguments in background
    if [[ -n "$VIDEO_PATH" ]]; then
        nohup ./run_live_native.sh -v "$VIDEO_PATH" > backend.log 2>&1 &
    else
        nohup ./run_live_native.sh -c "$CAMERA_INDEX" > backend.log 2>&1 &
    fi
    
    # Store the PID for reference
    BACKEND_PID=$!
    echo -e "${GREEN}✅ Backend started with PID: $BACKEND_PID${NC}"
    
    # Wait a moment to check if backend started successfully
    sleep 3
    
    # Check if the process is still running
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${GREEN}✅ Backend is running successfully${NC}"
    else
        echo -e "${RED}❌ Backend failed to start. Check backend.log for details${NC}"
        exit 1
    fi
}

# Function to show status
show_status() {
    echo -e "${GREEN}🎉 Development environment is ready!${NC}"
    echo ""
    echo -e "${BLUE}📱 Frontend (React):${NC} http://localhost:5173"
    echo -e "${BLUE}🔧 Backend (Python):${NC} http://localhost:8001"
    echo ""
    echo -e "${YELLOW}The frontend container can communicate with the backend via:${NC}"
    echo -e "${YELLOW}  - API calls: http://host.docker.internal:8001${NC}"
    echo -e "${YELLOW}  - WebSocket: ws://host.docker.internal:8001${NC}"
    echo ""
}

# Main execution
echo -e "${BLUE}🔍 Running pre-flight checks...${NC}"
echo ""

check_docker
echo ""

check_ports
echo ""

start_frontend
echo ""

show_status

# Start backend (runs in background)
start_backend

# Final status and instructions
echo ""
echo -e "${GREEN}🎉 Both services are now running in the background!${NC}"
echo ""
echo -e "${BLUE}📱 Frontend:${NC} http://localhost:5173 (Docker container)"
echo -e "${BLUE}🔧 Backend:${NC} http://localhost:8001 (Native process, PID: $BACKEND_PID)"
echo ""
echo -e "${YELLOW}📋 Management Commands:${NC}"
echo -e "${BLUE}  Check frontend status:${NC} docker compose ps"
echo -e "${BLUE}  Stop frontend:${NC} docker compose down"
echo -e "${BLUE}  Check backend logs:${NC} tail -f backend.log"
echo -e "${BLUE}  Stop backend:${NC} kill $BACKEND_PID"
echo -e "${BLUE}  Stop both:${NC} docker compose down && kill $BACKEND_PID"
echo ""
echo -e "${YELLOW}💡 Backend logs are saved to: backend.log${NC}"
