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

echo -e "${BLUE}🚀 Live Bib Tracking - Hybrid Development Setup${NC}"
echo "=============================================================="
echo -e "${YELLOW}Frontend: Docker Container (port 5173)${NC}"
echo -e "${YELLOW}Backend:  Native macOS (port 8001)${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up...${NC}"
    echo -e "${BLUE}Stopping frontend container...${NC}"
    docker-compose down
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

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
    docker-compose up -d --build
    
    # Wait a moment for the container to start
    sleep 3
    
    # Check if container is running
    if docker-compose ps | grep -q "Up"; then
        echo -e "${GREEN}✅ Frontend container started successfully${NC}"
        echo -e "${BLUE}🌐 Frontend available at: http://localhost:5173${NC}"
    else
        echo -e "${RED}❌ Failed to start frontend container${NC}"
        docker-compose logs
        exit 1
    fi
}

# Function to start backend natively
start_backend() {
    echo -e "${YELLOW}🐍 Starting backend natively...${NC}"
    echo -e "${BLUE}Running native backend script...${NC}"
    
    # Check if the native backend script exists
    if [[ ! -f "run_live_native.sh" ]]; then
        echo -e "${RED}❌ Backend script 'run_live_native.sh' not found${NC}"
        exit 1
    fi
    
    # Make sure the script is executable
    chmod +x run_live_native.sh
    
    # Run the native backend script
    echo -e "${BLUE}🌐 Backend will be available at: http://localhost:8001${NC}"
    echo -e "${YELLOW}💡 Press Ctrl+C to stop both frontend and backend${NC}"
    echo ""
    
    # Execute the backend script (this will block until stopped)
    ./run_live_native.sh
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

# Start backend (this will block)
start_backend
