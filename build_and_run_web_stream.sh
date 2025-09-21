#!/bin/bash

# Live Bib Tracking - Web Streaming Version
# Build and Run Script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="live-bib-tracking"
CONTAINER_NAME="live-bib-tracking-web"
PORT=8001

# Default paths (can be overridden with environment variables)
VIDEO_PATH=${VIDEO_PATH:-"$(pwd)/data"}
MODEL_PATH=${MODEL_PATH:-"$(pwd)/runs"}

echo -e "${BLUE}🏃‍♂️ Live Bib Tracking - Web Streaming Version 🏁${NC}"
echo "=================================================="

# Function to print usage
print_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 [OPTIONS]"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  -h, --help     Show this help message"
    echo "  -v, --video    Path to video directory (default: ./data)"
    echo "  -m, --models   Path to models directory (default: ./runs)"
    echo "  -p, --port     Port to expose (default: 8001)"
    echo "  --build-only   Only build the Docker image, don't run"
    echo "  --run-only     Only run the container (skip build)"
    echo ""
    echo -e "${YELLOW}Environment Variables:${NC}"
    echo "  VIDEO_PATH     Path to video directory"
    echo "  MODEL_PATH     Path to models directory"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0                                    # Use default paths"
    echo "  $0 -v /path/to/videos -m /path/to/models"
    echo "  VIDEO_PATH=/data MODEL_PATH=/models $0"
}

# Parse command line arguments
BUILD_ONLY=false
RUN_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -v|--video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        -m|--models)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --run-only)
            RUN_ONLY=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Convert relative paths to absolute paths
VIDEO_PATH=$(realpath "$VIDEO_PATH" 2>/dev/null || echo "$VIDEO_PATH")
MODEL_PATH=$(realpath "$MODEL_PATH" 2>/dev/null || echo "$MODEL_PATH")

echo -e "${BLUE}Configuration:${NC}"
echo "  Image Name: $IMAGE_NAME"
echo "  Container Name: $CONTAINER_NAME"
echo "  Port: $PORT"
echo "  Video Path: $VIDEO_PATH"
echo "  Model Path: $MODEL_PATH"
echo ""

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
        exit 1
    fi
}

# Function to stop and remove existing container
cleanup_container() {
    if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}🧹 Stopping and removing existing container...${NC}"
        docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
        echo -e "${GREEN}✅ Cleanup completed${NC}"
    fi
}

# Function to build Docker image
build_image() {
    echo -e "${YELLOW}🔨 Building Docker image...${NC}"
    if docker build -f docker/Dockerfile -t "$IMAGE_NAME" .; then
        echo -e "${GREEN}✅ Docker image built successfully${NC}"
    else
        echo -e "${RED}❌ Failed to build Docker image${NC}"
        exit 1
    fi
}

# Function to run container
run_container() {
    echo -e "${YELLOW}🚀 Starting container...${NC}"
    
    # Check if directories exist
    if [[ ! -d "$VIDEO_PATH" ]]; then
        echo -e "${YELLOW}⚠️  Video directory does not exist: $VIDEO_PATH${NC}"
        echo -e "${YELLOW}   Creating directory...${NC}"
        mkdir -p "$VIDEO_PATH"
    fi
    
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo -e "${YELLOW}⚠️  Model directory does not exist: $MODEL_PATH${NC}"
        echo -e "${YELLOW}   Creating directory...${NC}"
        mkdir -p "$MODEL_PATH"
    fi
    
    # Run the container with proper arguments for video processing
    if docker run -d \
        --name "$CONTAINER_NAME" \
        -p "$PORT:8001" \
        -v "$VIDEO_PATH:/app/data" \
        -v "$MODEL_PATH:/app/runs" \
        "$IMAGE_NAME" \
        --video /app/data/raw/race_1080p.mp4 \
        --model /app/runs/detect/yolo11_white_bibs/weights/last.pt; then
        
        echo -e "${GREEN}✅ Container started successfully${NC}"
        echo ""
        echo -e "${BLUE}🌐 Access the web interface at:${NC}"
        echo -e "${GREEN}   http://localhost:$PORT${NC}"
        echo ""
        echo -e "${BLUE}📋 Container Management:${NC}"
        echo "   View logs:    docker logs $CONTAINER_NAME"
        echo "   Stop:         docker stop $CONTAINER_NAME"
        echo "   Remove:       docker rm $CONTAINER_NAME"
        echo ""
        
        # Wait a moment and check if container is still running
        sleep 2
        if docker ps --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo -e "${GREEN}✅ Container is running successfully${NC}"
            
            # Show recent logs
            echo -e "${BLUE}📝 Recent logs:${NC}"
            docker logs --tail 10 "$CONTAINER_NAME"
        else
            echo -e "${RED}❌ Container failed to start. Check logs:${NC}"
            docker logs "$CONTAINER_NAME"
            exit 1
        fi
    else
        echo -e "${RED}❌ Failed to start container${NC}"
        exit 1
    fi
}

# Main execution
check_docker

if [[ "$RUN_ONLY" == "true" ]]; then
    echo -e "${BLUE}🏃 Run-only mode${NC}"
    cleanup_container
    run_container
elif [[ "$BUILD_ONLY" == "true" ]]; then
    echo -e "${BLUE}🔨 Build-only mode${NC}"
    build_image
else
    echo -e "${BLUE}🔄 Build and run mode${NC}"
    cleanup_container
    build_image
    run_container
fi

echo ""
echo -e "${GREEN}🎉 Done!${NC}"
