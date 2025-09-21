#!/bin/bash

# Live Bib Tracking - Live Camera Feed Version
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
CONTAINER_NAME="live-bib-tracking-live"
PORT=8001

# Default paths (can be overridden with environment variables)
MODEL_PATH=${MODEL_PATH:-"$(pwd)/runs"}
CAMERA_INDEX=${CAMERA_INDEX:-1}

echo -e "${BLUE}🏃‍♂️ Live Bib Tracking - Live Camera Feed Version 📹${NC}"
echo "======================================================="

# macOS users note
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}🍎 macOS Users:${NC}"
    echo -e "${BLUE}   Docker camera access on macOS can be challenging.${NC}"
    echo -e "${BLUE}   If you encounter camera issues, try the native runner:${NC}"
    echo -e "${GREEN}   ./run_live_native.sh -c $CAMERA_INDEX${NC}"
    echo ""
fi

# Function to print usage
print_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 [OPTIONS]"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  -h, --help       Show this help message"
    echo "  -m, --models     Path to models directory (default: ./runs)"
    echo "  -c, --camera     Camera index (default: 1, iPhone; 0 for built-in)"
    echo "  -p, --port       Port to expose (default: 8001)"
    echo "  --build-only     Only build the Docker image, don't run"
    echo "  --run-only       Only run the container (skip build)"
    echo ""
    echo -e "${YELLOW}Environment Variables:${NC}"
    echo "  MODEL_PATH       Path to models directory"
    echo "  CAMERA_INDEX     Camera index to use (0=built-in, 1=iPhone, etc.)"
    echo ""
    echo -e "${YELLOW}Camera Index Guide:${NC}"
    echo "  0 - Built-in camera (laptop/desktop webcam)"
    echo "  1 - External camera (iPhone via USB/wireless)"
    echo "  2+ - Additional cameras if available"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0                                    # Use iPhone camera (index 1)"
    echo "  $0 -c 0                              # Use built-in camera"
    echo "  $0 -m /path/to/models -c 1           # Custom model path with iPhone"
    echo "  CAMERA_INDEX=0 MODEL_PATH=/models $0 # Environment variables"
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
        -m|--models)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c|--camera)
            CAMERA_INDEX="$2"
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

# Validate camera index
if ! [[ "$CAMERA_INDEX" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}❌ Camera index must be a number (0, 1, 2, etc.)${NC}"
    exit 1
fi

# Convert relative paths to absolute paths
MODEL_PATH=$(realpath "$MODEL_PATH" 2>/dev/null || echo "$MODEL_PATH")

echo -e "${BLUE}Configuration:${NC}"
echo "  Image Name: $IMAGE_NAME"
echo "  Container Name: $CONTAINER_NAME"
echo "  Port: $PORT"
echo "  Model Path: $MODEL_PATH"
echo "  Camera Index: $CAMERA_INDEX"
if [[ "$CAMERA_INDEX" == "0" ]]; then
    echo "  Camera Type: Built-in camera"
elif [[ "$CAMERA_INDEX" == "1" ]]; then
    echo "  Camera Type: External camera (iPhone/USB)"
else
    echo "  Camera Type: Camera device $CAMERA_INDEX"
fi
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

# Function to check camera availability (best effort)
check_camera() {
    echo -e "${YELLOW}📹 Checking camera availability...${NC}"
    echo -e "${BLUE}ℹ️  Note: Camera access will be tested when the container starts${NC}"
    
    # On macOS, we can check if camera permissions might be needed
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${YELLOW}⚠️  macOS detected: Ensure camera permissions are granted to Docker/Terminal${NC}"
        echo -e "${BLUE}   Go to System Preferences > Security & Privacy > Camera${NC}"
    fi
}

# Function to run container
run_container() {
    echo -e "${YELLOW}🚀 Starting container with live camera feed...${NC}"
    
    # Check if model directory exists
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo -e "${YELLOW}⚠️  Model directory does not exist: $MODEL_PATH${NC}"
        echo -e "${YELLOW}   Creating directory...${NC}"
        mkdir -p "$MODEL_PATH"
    fi
    
    # Check if the specific model file exists
    MODEL_FILE="$MODEL_PATH/detect/yolo11_white_bibs/weights/last.pt"
    if [[ -f "$MODEL_FILE" ]]; then
        echo -e "${GREEN}✅ Model file found: $MODEL_FILE${NC}"
        MODEL_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null || echo "0")
        MODEL_SIZE_MB=$((MODEL_SIZE / 1024 / 1024))
        echo -e "${GREEN}   Model size: ${MODEL_SIZE_MB}MB${NC}"
    else
        echo -e "${RED}❌ Model file not found: $MODEL_FILE${NC}"
        echo -e "${YELLOW}💡 Available files in model directory:${NC}"
        find "$MODEL_PATH" -name "*.pt" 2>/dev/null | head -10 || echo "   No .pt files found"
        echo -e "${BLUE}   Make sure you have trained a model first${NC}"
        exit 1
    fi
    
    check_camera
    
    # Run the container with proper arguments for live camera processing
    echo -e "${BLUE}🔧 Configuring camera access for Docker container...${NC}"
    
    # Platform-specific camera configuration
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS specific configuration
        echo -e "${BLUE}🍎 macOS detected - configuring for macOS camera access${NC}"
        echo -e "${YELLOW}⚠️  Important: macOS camera access in Docker has limitations${NC}"
        echo -e "${BLUE}   For best results, consider running directly on macOS without Docker${NC}"
        
        # macOS Docker camera access is limited, but we'll try our best
        CAMERA_ARGS="--privileged"
        EXTRA_ARGS="--security-opt seccomp=unconfined --cap-add=ALL"
        
        # Don't use host network as it causes port issues
        NETWORK_ARGS="-p $PORT:8001"
        
        echo -e "${YELLOW}💡 macOS Camera Access Tips:${NC}"
        echo -e "${BLUE}   1. Ensure Docker Desktop has camera permissions${NC}"
        echo -e "${BLUE}   2. Go to System Preferences > Security & Privacy > Camera${NC}"
        echo -e "${BLUE}   3. Add Docker Desktop to allowed apps${NC}"
        echo -e "${BLUE}   4. If issues persist, try running without Docker:${NC}"
        echo -e "${BLUE}      python src/image_processor/video_inference.py --inference_mode live -c $CAMERA_INDEX${NC}"
        
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux specific configuration
        echo -e "${BLUE}🐧 Linux detected - configuring video device access${NC}"
        
        # Check if video devices exist
        if ls /dev/video* >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Found video devices: $(ls /dev/video* | tr '\n' ' ')${NC}"
            CAMERA_ARGS="--device=/dev/video0 --device=/dev/video1 --device=/dev/video2 --device=/dev/video3"
        else
            echo -e "${YELLOW}⚠️  No /dev/video* devices found${NC}"
            CAMERA_ARGS="--privileged -v /dev:/dev"
        fi
        
        EXTRA_ARGS="--group-add video --device-cgroup-rule='c 81:* rmw'"
        NETWORK_ARGS="-p $PORT:8001"
        
    else
        # Other platforms (Windows, etc.)
        echo -e "${BLUE}🖥️  Other platform detected - using privileged mode${NC}"
        CAMERA_ARGS="--privileged -v /dev:/dev"
        EXTRA_ARGS=""
        NETWORK_ARGS="-p $PORT:8001"
    fi
    
    echo -e "${BLUE}🚀 Starting Docker container with camera access...${NC}"
    if docker run -d \
        --name "$CONTAINER_NAME" \
        $CAMERA_ARGS \
        $EXTRA_ARGS \
        $NETWORK_ARGS \
        -v "$MODEL_PATH:/app/runs" \
        "$IMAGE_NAME" \
        --inference_mode live \
        --camera_index "$CAMERA_INDEX" \
        --model /app/runs/detect/yolo11_white_bibs/weights/last.pt; then
        
        echo -e "${GREEN}✅ Container started successfully${NC}"
        echo ""
        echo -e "${BLUE}🌐 Access the live camera feed at:${NC}"
        echo -e "${GREEN}   http://localhost:$PORT${NC}"
        echo ""
        echo -e "${BLUE}📋 Container Management:${NC}"
        echo "   View logs:    docker logs $CONTAINER_NAME"
        echo "   Follow logs:  docker logs -f $CONTAINER_NAME"
        echo "   Stop:         docker stop $CONTAINER_NAME"
        echo "   Remove:       docker rm $CONTAINER_NAME"
        echo ""
        
        # Wait a moment and check if container is still running
        sleep 3
        if docker ps --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo -e "${GREEN}✅ Container is running successfully${NC}"
            
            # Show recent logs
            echo -e "${BLUE}📝 Recent logs:${NC}"
            docker logs --tail 15 "$CONTAINER_NAME"
            
            echo ""
            echo -e "${YELLOW}💡 Troubleshooting Tips:${NC}"
            echo "   • If camera access fails, check Docker has camera permissions"
            echo "   • Try different camera indices (0, 1, 2) if current one doesn't work"
            echo "   • On macOS: System Preferences > Security & Privacy > Camera"
            echo "   • On Linux: Ensure user is in 'video' group"
            echo "   • Check logs with: docker logs -f $CONTAINER_NAME"
        else
            echo -e "${RED}❌ Container failed to start. Check logs:${NC}"
            docker logs "$CONTAINER_NAME"
            echo ""
            echo -e "${YELLOW}💡 Common issues:${NC}"
            echo "   • Camera not accessible (try different camera index)"
            echo "   • Model file not found (check model path)"
            echo "   • Permissions issue (Docker needs camera access)"
            echo ""
            if [[ "$OSTYPE" == "darwin"* ]]; then
                echo -e "${BLUE}🍎 macOS Alternative Solution:${NC}"
                echo -e "${GREEN}   Try running natively without Docker for better camera access:${NC}"
                echo -e "${BLUE}   ./run_live_native.sh -c $CAMERA_INDEX${NC}"
                echo ""
            fi
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
echo -e "${GREEN}🎉 Live camera feed is ready!${NC}"
echo -e "${BLUE}📹 Camera Index $CAMERA_INDEX is being used for live bib tracking${NC}"
