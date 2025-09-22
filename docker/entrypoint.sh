#!/bin/bash

# Live Bib Tracking - Docker Entrypoint Script
# This script starts the FastAPI server in the background and then runs the video inference script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏃‍♂️ Live Bib Tracking - Starting Services 🏁${NC}"
echo "=================================================="

# Function to handle cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up services...${NC}"
    if [[ -n "$FASTAPI_PID" ]]; then
        echo -e "${YELLOW}Stopping FastAPI server (PID: $FASTAPI_PID)...${NC}"
        kill $FASTAPI_PID 2>/dev/null || true
        wait $FASTAPI_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}✅ Cleanup completed${NC}"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGTERM SIGINT

# Function to check if FastAPI server is running
check_fastapi_health() {
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}🔍 Waiting for FastAPI server to start...${NC}"
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s -f http://localhost:8001/ >/dev/null 2>&1; then
            echo -e "${GREEN}✅ FastAPI server is ready!${NC}"
            return 0
        fi
        
        if [[ $((attempt % 5)) -eq 0 ]]; then
            echo -e "${YELLOW}   Still waiting... (attempt $attempt/$max_attempts)${NC}"
        fi
        
        sleep 1
        ((attempt++))
    done
    
    echo -e "${RED}❌ FastAPI server failed to start within $max_attempts seconds${NC}"
    return 1
}

# Parse command line arguments for the video inference script
if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo -e "${BLUE}📋 Video Inference Script Usage:${NC}"
    echo ""
    echo -e "${YELLOW}Available options for video processing:${NC}"
    echo "  --video PATH     Path to input video file (default: data/raw/race_1080p.mp4)"
    echo "  --model PATH     Path to trained YOLO model (default: runs/detect/yolo11_white_bibs/weights/last.pt)"
    echo "  --fps INT        Target processing frame rate (default: 8)"
    echo "  --conf FLOAT     YOLO confidence threshold (default: 0.3)"
    echo "  --help           Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # Process with default settings"
    echo "  docker run -p 8001:8001 -v /path/to/data:/app/data -v /path/to/models:/app/runs live-bib-tracking"
    echo ""
    echo "  # Process with custom video and model"
    echo "  docker run -p 8001:8001 -v /path/to/data:/app/data -v /path/to/models:/app/runs live-bib-tracking --video data/my_race.mp4 --model runs/my_model.pt"
    echo ""
    echo -e "${BLUE}🌐 Web Interface:${NC}"
    echo "  Access the live video stream at: http://localhost:8001"
    echo ""
    echo -e "${YELLOW}⚠️  Note: Video processing requires actual video and model files to be mounted.${NC}"
    echo "   The web interface will show a 503 error for the video feed until processing starts."
    echo ""
fi

# Check if required files exist
VIDEO_FILE=""
MODEL_FILE=""

# Parse arguments to find video and model paths
ARGS_TO_PASS=()
INFERENCE_MODE_ARG="test"
CAMERA_INDEX_ARG="1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            VIDEO_FILE="$2"
            ARGS_TO_PASS+=("$1" "$2")
            shift 2
            ;;
        --model)
            MODEL_FILE="$2"
            ARGS_TO_PASS+=("$1" "$2")
            shift 2
            ;;
        --inference_mode)
            INFERENCE_MODE_ARG="$2"
            ARGS_TO_PASS+=("$1" "$2")
            shift 2
            ;;
        --camera_index)
            CAMERA_INDEX_ARG="$2"
            ARGS_TO_PASS+=("$1" "$2")
            shift 2
            ;;
        --fps|--conf)
            ARGS_TO_PASS+=("$1" "$2")
            shift 2
            ;;
        --help|-h)
            # Help was already shown above, just exit
            exit 0
            ;;
        *)
            ARGS_TO_PASS+=("$1")
            shift
            ;;
    esac
done

# Set defaults if not specified
VIDEO_FILE=${VIDEO_FILE:-"data/raw/race_1080p.mp4"}
MODEL_FILE=${MODEL_FILE:-"runs/detect/yolo11_white_bibs/weights/last.pt"}

# Check if files exist
if [[ ! -f "$VIDEO_FILE" ]]; then
    echo -e "${RED}❌ Video file not found: $VIDEO_FILE${NC}"
    echo -e "${YELLOW}💡 Make sure to mount your video directory with: -v /path/to/videos:/app/data${NC}"
    echo -e "${BLUE}🌐 Starting web interface anyway at: http://localhost:8001${NC}"
    echo -e "${YELLOW}The video feed will show an error until proper files are provided.${NC}"
fi

if [[ ! -f "$MODEL_FILE" ]]; then
    echo -e "${RED}❌ Model file not found: $MODEL_FILE${NC}"
    echo -e "${YELLOW}💡 Make sure to mount your models directory with: -v /path/to/models:/app/runs${NC}"
    echo -e "${BLUE}🌐 Starting web interface anyway at: http://localhost:8001${NC}"
    echo -e "${YELLOW}The video feed will show an error until proper files are provided.${NC}"
fi

if [[ -f "$VIDEO_FILE" && -f "$MODEL_FILE" ]]; then
    echo -e "${GREEN}✅ Found video file: $VIDEO_FILE${NC}"
    echo -e "${GREEN}✅ Found model file: $MODEL_FILE${NC}"
fi

echo ""
echo -e "${YELLOW}🚀 Starting FastAPI server with video processing...${NC}"
echo -e "${BLUE}Arguments: ${ARGS_TO_PASS[*]}${NC}"
echo ""

# Export arguments as environment variables for the FastAPI app to use
export VIDEO_PATH="$VIDEO_FILE"
export MODEL_PATH="/app/$MODEL_FILE"
export INFERENCE_MODE="$INFERENCE_MODE_ARG"
export CAMERA_INDEX="$CAMERA_INDEX_ARG"

# Parse other arguments and export them
for ((i=0; i<${#ARGS_TO_PASS[@]}; i++)); do
    case "${ARGS_TO_PASS[i]}" in
        --fps)
            export TARGET_FPS="${ARGS_TO_PASS[i+1]}"
            ;;
        --conf)
            export CONFIDENCE_THRESHOLD="${ARGS_TO_PASS[i+1]}"
            ;;
    esac
done

# Set defaults for environment variables if not set
export TARGET_FPS="${TARGET_FPS:-8}"
export CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.3}"

echo -e "${BLUE}Environment variables set:${NC}"
echo -e "  VIDEO_PATH: $VIDEO_PATH"
echo -e "  MODEL_PATH: $MODEL_PATH"
echo -e "  INFERENCE_MODE: $INFERENCE_MODE"
echo -e "  CAMERA_INDEX: $CAMERA_INDEX"
echo -e "  TARGET_FPS: $TARGET_FPS"
echo -e "  CONFIDENCE_THRESHOLD: $CONFIDENCE_THRESHOLD"
echo ""

# Start the FastAPI server
exec python -m uvicorn src.image_processor.video_inference:app --host 0.0.0.0 --port 8001
