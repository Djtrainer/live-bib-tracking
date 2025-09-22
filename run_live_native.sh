#!/bin/bash

# Live Bib Tracking - Native macOS Live Camera Feed
# This script runs the live bib tracking directly on macOS without Docker
# for better camera access compatibility

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PORT=8001

# Default paths (can be overridden with environment variables)
MODEL_PATH=${MODEL_PATH:-"$(pwd)/runs/detect/yolo11_white_bibs/weights/last.pt"}
CAMERA_INDEX=${CAMERA_INDEX:-1}
VIDEO_PATH=""

echo -e "${BLUE}🏃‍♂️ Live Bib Tracking - Native macOS Processing 📹${NC}"
echo "=============================================================="

# Function to print usage
print_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 [OPTIONS]"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  -h, --help       Show this help message"
    echo "  -m, --model      Path to model file (default: ./runs/detect/yolo11_white_bibs/weights/last.pt)"
    echo "  -c, --camera     Camera index (default: 1, iPhone; 0 for built-in)"
    echo "  -v, --video      Path to video file for testing (overrides camera)"
    echo "  -p, --port       Port to bind server to (default: 8001)"
    echo ""
    echo -e "${YELLOW}Environment Variables:${NC}"
    echo "  MODEL_PATH       Path to model file"
    echo "  CAMERA_INDEX     Camera index to use (0=built-in, 1=iPhone, etc.)"
    echo ""
    echo -e "${YELLOW}Camera Index Guide:${NC}"
    echo "  0 - Built-in camera (MacBook webcam)"
    echo "  1 - External camera (iPhone via USB/wireless, external webcam)"
    echo "  2+ - Additional cameras if available"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0                                    # Use iPhone camera (index 1)"
    echo "  $0 -c 0                              # Use built-in camera"
    echo "  $0 -v test_race.mp4                  # Process video file"
    echo "  $0 -m /path/to/model.pt -c 1         # Custom model with iPhone"
    echo "  $0 -v /path/to/video.mp4 -m /path/to/model.pt  # Custom model with video"
    echo "  CAMERA_INDEX=0 MODEL_PATH=/path/to/model.pt $0"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c|--camera)
            CAMERA_INDEX="$2"
            shift 2
            ;;
        -v|--video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
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
if [[ -n "$VIDEO_PATH" ]]; then
    VIDEO_PATH=$(realpath "$VIDEO_PATH" 2>/dev/null || echo "$VIDEO_PATH")
fi

echo -e "${BLUE}Configuration:${NC}"
echo "  Model Path: $MODEL_PATH"
echo "  Port: $PORT"

if [[ -n "$VIDEO_PATH" ]]; then
    echo "  Input Source: Video file"
    echo "  Video Path: $VIDEO_PATH"
else
    echo "  Input Source: Live camera"
    echo "  Camera Index: $CAMERA_INDEX"
    if [[ "$CAMERA_INDEX" == "0" ]]; then
        echo "  Camera Type: Built-in camera (MacBook)"
    elif [[ "$CAMERA_INDEX" == "1" ]]; then
        echo "  Camera Type: External camera (iPhone/USB)"
    else
        echo "  Camera Type: Camera device $CAMERA_INDEX"
    fi
fi
echo ""

# Function to check Python environment
check_python_env() {
    echo -e "${YELLOW}🐍 Checking Python environment...${NC}"
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is not installed or not in PATH${NC}"
        echo -e "${BLUE}💡 Install Python 3: https://www.python.org/downloads/${NC}"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"
    
    # Check if we're in a virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo -e "${GREEN}✅ Virtual environment active: $VIRTUAL_ENV${NC}"
    else
        echo -e "${YELLOW}⚠️  No virtual environment detected${NC}"
        echo -e "${BLUE}💡 Consider using: python3 -m venv venv && source venv/bin/activate${NC}"
    fi
}

# Function to check required packages
check_dependencies() {
    echo -e "${YELLOW}📦 Checking required packages...${NC}"
    
    REQUIRED_PACKAGES=("cv2" "ultralytics" "easyocr" "fastapi" "uvicorn" "numpy" "requests")
    MISSING_PACKAGES=()
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if python3 -c "import ${package//-/_}" 2>/dev/null; then
            echo -e "${GREEN}✅ $package${NC}"
        else
            echo -e "${RED}❌ $package${NC}"
            MISSING_PACKAGES+=("$package")
        fi
    done
    
    if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
        echo -e "${YELLOW}⚠️  Missing packages detected${NC}"
        echo -e "${BLUE}💡 Install missing packages with:${NC}"
        echo -e "${BLUE}   pip install ${MISSING_PACKAGES[*]}${NC}"
        echo -e "${BLUE}   or: pip install -r requirements.txt${NC}"
        
        read -p "$(echo -e ${YELLOW}Would you like to install missing packages now? [y/N]: ${NC})" -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}📦 Installing missing packages...${NC}"
            pip install "${MISSING_PACKAGES[@]}"
            echo -e "${GREEN}✅ Packages installed${NC}"
        else
            echo -e "${RED}❌ Cannot proceed without required packages${NC}"
            exit 1
        fi
    fi
}

# Function to check model file
check_model() {
    echo -e "${YELLOW}🤖 Checking model file...${NC}"
    
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo -e "${RED}❌ Model file not found: $MODEL_PATH${NC}"
        echo -e "${BLUE}💡 Make sure you have trained a model or download a pre-trained one${NC}"
        exit 1
    fi
    
    # Check file size
    MODEL_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || echo "0")
    if [[ "$MODEL_SIZE" -eq 0 ]]; then
        echo -e "${RED}❌ Model file is empty: $MODEL_PATH${NC}"
        exit 1
    fi
    
    MODEL_SIZE_MB=$((MODEL_SIZE / 1024 / 1024))
    echo -e "${GREEN}✅ Model file found: $MODEL_PATH (${MODEL_SIZE_MB}MB)${NC}"
}

# Function to check video file
check_video() {
    echo -e "${YELLOW}🎬 Checking video file...${NC}"
    
    if [[ ! -f "$VIDEO_PATH" ]]; then
        echo -e "${RED}❌ Video file not found: $VIDEO_PATH${NC}"
        echo -e "${BLUE}💡 Make sure the video file exists and the path is correct${NC}"
        exit 1
    fi
    
    # Check file size
    VIDEO_SIZE=$(stat -f%z "$VIDEO_PATH" 2>/dev/null || echo "0")
    if [[ "$VIDEO_SIZE" -eq 0 ]]; then
        echo -e "${RED}❌ Video file is empty: $VIDEO_PATH${NC}"
        exit 1
    fi
    
    VIDEO_SIZE_MB=$((VIDEO_SIZE / 1024 / 1024))
    echo -e "${GREEN}✅ Video file found: $VIDEO_PATH (${VIDEO_SIZE_MB}MB)${NC}"
    
    # Test video file access with OpenCV
    VIDEO_TEST=$(python3 -c "
import cv2
import sys

try:
    cap = cv2.VideoCapture('$VIDEO_PATH')
    if cap.isOpened():
        ret, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if ret and frame is not None:
            print(f'SUCCESS:{frame_count}:{fps:.2f}')
        else:
            print('OPEN_BUT_NO_FRAME')
    else:
        print('CANNOT_OPEN')
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null)

    if [[ "$VIDEO_TEST" =~ ^SUCCESS:([0-9]+):([0-9.]+)$ ]]; then
        FRAME_COUNT="${BASH_REMATCH[1]}"
        VIDEO_FPS="${BASH_REMATCH[2]}"
        DURATION=$(python3 -c "print(f'{int($FRAME_COUNT / $VIDEO_FPS // 60):02d}:{int($FRAME_COUNT / $VIDEO_FPS % 60):02d}')")
        echo -e "${GREEN}✅ Video is readable: ${FRAME_COUNT} frames, ${VIDEO_FPS} FPS, ${DURATION} duration${NC}"
    else
        case "$VIDEO_TEST" in
            "OPEN_BUT_NO_FRAME")
                echo -e "${RED}❌ Video file opens but cannot read frames${NC}"
                echo -e "${BLUE}💡 The video file may be corrupted or in an unsupported format${NC}"
                exit 1
                ;;
            "CANNOT_OPEN")
                echo -e "${RED}❌ Cannot open video file${NC}"
                echo -e "${BLUE}💡 Check if the video format is supported by OpenCV${NC}"
                exit 1
                ;;
            *)
                echo -e "${RED}❌ Video test failed: $VIDEO_TEST${NC}"
                echo -e "${BLUE}💡 The video file may be corrupted or in an unsupported format${NC}"
                exit 1
                ;;
        esac
    fi
}

# Function to check camera access
check_camera() {
    echo -e "${YELLOW}📹 Checking camera access...${NC}"
    
    # Test camera access with a simple Python script
    CAMERA_TEST=$(python3 -c "
import cv2
import sys

try:
    cap = cv2.VideoCapture($CAMERA_INDEX)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print('SUCCESS')
        else:
            print('OPEN_BUT_NO_FRAME')
    else:
        print('CANNOT_OPEN')
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null)

    case "$CAMERA_TEST" in
        "SUCCESS")
            echo -e "${GREEN}✅ Camera $CAMERA_INDEX is accessible and working${NC}"
            ;;
        "OPEN_BUT_NO_FRAME")
            echo -e "${YELLOW}⚠️  Camera $CAMERA_INDEX opens but cannot read frames${NC}"
            echo -e "${BLUE}💡 Try a different camera index or check camera permissions${NC}"
            ;;
        "CANNOT_OPEN")
            echo -e "${RED}❌ Cannot open camera $CAMERA_INDEX${NC}"
            echo -e "${BLUE}💡 Try different camera indices: 0, 1, 2, etc.${NC}"
            ;;
        *)
            echo -e "${RED}❌ Camera test failed: $CAMERA_TEST${NC}"
            echo -e "${BLUE}💡 Check camera permissions and try different indices${NC}"
            ;;
    esac
    
    # macOS specific camera permission check
    echo -e "${BLUE}🍎 macOS Camera Permissions:${NC}"
    echo -e "${BLUE}   1. Go to System Preferences > Security & Privacy > Camera${NC}"
    echo -e "${BLUE}   2. Ensure Terminal (or your terminal app) has camera access${NC}"
    echo -e "${BLUE}   3. If using VS Code integrated terminal, grant VS Code camera access${NC}"
}

# Function to run the processing
run_processing() {
    echo -e "${YELLOW}🚀 Starting bib tracking processing...${NC}"
    echo -e "${BLUE}🌐 Server will be available at: http://localhost:$PORT${NC}"
    echo -e "${BLUE}🤖 Using model: $MODEL_PATH${NC}"
    
    if [[ -n "$VIDEO_PATH" ]]; then
        echo -e "${BLUE}🎬 Processing video file: $VIDEO_PATH${NC}"
        echo ""
        echo -e "${YELLOW}💡 Press Ctrl+C to stop the server${NC}"
        echo ""
        
        # Run with video file (test mode)
        python3 src/api_backend/local_server.py \
            --inference_mode test \
            --video "$VIDEO_PATH" \
            --model "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --fps 20 \
            --conf 0.3
    else
        echo -e "${BLUE}📹 Using camera index: $CAMERA_INDEX${NC}"
        echo ""
        echo -e "${YELLOW}💡 Press Ctrl+C to stop the server${NC}"
        echo ""
        
        # Run with live camera
        python3 src/api_backend/local_server.py \
            --inference_mode live \
            --camera_index "$CAMERA_INDEX" \
            --model "$MODEL_PATH" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --fps 20 \
            --conf 0.3
    fi
}

# Main execution
echo -e "${BLUE}🔍 Running pre-flight checks...${NC}"
echo ""

check_python_env
echo ""

check_dependencies
echo ""

check_model
echo ""

# Check input source (video file or camera)
if [[ -n "$VIDEO_PATH" ]]; then
    check_video
else
    check_camera
fi
echo ""

echo -e "${GREEN}✅ All checks passed!${NC}"
echo ""

run_processing
