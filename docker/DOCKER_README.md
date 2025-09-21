# Docker Setup for Live Bib Tracking Application

This document explains how to build and run the Live Bib Tracking application using Docker.

## Overview

The application is containerized using a multi-stage Docker build that:
- Uses `python:3.10-slim-bullseye` as the base image
- Builds dependencies in a separate stage to reduce final image size
- Runs as a non-root user for enhanced security
- Includes all necessary system dependencies for computer vision and OCR

## Files

- `Dockerfile`: Multi-stage build configuration
- `.dockerignore`: Excludes unnecessary files from build context
- `requirements.txt`: Python dependencies
- `src/image_processor/video_inference.py`: Main application entry point

## Building the Docker Image

### Basic Build
```bash
docker build -t live-bib-tracking .
```

### Build with Custom Tag
```bash
docker build -t live-bib-tracking:v1.0.0 .
```

### Build with Build Arguments (if needed)
```bash
docker build --build-arg PYTHON_VERSION=3.10 -t live-bib-tracking .
```

## Running the Container

### Show Help (Default Command)
```bash
docker run --rm live-bib-tracking
```

### Run with Video File
```bash
docker run --rm \
  -v /Users/dantrainer/projects/live-bib-tracking/data:/app/data \
  -v /Users/dantrainer/projects/live-bib-tracking/runs/detect/yolo11_white_bibs/weights:/app/models \
  live-bib-tracking \
  python src/image_processor/video_inference.py \
  --video /app/data/raw/2024_race.MOV \
  --model /app/models/last.pt
```

### Run with Display (X11 forwarding on Linux)
```bash
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /path/to/your/data:/app/data \
  live-bib-tracking \
  python src/image_processor/video_inference.py \
  --video /app/data/your_video.mov
```

### Run with Environment Variables
```bash
docker run --rm \
  -e IVS_PLAYBACK_URL="your_stream_url" \
  -v /path/to/your/data:/app/data \
  live-bib-tracking \
  python src/image_processor/video_inference.py \
  --from-stream true
```

### Interactive Mode for Debugging
```bash
docker run --rm -it \
  -v /path/to/your/data:/app/data \
  live-bib-tracking \
  /bin/bash
```

## Volume Mounts

The application expects certain directories to be available:

- `/app/data`: Input video files and datasets
- `/app/models`: YOLO model files (.pt)
- `/app/runs`: Training outputs and results
- `/app/config`: Configuration files

Example with all volumes:
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/config:/app/config \
  live-bib-tracking \
  python src/image_processor/video_inference.py \
  --video /app/data/race_video.mov \
  --model /app/models/yolo_model.pt
```

## Environment Variables

The application supports these environment variables:

- `IVS_PLAYBACK_URL`: AWS IVS stream URL for live processing
- `PYTHONPATH`: Python module search path (set automatically)

## Application Parameters

The containerized application supports all the same parameters as the standalone version:

- `--video`: Path to input video file
- `--model`: Path to YOLO model weights
- `--fps`: Target processing frame rate (default: 8)
- `--conf`: YOLO confidence threshold (default: 0.3)
- `--output`: Path to save output video
- `--no-display`: Disable video display
- `--from-stream`: Process live stream instead of file

## Docker Compose Example

Create a `docker-compose.yml` file for easier management:

```yaml
version: '3.8'
services:
  live-bib-tracking:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./runs:/app/runs
      - ./config:/app/config
    environment:
      - IVS_PLAYBACK_URL=${IVS_PLAYBACK_URL}
    command: >
      python src/image_processor/video_inference.py
      --video /app/data/race_video.mov
      --model /app/models/yolo_model.pt
      --fps 5
      --conf 0.25
```

Run with:
```bash
docker-compose up
```

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure your data directories have proper permissions
   ```bash
   chmod -R 755 ./data ./models ./runs ./config
   ```

2. **Display Issues**: For GUI applications, ensure X11 forwarding is properly configured
   ```bash
   xhost +local:docker  # Allow Docker to access X11
   ```

3. **Large Image Size**: The image includes OpenCV, PyTorch, and other ML libraries, so expect 2-3GB
   
4. **Memory Issues**: Ensure Docker has sufficient memory allocated (recommend 4GB+)

### Debugging

1. **Check container logs**:
   ```bash
   docker logs <container_id>
   ```

2. **Run in interactive mode**:
   ```bash
   docker run --rm -it live-bib-tracking /bin/bash
   ```

3. **Inspect the image**:
   ```bash
   docker inspect live-bib-tracking
   ```

## Security Considerations

- The container runs as a non-root user (`appuser`)
- Sensitive files are excluded via `.dockerignore`
- No secrets are baked into the image
- Use environment variables for configuration

## Performance Tips

1. **Use .dockerignore**: Reduces build context size
2. **Multi-stage build**: Keeps final image smaller
3. **Layer caching**: Order Dockerfile commands for optimal caching
4. **Volume mounts**: Use for large data files instead of copying into image

## Production Deployment

For production use:

1. Use specific version tags instead of `latest`
2. Set resource limits:
   ```bash
   docker run --memory=4g --cpus=2 live-bib-tracking
   ```
3. Use health checks and restart policies
4. Consider using Docker Swarm or Kubernetes for orchestration
