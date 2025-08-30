# Multi-stage Dockerfile for Live Bib Tracking Application
# Stage 1: Build stage with all build tools and dependencies
FROM python:3.10-slim-bullseye AS builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdal-dev \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage with minimal dependencies
FROM python:3.10-slim-bullseye AS runtime

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    tesseract-ocr \
    libgdal28 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml .

# Create necessary directories and set permissions
RUN mkdir -p /app/data /app/runs /app/models && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /home/appuser

# Create EasyOCR cache directory with proper permissions
RUN mkdir -p /home/appuser/.EasyOCR && \
    chown -R appuser:appuser /home/appuser/.EasyOCR

# Switch to non-root user
USER appuser

# Set EasyOCR cache directory
ENV EASYOCR_MODULE_PATH="/home/appuser/.EasyOCR"

# Disable Ultralytics auto-update to prevent permission issues
ENV YOLO_AUTOINSTALL="false"

# Set Python path to include src directory
ENV PYTHONPATH="/app/src"

# Expose port if the application serves HTTP (optional)
EXPOSE 8000

# Set default command to run the video inference application
CMD ["python", "src/image_processor/video_inference.py", "--help"]

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"
