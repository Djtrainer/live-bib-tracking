# Multi-stage Dockerfile for Live Bib Tracking Application
# Stage 1: Frontend Build
FROM node:18-alpine AS frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend package files
COPY src/frontend/package*.json ./
COPY src/frontend/bun.lockb* ./

# Install frontend dependencies
RUN npm install

# Copy frontend source code
COPY src/frontend/ ./

# Build the frontend
RUN npm run build

# Stage 2: Python Build stage with all build tools and dependencies
FROM python:3.10-slim-bullseye AS python-builder

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

# Stage 3: Final runtime stage
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
    libgtk-3-0 \
    libxcb1 \
    libxcb-xinerama0 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-util1 \
    xvfb \
    fluxbox \
    x11vnc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Copy virtual environment from python builder stage
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy built frontend from frontend builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy backend application code
COPY src/api_backend/ ./src/api_backend/
COPY src/image_processor/ ./src/image_processor/
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

# Expose ports
EXPOSE 8000 5900

# Set default command to start the FastAPI server
CMD ["python", "-m", "uvicorn", "src.api_backend.local_server:app", "--host", "0.0.0.0", "--port", "8000"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/results')" || exit 1
