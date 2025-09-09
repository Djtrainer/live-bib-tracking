# Multi-stage Dockerfile for Live Bib Tracking Application

# Stage 1: Frontend Build (No changes needed here)
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY src/frontend/package*.json ./
COPY src/frontend/bun.lockb* ./
RUN npm install
COPY src/frontend/ ./
RUN npm run build


# Stage 2: Python Build stage with GPU support
# --- FIX: Use an NVIDIA CUDA base image ---
FROM nvidia/cuda:12.1.1-base-ubuntu22.04 AS python-builder

# Prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Create and activate virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- FIX: Install the GPU version of PyTorch first, then other requirements ---
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt


# Stage 3: Final runtime stage with GPU support
# --- FIX: Use the same NVIDIA CUDA base image for runtime ---
FROM nvidia/cuda:12.1.1-base-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install only necessary runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
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

# Copy backend application code and config
COPY src/api_backend/ ./src/api_backend/
COPY src/image_processor/ ./src/image_processor/
COPY config/ ./config/

# Create necessary directories and set permissions
RUN mkdir -p /app/data /app/runs /app/models && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH="/app/src"
ENV YOLO_AUTOINSTALL="false"

# Expose ports
EXPOSE 8000 5900

# Set default command to start the FastAPI server
CMD ["python", "-m", "uvicorn", "src.api_backend.local_server:app", "--host", "0.0.0.0", "--port", "8000"]