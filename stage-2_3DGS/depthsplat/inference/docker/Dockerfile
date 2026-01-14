# DepthSplat Inference Pipeline - Production Container
#
# Based on NVIDIA DeepStream SDK for optimized video pipeline processing.
#
# Build:
#   docker build -t depthsplat-inference -f docker/Dockerfile .
#
# Run:
#   docker run --gpus all -v ./models:/app/models:ro depthsplat-inference

FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH
ENV GST_DEBUG=2

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-gl \
    gstreamer1.0-rtsp \
    libcairo2-dev \
    libgirepository1.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
COPY . /app/

# Create directories for outputs
RUN mkdir -p /app/output/gaussians /app/logs /app/models

# Set permissions
RUN chmod +x /app/docker/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:9090/metrics')" || exit 1

# Expose ports
# 9090: Prometheus metrics
EXPOSE 9090

# Entrypoint
ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Default command
CMD ["--config", "config/pipeline_config.yaml", "--mode", "prod"]
