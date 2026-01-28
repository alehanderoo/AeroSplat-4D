#!/bin/bash
# DepthSplat Inference Pipeline - Container Entrypoint
#
# This script initializes the container environment and starts the pipeline.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
echo "========================================"
echo "  DepthSplat Inference Pipeline"
echo "========================================"
echo ""

# Check GPU availability
log_info "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    log_warn "nvidia-smi not available. GPU status unknown."
fi

# Check GStreamer
log_info "Checking GStreamer..."
if command -v gst-inspect-1.0 &> /dev/null; then
    GST_PLUGINS=$(gst-inspect-1.0 | wc -l)
    log_info "GStreamer available with $GST_PLUGINS plugins"
else
    log_warn "GStreamer not available"
fi

# Check PyTorch
log_info "Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || \
    log_warn "PyTorch not available"

# Check configuration
CONFIG_FILE="${1:-config/pipeline_config.yaml}"
if [[ "$1" == "--config" ]]; then
    CONFIG_FILE="$2"
    shift 2
fi

log_info "Using configuration: $CONFIG_FILE"
if [ -f "/app/$CONFIG_FILE" ]; then
    log_info "Configuration file found"
else
    log_warn "Configuration file not found, using defaults"
fi

# Set up signal handling
trap 'log_info "Received shutdown signal"; exit 0' SIGTERM SIGINT

# Create output directories
mkdir -p /app/output/gaussians /app/logs

# Set environment variables
export PYTHONPATH=/app:$PYTHONPATH
export GST_DEBUG=${GST_DEBUG:-2}

# Start the pipeline
log_info "Starting inference pipeline..."
echo ""

exec python3 /app/main.py "$@"
