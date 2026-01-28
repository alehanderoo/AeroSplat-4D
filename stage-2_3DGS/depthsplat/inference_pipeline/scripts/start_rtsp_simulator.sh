#!/bin/bash
#
# Start RTSP Stream Simulator for DepthSplat Inference Pipeline
#
# This script starts the RTSP server that simulates 5 IP camera feeds
# from pre-rendered IsaacSim frames using GStreamer.
#
# Configuration is read from config/pipeline_config.yaml
#
# Usage:
#   ./start_rtsp_simulator.sh                    # Use config defaults
#   ./start_rtsp_simulator.sh --fps 60           # Override FPS
#
# Streams will be available at:
#   rtsp://localhost:8554/cam_01
#   rtsp://localhost:8554/cam_02
#   rtsp://localhost:8554/cam_03
#   rtsp://localhost:8554/cam_04
#   rtsp://localhost:8554/cam_05

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$INFERENCE_DIR/config/pipeline_config.yaml"

# GStreamer environment variables for conda compatibility
# These must be set BEFORE checking dependencies or running the server
export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0
export GST_PLUGIN_SYSTEM_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0
export GST_PLUGIN_PATH=""

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

# Read config values from YAML using Python
read_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi

    log_info "Reading configuration from: $CONFIG_FILE"

    # Use Python to parse YAML and extract values
    CONFIG_VALUES=$(python3 << EOF
import yaml
import sys

try:
    with open("$CONFIG_FILE", 'r') as f:
        config = yaml.safe_load(f)

    # Extract values with defaults
    render_dir = config.get('render_dir', '/home/sandro/aeroSplat-4D/renders/5cams_drone_50m')

    # Get simulator config
    simulator = config.get('sources', {}).get('simulator', {})
    port = simulator.get('port', 8554)
    fps = simulator.get('fps', 30)

    # Get file source config for num_frames
    file_source = config.get('visualization', {}).get('file_source', {})
    num_frames = file_source.get('num_frames', 120)

    print(f"RENDER_DIR={render_dir}")
    print(f"PORT={port}")
    print(f"FPS={fps}")
    print(f"NUM_FRAMES={num_frames}")
except Exception as e:
    print(f"ERROR={e}", file=sys.stderr)
    sys.exit(1)
EOF
)

    if [ $? -ne 0 ]; then
        log_error "Failed to parse config file"
        exit 1
    fi

    # Export the values
    eval "$CONFIG_VALUES"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    # Python
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Activate conda environment first."
        exit 1
    fi

    # Check for GStreamer RTSP Server
    if python -c "import gi; gi.require_version('GstRtspServer', '1.0')" 2>/dev/null; then
        log_info "GStreamer RTSP Server: Available"
    else
        log_error "GStreamer RTSP Server not available."
        echo ""
        echo "Install with:"
        echo "  sudo apt install gir1.2-gst-rtsp-server-1.0 libgstrtspserver-1.0-dev \\"
        echo "      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \\"
        echo "      gstreamer1.0-plugins-base gstreamer1.0-plugins-good \\"
        echo "      gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav"
        echo ""
        echo "For conda environments, ensure these are set:"
        echo "  export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0"
        echo "  export GST_PLUGIN_SYSTEM_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
        echo "  export GST_PLUGIN_PATH=''"
        exit 1
    fi
}

# Install GStreamer dependencies (Ubuntu/Debian)
install_gstreamer() {
    log_info "Installing GStreamer RTSP Server..."
    sudo apt update
    sudo apt install -y \
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
        gstreamer1.0-alsa \
        gstreamer1.0-gl \
        gstreamer1.0-gtk3 \
        gstreamer1.0-qt5 \
        gstreamer1.0-pulseaudio \
        gir1.2-gst-rtsp-server-1.0 \
        libgstrtspserver-1.0-dev

    log_info "GStreamer installed successfully"
    log_info "Note: If using conda, set these environment variables:"
    echo "  export GI_TYPELIB_PATH=/usr/lib/x86_64-linux-gnu/girepository-1.0"
    echo "  export GST_PLUGIN_SYSTEM_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
    echo "  export GST_PLUGIN_PATH=''"
}

# Print banner
print_banner() {
    echo ""
    echo "============================================================"
    echo "       DepthSplat RTSP Stream Simulator"
    echo "============================================================"
    echo ""
}

# Main
main() {
    print_banner

    # Check for install flag first (before reading config)
    for arg in "$@"; do
        if [ "$arg" = "--install-gstreamer" ]; then
            install_gstreamer
            exit 0
        fi
    done

    # Read config from YAML file
    read_config

    # Parse command line arguments (can override config values)
    while [[ $# -gt 0 ]]; do
        case $1 in
            --render-dir)
                RENDER_DIR="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --fps)
                FPS="$2"
                shift 2
                ;;
            --config)
                # Already handled by read_config, but allow explicit override
                CONFIG_FILE="$2"
                read_config
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Configuration is read from: $CONFIG_FILE"
                echo ""
                echo "Options (override config values):"
                echo "  --render-dir PATH     Path to renders directory"
                echo "  --port PORT           RTSP server port"
                echo "  --fps FPS             Frames per second"
                echo "  --config PATH         Use different config file"
                echo "  --install-gstreamer   Install GStreamer dependencies"
                echo "  --help                Show this help"
                exit 0
                ;;
            --install-gstreamer)
                # Already handled above
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Check dependencies
    check_dependencies

    # Validate render directory
    if [ ! -d "$RENDER_DIR" ]; then
        log_error "Render directory not found: $RENDER_DIR"
        exit 1
    fi

    # Validate cameras exist
    for i in 01 02 03 04 05; do
        if [ ! -d "$RENDER_DIR/cam_$i/rgb" ]; then
            log_error "Camera directory not found: $RENDER_DIR/cam_$i/rgb"
            exit 1
        fi
    done

    FRAME_COUNT=$(ls "$RENDER_DIR/cam_01/rgb/"rgb_*.png 2>/dev/null | wc -l)
    log_info "Found $FRAME_COUNT frames per camera"

    # Print configuration
    log_info "Configuration (from $CONFIG_FILE):"
    echo "  Render directory: $RENDER_DIR"
    echo "  Port: $PORT"
    echo "  FPS: $FPS"
    echo "  Frames: $FRAME_COUNT"
    echo ""

    # Change to inference directory
    cd "$INFERENCE_DIR"

    # Start server
    log_info "Starting RTSP server..."
    echo ""

    # Run the Python server directly (not with exec to allow signal handling)
    python -m stream_simulator.rtsp_server \
        --render-dir "$RENDER_DIR" \
        --port "$PORT" \
        --fps "$FPS"
}

main "$@"
