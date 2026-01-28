#!/bin/bash
# =============================================================================
# DepthSplat Visualization Server Launcher
# =============================================================================
# Starts the real-time visualization server with:
# - Inference pipeline with Gaussian rendering
# - WebSocket server for browser streaming
# - HTTP server for frontend
#
# Usage:
#   ./scripts/start_visualization.sh           # Default (dev mode)
#   ./scripts/start_visualization.sh prod      # Production mode
#   ./scripts/start_visualization.sh --test    # Quick test (2 seconds)
#   ./scripts/start_visualization.sh --help    # Show options
#
# Prerequisites:
#   - Conda environment 'aeroSplat' activated
#   - RTSP streams running (in dev mode, use start_rtsp_simulator.sh)
# =============================================================================

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
MODE="${1:-dev}"
WS_PORT="${WS_PORT:-8765}"
HTTP_PORT="${HTTP_PORT:-8080}"
CONFIG="${CONFIG:-config/pipeline_config.yaml}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "=============================================="
echo "   DepthSplat Visualization Server"
echo "=============================================="
echo -e "${NC}"

# Show help
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [mode] [options]"
    echo ""
    echo "Modes:"
    echo "  dev     Development mode (expects RTSP simulator)"
    echo "  prod    Production mode (real cameras)"
    echo ""
    echo "Environment Variables:"
    echo "  WS_PORT     WebSocket port (default: 8765)"
    echo "  HTTP_PORT   Frontend HTTP port (default: 8080)"
    echo "  CONFIG      Config file path"
    echo "  LOG_LEVEL   DEBUG, INFO, WARNING, ERROR"
    echo ""
    echo "Examples:"
    echo "  $0                    # Dev mode, default ports"
    echo "  WS_PORT=9000 $0       # Custom WebSocket port"
    echo "  $0 prod               # Production mode"
    exit 0
fi

# Check mode
if [ "$MODE" != "dev" ] && [ "$MODE" != "prod" ]; then
    echo -e "${RED}Error: Invalid mode '$MODE'. Use 'dev' or 'prod'${NC}"
    exit 1
fi

# Change to inference directory
cd "$INFERENCE_DIR"

# Check Python environment
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found. Please activate the depthsplat environment.${NC}"
    echo "  conda activate depthsplat"
    exit 1
fi

# Check required packages
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import websockets" 2>/dev/null || {
    echo -e "${RED}Missing 'websockets' package. Installing...${NC}"
    pip install websockets
}

python -c "import PIL" 2>/dev/null || {
    echo -e "${RED}Missing 'Pillow' package. Installing...${NC}"
    pip install Pillow
}

# Print configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Mode:           $MODE"
echo "  WebSocket Port: $WS_PORT"
echo "  HTTP Port:      $HTTP_PORT"
echo "  Config:         $CONFIG"
echo "  Log Level:      $LOG_LEVEL"
echo ""

# Print URLs
echo -e "${GREEN}Access URLs:${NC}"
echo -e "  Frontend:       ${BLUE}http://localhost:${HTTP_PORT}${NC}"
echo -e "  WebSocket:      ${BLUE}ws://localhost:${WS_PORT}${NC}"
echo ""

# Reminder for dev mode
if [ "$MODE" == "dev" ]; then
    echo -e "${YELLOW}Note: In dev mode, RTSP streams should be running.${NC}"
    echo "  Start with: ./scripts/start_rtsp_simulator.sh"
    echo ""
fi

# Start the server
echo -e "${GREEN}Starting visualization server...${NC}"
echo ""

python main.py \
    --config "$CONFIG" \
    --mode "$MODE" \
    --ws-port "$WS_PORT" \
    --http-port "$HTTP_PORT" \
    --log-level "$LOG_LEVEL"
