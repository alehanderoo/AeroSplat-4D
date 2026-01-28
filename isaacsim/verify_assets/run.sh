#!/bin/bash
# Run asset verification with Isaac Sim Python
# Usage: ./run.sh [--category birds|drones|all] [--render-size 512] [--grid-cols 6]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find Isaac Sim installation
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-}"

if [ -z "$ISAAC_SIM_PATH" ] || [ ! -d "$ISAAC_SIM_PATH" ]; then
    for candidate in \
        "$HOME/isaacsim" \
        "$HOME/.local/share/ov/pkg/isaac-sim-4.2.0" \
        "$HOME/.local/share/ov/pkg/isaac-sim-4.5.0" \
        "/isaac-sim"; do
        if [ -d "$candidate" ] && [ -f "$candidate/python.sh" ]; then
            ISAAC_SIM_PATH="$candidate"
            break
        fi
    done
fi

if [ ! -d "$ISAAC_SIM_PATH" ]; then
    echo "ERROR: Could not find Isaac Sim installation"
    echo "Set ISAAC_SIM_PATH environment variable to your Isaac Sim installation directory"
    exit 1
fi

PYTHON_SH="$ISAAC_SIM_PATH/python.sh"

echo "Using Isaac Sim at: $ISAAC_SIM_PATH"
echo "Running asset verification..."
echo ""

"$PYTHON_SH" "$SCRIPT_DIR/verify_assets.py" "$@"
