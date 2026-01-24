#!/bin/bash

# run_batch.sh - Convenience script for batch rendering
#
# Usage:
#   ./run_batch.sh                          # Run all combinations
#   ./run_batch.sh --dry-run                # Preview what would be rendered
#   ./run_batch.sh --asset-type bird        # Render only birds
#   ./run_batch.sh --asset-type drone       # Render only drones
#   ./run_batch.sh --list-assets            # List available assets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find Isaac Sim installation (check common locations)
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-}"

if [ -z "$ISAAC_SIM_PATH" ] || [ ! -d "$ISAAC_SIM_PATH" ]; then
    # Check common installation locations
    for candidate in \
        "$HOME/isaacsim" \
        "$HOME/.local/share/ov/pkg/isaac-sim-4.2.0" \
        "$HOME/.local/share/ov/pkg/isaac-sim-4.5.0" \
        "$HOME/.local/share/ov/pkg/isaac-sim-4.1.0" \
        "$HOME/.local/share/ov/pkg/isaac-sim-4.0.0" \
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
    echo "Example: export ISAAC_SIM_PATH=~/isaacsim"
    exit 1
fi

if [ ! -f "$ISAAC_SIM_PATH/python.sh" ]; then
    echo "ERROR: python.sh not found in $ISAAC_SIM_PATH"
    exit 1
fi

echo "Using Isaac Sim at: $ISAAC_SIM_PATH"
export ISAAC_SIM_PATH

# Run the batch render script
# The batch_render.py script uses standard Python (not Isaac Sim Python)
# It launches headless_runner.py with Isaac Sim Python for each render
python3 "$SCRIPT_DIR/batch_render.py" --isaac-sim-path "$ISAAC_SIM_PATH" "$@"
