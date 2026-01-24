#!/bin/bash

# run_single.sh - Run a single headless render
#
# Usage:
#   ./run_single.sh                                    # Use config.yaml as-is
#   ./run_single.sh --asset "bald-eagle-med-poly.usdc" # Override asset
#   ./run_single.sh --side-meters 20 --flight-height 25
#   ./run_single.sh --crop-depth                       # Enable depth cropping post-process
#
# All options from headless_runner.py are supported:
#   --asset           Asset name from asset_config.yaml
#   --scene           Scene USD path or URL
#   --side-meters     Camera rig diameter
#   --cam-height      Camera height above ground
#   --flight-height   Flight height offset
#   --rotation-offset Camera rotation offset (radians)
#   --camera-type     Camera type from cam_intrinsics.yaml
#   --num-frames      Number of frames to render
#   --num-cameras     Number of cameras
#   --output-dir      Output directory
#   --config          Path to config.yaml
#
# Post-processing options (handled by this script, not passed to headless_runner):
#   --crop-depth      Crop depth maps to object bounding box (saves ~99% storage)
#   --no-crop-depth   Disable depth cropping (default)
#   --keep-full-depth Keep original full-scene depth files after cropping

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse our custom arguments and separate from headless_runner arguments
CROP_DEPTH=false
KEEP_FULL_DEPTH=false
HEADLESS_ARGS=()
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --crop-depth)
            CROP_DEPTH=true
            shift
            ;;
        --no-crop-depth)
            CROP_DEPTH=false
            shift
            ;;
        --keep-full-depth)
            KEEP_FULL_DEPTH=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            HEADLESS_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            HEADLESS_ARGS+=("$1")
            shift
            ;;
    esac
done

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

PYTHON_SH="$ISAAC_SIM_PATH/python.sh"

if [ ! -f "$PYTHON_SH" ]; then
    echo "ERROR: Could not find python.sh at $PYTHON_SH"
    exit 1
fi

echo "Using Isaac Sim at: $ISAAC_SIM_PATH"
echo "Running headless render..."
if [ "$CROP_DEPTH" = true ]; then
    echo "Depth cropping: ENABLED (will run after render)"
fi
echo ""

# Run the headless runner with Isaac Sim Python
"$PYTHON_SH" "$SCRIPT_DIR/headless_runner.py" "${HEADLESS_ARGS[@]}"
RENDER_EXIT_CODE=$?

# Post-processing: Crop depth maps if enabled and render succeeded
if [ $RENDER_EXIT_CODE -eq 0 ] && [ "$CROP_DEPTH" = true ]; then
    echo ""
    echo "========================================"
    echo "Post-processing: Cropping depth maps..."
    echo "========================================"

    # Determine output directory
    # If --output-dir was specified, use it; otherwise try to find the most recent render
    if [ -n "$OUTPUT_DIR" ]; then
        RENDER_DIR="$OUTPUT_DIR"
    else
        # Find most recent render directory in default location
        BASE_PATH="/home/sandro/aeroSplat-4D/renders"
        RENDER_DIR=$(ls -td "$BASE_PATH"/*cams_* 2>/dev/null | head -1)
    fi

    if [ -n "$RENDER_DIR" ] && [ -d "$RENDER_DIR" ]; then
        CROP_ARGS="--render-dir $RENDER_DIR"
        if [ "$KEEP_FULL_DEPTH" = true ]; then
            CROP_ARGS="$CROP_ARGS --keep-originals"
        fi

        echo "Cropping depth in: $RENDER_DIR"
        python3 "$SCRIPT_DIR/crop_depth.py" $CROP_ARGS
        CROP_EXIT_CODE=$?

        if [ $CROP_EXIT_CODE -eq 0 ]; then
            echo "Depth cropping complete!"
        else
            echo "WARNING: Depth cropping failed (exit code $CROP_EXIT_CODE)"
        fi
    else
        echo "WARNING: Could not determine render output directory for depth cropping"
    fi
fi

exit $RENDER_EXIT_CODE
