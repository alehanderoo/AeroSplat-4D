#!/bin/bash

# Start tmux session for rendering directly to .torch files
SESSION_NAME="objaverse_render_torch"

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    # Start X server if not already running (outside tmux to handle sudo)
    echo "Checking X server status..."
    sudo python3 start_x_server.py start 2 2>&1 | grep -q "already running"
    if [ $? -eq 0 ]; then
        echo "X server is already running on display :2"
    else
        echo "X server started on display :2"
    fi

    # Create new tmux session
    tmux new-session -d -s $SESSION_NAME

    # Run the rendering commands in the tmux session
    tmux send-keys -t $SESSION_NAME "export DISPLAY=:2" C-m
    tmux send-keys -t $SESSION_NAME "uv run render_to_torch.py \
        --gpu_devices=1 \
        --render_depth=True \
        --render_mask=True \
        --render_normals=False \
        --num_workers=3 \
        --num_objects=30000 \
        --scenes_per_chunk=100 \
        --train_split=0.9 \
        --output_dir=/home/sandro/.objaverse/depthsplat" C-m

    echo "Tmux session '$SESSION_NAME' created and rendering started."
    echo "To attach to the session, run: tmux attach -t $SESSION_NAME"
    echo "To detach from the session, press: Ctrl+b then d"
else
    echo "Tmux session '$SESSION_NAME' already exists."
    echo "To attach to it, run: tmux attach -t $SESSION_NAME"
fi
