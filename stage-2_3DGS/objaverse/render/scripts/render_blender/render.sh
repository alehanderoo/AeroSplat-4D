#!/bin/bash

# Start tmux session for rendering
SESSION_NAME="objaverse_render"

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    # Start X server if not already running (outside tmux to handle sudo)
    echo "Checking X server status..."
    sudo uv run start_x_server.py start 2 2>&1 | grep -q "already running"
    if [ $? -eq 0 ]; then
        echo "X server is already running on display :2"
    else
        echo "X server started on display :2"
    fi
    
    # Create new tmux session
    tmux new-session -d -s $SESSION_NAME
    
    # Run the rendering commands in the tmux session
    tmux send-keys -t $SESSION_NAME "export DISPLAY=:2" C-m
    tmux send-keys -t $SESSION_NAME "uv run render_objaverse1.py --gpu_devices=1 --render_normals True --render_mask True --render_depth True --num_workers=3" C-m
    
    echo "Tmux session '$SESSION_NAME' created and rendering started."
    echo "To attach to the session, run: tmux attach -t $SESSION_NAME"
    echo "To detach from the session, press: Ctrl+b then d"
else
    echo "Tmux session '$SESSION_NAME' already exists."
    echo "To attach to it, run: tmux attach -t $SESSION_NAME"
fi
