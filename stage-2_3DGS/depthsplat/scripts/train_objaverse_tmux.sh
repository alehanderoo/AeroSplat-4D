#!/bin/bash
# Script to run objaverse training in a tmux session

SESSION_NAME="objaverse_run"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$DIR/.."

# Check if session exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Attaching..."
    tmux attach-session -t "$SESSION_NAME"
    exit 0
fi

# Create new session
echo "Creating new tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT"

# Activate conda environment inside tmux session, then run training
tmux send-keys -t "$SESSION_NAME" "source ~/anaconda3/etc/profile.d/conda.sh && conda activate depthsplat && \\" C-m
tmux send-keys -t "$SESSION_NAME" "python -m src.main +experiment=objaverse_white_small_gauss_depthMask \\" C-m
tmux send-keys -t "$SESSION_NAME" "    wandb.entity=a-a-f-verdiesen-tu-delft " C-m
# tmux send-keys -t "$SESSION_NAME" "    wandb.name=objaverse_white_bg_118k " C-m
# tmux send-keys -t "$SESSION_NAME" "    trainer.max_steps=100000" C-m

# Attach to the session
tmux attach-session -t "$SESSION_NAME"
