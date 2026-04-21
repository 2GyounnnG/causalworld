#!/bin/bash
# Usage: ./run_in_tmux.sh <session_name> <script_path> [args]

set -e

SESSION="$1"
SCRIPT="$2"
shift 2
ARGS="$@"

if [ -z "$SESSION" ] || [ -z "$SCRIPT" ]; then
  echo "Usage: $0 <session_name> <script_path> [args]"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  echo "Or kill first: tmux kill-session -t $SESSION"
  exit 1
fi

LOG="logs/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Find conda.sh from all common locations (miniforge3 first)
CONDA_SH=""
for path in ~/miniforge3 ~/miniconda3 ~/anaconda3 /opt/conda; do
  if [ -f "$path/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$path/etc/profile.d/conda.sh"
    break
  fi
done

if [ -z "$CONDA_SH" ]; then
  echo "ERROR: could not find conda.sh in any standard location"
  exit 1
fi

tmux new-session -d -s "$SESSION" -c "$(pwd)" \
  "source $CONDA_SH && \
   conda activate causalworld && \
   echo '=== Starting $SCRIPT at $(date) ===' | tee -a $LOG && \
   echo '=== Python: '\$(which python) | tee -a $LOG && \
   echo '=== GPU: '\$(nvidia-smi --query-gpu=name --format=csv,noheader) | tee -a $LOG && \
   python -u $SCRIPT $ARGS 2>&1 | tee -a $LOG; \
   echo '=== Finished at '\$(date)' (exit='\$?')' | tee -a $LOG; \
   echo; echo 'Task complete. Press any key to close.'; read -n 1"

echo ""
echo "✓ Started tmux session '$SESSION'"
echo "  Log: $LOG"
echo ""
echo "Monitor:  tmux attach -t $SESSION      # Ctrl+B D to detach"
echo "          tail -f $LOG"
