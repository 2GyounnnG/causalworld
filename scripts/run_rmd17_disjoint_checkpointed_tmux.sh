#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

echo "Aspirin disjoint checkpointed:"
echo "  tmux new-session -d -s rmd17_aspirin_disjoint 'cd $ROOT && source ~/miniforge3/etc/profile.d/conda.sh && conda activate causalworld && python scripts/run_rmd17_aspirin_disjoint_checkpointed.py 2>&1 | tee logs/rmd17_aspirin_disjoint_checkpointed.log'"
echo "Multimolecule disjoint checkpointed:"
echo "  tmux new-session -d -s rmd17_multimolecule_disjoint 'cd $ROOT && source ~/miniforge3/etc/profile.d/conda.sh && conda activate causalworld && python scripts/run_rmd17_multimolecule_disjoint_checkpointed.py 2>&1 | tee logs/rmd17_multimolecule_disjoint_checkpointed.log'"

if [[ "${1:-}" == "--start" ]]; then
  start_session() {
    local session="$1"
    local script="$2"
    local log="$3"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "session already running: $session"
      return
    fi
    tmux new-session -d -s "$session" \
      "cd $ROOT && source ~/miniforge3/etc/profile.d/conda.sh && conda activate causalworld && python $script 2>&1 | tee $log"
  }

  start_session rmd17_aspirin_disjoint scripts/run_rmd17_aspirin_disjoint_checkpointed.py logs/rmd17_aspirin_disjoint_checkpointed.log
  start_session rmd17_multimolecule_disjoint scripts/run_rmd17_multimolecule_disjoint_checkpointed.py logs/rmd17_multimolecule_disjoint_checkpointed.log
  echo "started tmux sessions: rmd17_aspirin_disjoint, rmd17_multimolecule_disjoint"
fi
