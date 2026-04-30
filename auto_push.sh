#!/bin/bash
cd /workspace/causalworld
git add *.json checkpoints/ logs/
git commit -m "auto: cloud partial $(date +%Y%m%d_%H%M)" 2>/dev/null
git push origin mechanism-ablation 2>&1 | tail -3
