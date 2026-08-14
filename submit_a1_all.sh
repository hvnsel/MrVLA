#!/bin/bash
# Submit one Path-A1 action-residual collection job per suite.
#   cd ~/mrvla && bash submit_a1_all.sh
set -euo pipefail

for M in goal spatial object 10; do
  sbatch --job-name="a1_${M}" run_action_collect.slurm "${M}"
done

squeue -u "$USER"
