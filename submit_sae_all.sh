#!/bin/bash
# Train SAE + extract codes + classify for every suite (one job each).
#   cd ~/mrvla && bash submit_sae_all.sh
set -euo pipefail

for suite in libero_goal libero_spatial libero_object libero_10; do
  sbatch --job-name="sae_${suite}" run_sae_suite.slurm "${suite}"
done

squeue -u "$USER"
