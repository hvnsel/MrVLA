#!/bin/bash
# Submit one Path-A2 (adapt + SAE-train) job per suite.
#   cd ~/mrvla && bash submit_a2_all.sh
set -euo pipefail

for M in goal spatial object 10; do
  sbatch --job-name="a2_${M}" run_a2_sae.slurm "${M}"
done

squeue -u "$USER"
