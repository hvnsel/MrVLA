#!/bin/bash
# Submit one activation-collection job per remaining LIBERO suite.
#   cd ~/mrvla && bash submit_remaining_suites.sh
set -euo pipefail

for suite in libero_spatial libero_object libero_10; do
  sbatch --job-name="collect_${suite}" run_collect_suite.slurm "${suite}"
done

squeue -u "$USER"
