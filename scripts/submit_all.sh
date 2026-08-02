#!/bin/bash
# Submit one activation-collection job per LIBERO suite on DeltaAI.
# Run from the repo root:  bash scripts/submit_all.sh
set -euo pipefail

SUITES=(libero_spatial libero_object libero_10)

for suite in "${SUITES[@]}"; do
  sbatch --job-name="collect_${suite}" scripts/collect_suite.sbatch "${suite}"
done
