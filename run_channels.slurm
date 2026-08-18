#!/bin/bash
#SBATCH --job-name=channels
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/channels_%x_%j.out
#
# B1 step 2: slot-resolved attribution + exact per-feature necessity, for ONE suite.
#   sbatch run_channels.slurm goal
# Tag: goal | spatial | object | 10
#
# GPU is only for the SAE encode; the counterfactual layer is numpy. Needs no LIBERO and no
# rollouts. Analysis afterwards (analyze_channels.py) is CPU and runs on the login node.

set -euo pipefail

M="${1:?Pass a suite tag: goal|spatial|object|10}"

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:${PYTHONPATH:-}

mkdir -p /u/hzhang77/mrvla/logs
BASE=/work/hdd/bhhv/hzhang77

python /u/hzhang77/mrvla/run_channel_attribution.py \
  --acts-dir "${BASE}/ACT_ACTION/${M}" \
  --sae-dir  "${BASE}/ACT_ACTION_SAE/${M}/sae" \
  --attr     "${BASE}/ATTR/${M}_k100/layer_31_attribution.npz" \
  --layer 31 --top 100 --coeff both --device cuda:0 \
  --out "${BASE}/CHANNELS/${M}"

echo "Done ${M}.  Next (login node):"
echo "  python /u/hzhang77/mrvla/analyze_channels.py \\"
echo "    --chan ${BASE}/CHANNELS/${M}/layer_31_channels.npz \\"
echo "    --attr ${BASE}/ATTR/${M}_k100/layer_31_attribution.npz \\"
echo "    --out  ${BASE}/CHANNELS/${M}/analysis.json"
