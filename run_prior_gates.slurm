#!/bin/bash
#SBATCH --job-name=priorgates
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/priorgates_%x_%j.out
#
# Gates 0-2 of the prior ladder, for ONE suite.
#   sbatch run_prior_gates.slurm goal
# Tag: goal | spatial | object | 10
#
# GPU is only for the SAE encode; every gate is numpy. Needs no LIBERO and no rollouts.
#
# RUN THIS FIRST AND READ GATE 1 BEFORE DOING ANYTHING ELSE. Gate 1 is the structural kill:
# phi carries an (l2/r) prefactor, so if mu tracks r then mu/r is constant, the lever cancels,
# and Gates 2-4 are computing properties of a quantity that does not exist. Do not set up
# norm_stats or the LIBERO stack for Gates 3-4 until Gate 1 passes.

set -euo pipefail

M="${1:?Pass a suite tag: goal|spatial|object|10}"

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:${PYTHONPATH:-}

mkdir -p /u/hzhang77/mrvla/logs
BASE=/work/hdd/bhhv/hzhang77

python /u/hzhang77/mrvla/prior_gates.py \
  --acts-dir "${BASE}/ACT_ACTION/${M}" \
  --sae-dir  "${BASE}/ACT_ACTION_SAE/${M}/sae" \
  --layer 31 --device cuda:0 \
  --out "${BASE}/ATTR/${M}_k100/prior_gates.json"

echo
echo "Gates 0-2 done for ${M}.  If Gate 1 PASSED, Gates 3-4 need two more things:"
echo "  1. export the checkpoint's norm_stats once (GPU node, ~2 min):"
echo "       python -c \"import json; from mrvla.model_utils import load_openvla; \\"
echo "         m,_=load_openvla('openvla/openvla-7b-finetuned-libero-${M}'); \\"
echo "         json.dump({k:{'action':v['action']} for k,v in m.norm_stats.items()}, \\"
echo "                   open('${BASE}/norm_stats_${M}.json','w'), default=list)\""
echo "  2. re-run with the LIBERO stack importable:"
echo "       python /u/hzhang77/mrvla/prior_gates.py \\"
echo "         --acts-dir ${BASE}/ACT_ACTION/${M} --sae-dir ${BASE}/ACT_ACTION_SAE/${M}/sae \\"
echo "         --task-suite libero_${M} --norm-stats ${BASE}/norm_stats_${M}.json \\"
echo "         --unnorm-key libero_${M} \\"
echo "         --out ${BASE}/ATTR/${M}_k100/prior_gates_full.json"
