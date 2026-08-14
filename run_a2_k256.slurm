#!/bin/bash
#SBATCH --job-name=a2-k256
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/a2_k256_%x_%j.out
#
# Path A gate fix: retrain goal's action-position SAE with k=256 (was k=100, L1=0.72).
#   sbatch run_a2_k256.slurm goal 256
# Tag: goal | spatial | object | 10   K: sparsity level to try

set -euo pipefail

M="${1:?Pass a suite tag: goal|spatial|object|10}"
K="${2:?Pass a k value, e.g. 256}"

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:${PYTHONPATH:-}

mkdir -p /u/hzhang77/mrvla/logs

SAEIN=/work/hdd/bhhv/hzhang77/ACT_ACTION_SAE/${M}
SAEOUT=${SAEIN}/sae_k${K}

echo "== A2b retrain SAE: ${M} k=${K} =="
time python /u/hzhang77/mrvla/mrvla/train_sae.py \
  --acts-dir "${SAEIN}" \
  --out-dir  "${SAEOUT}" \
  --layer 31 --seed 0 --k "${K}"

echo "Done ${M} k=${K}"
