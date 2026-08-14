#!/bin/bash
#SBATCH --job-name=a2-sae
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/a2_%x_%j.out
#
# Path A2: adapt A1 residuals -> SAE-trainable shards, then train the SAE, for ONE suite.
#   sbatch --job-name=a2_goal run_a2_sae.slurm goal
# Tag: goal | spatial | object | 10

set -euo pipefail

M="${1:?Pass a suite tag: goal|spatial|object|10}"

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:${PYTHONPATH:-}

mkdir -p /u/hzhang77/mrvla/logs

A1=/work/hdd/bhhv/hzhang77/ACT_ACTION/${M}
SAEIN=/work/hdd/bhhv/hzhang77/ACT_ACTION_SAE/${M}
SAEOUT=${SAEIN}/sae

echo "== A2a adapt: ${M} =="
python /u/hzhang77/mrvla/adapt_action_acts_for_sae.py \
  --in-dir  "${A1}" \
  --out-dir "${SAEIN}"

echo "== A2b train SAE: ${M} =="
time python /u/hzhang77/mrvla/mrvla/train_sae.py \
  --acts-dir "${SAEIN}" \
  --out-dir  "${SAEOUT}" \
  --layer 31 --seed 0

echo "Done ${M}"
