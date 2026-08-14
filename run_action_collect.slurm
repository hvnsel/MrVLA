#!/bin/bash
#SBATCH --job-name=a1-collect
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/a1_%x_%j.out
#
# Path A1: collect action-position residuals for ONE suite.
#   sbatch --job-name=a1_goal run_action_collect.slurm goal
# Tag is the short suite name: goal | spatial | object | 10

set -euo pipefail

M="${1:?Pass a suite tag: goal|spatial|object|10}"

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:/u/hzhang77/LIBERO:${PYTHONPATH:-}
export HF_HOME=/work/hdd/bhhv/hzhang77/hf_cache
export HF_HUB_OFFLINE=1
export LIBERO_DATA_PATH=/work/hdd/bhhv/hzhang77/datasets

mkdir -p /u/hzhang77/mrvla/logs

echo "== A1 collect: libero_${M} =="
time python /u/hzhang77/mrvla/collect_action_activations.py \
  --model "openvla/openvla-7b-finetuned-libero-${M}" \
  --task-suite "libero_${M}" \
  --unnorm-key "libero_${M}" \
  --layer 31 \
  --out "/work/hdd/bhhv/hzhang77/ACT_ACTION/${M}" \
  --no-flash-attn

echo "Done libero_${M}"
