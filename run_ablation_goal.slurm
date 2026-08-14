#!/bin/bash
#SBATCH --job-name=ablation-goal
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --array=0-6
#SBATCH --output=/u/hzhang77/mrvla/logs/ablation_%x_%A_%a.out
#
# Single-feature ablation: goal suite, baseline + 6 named singleton conditions
# (only_1167/1235/1628/1999/1140/1134), 20 episodes/task, 7 array tasks x 10 jobs
# each (one GPU per task -- use an array instead of a same-node multi-GPU loop).
#   sbatch run_ablation_goal.slurm

set -euo pipefail

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:/u/hzhang77/LIBERO:${PYTHONPATH:-}
export HF_HOME=/work/hdd/bhhv/hzhang77/hf_cache
export HF_HUB_OFFLINE=1
export LIBERO_DATA_PATH=/work/hdd/bhhv/hzhang77/datasets

mkdir -p /u/hzhang77/mrvla/logs

BASE=/work/hdd/bhhv/hzhang77

python /u/hzhang77/mrvla/run_ablation.py \
  --model openvla/openvla-7b-finetuned-libero-goal \
  --task-suite libero_goal --unnorm-key libero_goal \
  --sae-dir "${BASE}/ACT_ACTION_SAE/goal/sae" \
  --attr    "${BASE}/ATTR/goal_k100/layer_31_attribution.npz" \
  --conditions baseline \
  --ablate-each 1167,1235,1628,1999,1140,1134 \
  --episodes-per-task 20 \
  --worker-id "${SLURM_ARRAY_TASK_ID}" --n-workers 7 --device cuda:0 \
  --no-flash-attn \
  --out "${BASE}/ABLATION/goal_singles"

echo "Done (array task ${SLURM_ARRAY_TASK_ID})"
