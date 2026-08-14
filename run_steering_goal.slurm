#!/bin/bash
#SBATCH --job-name=steer-goal
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --array=0-6
#SBATCH --output=/u/hzhang77/mrvla/logs/steer_steer-goal_%A_%a.out
#
# Additive steering: goal suite, baseline + 6 named steered conditions, 10 episodes/task,
# 2 videos/task, 7 array tasks x 10 jobs each (one GPU per task).
# Fill in ALPHA below with the value printed by run_steering_calibrate.slurm before submitting.
#   sbatch run_steering_goal.slurm

set -euo pipefail

ALPHA=REPLACE_ME

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:/u/hzhang77/LIBERO:${PYTHONPATH:-}
export HF_HOME=/work/hdd/bhhv/hzhang77/hf_cache
export HF_HUB_OFFLINE=1
export LIBERO_DATA_PATH=/work/hdd/bhhv/hzhang77/datasets

mkdir -p /u/hzhang77/mrvla/logs

BASE=/work/hdd/bhhv/hzhang77

python /u/hzhang77/mrvla/run_steering.py \
  --model openvla/openvla-7b-finetuned-libero-goal \
  --task-suite libero_goal --unnorm-key libero_goal \
  --sae-dir "${BASE}/ACT_ACTION_SAE/goal/sae" \
  --features 1167,1235,1628,1999,1140,1134 \
  --alpha "${ALPHA}" --episodes-per-task 10 --video-episodes 2 \
  --worker-id "${SLURM_ARRAY_TASK_ID}" --n-workers 7 --device cuda:0 \
  --no-flash-attn \
  --out "${BASE}/STEER/goal"

echo "Done (array task ${SLURM_ARRAY_TASK_ID})"
