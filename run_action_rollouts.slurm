#!/bin/bash
#SBATCH --job-name=actroll
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/actroll_%x_%j.out
#
# Closed-loop rollouts storing action-position residuals AND success labels.
#   sbatch run_action_rollouts.slurm goal
#
# 50 init states x 10 tasks = 500 episodes, sharded over 4 GPUs by task. ~5 GB.
# Sharding is by TASK, so each worker's init-state ordering matches a single-process run
# and success rates stay comparable with the ablation runs.
#
# SMOKE TEST FIRST on ghx4-interactive -- this has never run:
#   python collect_action_rollouts.py --model openvla/openvla-7b-finetuned-libero-goal \
#     --task-suite libero_goal --unnorm-key libero_goal \
#     --max-tasks 1 --trials-per-task 2 --out /tmp/rollout_smoke
# Check: shard shapes are [n, 7, 4096], success is not all-1, and head_constants.npz exists.

set -euo pipefail

M="${1:?Pass a suite tag: goal|spatial|object|10}"

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:${PYTHONPATH:-}

mkdir -p /u/hzhang77/mrvla/logs
BASE=/work/hdd/bhhv/hzhang77
OUT="${BASE}/ROLLOUT_ACTION/${M}"

for i in 0 1 2 3; do
  python /u/hzhang77/mrvla/collect_action_rollouts.py \
    --model "openvla/openvla-7b-finetuned-libero-${M}" \
    --task-suite "libero_${M}" --unnorm-key "libero_${M}" \
    --layer 31 --trials-per-task 50 \
    --worker-id "$i" --n-workers 4 --device "cuda:$i" \
    --out "$OUT" &
done
wait

echo "Done ${M}. Episodes and failures per worker:"
cat "${OUT}"/rollout_summary_w*.json
