#!/bin/bash
#SBATCH --job-name=recur-action
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/recur_action_%j.out
#
# Action-position Path B: shared probe (pool=last) -> recurrence -> identify recurrent features.
#   sbatch run_recurrence_action.slurm

set -euo pipefail

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:/u/hzhang77/LIBERO:${PYTHONPATH:-}
export HF_HOME=/work/hdd/bhhv/hzhang77/hf_cache
export HF_HUB_OFFLINE=1
export LIBERO_DATA_PATH=/work/hdd/bhhv/hzhang77/datasets

mkdir -p /u/hzhang77/mrvla/logs
BASE=/work/hdd/bhhv/hzhang77

echo "=== 1. collect action-position shared probe ==="
python /u/hzhang77/mrvla/collect_shared_probe.py \
  --out "${BASE}/PROBE_ACTION" --pool last --layers 31 --no-flash-attn

echo "=== 2. recurrence on action-position SAEs ==="
python /u/hzhang77/mrvla/run_recurrence.py \
  --probe-dir "${BASE}/PROBE_ACTION" \
  --sae-map "goal=${BASE}/ACT_ACTION_SAE/goal/sae,spatial=${BASE}/ACT_ACTION_SAE/spatial/sae,object=${BASE}/ACT_ACTION_SAE/object/sae,10=${BASE}/ACT_ACTION_SAE/10/sae" \
  --base-key base --layers 31 \
  --out "${BASE}/RECURRENCE_ACTION"

echo "=== 3. identify recurrent features (offline) ==="
python /u/hzhang77/mrvla/identify_recurrent_features.py \
  --rec-dir "${BASE}/RECURRENCE_ACTION" \
  --layer 31 --target goal --top 12 \
  --out "${BASE}/RECURRENCE_ACTION/recurrent_features_goal.json"

echo "Done"
