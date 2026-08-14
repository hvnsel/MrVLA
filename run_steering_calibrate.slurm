#!/bin/bash
#SBATCH --job-name=steer-calibrate
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/u/hzhang77/mrvla/logs/steer_calibrate_%j.out
#
# One forward pass to measure mean ||h|| at layer 31 and print the implied alpha for gamma=1.0.
# Read the printed alpha and pass it as --alpha to run_steering_goal.slurm.
#   sbatch run_steering_calibrate.slurm

set -euo pipefail

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
  --gamma 1.0 --calibrate-only --no-flash-attn \
  --out "${BASE}/STEER/goal"
