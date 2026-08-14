#!/bin/bash
#SBATCH --job-name=a3-pipe
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/a3_%x_%j.out
#
# A3/A4 pipeline: attribution gate + identify features + capture frames, for ONE suite.
#   sbatch --dependency=afterok:<A2_JOBID> run_a3_pipeline.slurm spatial
# Tag: goal | spatial | object | 10

set -euo pipefail

M="${1:?Pass a suite tag: goal|spatial|object|10}"

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:/u/hzhang77/LIBERO:${PYTHONPATH:-}
export HF_HOME=/work/hdd/bhhv/hzhang77/hf_cache
export HF_HUB_OFFLINE=1
export LIBERO_DATA_PATH=/work/hdd/bhhv/hzhang77/datasets

mkdir -p /u/hzhang77/mrvla/logs

BASE=/work/hdd/bhhv/hzhang77

echo "=== A3/A4 attribution: ${M} ==="
python /u/hzhang77/mrvla/run_attribution.py \
  --acts-dir "${BASE}/ACT_ACTION/${M}" \
  --sae-dir  "${BASE}/ACT_ACTION_SAE/${M}/sae" \
  --layer 31 --out "${BASE}/ATTR/${M}_k100"

echo "=== identify features: ${M} ==="
python /u/hzhang77/mrvla/identify_features.py \
  --acts-dir  "${BASE}/ACT_ACTION/${M}" \
  --sae-dir   "${BASE}/ACT_ACTION_SAE/${M}/sae" \
  --attr      "${BASE}/ATTR/${M}_k100/layer_31_attribution.npz" \
  --layer 31 --top 8 --exemplars 12 \
  --out "${BASE}/FEATURES/${M}_k100"

echo "=== capture frames: ${M} ==="
python /u/hzhang77/mrvla/capture_feature_frames.py \
  --acts-dir  "${BASE}/ACT_ACTION/${M}" \
  --exemplars "${BASE}/FEATURES/${M}_k100/exemplars.json" \
  --task-suite "libero_${M}" \
  --out "${BASE}/FEATURES/${M}_k100/frames"

echo "Done ${M}"
