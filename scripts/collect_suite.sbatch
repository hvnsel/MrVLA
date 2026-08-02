#!/bin/bash
#SBATCH --job-name=mrvla-collect
#SBATCH --partition=ghx4
#SBATCH --account=CHANGE_ME-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=logs/collect_%x_%j.out
#SBATCH --error=logs/collect_%x_%j.err
#
# Collect OpenVLA residual-stream activations for one LIBERO suite on NCSA DeltaAI.
# Usage:
#   sbatch --job-name=collect_libero_spatial scripts/collect_suite.sbatch libero_spatial
#
# Fill in --account above (e.g. bbXX-dtai-gh) and CONDA_ENV below before submitting.

set -euo pipefail

SUITE="${1:?Pass a LIBERO suite, e.g. libero_spatial}"
CONDA_ENV="mrvla"   # <-- change to the env you used for libero_goal

# DeltaAI is a Grace-Hopper (aarch64) system; load conda then activate the env.
module purge
module load anaconda3_gpu 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "Node:  $(hostname)"
echo "GPU:   ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Suite: ${SUITE}"
nvidia-smi -L || true

python collect_activations.py \
  --source libero-demos \
  --model "openvla/openvla-7b-finetuned-${SUITE//_/-}" \
  --task-suite "${SUITE}" \
  --unnorm-key "${SUITE}" \
  --layers 0,8,16,24,31 \
  --out "./activations/${SUITE}_demos"

echo "Done: ${SUITE}"
