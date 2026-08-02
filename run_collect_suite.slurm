#!/bin/bash
#SBATCH --job-name=mrvla-collect
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/collect_%x_%j.out
#
# Same as run_collect.slurm but the LIBERO suite is passed as an argument:
#   sbatch --job-name=collect_libero_spatial run_collect_suite.slurm libero_spatial

set -euo pipefail

SUITE="${1:?Pass a LIBERO suite, e.g. libero_spatial}"
MODEL_SUITE="${SUITE//_/-}"

mkdir -p /u/hzhang77/mrvla/logs

module load python/miniforge3_pytorch
conda activate mrvla

export HF_HOME=/u/hzhang77/.cache/huggingface
export LIBERO_DATA_PATH=/u/hzhang77/LIBERO/libero/datasets

echo "Suite: ${SUITE}  Model: openvla/openvla-7b-finetuned-${MODEL_SUITE}"

time srun python /u/hzhang77/mrvla/collect_activations.py \
  --source libero-demos \
  --model "openvla/openvla-7b-finetuned-${MODEL_SUITE}" \
  --task-suite "${SUITE}" \
  --unnorm-key "${SUITE}" \
  --layers 0,8,16,24,31 \
  --out "/u/hzhang77/mrvla/activations/${SUITE}_demos"
