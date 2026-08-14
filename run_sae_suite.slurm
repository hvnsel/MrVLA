#!/bin/bash
#SBATCH --job-name=sae
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/sae_%x_%j.out
#
# Train SAE (defaults) -> extract codes -> generality classifier, for ONE suite.
#   sbatch --job-name=sae_libero_spatial run_sae_suite.slurm libero_spatial

set -euo pipefail

SUITE="${1:?Pass a LIBERO suite, e.g. libero_spatial}"

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:${PYTHONPATH:-}

mkdir -p /u/hzhang77/mrvla/logs

ACTS=/work/hdd/bhhv/hzhang77/activations/${SUITE}_demos
ROOT=/work/hdd/bhhv/hzhang77/${SUITE}_demos
SAE=${ROOT}/sae_v4
CODES=${ROOT}/codes_v4
GEN=${ROOT}/generality_v4

echo "== ${SUITE}: train SAE =="
time srun python /u/hzhang77/mrvla/mrvla/train_sae.py \
  --acts-dir "${ACTS}" \
  --out-dir "${SAE}" \
  --layer all

echo "== ${SUITE}: extract codes =="
time srun python /u/hzhang77/mrvla/mrvla/extract_codes_and_metrics.py \
  --acts-dir "${ACTS}" \
  --sae-dir "${SAE}" \
  --out-dir "${CODES}"

echo "== ${SUITE}: generality classifier =="
srun python /u/hzhang77/mrvla/mrvla/generality_classifier.py \
  --codes-dir "${CODES}" \
  --out-dir "${GEN}" \
  --dataset libero

echo "Done ${SUITE}. Summary:"
cat "${GEN}/summary.json"
