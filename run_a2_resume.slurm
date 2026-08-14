#!/bin/bash
#SBATCH --job-name=a2-resume
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/a2_resume_%x_%j.out
#
# Path A gate fix: resume a previously trained action-position SAE for more epochs.
#   sbatch run_a2_resume.slurm <suite> <k> <prev_out_dir_name> <target_total_epochs>
#   e.g.  sbatch run_a2_resume.slurm goal 256 sae_k256 300
#
# Reads checkpoint from ACT_ACTION_SAE/<suite>/<prev_out_dir_name>/layer_31/final.pt
# and writes the extended run to ACT_ACTION_SAE/<suite>/<prev_out_dir_name>_e<target_total_epochs>/

set -euo pipefail

M="${1:?Pass a suite tag: goal|spatial|object|10}"
K="${2:?Pass a k value, e.g. 256}"
PREV="${3:?Pass the previous out-dir name, e.g. sae_k256}"
EPOCHS="${4:?Pass the target TOTAL epoch count, e.g. 300}"

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:${PYTHONPATH:-}

mkdir -p /u/hzhang77/mrvla/logs

SAEIN=/work/hdd/bhhv/hzhang77/ACT_ACTION_SAE/${M}
RESUME_FROM=${SAEIN}/${PREV}
SAEOUT=${SAEIN}/${PREV}_e${EPOCHS}

echo "== A2b resume SAE: ${M} k=${K} resume-from=${RESUME_FROM} -> epochs=${EPOCHS} =="
time python /u/hzhang77/mrvla/mrvla/train_sae.py \
  --acts-dir "${SAEIN}" \
  --out-dir  "${SAEOUT}" \
  --layer 31 --seed 0 --k "${K}" \
  --epochs "${EPOCHS}" \
  --resume-from "${RESUME_FROM}"

echo "Done ${M} k=${K} epochs=${EPOCHS}"
