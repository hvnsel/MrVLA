#!/bin/bash
#SBATCH --job-name=sae-seed1
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/sae_seed1_%x_%j.out
#
# Second-seed ACTION-POSITION SAE, for ONE suite.
#   sbatch run_sae_seed1.slurm goal
#
# WHY THIS IS A PREREQUISITE, not an extra. Every A4 number is a cosine, and a cosine has no
# scale on its own. The same-model different-seed match is the CEILING -- the most any
# cross-model comparison could reach given that SAE dictionaries are only ~60% seed-reproducible
# (EXPERIMENT_PLAN section 2.5). Without it, inventory_recurrence.py and inventory_clusters.py
# will run but will report their cosines as uncalibrated, and chance-corrected retention (the
# ret_cc statistic of section 2.4) cannot be computed at all.
#
# NB the existing seed-1 SAE from section 2.5 was trained on the POOLED activations. A4 works on
# ACTION-POSITION SAEs, so it needs its own second seed with matching feature indices; the
# pooled one is not a substitute.

set -euo pipefail

M="${1:?Pass a suite tag: goal|spatial|object|10}"

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:${PYTHONPATH:-}

mkdir -p /u/hzhang77/mrvla/logs
BASE=/work/hdd/bhhv/hzhang77

python /u/hzhang77/mrvla/mrvla/train_sae.py \
  --acts-dir "${BASE}/ACT_ACTION_SAE/${M}" \
  --out-dir  "${BASE}/ACT_ACTION_SAE/${M}/sae_seed1" \
  --layer 31 --seed 1 --epochs 250

echo "Done ${M} seed 1 -> ${BASE}/ACT_ACTION_SAE/${M}/sae_seed1"
