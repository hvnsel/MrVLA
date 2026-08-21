#!/bin/bash
#SBATCH --job-name=channels
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/u/hzhang77/mrvla/logs/channels_%x_%j.out
#
# B1 step 2: slot-resolved attribution + exact per-feature necessity, for ONE suite.
#   sbatch run_channels.slurm goal            # the published Step 5 run, unchanged
#   sbatch run_channels.slurm goal all        # all 2048 features, coded only -> ${M}_all
# Tag: goal | spatial | object | 10
#
# GPU is only for the SAE encode; the counterfactual layer is numpy. Needs no LIBERO and no
# rollouts. Analysis afterwards (analyze_channels.py) is CPU and runs on the login node.
#
# WHY MODE `all` WRITES SOMEWHERE ELSE. results.md P5, P5b, P5c, P5d, P6 and P6a are all
# computed from ${BASE}/CHANNELS/${M}. This pass changes what `_trans` MEANS -- the mask is now
# each channel's own emitted bins rather than the gripper's broadcast to all seven -- so
# overwriting that directory would make six published sections unreproducible from their own
# inputs. New output dir, always.
#
# COST. Going from 396 features to 2048 is NOT 5x. `single_feature_flips` prunes with an exact
# bound, and a CODED coefficient is exactly zero wherever the feature does not fire, so ~90.6%
# of rows are pruned before the argmax; TopK also fixes the number of active (row, feature)
# pairs at n_rows * k regardless of F. Dropping `projection` -- the mode that barely prunes,
# because <h, w_j> is non-zero everywhere -- removes the expensive half of the current job. The
# walltime here is a margin, not an estimate.

set -euo pipefail

M="${1:?Pass a suite tag: goal|spatial|object|10}"
MODE="${2:-candidates}"

export PATH=/u/hzhang77/.conda/envs/mrvla/bin:$PATH
export PYTHONPATH=/u/hzhang77/mrvla:${PYTHONPATH:-}

mkdir -p /u/hzhang77/mrvla/logs
BASE=/work/hdd/bhhv/hzhang77

case "${MODE}" in
  candidates) EXTRA=(--coeff both);            OUT="${BASE}/CHANNELS/${M}" ;;
  all)        EXTRA=(--all-features --coeff coded); OUT="${BASE}/CHANNELS/${M}_all" ;;
  smoke)      EXTRA=(--all-features --coeff coded --max-decisions 8192)
              OUT="${BASE}/CHANNELS/${M}_smoke" ;;
  *) echo "unknown mode '${MODE}': expected candidates | all | smoke" >&2; exit 2 ;;
esac

python /u/hzhang77/mrvla/run_channel_attribution.py \
  --acts-dir "${BASE}/ACT_ACTION/${M}" \
  --sae-dir  "${BASE}/ACT_ACTION_SAE/${M}/sae" \
  --attr     "${BASE}/ATTR/${M}_k100/layer_31_attribution.npz" \
  --layer 31 --top 100 --device cuda:0 "${EXTRA[@]}" \
  --out "${OUT}"

echo "Done ${M} (${MODE}) -> ${OUT}.  Next (login node):"
echo "  python /u/hzhang77/mrvla/analyze_channels.py \\"
echo "    --chan ${OUT}/layer_31_channels.npz \\"
echo "    --attr ${BASE}/ATTR/${M}_k100/layer_31_attribution.npz \\"
echo "    --out  ${OUT}/analysis.json"
if [ "${MODE}" != "candidates" ]; then
echo "  python /u/hzhang77/mrvla/causal_breadth.py \\"
echo "    --chan ${OUT}/layer_31_channels.npz \\"
echo "    --attr ${BASE}/ATTR/${M}_k100/layer_31_attribution.npz \\"
echo "    --suite ${M} --n-perm 1000 --out ${OUT}/causal_breadth.json"
fi
