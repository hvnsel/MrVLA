#!/usr/bin/env bash
# Run the whole cheap-diagnostic suite over existing artifacts. CPU only, no GPU, no
# rollouts, no retraining -- everything here is re-analysis of files already on disk and
# finishes in minutes on a login node.
#
#   BASE=/work/.../mrvla ./run_diagnostics.sh [output_dir]
#
# Skips any stage whose inputs are missing, so it is safe to run before every suite has
# landed. See notes/elevation_diagnostics.md for what each stage is for.

set -uo pipefail

BASE="${BASE:?set BASE to the artifact root (the dir holding ATTR/, ABLATION/, RECURRENCE_ACTION/)}"
OUT="${1:-$BASE/DIAGNOSTICS}"
mkdir -p "$OUT"

SUITES=(goal spatial object 10)

attr_path() { echo "$BASE/ATTR/$1_k100/layer_31_attribution.npz"; }

echo "=== 1. Path A negative control: is partial|both above its floor? ==="
ATTR_ARGS=()
for s in "${SUITES[@]}"; do
    p="$(attr_path "$s")"
    [ -f "$p" ] && ATTR_ARGS+=(--attr "$s=$p")
done
if [ ${#ATTR_ARGS[@]} -gt 0 ]; then
    python permutation_null.py "${ATTR_ARGS[@]}" --n-perm 1000 \
        --show-invalid-null --out "$OUT/permutation_null.json" | tee "$OUT/permutation_null.txt"
else
    echo "  (skip: no attribution npz found under $BASE/ATTR)"
fi

echo
echo "=== 2. Concentration and cross-task reproducibility of causal influence ==="
if [ ${#ATTR_ARGS[@]} -gt 0 ]; then
    python causal_concentration.py "${ATTR_ARGS[@]}" --top 50 \
        --out "$OUT/concentration.json" | tee "$OUT/concentration.txt"
else
    echo "  (skip: no attribution npz found)"
fi

echo
echo "=== 3. Breadth reliability (split-half) ==="
if [ ${#ATTR_ARGS[@]} -gt 0 ]; then
    python split_half_breadth.py "${ATTR_ARGS[@]}" --n-splits 200 \
        --out "$OUT/split_half_breadth.png" | tee "$OUT/split_half.txt"
else
    echo "  (skip: no attribution npz found)"
fi

echo
echo "=== 4. Ablation: paired tests, damage intervals, and the power bound ==="
shopt -s nullglob
for d in "$BASE"/ABLATION/*/; do
    [ -f "$d/manifest.json" ] || continue
    name="$(basename "$d")"
    echo "--- $name ---"
    python ablation_power.py --dir "$d" --per-task --target-effect 0.05 \
        --out "$OUT/power_$name.json" | tee "$OUT/power_$name.txt"
done
shopt -u nullglob

echo
echo "=== 5. Attenuation correction on the A x B null ==="
SH="$OUT/split_half_breadth.json"
JOIN="$BASE/RECURRENCE_ACTION/join_pathA_pathB.json"
if [ -f "$SH" ] && [ -f "$JOIN" ]; then
    python reliability_ceiling.py --split-half "$SH" --join "$JOIN" \
        --partial 0.493 --out "$OUT/reliability.json" | tee "$OUT/reliability.txt"
else
    echo "  (skip: need $SH and $JOIN)"
fi

echo
echo "=== 6. Causal breadth: does breadth predict DECISIVENESS on a held-out task? ==="
# Needs a channel run carrying the per-(feature, task, slot) counters, i.e. one produced with
# --all-features (sbatch run_channels.slurm <suite> all). The candidate-only runs under
# CHANNELS/<suite> cover the two extremes of the predictor under test and are deliberately not
# used here: extreme-group sampling inflates the correlation by construction.
shopt -s nullglob
found_cb=0
for s in "${SUITES[@]}"; do
    cp="$BASE/CHANNELS/${s}_all/layer_31_channels.npz"
    ap="$(attr_path "$s")"
    [ -f "$cp" ] && [ -f "$ap" ] || continue
    found_cb=1
    echo "--- $s ---"
    python causal_breadth.py --chan "$cp" --attr "$ap" --suite "$s" --n-perm 1000 \
        --out "$OUT/causal_breadth_$s.json" | tee "$OUT/causal_breadth_$s.txt"
done
shopt -u nullglob
if [ "$found_cb" -eq 0 ]; then
    echo "  (skip: no CHANNELS/<suite>_all run found -- sbatch run_channels.slurm <suite> all)"
fi

echo
echo "wrote $OUT"
