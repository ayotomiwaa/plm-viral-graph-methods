#!/usr/bin/env bash
set -euo pipefail

ROOT="/alina-data1/Ezekiel/Protein_embeddings"
PYTHON="/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python"
OUT_ROOT="${OUT_ROOT:-analysis/cohort_validation/31_seed42_20k_rng_timestamp_range_directionality/random_full_dataset_seed42/seed_42}"
PERMUTATIONS="${PERMUTATIONS:-1000}"

cd "$ROOT"
mkdir -p "$OUT_ROOT/logs" /tmp/ezekiel-mpl
export MPLCONFIGDIR="/tmp/ezekiel-mpl"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_ROOT/logs/timestamp_range_directionality_${STAMP}.log"

"$PYTHON" scripts/validation/evaluate_seed42_rng_timestamp_range_directionality.py \
  --out-root "$OUT_ROOT" \
  --permutations "$PERMUTATIONS" \
  2>&1 | tee "$LOG"
