#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python}"
DIMENSIONS="${DIMENSIONS:-2,3,4,5,6,7}"
SAMPLE_SIZES="${SAMPLE_SIZES:-20000}"
REPLICATES="${REPLICATES:-3}"
OUT_ROOT="${OUT_ROOT:-analysis/cohort_validation/29_seed42_20k_rng_degree_dimension_calibration/random_full_dataset_seed42/seed_42}"
ADD_BIOLOGICAL_UNIQUE_SIZE="${ADD_BIOLOGICAL_UNIQUE_SIZE:-1}"
FORCE="${FORCE:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/alina-data1/Ezekiel/tmp/matplotlib_rng_degree}"

mkdir -p "$OUT_ROOT/logs"
mkdir -p "$MPLCONFIGDIR"
LOG_PATH="$OUT_ROOT/logs/rng_degree_calibration_$(date +%Y%m%d_%H%M%S).log"

args=(
  scripts/validation/evaluate_seed42_rng_degree_dimension_calibration.py
  --dimensions "$DIMENSIONS"
  --sample-sizes "$SAMPLE_SIZES"
  --replicates "$REPLICATES"
  --out-root "$OUT_ROOT"
)

if [[ "$ADD_BIOLOGICAL_UNIQUE_SIZE" != "1" ]]; then
  args+=(--no-add-biological-unique-size)
fi
if [[ "$FORCE" == "1" ]]; then
  args+=(--force)
fi

echo "Writing log to $LOG_PATH"
"$PYTHON" "${args[@]}" 2>&1 | tee "$LOG_PATH"
"$PYTHON" scripts/validation/plot_seed42_rng_degree_distributions.py \
  --analysis-root "$OUT_ROOT" 2>&1 | tee -a "$LOG_PATH"
