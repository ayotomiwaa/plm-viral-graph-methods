#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python}"
DIMENSION="${DIMENSION:-2}"
REPLICATES="${REPLICATES:-3}"
OUT_ROOT="${OUT_ROOT:-analysis/cohort_validation/29_seed42_20k_rng_degree_dimension_calibration/random_full_dataset_seed42/seed_42/degree_structure_comparison}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/alina-data1/Ezekiel/tmp/matplotlib_rng_degree_structure}"

mkdir -p "$OUT_ROOT/logs"
mkdir -p "$MPLCONFIGDIR"
LOG_PATH="$OUT_ROOT/logs/degree_structure_comparison_$(date +%Y%m%d_%H%M%S).log"

echo "Writing log to $LOG_PATH"
"$PYTHON" scripts/validation/compare_seed42_rng_degree_structure.py \
  --dimension "$DIMENSION" \
  --replicates "$REPLICATES" \
  --out-root "$OUT_ROOT" 2>&1 | tee "$LOG_PATH"
