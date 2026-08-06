#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python"
WORKSPACE="${1:-analysis/cohort_validation/15_seed42_20k/raw_distance_ball_temporal_spread/pow2_local_shuffle}"
TEMP_ROOT="/alina-data1/Ezekiel/tmp"

mkdir -p "${TEMP_ROOT}/mplconfig" "${WORKSPACE}/logs"
export TMPDIR="${TEMP_ROOT}"
export MPLCONFIGDIR="${TEMP_ROOT}/mplconfig"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/validation/summarize_raw_distance_ball_temporal_spread.py \
  --workspace "${WORKSPACE}" \
  --source-root analysis/cohort_validation/07_sampling_design_20k \
  --panels random_full_dataset_seed42 \
  --seeds 42 \
  --sample-label pool_n20000 \
  --metric-names raw_hamming,raw_embedding_cityblock \
  --radius-mode powers-of-two \
  --date-shuffle-count 1 \
  --date-shuffle-max-window-days 62 \
  --date-shuffle-attempts-per-node 20 \
  --progress-every 250 \
  2>&1 | tee "${WORKSPACE}/logs/raw_distance_ball_temporal_spread.log"

"${PYTHON_BIN}" scripts/validation/plot_rng_ball_normalized_spread_correlations.py \
  --comparison-kind raw \
  --embedding-summary-csv "${WORKSPACE}/all_raw_embedding_cityblock_raw_distance_ball_temporal_spread_radius_summary.csv" \
  --hamming-summary-csv "${WORKSPACE}/all_raw_hamming_raw_distance_ball_temporal_spread_radius_summary.csv" \
  --out-dir "${WORKSPACE}/normalized_spread_correlations" \
  --bins 20 \
  2>&1 | tee "${WORKSPACE}/logs/normalized_raw_distance_spread_correlations.log"
