#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python"
WORKSPACE="${1:-analysis/cohort_validation/15_seed42_20k/zero_skew_constructive_lower_bound/hamming_embedding_knn05_knn50_rng}"
TEMP_ROOT="/alina-data1/Ezekiel/tmp"

mkdir -p "${TEMP_ROOT}/mplconfig" "${WORKSPACE}/logs"
export TMPDIR="${TEMP_ROOT}"
export MPLCONFIGDIR="${TEMP_ROOT}/mplconfig"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/validation/compute_zero_skew_constructive_lower_bound.py \
  prepare-graph-distances \
  --workspace "${WORKSPACE}" \
  --metrics hamming_knn_k05,hamming_knn_k50,embedding_knn_k05,embedding_knn_k50 \
  --batch-size "${BATCH_SIZE:-32}" \
  2>&1 | tee "${WORKSPACE}/logs/prepare_knn_weighted_shortest_paths.log"

"${PYTHON_BIN}" scripts/validation/compute_zero_skew_constructive_lower_bound.py \
  compute \
  --workspace "${WORKSPACE}" \
  --metrics all \
  --progress-every 500 \
  2>&1 | tee "${WORKSPACE}/logs/compute_zero_skew_constructive_lower_bound.log"
