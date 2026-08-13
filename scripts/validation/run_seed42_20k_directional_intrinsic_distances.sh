#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python"
PHASE="${1:-all}"
OUT_ROOT="${2:-analysis/cohort_validation/24_seed42_20k_directional_intrinsic_distances/random_full_dataset_seed42/seed_42}"
TEMP_ROOT="/alina-data1/Ezekiel/tmp"
SCRIPT="scripts/validation/build_directional_intrinsic_distances.py"

mkdir -p "${TEMP_ROOT}/directional_intrinsic" "${OUT_ROOT}/logs"
export TMPDIR="${TEMP_ROOT}/directional_intrinsic"
export OMP_NUM_THREADS="${DIRECTIONAL_BLAS_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${DIRECTIONAL_BLAS_THREADS:-8}"
export MKL_NUM_THREADS="${DIRECTIONAL_BLAS_THREADS:-8}"

cd "${REPO_ROOT}"

run_validate() {
  "${PYTHON_BIN}" "${SCRIPT}" validate-inputs \
    --out-root "${OUT_ROOT}" \
    --candidate-fraction 0.10 \
    --delta 0.01 \
    2>&1 | tee "${OUT_ROOT}/logs/validate_inputs.log"
}

run_filter() {
  "${PYTHON_BIN}" "${SCRIPT}" filter-graphs \
    --out-root "${OUT_ROOT}" \
    --candidate-fraction 0.10 \
    --delta 0.01 \
    --node-chunk-size "${NODE_CHUNK_SIZE:-64}" \
    2>&1 | tee "${OUT_ROOT}/logs/filter_graphs.log"
}

run_distances() {
  "${PYTHON_BIN}" "${SCRIPT}" prepare-distances \
    --out-root "${OUT_ROOT}" \
    --candidate-fraction 0.10 \
    --delta 0.01 \
    --variants baseline,refined \
    --batch-size "${DIJKSTRA_BATCH_SIZE:-16}" \
    2>&1 | tee "${OUT_ROOT}/logs/prepare_distances.log"
}

run_summarize() {
  "${PYTHON_BIN}" "${SCRIPT}" summarize \
    --out-root "${OUT_ROOT}" \
    --candidate-fraction 0.10 \
    --delta 0.01 \
    --sample-pairs "${SUMMARY_SAMPLE_PAIRS:-200000}" \
    --sample-seed 42 \
    2>&1 | tee "${OUT_ROOT}/logs/summarize.log"
}

case "${PHASE}" in
  validate|preflight)
    run_validate
    ;;
  filter)
    run_filter
    ;;
  distances)
    run_distances
    ;;
  summarize)
    run_summarize
    ;;
  all)
    run_validate
    run_filter
    run_distances
    run_summarize
    ;;
  *)
    echo "Usage: $0 {validate|filter|distances|summarize|all} [output_root]" >&2
    exit 2
    ;;
esac
