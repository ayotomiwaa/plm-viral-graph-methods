#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python"
WORKSPACE="${1:-analysis/cohort_validation/15_seed42_20k/graph_box_counting/hamming_embedding_knn05_knn50_rng}"
TRIALS="${TRIALS:-100}"
TEMP_ROOT="/alina-data1/Ezekiel/tmp"

mkdir -p "${TEMP_ROOT}/mplconfig" "${WORKSPACE}/logs"
export TMPDIR="${TEMP_ROOT}"
export MPLCONFIGDIR="${TEMP_ROOT}/mplconfig"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/validation/evaluate_graph_box_counting.py \
  --workspace "${WORKSPACE}" \
  --source-root analysis/cohort_validation/07_sampling_design_20k \
  --panel random_full_dataset_seed42 \
  --seed 42 \
  --sample-label pool_n20000 \
  --box-sizes 3,5,7,9,11,13,15,17,19,21 \
  --trials "${TRIALS}" \
  --random-seed 42 \
  2>&1 | tee "${WORKSPACE}/logs/graph_box_counting.log"
