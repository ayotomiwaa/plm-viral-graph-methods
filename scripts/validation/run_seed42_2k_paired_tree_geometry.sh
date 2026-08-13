#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/alina-data1/Ezekiel/Protein_embeddings"
PYTHON_BIN="/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python"
SCRIPT="${REPO_ROOT}/scripts/validation/evaluate_seed42_2k_paired_tree_geometry.py"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/analysis/cohort_validation/25_seed42_2k_paired_tree_geometry/random_full_dataset_seed42/seed_42}"
LOG_DIR="${OUT_ROOT}/logs"
STAGES="${STAGES:-all}"
REPRESENTATIONS="${REPRESENTATIONS:-all}"
N_TIPS="${N_TIPS:-2000}"
SELECTION_SEED="${SELECTION_SEED:-42}"
GROMOV_SAMPLES="${GROMOV_SAMPLES:-50000}"
GROMOV_SEED="${GROMOV_SEED:-42}"
PATRISTIC_BLOCK_SIZE="${PATRISTIC_BLOCK_SIZE:-128}"
TREE_BUILDER="${TREE_BUILDER:-skbio}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" "${SCRIPT}" \
  --out-root "${OUT_ROOT}" \
  --stages "${STAGES}" \
  --representations "${REPRESENTATIONS}" \
  --n-tips "${N_TIPS}" \
  --selection-seed "${SELECTION_SEED}" \
  --gromov-samples "${GROMOV_SAMPLES}" \
  --gromov-seed "${GROMOV_SEED}" \
  --patristic-block-size "${PATRISTIC_BLOCK_SIZE}" \
  --prefer-tree-builder "${TREE_BUILDER}" \
  "$@" 2>&1 | tee "${LOG_DIR}/seed42_2k_paired_tree_geometry_${STAGES//,/+}.log"
