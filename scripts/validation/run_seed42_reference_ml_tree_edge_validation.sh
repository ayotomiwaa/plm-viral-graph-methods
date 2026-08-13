#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/alina-data1/Ezekiel/Protein_embeddings}"
PYTHON_BIN="${PYTHON_BIN:-/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/analysis/cohort_validation/26_seed42_reference_ml_tree_edge_validation/random_full_dataset_seed42/seed_42}"
LOG_DIR="${OUT_ROOT}/logs"
PHASE="${PHASE:-panel}"

GRAPH_PRIORITY="${GRAPH_PRIORITY:-rng,knn5,knn50}"
SELECTION_SEED="${SELECTION_SEED:-42}"
MAX_UNIQUE_SEQUENCES="${MAX_UNIQUE_SEQUENCES:-3000}"
MAX_TIPS="${MAX_TIPS:-6000}"
CALIPER_FRACTION="${CALIPER_FRACTION:-0.25}"
CONTROLS_PER_ENDPOINT="${CONTROLS_PER_ENDPOINT:-1}"
IQTREE_BINARY="${IQTREE_BINARY:-iqtree2}"
IQTREE_MODEL="${IQTREE_MODEL:-LG+F+G4}"
IQTREE_THREADS="${IQTREE_THREADS:-AUTO}"
IQTREE_MEMORY="${IQTREE_MEMORY:-16G}"

QUARTET_SAMPLES="${QUARTET_SAMPLES:-2000000}"
PERMUTATIONS="${PERMUTATIONS:-2000}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

case "${PHASE}" in
  panel)
    exec "${PYTHON_BIN}" scripts/validation/build_reference_ml_tree_panel.py \
      --out-root "${OUT_ROOT}" \
      --stages all \
      --graph-priority "${GRAPH_PRIORITY}" \
      --selection-seed "${SELECTION_SEED}" \
      --max-unique-sequences "${MAX_UNIQUE_SEQUENCES}" \
      --max-tips "${MAX_TIPS}" \
      --caliper-fraction "${CALIPER_FRACTION}" \
      --controls-per-endpoint "${CONTROLS_PER_ENDPOINT}" \
      --iqtree-binary "${IQTREE_BINARY}" \
      --iqtree-model "${IQTREE_MODEL}" \
      --iqtree-threads "${IQTREE_THREADS}" \
      --iqtree-memory "${IQTREE_MEMORY}" \
      "$@" 2>&1 | tee "${LOG_DIR}/build_panel.log"
    ;;
  tree)
    # long-running; run this one in its own tmux window
    exec bash "${OUT_ROOT}/tree/run_iqtree.sh" 2>&1 | tee "${LOG_DIR}/iqtree.log"
    ;;
  edges)
    exec "${PYTHON_BIN}" scripts/validation/evaluate_reference_ml_tree_edge_validation.py \
      --out-root "${OUT_ROOT}" \
      --stages all \
      --graphs "${GRAPH_PRIORITY}" \
      --caliper-fraction "${CALIPER_FRACTION}" \
      --permutations "${PERMUTATIONS}" \
      --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
      "$@" 2>&1 | tee "${LOG_DIR}/edge_validation.log"
    ;;
  fourpoint)
    exec "${PYTHON_BIN}" scripts/validation/evaluate_four_point_condition.py \
      --out-root "${OUT_ROOT}" \
      --stages all \
      --quartet-samples "${QUARTET_SAMPLES}" \
      "$@" 2>&1 | tee "${LOG_DIR}/four_point.log"
    ;;
  *)
    echo "Unknown PHASE=${PHASE}; expected one of panel, tree, edges, fourpoint" >&2
    exit 2
    ;;
esac
