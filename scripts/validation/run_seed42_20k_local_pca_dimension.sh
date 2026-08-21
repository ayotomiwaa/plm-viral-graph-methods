#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/alina-data1/Ezekiel/Protein_embeddings}"
PYTHON_BIN="${PYTHON_BIN:-/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python}"
OUT_ROOT="${OUT_ROOT:-analysis/cohort_validation/28_seed42_20k_local_pca_dimension/random_full_dataset_seed42/seed_42}"
NODE_SET="${NODE_SET:-affected}"
MAX_NODES="${MAX_NODES:-}"
MIN_DEGREE="${MIN_DEGREE:-3}"
MAX_DEGREE_FOR_SVD="${MAX_DEGREE_FOR_SVD:-256}"
RANDOM_PRUNE_REPLICATES="${RANDOM_PRUNE_REPLICATES:-50}"
INCLUDE_HAMMING="${INCLUDE_HAMMING:-1}"
WRITE_RANDOM_REPLICATES="${WRITE_RANDOM_REPLICATES:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/logs"

args=(
  scripts/validation/evaluate_seed42_local_pca_dimension.py
  --out-root "$OUT_ROOT"
  --node-set "$NODE_SET"
  --min-degree "$MIN_DEGREE"
  --max-degree-for-svd "$MAX_DEGREE_FOR_SVD"
  --random-prune-replicates "$RANDOM_PRUNE_REPLICATES"
  --progress-every "$PROGRESS_EVERY"
)

if [[ -n "$MAX_NODES" ]]; then
  args+=(--max-nodes "$MAX_NODES")
fi
if [[ "$INCLUDE_HAMMING" == "1" ]]; then
  args+=(--include-hamming)
else
  args+=(--no-include-hamming)
fi
if [[ "$WRITE_RANDOM_REPLICATES" == "1" ]]; then
  args+=(--write-random-replicates)
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="$OUT_ROOT/logs/local_pca_${timestamp}.log"
"$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$log_path"
