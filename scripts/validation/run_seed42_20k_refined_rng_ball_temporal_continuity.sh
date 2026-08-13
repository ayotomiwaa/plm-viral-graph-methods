#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python}"
TEMP_ROOT="${TEMP_ROOT:-/alina-data1/Ezekiel/tmp}"

WORKSPACE="${WORKSPACE:-analysis/cohort_validation/27_seed42_20k_refined_rng_temporal_ball_continuity/random_full_dataset_seed42/seed_42}"
SOURCE_ROOT="${SOURCE_ROOT:-analysis/cohort_validation/07_sampling_design_20k}"
PANEL="${PANEL:-random_full_dataset_seed42}"
SEED="${SEED:-42}"
SAMPLE_LABEL="${SAMPLE_LABEL:-pool_n20000}"
PARAMETER_TAG="${PARAMETER_TAG:-candidate_0p1_delta_0p01}"
REFINED_RNG_DIR="${REFINED_RNG_DIR:-analysis/cohort_validation/24_seed42_20k_directional_intrinsic_distances/random_full_dataset_seed42/seed_42/refined_graphs/${PARAMETER_TAG}/rng}"

RADII="${RADII:-0,1,2,4,8,16,32,64,128,256,512,1024}"
RADIUS_MODE="${RADIUS_MODE:-powers-of-two}"
DATE_SHUFFLE_COUNT="${DATE_SHUFFLE_COUNT:-1}"
DATE_SHUFFLE_MAX_WINDOW_DAYS="${DATE_SHUFFLE_MAX_WINDOW_DAYS:-62}"
DATE_SHUFFLE_ATTEMPTS_PER_NODE="${DATE_SHUFFLE_ATTEMPTS_PER_NODE:-20}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"
MAX_CENTERS="${MAX_CENTERS:-}"
CENTER_MODE="${CENTER_MODE:-first}"
CENTER_SEED="${CENTER_SEED:-42}"
PHASE="${PHASE:-all}"

mkdir -p "${TEMP_ROOT}/mplconfig" "${WORKSPACE}/logs"
export TMPDIR="${TEMP_ROOT}"
export MPLCONFIGDIR="${TEMP_ROOT}/mplconfig"

cd "${REPO_ROOT}"

extra_spec="embedding_cityblock_rng_refined|${REFINED_RNG_DIR}|embedding|cityblock|rng_exact"

run_summary() {
  local -a max_center_args=()
  if [[ -n "${MAX_CENTERS}" ]]; then
    max_center_args=(--max-centers "${MAX_CENTERS}")
  fi
  local -a radius_args=()
  if [[ -n "${RADII}" ]]; then
    radius_args=(--radii "${RADII}")
  else
    radius_args=(--radius-mode "${RADIUS_MODE}")
  fi

  "${PYTHON_BIN}" scripts/validation/summarize_rng_ball_temporal_spread.py \
    --workspace "${WORKSPACE}" \
    --source-root "${SOURCE_ROOT}" \
    --panels "${PANEL}" \
    --seeds "${SEED}" \
    --sample-label "${SAMPLE_LABEL}" \
    --graph-names "embedding_cityblock_rng_exact" \
    --extra-graph-specs "${extra_spec}" \
    "${radius_args[@]}" \
    --date-shuffle-count "${DATE_SHUFFLE_COUNT}" \
    --date-shuffle-max-window-days "${DATE_SHUFFLE_MAX_WINDOW_DAYS}" \
    --date-shuffle-attempts-per-node "${DATE_SHUFFLE_ATTEMPTS_PER_NODE}" \
    --center-mode "${CENTER_MODE}" \
    --center-seed "${CENTER_SEED}" \
    --progress-every "${PROGRESS_EVERY}" \
    "${max_center_args[@]}"
}

run_compare() {
  local seed_dir="${WORKSPACE}/${PANEL}/seed_${SEED}"
  "${PYTHON_BIN}" scripts/validation/plot_rng_ball_normalized_spread_correlations.py \
    --comparison-kind baseline_refined_rng \
    --hamming-summary-csv "${seed_dir}/${PANEL}_seed_${SEED}_embedding_cityblock_rng_exact_rng_ball_temporal_spread_radius_summary.csv" \
    --embedding-summary-csv "${seed_dir}/${PANEL}_seed_${SEED}_embedding_cityblock_rng_refined_rng_ball_temporal_spread_radius_summary.csv" \
    --out-dir "${WORKSPACE}/normalized_baseline_refined_rng_spread_correlations" \
    --bins 20
}

case "${PHASE}" in
  summary)
    run_summary 2>&1 | tee "${WORKSPACE}/logs/refined_rng_ball_temporal_summary.log"
    ;;
  compare)
    run_compare 2>&1 | tee "${WORKSPACE}/logs/refined_rng_ball_temporal_compare.log"
    ;;
  all)
    run_summary 2>&1 | tee "${WORKSPACE}/logs/refined_rng_ball_temporal_summary.log"
    run_compare 2>&1 | tee "${WORKSPACE}/logs/refined_rng_ball_temporal_compare.log"
    ;;
  *)
    echo "Unknown PHASE=${PHASE}; expected summary, compare, or all" >&2
    exit 2
    ;;
esac
