#!/bin/bash
# -----------------------------------------------------------------------------
# Score the 512 B.1.1.7 hypercube genotypes with ESM-2 on a Slurm GPU cluster.
#
# This is Experiment B stage 2. Stages 1 and 3 are Node scripts that run on the
# workstation; only the model scoring needs a GPU.
#
# Expected use on the cluster:
#
#   tar -xzf b117_hypercube_code.tar.gz
#   cd <project root>
#   sbatch b117_bridge/cluster/run_score_hypercube_plm_slurm.sh
#
# The array runs one scoring mode per task (see MODES below). Outputs:
#   b117_bridge/results/hypercube/esm2_650M_<mode>/
# -----------------------------------------------------------------------------
#SBATCH --job-name=b117_hypercube
#SBATCH --output=b117_hypercube_%A_%a.out
#SBATCH --error=b117_hypercube_%A_%a.err
#SBATCH --partition=qGPU24
#SBATCH --account=csc1106s182
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

BUNDLE_ROOT="${BUNDLE_ROOT:-$(pwd)}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${BUNDLE_ROOT}/b117_bridge/scripts}"
SCORE_SCRIPT="${SCORE_SCRIPT:-${SCRIPTS_DIR}/score_hypercube_plm.py}"
ENV_DIR="${ENV_DIR:-/home/users/aadeniyi6/esm2_lora_env}"
PYTHON="${PYTHON_BIN:-${ENV_DIR}/bin/python}"

HYPERCUBE_DIR="${HYPERCUBE_DIR:-${BUNDLE_ROOT}/b117_bridge/results/hypercube}"
FASTA="${FASTA_PATH:-${HYPERCUBE_DIR}/hypercube_512.fasta}"
MANIFEST="${MANIFEST_PATH:-${HYPERCUBE_DIR}/hypercube_512_manifest.tsv}"
CHANGES="${CHANGES_PATH:-${BUNDLE_ROOT}/b117_bridge/config/alpha_changes.tsv}"
MODEL_NAME="${MODEL_NAME:-facebook/esm2_t33_650M_UR50D}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EMBED_TAG="${EMBED_TAG:-esm2_650M}"

# One mode per array task. `wt` is the seconds-long sanity check, `sites` is the
# primary result. `full` is the hours-long pseudo-log-likelihood pass -- add it
# only deliberately, e.g.  MODES="full" sbatch --array=0-0 ...
IFS=' ' read -r -a MODE_LIST <<< "${MODES:-wt sites}"

export HF_HOME="${HF_HOME:-${BUNDLE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

module load miniconda3

echo "BUNDLE_ROOT=${BUNDLE_ROOT}"
echo "PYTHON=${PYTHON}"
"${PYTHON}" --version
echo "SCORE_SCRIPT=${SCORE_SCRIPT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "HF_HOME=${HF_HOME}"
echo "MODES=${MODE_LIST[*]}"

for required in "${SCORE_SCRIPT}" "${FASTA}" "${MANIFEST}" "${CHANGES}"; do
    if [ ! -f "${required}" ]; then
        echo "ERROR: missing required input: ${required}"
        exit 1
    fi
done

NSEQ=$(grep -c "^>" "${FASTA}")
if [ "${NSEQ}" -ne 512 ]; then
    echo "ERROR: expected 512 sequences in ${FASTA}, found ${NSEQ}"
    exit 1
fi

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

PRECISION_FLAG=$("${PYTHON}" - <<'PYEOF'
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print("--bf16" if cap[0] >= 8 else "--fp16")
else:
    print("")
PYEOF
)
echo "Precision: ${PRECISION_FLAG:-none}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${#MODE_LIST[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} outside 0-$((${#MODE_LIST[@]} - 1))"
    exit 1
fi

MODE="${MODE_LIST[${TASK_ID}]}"
OUTDIR="${HYPERCUBE_DIR}/${EMBED_TAG}_${MODE}"

echo ""
echo "=== Array task ${TASK_ID}: mode=${MODE} at $(date) ==="
echo "Input FASTA: ${FASTA} (${NSEQ} sequences)"
echo "Output: ${OUTDIR}"
mkdir -p "${OUTDIR}"

"${PYTHON}" "${SCORE_SCRIPT}" \
    --fasta "${FASTA}" \
    --manifest "${MANIFEST}" \
    --changes "${CHANGES}" \
    --weight-metric "${WEIGHT_METRIC:-auto}" \
    --out-dir "${OUTDIR}" \
    --model "${MODEL_NAME}" \
    --cache-dir "${HF_HOME}" \
    --mode "${MODE}" \
    --batch-size "${BATCH_SIZE}" \
    ${PRECISION_FLAG}

# QC. Nothing here re-derives a score; it only checks that the 512 vertices are
# all present, that the residues the scorer saw are the ones the manifest says
# are there, and that the weights are finite and not all equal (a constant
# weight column would make every threshold result trivially true).
"${PYTHON}" - <<PYEOF
import json
import math
from pathlib import Path

out = Path("${OUTDIR}")
manifest_path = Path("${MANIFEST}")

expected = {}
with manifest_path.open(encoding="utf-8") as handle:
    header = handle.readline().rstrip("\n").split("\t")
    for line in handle:
        row = dict(zip(header, line.rstrip("\n").split("\t")))
        expected[row["pattern"]] = row["site_residues"]

scores = {}
with (out / "hypercube_scores.tsv").open(encoding="utf-8") as handle:
    header = handle.readline().rstrip("\n").split("\t")
    for line in handle:
        row = dict(zip(header, line.rstrip("\n").split("\t")))
        scores[row["pattern"]] = row

if len(scores) != 512:
    raise SystemExit(f"expected 512 scored vertices, got {len(scores)}")
missing = set(expected) - set(scores)
if missing:
    raise SystemExit(f"{len(missing)} vertices missing, e.g. {sorted(missing)[:3]}")
for pattern, residues in expected.items():
    if scores[pattern]["site_residues"] != residues:
        raise SystemExit(f"{pattern}: scorer saw {scores[pattern]['site_residues']}, manifest says {residues}")
for endpoint in ("000000000", "111111111"):
    if endpoint not in scores:
        raise SystemExit(f"endpoint {endpoint} not scored")

weights = {}
with (out / "weights_logprob.tsv").open(encoding="utf-8") as handle:
    handle.readline()
    for line in handle:
        pattern, value = line.rstrip("\n").split("\t")
        weights[pattern] = float(value)
if len(weights) != 512:
    raise SystemExit(f"weights_logprob.tsv has {len(weights)} rows, expected 512")
bad = [p for p, v in weights.items() if not math.isfinite(v)]
if bad:
    raise SystemExit(f"{len(bad)} non-finite weights, e.g. {bad[:3]}")
values = sorted(weights.values())
if values[0] == values[-1]:
    raise SystemExit("all 512 weights are identical -- scoring did not discriminate")

summary = {
    "mode": "${MODE}",
    "model": "${MODEL_NAME}",
    "n_vertices": len(scores),
    "weight_min": values[0],
    "weight_max": values[-1],
    "weight_wuhan": weights["000000000"],
    "weight_alpha": weights["111111111"],
    "n_distinct_weights": len(set(values)),
}
with (out / "hypercube_scoring_qc.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(f"QC ok: {summary}")
PYEOF

ls -lh "${OUTDIR}"

echo ""
echo "Hypercube scoring completed for mode=${MODE} at $(date)"
