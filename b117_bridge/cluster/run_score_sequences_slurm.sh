#!/bin/bash
# -----------------------------------------------------------------------------
# Island-baseline scoring: score the neutral 9-change mutants (plus the Wuhan
# reference) with ESM-2, whole-sequence pseudo-log-likelihood (full mode).
#
# This is the control the 2026-10-21 meeting asked for: how steep is the
# Wuhan->Alpha bridge compared to neutral 9-change drift?
#
# Expected use on the cluster:
#
#   tar -xzf b117_island_baseline_code.tar.gz
#   cd <project root>
#   sbatch b117_bridge/cluster/run_score_sequences_slurm.sh
#
# Input:  b117_bridge/results/island_baseline/neutral_baseline.fasta
# Output: b117_bridge/results/island_baseline/sequences_scores.tsv
# -----------------------------------------------------------------------------
#SBATCH --job-name=b117_island
#SBATCH --output=b117_island_%j.out
#SBATCH --error=b117_island_%j.err
#SBATCH --partition=qGPU24
#SBATCH --account=csc1106s182
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

BUNDLE_ROOT="${BUNDLE_ROOT:-$(pwd)}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${BUNDLE_ROOT}/b117_bridge/scripts}"
SCORE_SCRIPT="${SCORE_SCRIPT:-${SCRIPTS_DIR}/score_sequences_plm.py}"
ENV_DIR="${ENV_DIR:-/home/user/a/aadeniyi6/esm2_lora_env}"
PYTHON="${PYTHON_BIN:-${ENV_DIR}/bin/python}"

ISLAND_DIR="${ISLAND_DIR:-${BUNDLE_ROOT}/b117_bridge/results/island_baseline}"
FASTA="${FASTA_PATH:-${ISLAND_DIR}/neutral_baseline.fasta}"
MODEL_NAME="${MODEL_NAME:-facebook/esm2_t33_650M_UR50D}"
BATCH_SIZE="${BATCH_SIZE:-8}"

export HF_HOME="${HF_HOME:-${BUNDLE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

module load miniconda3

echo "BUNDLE_ROOT=${BUNDLE_ROOT}"
echo "PYTHON=${PYTHON}"
"${PYTHON}" --version
echo "SCORE_SCRIPT=${SCORE_SCRIPT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "HF_HOME=${HF_HOME}"

for required in "${SCORE_SCRIPT}" "${FASTA}"; do
    if [ ! -f "${required}" ]; then
        echo "ERROR: missing required input: ${required}"
        exit 1
    fi
done

NSEQ=$(grep -c "^>" "${FASTA}")
echo "Input FASTA: ${FASTA} (${NSEQ} sequences)"

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

mkdir -p "${ISLAND_DIR}"

"${PYTHON}" "${SCORE_SCRIPT}" \
    --fasta "${FASTA}" \
    --out-dir "${ISLAND_DIR}" \
    --model "${MODEL_NAME}" \
    --cache-dir "${HF_HOME}" \
    --batch-size "${BATCH_SIZE}" \
    ${PRECISION_FLAG}

# QC: every input sequence must be present, scores finite, not all identical.
"${PYTHON}" - <<PYEOF
import json, math
from pathlib import Path

out = Path("${ISLAND_DIR}")
fasta = Path("${FASTA}")

ids = []
with fasta.open() as fh:
    for line in fh:
        if line.startswith(">"):
            ids.append(line[1:].split("|")[0].split()[0])

rows = {}
with (out / "sequences_scores.tsv").open() as fh:
    header = fh.readline().rstrip("\n").split("\t")
    for line in fh:
        f = line.rstrip("\n").split("\t")
        rows[f[0]] = dict(zip(header, f))

if set(rows) != set(ids):
    raise SystemExit(f"scored {len(rows)} ids, expected {len(ids)}; missing={set(ids)-set(rows)}")

pll = [float(r["pll_mean"]) for r in rows.values()]
bad = [i for i, v in enumerate(pll) if not math.isfinite(v)]
if bad:
    raise SystemExit(f"{len(bad)} non-finite pll_mean values")
if min(pll) == max(pll):
    raise SystemExit("all pll_mean identical -- scoring did not discriminate")

ref = rows["REF"]["pll_mean"]
summary = {
    "model": "${MODEL_NAME}", "mode": "full", "n_sequences": len(rows),
    "pll_mean_ref": float(ref), "pll_mean_min": min(pll), "pll_mean_max": max(pll),
}
with (out / "island_scoring_qc.json").open("w") as fh:
    json.dump(summary, fh, indent=2)
print(f"QC ok: {summary}")
PYEOF

ls -lh "${ISLAND_DIR}"

echo ""
echo "Island-baseline scoring completed at $(date)"
echo "Transfer ${ISLAND_DIR}/sequences_scores.tsv back and run:"
echo "  node b117_bridge/scripts/prototype/island_baseline.mjs --scores ... --positions ... --out-dir ..."
