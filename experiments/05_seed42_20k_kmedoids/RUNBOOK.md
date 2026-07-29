# Seed-42 20k paired k-medoids runbook

This workflow compares four k-medoids metrics on the seed-42 20k panel:

- `hamming`: raw Hamming dense matrix, with position-level amino-acid frequency/profile medoid updates.
- `embedding`: raw ESM-2 cityblock dense matrix, with exact dense-distance medoid updates.
- `hamming_rng`: weighted shortest-path distance on the exact Hamming RNG, with all-node medoid updates.
- `embedding_rng`: weighted shortest-path distance on the exact ESM-2 cityblock RNG, with all-node medoid updates.

The ground-truth label is `cohort_id`, which gives the 9 broad lineage/cohort buckets. Each of 200 repeats initializes one uniformly sampled medoid from each bucket, with replacement across repeats. The same accession-level initial medoids are reused across all four metrics. ARI is permutation invariant; the mislabeled count uses an optimal one-to-one Hungarian mapping from clusters to the 9 labels.

Within a repeat, the implementation alternates nearest-medoid assignment and exact within-cluster medoid updates until the medoids stop changing or `--max-iter` is reached. This is the alternate k-medoids algorithm. For Hamming, the frequency-profile score is algebraically identical to the sum of pairwise Hamming distances while avoiding a dense within-cluster scan.

RNG note: the production design uses all 20,000 nodes as candidate medoids, so the RNG updates are exact for the cached weighted shortest-path metric. This requires two 20,000 x 20,000 float32 caches, about 3.2 GB total, and substantial Dijkstra compute. A smaller `--candidate-pool-size` is supported only for explicitly approximate smoke or sensitivity runs.

## Paths

```bash
cd /alina-data1/Ezekiel/Protein_embeddings

PY=/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python
SCRIPT=scripts/validation/paired_kmedoids_comparison.py
OUT=analysis/cohort_validation/16_seed42_20k_kmedoids/random_full_dataset_seed42/seed_42
TMPDIR=/alina-data1/Ezekiel/tmp
```

## Cheap checks and design generation

```bash
$PY $SCRIPT validate-inputs
$PY -m unittest -v tests/test_paired_kmedoids_comparison.py
$PY $SCRIPT prepare-design --repeats 200 --seed 42 --candidate-pool-size 20000
```

Run `prepare-design` once for the production output root. Its fingerprints protect later cache and resume steps from silently mixing designs.

Primary design outputs:

- `$OUT/input_validation.json`
- `$OUT/design/initial_medoids.csv`
- `$OUT/design/candidate_pool.csv`
- `$OUT/design/manifest.json`

## Heavy RNG distance cache

Run this in tmux. Increase `--batch-size` if memory is comfortable; reduce it if the job is too memory hungry.

```bash
tmux new -s seed42_kmedoids_rng
cd /alina-data1/Ezekiel/Protein_embeddings
export TMPDIR=/alina-data1/Ezekiel/tmp
PY=/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python
SCRIPT=scripts/validation/paired_kmedoids_comparison.py

$PY $SCRIPT prepare-rng-distances --metrics hamming_rng embedding_rng --batch-size 16
```

RNG cache outputs:

- `$OUT/distance_rows/hamming_rng_candidate_to_all_float32.npy`
- `$OUT/distance_rows/hamming_rng_candidate_to_all_checkpoint.json`
- `$OUT/distance_rows/embedding_rng_candidate_to_all_float32.npy`
- `$OUT/distance_rows/embedding_rng_candidate_to_all_checkpoint.json`

The command is resumable. Rerun it without `--force` to continue from the checkpoint.

## Heavy paired k-medoids run

```bash
tmux new -s seed42_kmedoids_run
cd /alina-data1/Ezekiel/Protein_embeddings
export TMPDIR=/alina-data1/Ezekiel/tmp
PY=/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python
SCRIPT=scripts/validation/paired_kmedoids_comparison.py

$PY $SCRIPT run --metrics all --repeats 200 --max-iter 100 --resume
$PY $SCRIPT summarize
```

Main result outputs:

- `$OUT/runs/<metric>/repeat_000.json` through `repeat_199.json`
- `$OUT/kmedoids_runs.csv`
- `$OUT/summary_by_metric.csv`
- `$OUT/paired_metric_deltas.csv`
- `$OUT/summary_by_metric.md`

## Smoke option

This is only a smoke run. It is not the full analysis, and its 64-node RNG candidate pool makes the RNG branches approximate by design.

```bash
SMOKE_OUT=analysis/cohort_validation/16_seed42_20k_kmedoids/smoke/random_full_dataset_seed42/seed_42

$PY $SCRIPT --out-root $SMOKE_OUT validate-inputs
$PY $SCRIPT --out-root $SMOKE_OUT prepare-design --repeats 2 --seed 42 --candidate-pool-size 64
$PY $SCRIPT --out-root $SMOKE_OUT prepare-rng-distances --metrics hamming_rng embedding_rng --batch-size 8
$PY $SCRIPT --out-root $SMOKE_OUT run --metrics hamming embedding hamming_rng embedding_rng --repeats 2 --max-iter 3 --resume
$PY $SCRIPT --out-root $SMOKE_OUT summarize
```
