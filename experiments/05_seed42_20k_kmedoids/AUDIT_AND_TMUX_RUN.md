# Seed-42 20k Paired K-Medoids Comparison

## Objective

Compare clustering agreement with the nine ground-truth variant/cohort groups on the seed-42 20,000-sequence panel using four distance representations:

1. Raw Hamming distance
2. Raw ESM-2 cityblock embedding distance
3. Hamming exact-RNG weighted shortest-path distance
4. ESM-2 cityblock exact-RNG weighted shortest-path distance

The comparison uses 200 paired random initializations. Within each repeat, one starting medoid is sampled uniformly from each of the nine `cohort_id` groups. The same nine accession-level starting medoids are then used for all four metrics.

The primary evaluation measures are:

- Adjusted Rand Index (ARI), which is invariant to cluster-label permutations.
- Number and fraction of mislabeled observations after an optimal one-to-one Hungarian mapping between the nine clusters and nine ground-truth groups.

## Final Method

The implementation uses alternating k-medoids:

1. Assign every observation to its nearest current medoid.
2. Within each cluster, select the member minimizing the sum of distances to the other cluster members.
3. Repeat until the medoids stop changing or the iteration limit is reached.

This is the standard alternate assignment/update form of k-medoids. It is not the PAM swap algorithm.

### Metric-specific medoid updates

| Metric | Assignment distance | Medoid update |
|---|---|---|
| `hamming` | Raw precomputed Hamming distance | Exact position-level residue-frequency/profile medoid over all cluster members |
| `embedding` | Raw precomputed ESM-2 cityblock distance | Exact dense-distance medoid over all cluster members |
| `hamming_rng` | Weighted shortest-path distance on the exact Hamming RNG | Exact all-node medoid update using cached graph distances |
| `embedding_rng` | Weighted shortest-path distance on the exact ESM-2 cityblock RNG | Exact all-node medoid update using cached graph distances |

For Hamming distance, minimizing the profile-disagreement score is algebraically identical to minimizing the sum of pairwise Hamming distances. The frequency calculation avoids repeatedly scanning a dense within-cluster Hamming submatrix.

## Corrections Made During the Code Audit

### 1. Approximate RNG medoid search

The first implementation restricted RNG medoid updates to 2,048 candidate nodes. That produced an approximation and was not methodologically equivalent to the all-member updates used for the raw metrics.

The production design now uses all 20,000 nodes as RNG medoid candidates:

```bash
--candidate-pool-size 20000
```

Smaller candidate pools remain available only for explicitly approximate smoke tests or sensitivity analyses.

### 2. Zero-distance tie handling

The dataset contains zero-distance relationships. A plain `argmin` can assign multiple medoid observations to the first tied cluster and leave another cluster empty.

The revised assignment step forces each medoid to own its corresponding observation when distances tie. This preserves nine nonempty clusters without changing the clustering objective because the reassigned distances are tied at zero.

### 3. Cache and resume provenance

The original resumable workflow did not fully protect against reusing a cache or result after changing the design.

The revised code records and validates:

- Ordered initialization fingerprint
- Ordered candidate-pool fingerprint
- RNG graph path, size, and modification-time signature
- Expected cache dimensions
- Completed shortest-path rows
- Initial medoids used by each saved result
- `max_iter` used by each saved result

A stale or incompatible cache/result now raises an error instead of being silently reused.

### 4. Input validation

Validation was expanded to check:

- Exactly 20,000 graph nodes
- Exactly nine non-missing `cohort_id` groups
- Metadata, Hamming-node, and embedding-node accession order
- Metadata and node-table ground-truth label agreement
- Raw matrix dimensions and data types
- Sampled off-diagonal matrix distances for finiteness, non-negativity, and symmetry
- Hamming diagonal sentinel handling
- Embedding diagonal `+inf` handling in memory
- RNG adjacency dimensions
- Finite, non-negative RNG edge weights
- Structural and weighted RNG adjacency symmetry, including explicit zero-weight edges
- A single connected component containing all 20,000 nodes

### 5. Convergence and result integrity

- The production iteration limit was increased from 25 to 100.
- The clustering objective is checked for monotonic non-increase.
- Final assignments and objectives are recomputed from the final medoids.
- Summary generation requires all four metrics and all 200 paired repeats by default.
- Partial summaries require the explicit `--allow-incomplete` flag.
- Paired ARI and mislabeled-count differences are joined by repeat ID.

## Confirmed Input Contract

The validated panel contains nine nearly balanced `cohort_id` groups:

| `cohort_id` | Count |
|---|---:|
| A | 2,223 |
| B_clean | 2,223 |
| C | 2,222 |
| D | 2,222 |
| E1 | 2,222 |
| E2 | 2,222 |
| E3 | 2,222 |
| F | 2,222 |
| G | 2,222 |
| **Total** | **20,000** |

Both exact RNG graphs contain all 20,000 nodes in one connected component. The raw Hamming and ESM-2 cityblock matrices are both 20,000 by 20,000 and aligned to the same accession order.

## Production Run in tmux

Start a tmux session:

```bash
tmux new -s seed42_kmedoids
```

Inside the session, run:

```bash
cd /alina-data1/Ezekiel/Protein_embeddings
export TMPDIR=/alina-data1/Ezekiel/tmp

PY=/alina-data1/Ezekiel/conda/environments/prot_embed/bin/python
SCRIPT=scripts/validation/paired_kmedoids_comparison.py

$PY $SCRIPT validate-inputs

$PY $SCRIPT prepare-design \
  --repeats 200 \
  --seed 42 \
  --candidate-pool-size 20000

$PY $SCRIPT prepare-rng-distances \
  --metrics hamming_rng embedding_rng \
  --batch-size 16

$PY $SCRIPT run \
  --metrics all \
  --repeats 200 \
  --max-iter 100 \
  --resume

$PY $SCRIPT summarize
```

Detach from the running session with `Ctrl-b d`.

Reconnect later with:

```bash
tmux attach -t seed42_kmedoids
```

Inspect existing sessions with:

```bash
tmux ls
```

## Runtime and Storage Expectations

The most expensive step is `prepare-rng-distances`. Exact all-node RNG k-medoids requires a shortest-path row from every node to all 20,000 nodes for each RNG graph.

The two 20,000 by 20,000 float32 caches require approximately 3.2 GB total. Dijkstra computation may take considerable time. The cache step is checkpointed and resumable: rerun the same command without `--force` after an interruption.

Do not rerun `prepare-design` after RNG cache generation has started unless the intention is to create a new experimental design and regenerate its caches and clustering results.

## Output Locations

The production output root is:

```text
analysis/cohort_validation/16_seed42_20k_kmedoids/random_full_dataset_seed42/seed_42/
```

Important outputs:

```text
input_validation.json
design/initial_medoids.csv
design/candidate_pool.csv
design/manifest.json
distance_rows/hamming_rng_candidate_to_all_float32.npy
distance_rows/hamming_rng_candidate_to_all_checkpoint.json
distance_rows/embedding_rng_candidate_to_all_float32.npy
distance_rows/embedding_rng_candidate_to_all_checkpoint.json
runs/hamming/repeat_000.json ... repeat_199.json
runs/embedding/repeat_000.json ... repeat_199.json
runs/hamming_rng/repeat_000.json ... repeat_199.json
runs/embedding_rng/repeat_000.json ... repeat_199.json
kmedoids_runs.csv
summary_by_metric.csv
paired_metric_deltas.csv
summary_by_metric.md
```

## Interpretation Notes

- Compare the four methods primarily using ARI and Hungarian-mapped mislabeled counts.
- Do not compare raw objective values across distance families because their numerical scales differ.
- Variation across the 200 repeats measures sensitivity to initialization. It is not uncertainty from resampling the underlying dataset.
- Inspect `converged_rate` before interpreting summary differences. A rate below 1.0 indicates that some starts reached `--max-iter` before the medoids stabilized.
- The RNG results are exact with respect to the cached weighted shortest-path matrices when the candidate-pool size is 20,000. A smaller candidate pool changes the method to approximate candidate-restricted k-medoids.

## Files

- Main implementation: [`scripts/validation/paired_kmedoids_comparison.py`](../../scripts/validation/paired_kmedoids_comparison.py)
- Synthetic sequence-free checks: [`tests/test_paired_kmedoids_comparison.py`](../../tests/test_paired_kmedoids_comparison.py)
- Short runbook: [`RUNBOOK.md`](RUNBOOK.md)

## Validation Status

The main algorithm corrections were checked with synthetic, sequence-free tests covering:

- Equivalence of the profile-based and dense Hamming medoid definitions
- Preservation of medoid ownership under zero-distance ties
- Equivalence of all-node cached-row and dense exact medoid updates
- Permutation-invariant ARI and Hungarian mislabeled counts
- Candidate-pool fingerprint sensitivity to row order

No production RNG cache generation or 200-repeat clustering run was performed during the audit. Those heavy steps remain under user control through the tmux commands above.
