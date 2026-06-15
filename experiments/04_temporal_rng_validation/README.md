# Temporal RNG Validation

## Question

Does orienting RNG edges by collection date reveal meaningful temporal source, bridge, and sink structure beyond date-shuffled null expectations?

## Required Contract

- Keep the same RNG topology.
- Reshuffle collection dates for null models.
- Report observed-vs-null counts and z-scores.
- Preserve source, bridge, sink, and temporal edge-distance summaries.

## Local Sources

Historical local workspaces include:

- `analysis/cohort_validation/13_random_full_dataset_2k_nj_tree_validation/`
- `analysis/cohort_validation/14_seed42_20k_temporal_rng_dag_validation/`

Large DAG node tables, edge tables, and null outputs remain local and untracked.
