# Seed-42 20k Paired K-Medoids Validation

## Question

Does clustering the same 20,000-sequence seed-42 panel with paired k-medoids favor raw Hamming distance, raw ESM-2 embedding distance, Hamming RNG geodesic distance, or ESM-2 RNG geodesic distance when evaluated against the nine broad `cohort_id` groups?

## Method Scope

The comparison uses four distance representations:

- `hamming`: raw Hamming dense matrix with profile-based medoid updates.
- `embedding`: raw ESM-2 cityblock dense matrix with exact dense-distance medoid updates.
- `hamming_rng`: weighted shortest-path distance on the exact Hamming RNG with all-node medoid updates.
- `embedding_rng`: weighted shortest-path distance on the exact ESM-2 cityblock RNG with all-node medoid updates.

Each of 200 repeats initializes one medoid from each of the nine `cohort_id` groups. The same accession-level initial medoids are reused across all four metrics, so metric comparisons are paired by repeat.

## Tracked Shareable Files

- `RUNBOOK.md`: commands and output contract for reproducing the workflow locally.
- `AUDIT_AND_TMUX_RUN.md`: method audit, corrections, validation status, and tmux run notes.
- `summary_by_metric.md`: compact human-readable production summary.
- `input_validation.json`: lightweight validation manifest for the seed-42 20k inputs.
- `results/summaries/seed42_20k_kmedoids/summary_by_metric.csv`: compact metric summary.
- `results/summaries/seed42_20k_kmedoids/paired_metric_deltas.csv`: paired mean deltas between metrics.
- `scripts/validation/paired_kmedoids_comparison.py`: reusable runner.
- `tests/test_paired_kmedoids_comparison.py`: synthetic sequence-free tests for key algorithm behavior.

## Local-Only Files

The original run folder remains ignored:

```text
analysis/cohort_validation/16_seed42_20k_kmedoids/
```

Do not track:

- `distance_rows/*.npy` RNG shortest-path caches
- per-repeat JSON files under `runs/`
- large design tables such as full candidate pools
- `__pycache__/`

## Primary Interpretation

All four production runs completed 200 paired repeats with `converged_rate = 1.0`.

Mean ARI ranking:

1. `hamming`: 0.4742
2. `hamming_rng`: 0.4295
3. `embedding`: 0.3453
4. `embedding_rng`: 0.3369

In this seed-42 20k clustering validation, Hamming remains stronger than ESM-2 cityblock embedding distance for broad `cohort_id` recovery. Exact RNG geodesics do not improve mean ARI over their corresponding raw distance metric in this task.

## Notes

Do not compare raw k-medoids objective values across distance families because the distance scales differ. Use ARI and Hungarian-mapped mislabeled counts for method comparison.
