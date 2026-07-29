# Script Provenance

The initial script surface was copied from local experiment folders on 2026-06-15.

These copies are intended as reusable workflow entry points, not as a fully refactored Python package. Keep source provenance visible until each script is promoted into `src/` with tests.

## Preprocessing

| New path | Source path |
|---|---|
| `scripts/preprocessing/processing.py` | `preprocessing/processing.py` |
| `scripts/preprocessing/esm_embed.py` | `preprocessing/esm_embed.py` |
| `scripts/preprocessing/prot_embed.py` | `preprocessing/prot_embed.py` |

## Graph Construction

| New path | Source path |
|---|---|
| `scripts/graph_construction/run_graph_variants_rng.py` | `spectral_backbone_revised/run_graph_variants_rng.py` |
| `scripts/graph_construction/run_graph_variants_hamming.py` | `spectral_backbone_revised/run_graph_variants_hamming.py` |
| `scripts/graph_construction/build_graph_family_coleman.py` | `spectral_backbone_revised/build_graph_family_coleman.py` |
| `scripts/graph_construction/build_cohort_embedding_graphs.py` | `analysis/cohort_validation/scripts/build_cohort_embedding_graphs.py` |
| `scripts/graph_construction/build_cohort_hamming_graphs.py` | `analysis/cohort_validation/scripts/build_cohort_hamming_graphs.py` |
| `scripts/graph_construction/build_panel_nj_distance_reference_trees.py` | `analysis/cohort_validation/scripts/build_panel_nj_distance_reference_trees.py` |
| `scripts/graph_construction/build_panel_spike_reference_tree.py` | `analysis/cohort_validation/scripts/build_panel_spike_reference_tree.py` |

## Validation

| New path | Source path |
|---|---|
| `scripts/validation/paired_kmedoids_comparison.py` | `analysis/cohort_validation/16_seed42_20k_kmedoids/paired_kmedoids_comparison.py` |
| `scripts/validation/make_assortativity_summary_table.py` | `analysis/cohort_validation/scripts/make_assortativity_summary_table.py` |
| `scripts/validation/nextstrain_spike_tree_validation.py` | `analysis/cohort_validation/scripts/nextstrain_spike_tree_validation.py` |
| `scripts/validation/time_dated_tree_validation.py` | `analysis/cohort_validation/scripts/time_dated_tree_validation.py` |
| `scripts/validation/evaluate_gromov_hyperbolicity_metrics.py` | `analysis/cohort_validation/scripts/evaluate_gromov_hyperbolicity_metrics.py` |
| `scripts/validation/evaluate_nj_tree_distortion_metrics.py` | `analysis/cohort_validation/scripts/evaluate_nj_tree_distortion_metrics.py` |
| `scripts/validation/evaluate_root_to_tip_regression_metrics.py` | `analysis/cohort_validation/scripts/evaluate_root_to_tip_regression_metrics.py` |
| `scripts/validation/evaluate_temporal_rng_dag_metrics.py` | `analysis/cohort_validation/scripts/evaluate_temporal_rng_dag_metrics.py` |
| `scripts/validation/summarize_rng_edge_temporal_distances.py` | `analysis/cohort_validation/scripts/summarize_rng_edge_temporal_distances.py` |

## Summaries

| New path | Source path |
|---|---|
| `scripts/summaries/summarize_embedding_vs_hamming_graphs.py` | `analysis/cohort_validation/scripts/summarize_embedding_vs_hamming_graphs.py` |
| `scripts/summaries/summarize_tree_validation_seed_correlations.py` | `analysis/cohort_validation/scripts/summarize_tree_validation_seed_correlations.py` |
| `scripts/summaries/summarize_exact_witness_runs.py` | `spectral_backbone_revised/summarize_exact_witness_runs.py` |
