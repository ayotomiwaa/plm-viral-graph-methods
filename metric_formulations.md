# Metric Formulations

## Notation

Given:

- `V = {1, ..., n}`: set of sequences.
- `y_i`: lineage label of sequence `i` for the lineage-homophily metric.
- `G = (V, E, w)`: weighted graph built from a distance representation.
- `d_G(i,j)`: shortest-path/geodesic distance between `i` and `j` on `G`.
- `D(i,j)`: raw pairwise distance between sequences `i` and `j`.
- `T`: Neighbor-Joining tree.
- `d_T(i,j)`: patristic distance between leaves `i` and `j` on `T`.
- `P = {(i,j): i < j}`: set of unordered sequence pairs.

Representations:

- Hamming distance: mutation-count distance between aligned spike sequences.
- ESM-2 cityblock distance: Manhattan distance between ESM-2 embedding vectors.

Unless stated otherwise, formulas are applied only to finite distances and to the upper-triangular pair set `P`, or to the sampled pair set produced by the code.

## Metric 1: Lineage Assortativity / Homophily Index

Given:

- A graph `G = (V, E, w)`.
- Lineage label `y_i` for each node, after dropping missing lineage labels such
  as empty/unknown labels.
- Valid-lineage edge set `E_y = {(i,j) in E: y_i and y_j are valid lineages}`.
- Lineage counts `n_l = |{i: y_i = l}|` among valid lineage-labeled nodes.
- Valid lineage-labeled node count `n_y = sum_l n_l`.
- Node degree `deg(i)` in the graph.

Find:

The observed graph same-lineage fraction is

```text
p_obs = (1 / |E_y|) * sum_{(i,j) in E_y} 1{y_i = y_j}.
```

The Coleman-style null uses graph endpoints/stubs. Let

```text
q_l = [sum_{i: y_i = l} deg(i)] / [sum_i deg(i)].
```

The expected same-lineage endpoint fraction is

```text
p_stub = sum_l q_l^2.
```

The assortativity/homophily index is the Coleman homophily index:

```text
h = (p_obs - p_stub) / (1 - p_stub).
```

Interpretation:

- `h = 0`: same-lineage edges occur at the endpoint/stub null rate.
- `h > 0`: same-lineage edges are enriched relative to the null.
- `h < 0`: same-lineage edges are depleted relative to the null.
- `h = 1`: all valid-lineage edges are same-lineage, when `p_stub < 1`.

Project-specific readout: in this project, higher positive `h` is better. Values
closer to `1` indicate stronger lineage-local graph structure, which is the
desired behavior when evaluating whether a graph preserves evolutionary
neighborhoods. Values near `0` mean little lineage homophily beyond the null.

The code also records a simpler node-pair enrichment ratio:

```text
p_pair = [sum_l n_l (n_l - 1) / 2] / [n_y (n_y - 1) / 2]
E_pair = p_obs / p_pair.
```

This node-pair ratio is reported as `nodepair_enrichment_ratio` and, in some
tables, as `assortatirtivty_coefficient`. It is related to `h`, but not
generally equal to it because it uses a node-pair null and a ratio, while
Coleman `h` uses an endpoint/stub null and a normalized difference.

## Metric 2: Spearman Correlation with NJ Tree Patristic Distances

Given:

- A distance vector `x_ij`, either raw distances `D(i,j)` or graph geodesic distances `d_G(i,j)`.
- An NJ-tree patristic distance vector `t_ij = d_T(i,j)`.
- Pair set `P`, either all unordered pairs or sampled pairs.

Find:

After masking to finite pairs, the tree-signal score is

```text
rho = Spearman({x_ij}, {t_ij}) for (i,j) in P.
```

Equivalently,

```text
rho = corr(rank(x), rank(t)).
```

The implementation uses `scipy.stats.spearmanr`. If fewer than three finite pairs remain, the score is reported as `NaN`.

Project-specific readout: higher positive `rho` is better. Values closer to
`+1` mean the representation preserves the NJ tree patristic distance ordering;
values near `0` mean weak tree-distance signal, and negative values mean the
ordering is reversed.

## Metric 3: Relative Squared Deviation (RSD)

Given:

- A distance vector `d_ij`, usually `D(i,j)` or `d_G(i,j)`.
- An NJ-tree patristic distance vector `t_ij = d_T(i,j)`.
- Finite pair set `P_f`.

Find:

```text
RSD = sqrt( [sum_{(i,j) in P_f} (d_ij - t_ij)^2]
            / [sum_{(i,j) in P_f} d_ij^2] ).
```

The denominator is the squared norm of the compared distance vector `d`, not the tree-distance vector and not the number of pairs. In the current NJ distortion workflow, RSD is unscaled: it compares `d` directly to `t`.

Project-specific readout: lower RSD is better. The ideal value is `0`, meaning
the compared distances exactly match the NJ patristic distances on the evaluated
pairs. Larger values mean larger relative squared error.

## Metric 4: Maximum Tree Distortion

Given:

- Finite positive distance pairs `d_ij > eps` and `t_ij > eps`.
- Tree distances are first scaled to the compared distance scale.

Find:

The least-squares scale applied to tree distances is

```text
s* = [sum d_ij t_ij] / [sum t_ij^2].
```

For each finite positive pair,

```text
r_ij = d_ij / (s* t_ij + eps).
```

The maximum tree distortion is

```text
Delta_max = max_{(i,j)} max(r_ij, 1 / (r_ij + eps)).
```

The code also records a log distortion:

```text
Delta_log = max_{(i,j)} |log(r_ij + eps)|.
```

Only finite positive pairs with finite positive `s*` are used for the distortion maximum.

Project-specific readout: values closer to `1` are better. The ideal value is
`1`, meaning no multiplicative distortion after the fitted tree-distance scale
`s*`. Larger values indicate a worse worst-case pairwise distortion.

## Metric 5: Gromov Hyperbolicity

Given:

- A distance matrix `D`, from raw distances or graph geodesic distances.
- Random sampled quadruples `(a,b,c,d)` without replacement.

Find:

For each finite quadruple, compute the three four-point sums:

```text
s1 = D(a,b) + D(c,d)
s2 = D(a,c) + D(b,d)
s3 = D(a,d) + D(b,c)
```

Let `s_(1) <= s_(2) <= s_(3)` be the sorted sums. The quadruple hyperbolicity is

```text
delta_quad = (s_(3) - s_(2)) / 2.
```

The reported Gromov hyperbolicity is the maximum sampled quadruple value:

```text
delta = max delta_quad.
```

The normalized score is

```text
delta_norm = delta / IQR({D(i,j): i < j}).
```

The current finite-aware implementation skips sampled quadruples with non-finite sums and reports `NaN` for `delta_norm` if the distance IQR is not finite or is zero.

Project-specific readout: lower `delta` or `delta_norm` is better. A tree metric
has four-point hyperbolicity `0`; larger values indicate less tree-like metric
geometry. In plots, the normalized value `delta_norm` is usually easier to
compare across distance scales.

## Code Provenance

| Metric | Source file | Function(s) | Confirmed implementation detail |
|---|---|---|---|
| Assortativity / homophily index | `analysis/cohort_validation/scripts/make_assortativity_summary_table.py` | `endpoint_expected_same`, `build_table` | Computes Coleman h as `(observed_same - endpoint_expected_same) / (1 - endpoint_expected_same)`. |
| Coleman index implementation | `spectral_backbone_revised/build_graph_family_coleman.py` | `coleman_index` | Computes endpoint/stub expectation, Coleman `h`, node-pair enrichment, and global assortativity ratio. |
| Node-pair enrichment side metric | `analysis/cohort_validation/scripts/build_cohort_embedding_graphs.py` | `label_metrics` | Computes `observed_same_fraction`, `nodepair_expected_same_fraction`, and `nodepair_enrichment_ratio = observed / expected`. |
| NJ patristic distances and Spearman | `analysis/cohort_validation/scripts/build_panel_nj_distance_reference_trees.py` | `compute_patristic_matrix`, `upper_values`, `score_raw_vs_nj` | Patristic distances are leaf-to-leaf NJ tree distances; Spearman uses finite upper-triangle or sampled pairs. |
| Pair sampling convention | `analysis/cohort_validation/scripts/nextstrain_spike_tree_validation.py` | `make_pair_indices`, `spearman_score` | `pair_mode="all"` uses all `i < j`; otherwise sampled unordered pairs are generated and scored with `spearmanr`. |
| RSD | `analysis/cohort_validation/scripts/evaluate_nj_tree_distortion_metrics.py` | `distance_tree_metrics` | Current finite-aware RSD denominator is `sum d^2`. |
| RSD older helper | `geodesic/cc_geodesic.py`; `rel_distance/post_metrics.py` | `rsd` | Older helpers confirm the same denominator, `sum D^2`, over upper-triangular pairs. |
| Maximum tree distortion | `analysis/cohort_validation/scripts/evaluate_nj_tree_distortion_metrics.py` | `optimal_scale_from_vectors`, `distance_tree_metrics` | Uses least-squares scale `s* = <d,t>/<t,t>` and maximum multiplicative ratio after scaling tree distances. |
| Gromov hyperbolicity | `analysis/cohort_validation/scripts/evaluate_gromov_hyperbolicity_metrics.py` | `gromov_delta_hyperbolicity` | Uses the four-point max-over-sampled-quadruples convention and normalizes by distance IQR. |
| Gromov older helper | `rel_distance/post_metrics.py` | `gromov_delta_hyperbolicity` | Confirms the same four-point sum sorting convention: `(largest - middle) / 2`. |
