# Project Compass

## Scientific Scope

This project is a method-validation study for graph-based analysis of viral sequence representations.

The same SARS-CoV-2 spike sequences are represented in two primary ways:

1. **Aligned sequence space**, using Hamming distances.
2. **Protein language model space**, using distances between PLM embeddings.

The method asks whether embedding geometry captures useful evolutionary, temporal, or graph-geodesic structure that is not already captured by Hamming distance, and whether graph construction can expose or destroy that signal.

## Core Hypothesis

Embedding distances may add value when they preserve biologically meaningful neighborhoods, temporal progression, or tree-like structure beyond raw mutation counts. Hamming distance remains a strong baseline and should not be treated as a weak control.

The analysis should make three outcomes visible:

- cases where embeddings improve over Hamming
- cases where Hamming remains equal or stronger
- cases where both representations fail or expose artifacts

## Representations

Tracked comparisons should keep accession sets aligned across representations.

| Representation | Role |
|---|---|
| Hamming distance | Sequence baseline from aligned spike sequences. |
| PLM embedding distance | Learned protein representation baseline, usually ESM-2. |
| Graph geodesic distance | Sparse-graph induced distance from MST, RNG, or kNN graph families. |
| Tree patristic distance | Reference distance from Nextstrain, time-dated, or NJ trees. |

## Graph Families

The project emphasizes sparse graph construction from the same distance inputs:

- MST
- RNG
- kNN-5
- kNN-50
- exact-witness kNN-RNG controls
- approximate or local-witness RNG variants when needed for scale

Every graph result should record:

- distance representation
- graph family
- k or witness parameters
- node/accession manifest
- input artifact manifest
- connected-component policy
- whether distances are raw, weighted graph geodesics, or unweighted graph geodesics

## Validation Axes

### Lineage Structure

Use lineage assortativity, Coleman-style homophily, endpoint/stub nulls, and node-pair enrichment summaries to test whether graph neighborhoods preserve lineage-local structure.

### Tree Structure

Use Spearman correlations with tree patristic distances, NJ self-tree-likeness, relative squared deviation, maximum tree distortion, and Gromov hyperbolicity to test whether raw and graph-induced distances preserve tree-like relationships.

### Temporal Structure

Use collection dates to evaluate root-to-tip trends, temporal RNG DAG behavior, source/sink structure, bridge behavior, date-shuffle nulls, and temporal edge-distance summaries.

### Geodesic Behavior

Use graph shortest paths, geodesic distance summaries, diameter coordinates, and graph-specific diagnostics to test whether sparse graphs create useful global structure or shortcut artifacts.

## Decision Rules

- Compare Hamming and embedding results on matched accessions whenever possible.
- Prefer graph-family comparisons that keep construction parameters explicit.
- Treat raw distances and graph geodesics as different objects.
- Report missing accession counts and matching policies.
- Separate random-full-dataset and monthly-stratified panels.
- Do not interpret graph results without component and coverage QC.
- Keep seed-level summaries when repeated sampling is part of the claim.

