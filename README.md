# Protein Embedding Graph Validation

This repository organizes a method-validation workflow for comparing sequence-space and protein-language-model representations of SARS-CoV-2 spike sequences.

The central question is whether PLM embedding distances add useful biological and geometric signal beyond aligned-sequence Hamming distance when both representations are converted into sparse graphs and evaluated against lineage, time, and tree-structured references.

## Project Compass

The workflow compares the same viral sequences under two distance views:

- **Hamming distance** from aligned spike sequences.
- **PLM embedding distance** from protein language model embeddings, primarily ESM-2.

Those distances are used to build graph families:

- minimum spanning tree (MST)
- relative neighborhood graph (RNG)
- k-nearest-neighbor graphs, including kNN-5 and kNN-50
- exact-witness or approximate RNG variants where scalability requires controls

Graphs and raw distances are then evaluated with:

- lineage assortativity and homophily
- tree-distance correlation against reference or Neighbor-Joining trees
- temporal structure and root-to-tip behavior
- graph geodesic behavior, including hyperbolicity and shortest-path summaries

See [PROJECT_COMPASS.md](PROJECT_COMPASS.md) for the full method scope and decision rules.

## Repository Layout

```text
configs/       Reproducible run templates and parameter manifests.
data/          Manifest-only local data documentation. No raw data is tracked.
docs/          Method notes, workflow descriptions, and metric documentation.
experiments/   Lightweight summaries for major completed analysis families.
results/       Curated small summary tables and figures only.
scripts/       Reusable workflow scripts copied out of local experiment folders.
src/           Future package home for stable reusable library code.
tests/         Lightweight tests for stable code paths.
```

Large local artifacts are intentionally excluded: FASTA files, metadata dumps, embeddings, distance matrices, graph edge lists, Dijkstra outputs, PTU outputs, model checkpoints, logs, caches, and temporary experiment folders.

## Reproducibility Contract

Tracked files should be enough to understand and rerun the scientific workflow, but not enough to reconstruct private or bulky local artifacts from Git alone.

Use `data/manifests/` to document required local inputs, expected checksums when available, and the commands or scripts that generated derived artifacts. Use `experiments/` and `results/` for small summaries, final tables, and figures that support interpretation.

## Current Workflow Families

1. **Representation preparation**
   - Build or load aligned spike sequences.
   - Build or load PLM embeddings for the same accession set.
   - Keep explicit accession manifests so Hamming and embedding views are matched.

2. **Graph construction**
   - Construct Hamming and embedding graphs under comparable graph families.
   - Preserve graph parameters in config files and manifests.
   - Keep large graph outputs local.

3. **Method validation**
   - Compare lineage homophily, tree-distance correlation, temporal structure, root-to-tip behavior, and geodesic summaries.
   - Report where embeddings improve over Hamming, where Hamming remains stronger, and where both fail.

4. **Summarization**
   - Promote only small, interpretable summaries into `results/`.
   - Keep raw per-seed outputs and temporary workspaces out of Git.

## Getting Started

Create local data manifests first:

```bash
cp data/manifests/local_artifacts.example.yaml data/manifests/local_artifacts.yaml
```

Then edit the local manifest to point to your untracked FASTA, metadata, embedding, distance, and graph output locations.

Install the core environment:

```bash
python -m pip install -r requirements.txt
```

Run scripts from the repository root unless a script-specific README says otherwise. Most copied scripts were preserved as workflow entry points and may still expect the same command-line arguments used in the original local experiments.

## Development Notes

- Do not commit large artifacts.
- Add new method code under `src/` once it is stable enough to be imported and tested.
- Add runnable workflow entry points under `scripts/`.
- Add lightweight experiment summaries under `experiments/`.
- Add final small tables and figures under `results/`.
- Record local data dependencies in manifests instead of committing the data.
