# Data Policy

This repository does not track raw data or large derived artifacts.

Use this directory only for README files, `.gitkeep` placeholders, and lightweight manifests. Store local FASTA files, metadata dumps, embeddings, distance matrices, graph edge lists, graph adjacencies, Dijkstra outputs, PTU outputs, and model artifacts outside Git.

Expected local data classes:

- `raw/`: original sequence and metadata inputs
- `processed/`: cleaned or aligned sequence tables
- `embeddings/`: PLM embedding arrays and ID files
- `graphs/`: graph outputs and shortest-path artifacts
- `manifests/`: lightweight descriptions of the local artifacts required to rerun analyses
