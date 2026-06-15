# Git Staging Checklist

Use this checklist before creating commits.

1. Create or switch to a working branch.
2. Review `.gitignore`.
3. Run a dry-run staging check.
4. Confirm no large artifacts appear in the dry run.
5. Stage only the clean repository surface.
6. Commit in small, meaningful steps.

Suggested dry-run checks:

```bash
git add -n .
git status --short --ignored
```

Expected tracked surface for the restructuring pass:

- `.gitignore`
- `README.md`
- `PROJECT_COMPASS.md`
- `configs/`
- `src/`
- `scripts/`
- `experiments/`
- `results/`
- `docs/`
- `data/README.md`
- `data/*/README.md`
- `data/manifests/`
- `tests/`
- `metric_formulations.md`

Do not stage:

- `analysis/`
- `cleaned_data/`
- `embeddings_*`
- `graph_runs_*`
- `rel_distance/`
- `runs/`
- `spikeprot/`
- archives
- FASTA files
- arrays
- distance matrices
- edge lists
- logs
- caches
