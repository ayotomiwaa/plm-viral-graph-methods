# Cohort Validation Workflow

The cohort-validation workflow separates two biological questions:

1. Across cohorts, do graph methods separate major phenotypes or evolutionary regimes?
2. Within cohorts, do graph methods preserve fine-scale evolutionary, temporal, lineage, or regional structure?

## Cohort Families

| ID | Name | Intended role |
|---|---|---|
| A | Early U.S. ancestral/B.1.x | Early baseline regime. |
| B | Alpha | Clean Alpha or Alpha-to-Delta transition sensitivity. |
| C | Delta-dominant | Delta expansion and dominance. |
| D | Early Omicron | BA.1/BA.1.1/BA.2/BA.2.12.1. |
| E1 | BA.4/BA.5 era | BA.4, BA.5, BF, and related lineages. |
| E2 | BQ/XBB transition era | BQ.1, BQ.1.1, CH.1.1, XBB, and related lineages. |
| E3 | XBB/JN.1 era | XBB descendants, BA.2.86, JN.1, KP, LB.1, XEC. |
| F | Beta-focused | Beta-labeled sequences in a meaningful region/time context. |
| G | Gamma-focused | Gamma-labeled sequences in a meaningful region/time context. |
| H | Residual/minor lineage background | Heterogeneous exploratory background, not a clean primary cohort. |

## Metric Policy

- Pooled graph labels may use cohort, broad variant bucket, or major evolutionary regime.
- Within-cohort labels may use Pango lineage, collection month, region, or dominant-vs-background status.
- Keep node-pair enrichment and endpoint/stub homophily visible when reporting assortativity.
- Keep matched accession counts visible when comparing Hamming and embedding representations.

## Repository Policy

The original local cohort-validation folders contain many large artifacts and seed-level outputs. They should remain local. Promote only scripts, configs, small summary tables, final figures, and interpretation notes into the clean repository surface.
