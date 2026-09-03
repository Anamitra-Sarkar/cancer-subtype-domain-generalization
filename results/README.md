# Real training run — METABRIC / PAM50 (2026-09-03)

First real (non-synthetic) run of this repository's pipeline. Raw output:
`metabric-lodo-metrics.json`. Reproduce with `run_real_training_modal.py`.

## Data — real, public, no auth

METABRIC (Curtis et al. 2012; Pereira et al. 2016) via the **cBioPortal REST API**.

> Note: `docs/data_sources.md` points at the S3 datahub tarball
> (`https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz`), which now
> returns **403 Forbidden**. The official REST API at `https://www.cbioportal.org/api`
> serves the same public study and was used instead.

- Study `brca_metabric`, expression profile `brca_metabric_mrna`
  (Illumina HT-12 v3 microarray), sample list `brca_metabric_all`.
- **50/50 PAM50 genes** (Parker et al. 2009) resolved and present — 99,000 expression values.
- Subtype label: `CLAUDIN_SUBTYPE` (5 classes).
- **Domain: the real `COHORT` field (1–5).** METABRIC was assembled from several
  patient cohorts, so leave-one-domain-out here uses genuine acquisition batches
  rather than an invented or random split.
- **1,826 samples** after intersecting expression with usable labels.

## Result — random split vs. leave-one-domain-out

Pooled over all folds:

| Method | Random-split acc | Random-split macro-F1 | **LODO acc** | LODO macro-F1 | Acc gap |
|---|---|---|---|---|---|
| **ERM** | 0.8384 | 0.8036 | **0.8154** | 0.7812 | 0.0237 |
| domain_std | 0.8253 | 0.7876 | 0.7968 | 0.7563 | 0.0219 |
| DANN | 0.8231 | 0.7895 | 0.8023 | 0.7664 | 0.0228 |

Per-domain LODO accuracy (ERM): cohort 1 = 0.825, 2 = 0.816, 3 = 0.812,
4 = 0.809, 5 = 0.812 — notably stable across held-out cohorts.

## Honest reading

**Neither domain-generalization method beat plain ERM on LODO.** ERM is the best
LODO model (0.8154), with `domain_std` at 0.7968 and DANN at 0.8023.

This is not a bug, and it is worth stating plainly rather than reporting only the
DG numbers. It reproduces the central finding of Gulrajani & Lopez-Paz, *"In Search
of Lost Domain Generalization"* (ICLR 2021) — already cited in
`docs/data_sources.md` — that under a fair, identically-tuned evaluation protocol
ERM is competitive with or better than most proposed DG algorithms.

The mechanism here is visible in the numbers: the random-split → LODO gap is only
**~2–3 percentage points for every method**. Cross-cohort shift within METABRIC is
mild, because all five cohorts were profiled on the same Illumina HT-12 platform.
When there is little domain shift to correct, domain-invariance objectives mostly
spend capacity, and per-domain standardization discards genuinely informative
between-cohort scale. A meaningful DG benefit would require a harder shift —
e.g. METABRIC (microarray) → TCGA-BRCA (RNA-seq), a cross-platform comparison this
run does not make.

So the defensible claim from this run is: **PAM50 subtype classification transfers
across METABRIC cohorts at ~0.81 accuracy, and DG methods provide no advantage at
this level of shift.** It is not evidence that DANN or per-domain standardization
are ineffective in general.
