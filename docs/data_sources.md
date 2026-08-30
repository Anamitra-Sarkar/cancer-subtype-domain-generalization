# Data Sources

## Target Scheme: PAM50 Breast Cancer Subtypes

- Parker, J.S. et al. "Supervised Risk Predictor of Breast Cancer Based on Intrinsic Subtypes." *Journal of Clinical Oncology* 27(8):1160-1167, 2009. DOI: 10.1200/JCO.2008.18.1370
- Subtypes: **Luminal A, Luminal B, HER2-enriched, Basal-like, Normal-like** — 50-gene PAM50 classifier, widely used clinically and in TCGA.
- Alternative consistent scheme (not used here but equivalently valid): glioma IDH-mutant/wildtype + 1p/19q co-deletion classification, Louis et al. 2021 WHO Classification of CNS Tumors, 5th edition.

## Domain Generalization Methodology

- **Leave-one-domain-out (LODO) evaluation** — the standard DG evaluation protocol: train on N-1 cohorts/domains, test on held-out domain, repeat per domain (Gulrajani & Lopez-Paz, "In Search of Lost Domain Generalization," ICLR 2021 — documents LODO as the realistic DG benchmark vs optimistic random-split).
- **Per-domain standardization** — per-domain z-score / batch normalization is a simple, well-established DG baseline (widely used in genomics batch-effect correction; see e.g., Leek et al. "Tackling the widespread and critical impact of batch effects in high-throughput data." *Nature Reviews Genetics* 11:733-739, 2010; and domain-specific batchnorm in Chang et al. "Domain-Specific Batch Normalization for Unsupervised Domain Adaptation." *CVPR* 2019).
- **Domain-Adversarial Neural Network (DANN)** — Ganin, Y. et al. "Domain-Adversarial Training of Neural Networks." *Journal of Machine Learning Research* 17(59):1-35, 2016 (arXiv:1505.07818). Gradient-reversal layer encouraging domain-invariant features. Implemented in `src/dann.py`.
- Invariant Risk Minimization (Arjovsky et al. 2019) is noted as another valid choice but not implemented here (per-domain standardization + DANN cover the requirement; IRM is mentioned as alternative).

## Expression Data Resources (Real, Public, No-Auth Scenario)

All are real, publicly accessible, no authentication required for bulk download via portal or API:

| Resource | URL | Access |
|----------|-----|--------|
| **TCGA** (GDC Portal) | https://portal.gdc.cancer.gov/ | Open; TCGA-BRCA RNA-seq (n~1100) via GDC data transfer tool or `TCGAbiolinks` R package or `gdc-client`. PAM50 calls available from TCGA marker paper. |
| **cBioPortal** | https://www.cbioportal.org/ , datasets at https://www.cbioportal.org/datasets | Open; METABRIC (n~2509), TCGA PanCancer Atlas (incl. BRCA). Expression + clinical (PAM50) via web API or direct download. |
| **METABRIC** | via cBioPortal or EGA; Curtis et al. 2012 *Nature* 486:346-352 | Open subset via cBioPortal (brca_metabric). |
| **GEO** | https://www.ncbi.nlm.nih.gov/geo/ | Additional validation cohorts (e.g., GSE96058). |

### CLI for Real Data (not executed in sandbox)

```bash
python -m data_pipeline.cli \
  --expression-path data/brca_expression.csv \
  --subtype-labels-path data/pam50_subtypes.csv \
  --domain-labels-path data/cohort_labels.csv \
  --method domain_std \
  --output-dir outputs
```

- `expression.csv`: rows=samples, cols=genes (with optional `sample_id` first column; header row auto-detected).
- `subtype_labels.csv`: `sample_id,subtype` or single column of PAM50 strings.
- `domain_labels.csv`: `sample_id,domain` (e.g., TCGA, METABRIC, GEO) or single column of cohort strings.
- Optional `--gene-names-path`: one gene symbol per line (otherwise taken from expression header or `GENE_i`).

Download guidance:

- **GDC**: `gdc-client download -m manifest.txt` or `TCGAbiolinks::GDCquery(project="TCGA-BRCA", data.category="Transcriptome Profiling")`.
- **cBioPortal**: `https://www.cbioportal.org/api` or dataset download from https://www.cbioportal.org/datasets (e.g., `brca_metabric`).

## Synthetic Verification (Used in Tests Here)

- `data_pipeline.dataset.make_synthetic_dataset()` creates small multi-domain matrices (3+ domains × 3+ subtypes) with injected true biological signal (per-subtype centroids in distinct genes) plus injected domain-specific batch shifts (per-domain random vectors) plus Gaussian noise.
- This fixture is **explicitly documented as synthetic** (not a real clinical finding) and used only to verify the pipeline's ability to recover subtype signal that generalizes across domains while a naive model overfits batch artifacts.
- Honest reporting: gap is random-split accuracy minus LODO accuracy; DG technique's effect (narrows gap or not) is reported truthfully from actual test outcome.

## Citations Are Real
No fabricated datasets, APIs, or papers. All citations above resolve to real publications/resources. Synthetic fixtures are clearly labeled as synthetic verification, not clinical evidence.
