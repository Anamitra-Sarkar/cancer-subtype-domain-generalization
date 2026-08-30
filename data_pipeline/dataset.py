"""Dataset handling for multi-domain gene expression + PAM50 subtypes.

Real data scenario: TCGA/METABRIC expression matrices downloaded via
cBioPortal (https://www.cbioportal.org/) or GDC (https://portal.gdc.cancer.gov/).
See docs/data_sources.md for CLI usage: --expression-path, --domain-labels-path,
--subtype-labels-path.

Tests use small synthetic fixtures (see tests/fixtures/).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# PAM50 subtypes - Parker et al. 2009 JCO (canonical 5-class scheme)
PAM50_SUBTYPES = ["Luminal A", "Luminal B", "HER2-enriched", "Basal-like", "Normal-like"]
SUBTYPE_TO_IDX = {s: i for i, s in enumerate(PAM50_SUBTYPES)}
IDX_TO_SUBTYPE = {i: s for s, i in SUBTYPE_TO_IDX.items()}


@dataclass
class ExpressionDataset:
    """Multi-domain expression dataset.

    Attributes:
        X: (n_samples, n_genes) expression matrix (float)
        y: (n_samples,) subtype label indices (int 0..n_classes-1)
        domains: (n_samples,) domain/cohort labels (str)
        gene_names: list of gene symbols (len n_genes)
        subtype_names: list mapping idx -> subtype string
        sample_ids: optional sample identifiers
    """

    X: np.ndarray
    y: np.ndarray
    domains: np.ndarray
    gene_names: list[str]
    subtype_names: list[str]
    sample_ids: Optional[list[str]] = None

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        assert self.y.shape[0] == n, "y length mismatch"
        assert self.domains.shape[0] == n, "domains length mismatch"
        assert self.X.shape[1] == len(self.gene_names)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.subtype_names)

    @property
    def unique_domains(self) -> list[str]:
        return sorted(set(self.domains.tolist()))

    def subset_by_indices(self, indices: np.ndarray) -> "ExpressionDataset":
        return ExpressionDataset(
            X=self.X[indices],
            y=self.y[indices],
            domains=self.domains[indices],
            gene_names=self.gene_names,
            subtype_names=self.subtype_names,
            sample_ids=[self.sample_ids[i] for i in indices] if self.sample_ids else None,
        )


def load_dataset(
    expression_path: str | Path,
    subtype_labels_path: str | Path,
    domain_labels_path: str | Path,
    gene_names_path: Optional[str | Path] = None,
) -> ExpressionDataset:
    """Load dataset from flat files.

    expression_path: CSV/TSV with rows=samples, cols=genes (no header col for sample id by default).
        If first column is sample_id, it will be auto-detected if subtype file has matching ids.
    subtype_labels_path: CSV with columns sample_id,subtype  OR single column of subtype strings
    domain_labels_path: CSV with columns sample_id,domain  OR single column of domain strings
    """
    exp_path = Path(expression_path)
    # Detect delimiter
    suffix = exp_path.suffix.lower()
    # Load expression
    X = np.loadtxt(exp_path, delimiter="," if suffix == ".csv" else "\t")

    # Handle case where file has header
    try:
        # If first row is header (non-numeric), reload skipping header
        with open(exp_path) as f:
            first_line = f.readline().strip()
            # try parse as float
            parts = first_line.replace("\t", ",").split(",")
            try:
                [float(p) for p in parts]
                has_header = False
            except ValueError:
                has_header = True
        if has_header:
            X = np.loadtxt(exp_path, delimiter="," if suffix == ".csv" else "\t", skiprows=1)
            # Extract gene names from header
            header_genes = [p.strip() for p in first_line.replace("\t", ",").split(",")]
            # Heuristic: if first col is sample_id header like "sample_id"
            if header_genes[0].lower() in ("sample_id", "sample", "id"):
                header_genes = header_genes[1:]
                # First column was sample ids - need to parse separately
                sample_ids_from_exp: list[str] | None = []
                gene_names_from_header = header_genes
                # Re-read to get sample ids
                with open(exp_path) as f:
                    next(f)  # skip header
                    for line in f:
                        sample_ids_from_exp.append(line.strip().split("," if suffix == ".csv" else "\t")[0])
                # X already loaded includes sample_id column as string? Actually loadtxt would fail.
                # So we need genfromtxt approach - simpler: use csv module
                rows = []
                sids = []
                with open(exp_path) as f:
                    reader = csv.reader(f, delimiter="," if suffix == ".csv" else "\t")
                    header = next(reader)
                    # check if first col is sample id
                    gene_names_from_header = header[1:] if header[0].lower() in ("sample_id", "sample", "id") else header
                    for row in reader:
                        if header[0].lower() in ("sample_id", "sample", "id"):
                            sids.append(row[0])
                            rows.append([float(x) for x in row[1:]])
                        else:
                            rows.append([float(x) for x in row])
                X = np.array(rows)
                sample_ids_from_exp = sids if sids else None
            else:
                gene_names_from_header = header_genes
                sample_ids_from_exp = None
        else:
            gene_names_from_header = None
            sample_ids_from_exp = None
    except Exception:
        gene_names_from_header = None
        sample_ids_from_exp = None

    if gene_names_path is not None:
        with open(gene_names_path) as f:
            gene_names = [line.strip() for line in f if line.strip()]
    elif gene_names_from_header is not None:
        gene_names = gene_names_from_header
    else:
        gene_names = [f"GENE_{i}" for i in range(X.shape[1])]

    # Load subtype labels
    subtype_path = Path(subtype_labels_path)
    subtypes_raw, sample_ids_sub = _load_label_file(subtype_path)
    # Load domain labels
    domain_path = Path(domain_labels_path)
    domains_raw, sample_ids_dom = _load_label_file(domain_path)

    # Resolve sample ordering: if files have sample_id col, align by id
    if sample_ids_sub is not None and sample_ids_dom is not None:
        # both have ids - use intersection ordering from expression? For simplicity align to subtype order
        assert sample_ids_sub == sample_ids_dom or set(sample_ids_sub) == set(sample_ids_dom), \
            "subtype and domain sample_id sets must match"
        sample_ids = sample_ids_sub
        # If expression had sample ids, reorder X to match label order
        if sample_ids_from_exp is not None:
            id_to_idx = {sid: i for i, sid in enumerate(sample_ids_from_exp)}
            order = [id_to_idx[sid] for sid in sample_ids]
            X = X[np.array(order)]
        else:
            assert len(sample_ids) == X.shape[0], "sample count mismatch"
    elif sample_ids_sub is not None:
        sample_ids = sample_ids_sub
        assert len(sample_ids) == X.shape[0]
    else:
        sample_ids = None
        assert len(subtypes_raw) == X.shape[0]
        assert len(domains_raw) == X.shape[0]

    # Map subtype strings to indices
    # Build subtype_names from unique values in file, but ensure PAM50 ordering if applicable
    unique_subs = sorted(set(subtypes_raw))
    # If subtypes are PAM50, use canonical ordering
    if set(unique_subs).issubset(set(PAM50_SUBTYPES)):
        subtype_names = [s for s in PAM50_SUBTYPES if s in unique_subs]
    else:
        subtype_names = unique_subs
    sub_to_idx = {s: i for i, s in enumerate(subtype_names)}
    y = np.array([sub_to_idx[s] for s in subtypes_raw], dtype=int)
    domains = np.array(domains_raw, dtype=object)
    sample_ids_list = sample_ids

    return ExpressionDataset(
        X=X, y=y, domains=domains, gene_names=gene_names,
        subtype_names=subtype_names, sample_ids=sample_ids_list,
    )


def _load_label_file(path: Path) -> tuple[list[str], Optional[list[str]]]:
    """Returns (labels, sample_ids or None). Handles both 1-col and 2-col CSV."""
    labels: list[str] = []
    sample_ids: list[str] | None = None
    has_header = False
    with open(path) as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], None
    # Detect header: if first row contains known header words
    first = [c.strip() for c in rows[0]]
    header_keywords = {"sample_id", "sample", "id", "subtype", "label", "domain", "cohort", "batch"}
    if any(c.lower() in header_keywords for c in first) and len(first) >= 2:
        # check if second row looks like data
        has_header = True
        rows = rows[1:]
    if rows and len(rows[0]) >= 2:
        # Two columns: sample_id, label
        # But need to distinguish 1-col that happens to have comma inside subtype name? PAM50 has spaces not commas
        # Heuristic: if header present, it's 2-col. Else if all rows have 2 cols, treat as 2-col.
        # If file has only one distinct domain/subtype per row with comma, it's 2-col.
        # We check: does first col look like sample id (contains no PAM50 name)? Just treat as 2-col if 2 cols.
        sample_ids = []
        labels = []
        for r in rows:
            if len(r) >= 2:
                sample_ids.append(r[0].strip())
                labels.append(r[1].strip())
            else:
                labels.append(r[0].strip())
        # If labels look like sample ids (e.g., duplicates), fallback? Keep as is.
        return labels, sample_ids
    else:
        labels = [r[0].strip() for r in rows if r]
        return labels, None


def make_synthetic_dataset(
    n_genes: int = 20,
    n_per_domain_per_class: int = 10,
    domains: Optional[list[str]] = None,
    subtypes: Optional[list[str]] = None,
    seed: int = 0,
    batch_shift_scale: float = 2.0,
    signal_scale: float = 3.0,
    noise_scale: float = 0.5,
) -> ExpressionDataset:
    """Create synthetic multi-domain expression dataset with injected signal + batch shift.

    Each subtype has a distinct mean pattern in first `n_classes` genes (one-hot-like signal).
    Each domain adds a constant shift vector (batch effect) shared across subtypes.
    This creates a realistic DG challenge: naive model can overfit batch shift if not invariant.

    Args:
        n_genes: total genes
        n_per_domain_per_class: samples per (domain, subtype) combo
        domains: domain names
        subtypes: subtype names
        seed: RNG seed
        batch_shift_scale: magnitude of domain-specific shift (larger = harder DG)
        signal_scale: magnitude of subtype signal (larger = more separable)
        noise_scale: Gaussian noise std

    Returns:
        ExpressionDataset
    """
    rng = np.random.default_rng(seed)
    if domains is None:
        domains = ["cohort_A", "cohort_B", "cohort_C"]
    if subtypes is None:
        subtypes = PAM50_SUBTYPES[:3]  # default 3 for simplicity in tests
    n_classes = len(subtypes)
    n_domains = len(domains)

    # Subtype centroids: each class has high expression in one gene
    centroids = np.zeros((n_classes, n_genes))
    for k in range(n_classes):
        centroids[k, k % n_genes] = signal_scale
        # Also add secondary signal in another gene for robustness
        centroids[k, (k + n_classes) % n_genes] = signal_scale * 0.6

    # Domain shifts: random vector per domain
    domain_shifts = rng.normal(0, batch_shift_scale, size=(n_domains, n_genes))
    # Make domain shifts affect non-signal genes more to simulate batch artifact
    # Actually shift all genes, but keep as is for generality

    X_list, y_list, d_list, sid_list = [], [], [], []
    for di, dname in enumerate(domains):
        for ki, sname in enumerate(subtypes):
            for j in range(n_per_domain_per_class):
                x = centroids[ki] + domain_shifts[di] + rng.normal(0, noise_scale, size=n_genes)
                X_list.append(x)
                y_list.append(ki)
                d_list.append(dname)
                sid_list.append(f"{dname}_{sname}_{j}")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    dom_arr = np.array(d_list, dtype=object)
    gene_names = [f"GENE_{i}" for i in range(n_genes)]

    # Shuffle
    perm = rng.permutation(len(y))
    return ExpressionDataset(
        X=X[perm], y=y[perm], domains=dom_arr[perm],
        gene_names=gene_names, subtype_names=subtypes,
        sample_ids=[sid_list[i] for i in perm],
    )
