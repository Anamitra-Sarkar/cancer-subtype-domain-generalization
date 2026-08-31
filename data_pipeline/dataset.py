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


def _detect_delimiter(path: Path) -> str:
    """Sniff delimiter from first non-empty, non-comment line."""
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            # Count candidates
            comma = stripped.count(",")
            tab = stripped.count("\t")
            semicolon = stripped.count(";")
            if tab > comma and tab >= semicolon:
                return "\t"
            if semicolon > comma and semicolon > tab:
                return ";"
            if comma > 0:
                return ","
            if tab > 0:
                return "\t"
            return ","
    # Fallback by suffix
    suffix = path.suffix.lower()
    if suffix in (".tsv", ".txt", ".tab"):
        return "\t"
    return ","


def _is_header_row(parts: list[str]) -> bool:
    """Heuristic: header if any part fails float() after stripping."""
    if not parts:
        return False
    for p in parts:
        p = p.strip().strip('"').strip("'")
        if p == "" or p.lower() in ("nan", "na", "null", "none"):
            continue
        # allow empty / missing handled elsewhere
        try:
            float(p)
        except ValueError:
            return True
    return False


def load_dataset(
    expression_path: str | Path,
    subtype_labels_path: str | Path,
    domain_labels_path: str | Path,
    gene_names_path: Optional[str | Path] = None,
) -> ExpressionDataset:
    """Load dataset from flat files.

    expression_path: CSV/TSV with rows=samples, cols=genes (no header col for sample id by default).
        If first column is sample_id, it will be auto-detected if subtype file has matching ids.
        Handles:
        - CSV or TSV (auto-detected via sniffing + suffix fallback)
        - Header row with gene names, with or without leading sample_id column
        - Comment lines starting with # or //, blank lines, BOM, quoted fields, whitespace-trimmed values
    subtype_labels_path: CSV with columns sample_id,subtype  OR single column of subtype strings
    domain_labels_path: CSV with columns sample_id,domain  OR single column of domain strings
    """
    exp_path = Path(expression_path)
    if not exp_path.exists():
        raise FileNotFoundError(f"Expression file not found: {exp_path}")
    delimiter = _detect_delimiter(exp_path)

    # Read with csv module for robustness (handles quotes, whitespace, comments, BOM)
    rows: list[list[str]] = []
    header: list[str] | None = None
    sample_ids_from_exp: list[str] | None = None
    gene_names_from_header: list[str] | None = None

    with open(exp_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        # Collect non-empty, non-comment rows first
        raw_rows: list[list[str]] = []
        for row in reader:
            if not row:
                continue
            # Skip if all cells blank
            if all(c.strip() == "" for c in row):
                continue
            # Skip comment lines: first cell starts with # or //
            first = row[0].strip()
            if first.startswith("#") or first.startswith("//"):
                continue
            # Strip whitespace + surrounding quotes from each cell
            cleaned = [c.strip().strip('"').strip("'").strip() for c in row]
            # Skip if still all empty after cleaning
            if all(c == "" for c in cleaned):
                continue
            raw_rows.append(cleaned)

    if not raw_rows:
        raise ValueError(f"Expression file {exp_path} contains no data rows")

    # Detect header: if first non-comment row has any non-numeric entry (excluding sample_id col)
    first_row = raw_rows[0]
    # Check header by trying numeric parse; if column 0 looks like sample id (non-numeric), ignore it for header check
    header_candidate = first_row
    maybe_sample_id_col = False
    # If first cell non-numeric but rest also non-numeric -> header
    # If first cell non-numeric but rest numeric -> sample_id + data row (no header)
    # We disambiguate by checking if first row as a whole is non-numeric
    if _is_header_row(first_row):
        # Could be "sample_id,gene1,gene2" or "gene1,gene2"
        # Determine if first col is sample id header
        first_col_lower = first_row[0].strip().lower()
        if first_col_lower in ("sample_id", "sample", "id", ""):
            # Sample id header present
            gene_names_from_header = [c for c in first_row[1:] if c]
            if not gene_names_from_header:
                raise ValueError("Header row with sample_id must have at least one gene name")
            header = first_row
            sample_ids_from_exp = None  # will fill below
            maybe_sample_id_col = True
        else:
            # Check if first col looks like a gene name (non-numeric header) vs sample id
            # Heuristic: if header contains any known header words, treat first col as sample_id
            # Otherwise treat all as gene names
            # Also if second row first col looks like sample id (non-numeric), then first col is sample_id
            if len(raw_rows) > 1:
                second = raw_rows[1]
                # second row first col non-numeric -> likely sample ids present
                try:
                    float(second[0])
                    second_first_numeric = True
                except ValueError:
                    second_first_numeric = False
                if not second_first_numeric and len(second) == len(first_row):
                    # Sample ids present: next rows have string first col
                    gene_names_from_header = [c for c in first_row[1:] if c]
                    header = first_row
                    maybe_sample_id_col = True
                else:
                    gene_names_from_header = [c for c in first_row if c]
                    header = first_row
                    maybe_sample_id_col = False
            else:
                gene_names_from_header = [c for c in first_row if c]
                header = first_row
        # Remove header from data rows
        data_rows = raw_rows[1:]
    else:
        data_rows = raw_rows
        # No header; check if first column is sample ids (string column)
        # If first col is non-numeric for data rows, it's sample ids
        # Actually above already handled non-header; here first row is numeric
        gene_names_from_header = None
        maybe_sample_id_col = False
        # Peek if data rows have string first col: try parsing first col
        if data_rows:
            try:
                float(data_rows[0][0])
                maybe_sample_id_col = False
            except ValueError:
                maybe_sample_id_col = True
                # Edge: gene names missing -> first col is sample id without header
                # We'll treat it as such if subtype file has ids

    # Now parse numeric matrix; handle sample_id column if present
    numeric_rows: list[list[float]] = []
    sids: list[str] = []
    n_genes_expected: int | None = None
    has_sample_id_col = False

    # Decide has_sample_id_col: true if header indicated it, or if data suggests it
    if header is not None and header[0].lower() in ("sample_id", "sample", "id"):
        has_sample_id_col = True
    elif maybe_sample_id_col:
        # More robust: check all rows have first col non-numeric
        non_numeric_first = 0
        for r in data_rows[: min(5, len(data_rows))]:
            try:
                float(r[0])
            except ValueError:
                non_numeric_first += 1
        if non_numeric_first >= 3 or (len(data_rows) <= 2 and non_numeric_first >= 1):
            has_sample_id_col = True
        # Also if header was like gene names but data has extra column mismatch with gene_names_path
        # We'll also check if row length vs header length: header len = row len -1 -> sample_id
        if header is not None and len(data_rows) > 0 and len(header) == len(data_rows[0]) - 1:
            has_sample_id_col = True

    for idx, row in enumerate(data_rows):
        # Skip rows with wrong column count? Report clearly
        if has_sample_id_col:
            if len(row) < 2:
                raise ValueError(f"Row {idx+2} has too few columns (expected sample_id + genes): {row}")
            sids.append(row[0])
            vals_str = row[1:]
        else:
            vals_str = row
        # Filter empty trailing values (e.g., trailing comma)
        vals_str = [v for v in vals_str if v != ""]
        # Allow missing values? Treat empty as NaN then error
        vals: list[float] = []
        for v in vals_str:
            if v == "" or v.lower() in ("na", "nan", "null", "none", "."):
                vals.append(float("nan"))
            else:
                try:
                    vals.append(float(v))
                except ValueError as e:
                    raise ValueError(f"Non-numeric expression value '{v}' at row {idx+2}, col {vals_str.index(v)+1}: {e}") from e
        if n_genes_expected is None:
            n_genes_expected = len(vals)
            if n_genes_expected == 0:
                raise ValueError(f"Row {idx+2} has no numeric columns")
        elif len(vals) != n_genes_expected:
            raise ValueError(
                f"Inconsistent gene count at row {idx+2}: expected {n_genes_expected}, got {len(vals)} (row: {row[:6]}...)"
            )
        # Check for NaN/Inf in expression (common upload error)
        if any(not np.isfinite(x) for x in vals):
            # Allow but warn; keep NaN for now — caller can decide; we raise explicit
            # For expression, non-finite is invalid input
            raise ValueError(f"Non-finite value (NaN/Inf) at row {idx+2}: {row[:6]}")
        numeric_rows.append(vals)

    if not numeric_rows:
        raise ValueError(f"No numeric data rows found in {exp_path}")

    X = np.array(numeric_rows, dtype=float)
    if has_sample_id_col:
        sample_ids_from_exp = sids
        # If header present and it had sample_id col, gene_names already set; else need to infer
        if gene_names_from_header is None:
            # Check if we previously set it from header without sample_id
            pass
    else:
        sample_ids_from_exp = None

    if gene_names_from_header is not None:
        # Validate length matches data columns
        if len(gene_names_from_header) != X.shape[1]:
            # Mismatch: header may include extra sample_id col already stripped, or trailing empty
            # Try to handle: if header counted without sample_id but X includes it, already stripped above.
            # If still mismatch, warn and truncate/pad?
            if len(gene_names_from_header) > X.shape[1]:
                gene_names_from_header = gene_names_from_header[: X.shape[1]]
            elif len(gene_names_from_header) < X.shape[1]:
                # Pad missing gene names
                gene_names_from_header = gene_names_from_header + [f"GENE_{i}" for i in range(len(gene_names_from_header), X.shape[1])]
        gene_names_header = gene_names_from_header
    else:
        gene_names_header = None

    if gene_names_path is not None:
        gpath = Path(gene_names_path)
        if not gpath.exists():
            raise FileNotFoundError(f"Gene names file not found: {gpath}")
        with open(gpath, encoding="utf-8-sig") as f:
            gene_names: list[str] = []
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                    continue
                # Handle comma-separated gene list on one line as well
                if "," in stripped:
                    parts = [p.strip().strip('"').strip("'") for p in stripped.split(",") if p.strip()]
                    gene_names.extend(parts)
                else:
                    gene_names.append(stripped.strip('"').strip("'"))
            gene_names = [g for g in gene_names if g]
        if len(gene_names) != X.shape[1]:
            raise ValueError(f"Gene names file has {len(gene_names)} entries but expression has {X.shape[1]} genes")
    elif gene_names_header is not None:
        gene_names = gene_names_header
    else:
        gene_names = [f"GENE_{i}" for i in range(X.shape[1])]

    # Load subtype labels
    subtype_path = Path(subtype_labels_path)
    if not subtype_path.exists():
        raise FileNotFoundError(f"Subtype labels file not found: {subtype_path}")
    subtypes_raw, sample_ids_sub = _load_label_file(subtype_path)
    # Load domain labels
    domain_path = Path(domain_labels_path)
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain labels file not found: {domain_path}")
    domains_raw, sample_ids_dom = _load_label_file(domain_path)

    if not subtypes_raw:
        raise ValueError(f"No subtype labels found in {subtype_path}")
    if not domains_raw:
        raise ValueError(f"No domain labels found in {domain_path}")

    # Resolve sample ordering: if files have sample_id col, align by id
    if sample_ids_sub is not None and sample_ids_dom is not None:
        if set(sample_ids_sub) != set(sample_ids_dom):
            raise ValueError(
                f"Subtype and domain sample_id sets differ: "
                f"{len(sample_ids_sub)} subtype ids vs {len(sample_ids_dom)} domain ids; "
                f"example missing in domain: {list(set(sample_ids_sub) - set(sample_ids_dom))[:3]}"
            )
        sample_ids = sample_ids_sub
        # Enforce consistent order or reorder
        if sample_ids_sub != sample_ids_dom:
            # Reorder domains to match subtype order
            dom_map = {sid: d for sid, d in zip(sample_ids_dom, domains_raw)}
            domains_raw = [dom_map[sid] for sid in sample_ids]
            sample_ids_dom = sample_ids
        # If expression had sample ids, reorder X to match label order
        if sample_ids_from_exp is not None:
            if set(sample_ids) != set(sample_ids_from_exp):
                raise ValueError(
                    f"Expression sample_ids differ from label sample_ids: "
                    f"{len(sample_ids_from_exp)} in expression vs {len(sample_ids)} in labels"
                )
            id_to_idx = {sid: i for i, sid in enumerate(sample_ids_from_exp)}
            try:
                order = [id_to_idx[sid] for sid in sample_ids]
            except KeyError as e:
                raise ValueError(f"Sample id {e} in labels not found in expression file") from e
            X = X[np.array(order)]
        else:
            if len(sample_ids) != X.shape[0]:
                raise ValueError(f"Sample count mismatch: labels have {len(sample_ids)} but expression has {X.shape[0]} rows")
    elif sample_ids_sub is not None:
        sample_ids = sample_ids_sub
        if len(sample_ids) != X.shape[0]:
            # Try to see if expression had ids but labels did, maybe expression ids are just numeric?
            if sample_ids_from_exp is not None and len(sample_ids_from_exp) == len(sample_ids):
                # Align if same count but subtype has ids — assume order matches
                pass
            else:
                raise ValueError(f"Sample count mismatch: subtype labels have {len(sample_ids)} but expression has {X.shape[0]} rows")
        if sample_ids_from_exp is not None and set(sample_ids) == set(sample_ids_from_exp) and sample_ids != sample_ids_from_exp:
            # Reorder X to match subtype order
            id_to_idx = {sid: i for i, sid in enumerate(sample_ids_from_exp)}
            order = [id_to_idx[sid] for sid in sample_ids]
            X = X[np.array(order)]
    elif sample_ids_dom is not None:
        # Only domain has ids — unusual, but handle
        sample_ids = sample_ids_dom
        if len(sample_ids) != X.shape[0]:
            raise ValueError(f"Sample count mismatch: domain labels have {len(sample_ids)} but expression has {X.shape[0]} rows")
    else:
        sample_ids = None
        if len(subtypes_raw) != X.shape[0]:
            raise ValueError(f"Sample count mismatch: {len(subtypes_raw)} subtypes vs {X.shape[0]} expression rows")
        if len(domains_raw) != X.shape[0]:
            raise ValueError(f"Sample count mismatch: {len(domains_raw)} domains vs {X.shape[0]} expression rows")

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
    """Returns (labels, sample_ids or None). Handles both 1-col and 2-col CSV.

    Robust to:
    - Blank lines, comment lines (# or //)
    - BOM, whitespace, quoted fields
    - Header variants (sample_id/sample/id, subtype/label/class/type, domain/cohort/batch/dataset/source)
    - Trailing commas, extra whitespace columns
    - Single-column files with or without header
    """
    labels: list[str] = []
    sample_ids: list[str] | None = None
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows: list[list[str]] = []
        for row in reader:
            if not row:
                continue
            # Trim cells
            cleaned = [c.strip().strip('"').strip("'").strip() for c in row]
            if all(c == "" for c in cleaned):
                continue
            if cleaned[0].startswith("#") or cleaned[0].startswith("//"):
                continue
            # Remove trailing empty cells (trailing commas)
            while cleaned and cleaned[-1] == "":
                cleaned.pop()
            if not cleaned:
                continue
            rows.append(cleaned)
    if not rows:
        return [], None
    # Detect header: if first row contains known header words
    first = [c.strip() for c in rows[0]]
    header_keywords = {"sample_id", "sample", "id", "subtype", "label", "class", "type", "domain", "cohort", "batch", "dataset", "source", "group"}
    # Header if any cell matches keyword (case-insensitive) and row has >=2 cols OR single-col header like "subtype"
    first_lower = [c.lower() for c in first]
    has_header = any(c in header_keywords for c in first_lower)
    # Refine: single-column header "subtype"/"domain" should be treated as header only if second row is not also header-like
    if has_header:
        # If single column header and all rows single-col, treat first as header
        if len(first) == 1:
            # Need to ensure second row is not also header keyword (otherwise data is header word?)
            # We'll treat as header if file has >1 row and second row is likely data
            rows = rows[1:]
            if not rows:
                return [], None
        else:
            # Multi-column: check if first row looks like header (e.g., sample_id, subtype)
            # If first cell is header keyword and second cell is header keyword, definitely header
            if len(first) >= 2 and (first_lower[0] in header_keywords or first_lower[1] in header_keywords):
                rows = rows[1:]
            elif any(c in header_keywords for c in first_lower):
                # Fallback: treat as header if any keyword
                rows = rows[1:]
    if not rows:
        return [], None
    # Determine one-col vs two-col: if rows have >=2 cols, treat as 2-col (sample_id, label)
    # Single-col file: each row has 1 label
    if rows and len(rows[0]) >= 2:
        sample_ids_list: list[str] = []
        labels_list: list[str] = []
        for r in rows:
            if len(r) >= 2:
                # First col is sample_id, second is label; ignore extra cols beyond 2 (e.g., trailing)
                sample_ids_list.append(r[0].strip())
                labels_list.append(r[1].strip())
            elif len(r) == 1:
                # Unexpected single col in 2-col file — treat as label with generated id?
                labels_list.append(r[0].strip())
                sample_ids_list.append(f"sample_{len(sample_ids_list)}")
            else:
                continue
        # Validate: sample_ids should be unique
        if len(set(sample_ids_list)) != len(sample_ids_list):
            # Duplicate ids — still return but caller will error on mismatch; keep
            pass
        # Filter empty labels
        filtered_sids: list[str] = []
        filtered_labels: list[str] = []
        for sid, lab in zip(sample_ids_list, labels_list):
            if lab == "":
                continue
            filtered_sids.append(sid)
            filtered_labels.append(lab)
        return filtered_labels, filtered_sids
    else:
        # Single column: all rows are labels, no sample_ids
        labels_only = [r[0].strip() for r in rows if r and r[0].strip() != ""]
        # Also handle case where file has header single-col already removed; if first label equals header word duplicated, ignore?
        return labels_only, None


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
