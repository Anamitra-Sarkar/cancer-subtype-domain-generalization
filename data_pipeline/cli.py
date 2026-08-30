"""CLI for real data runs (TCGA/cBioPortal/GDC). Does not download in sandbox."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_pipeline.dataset import load_dataset
from data_pipeline.splits import leave_one_domain_out_splits, random_splits
from data_pipeline.preprocessing import DomainStandardizer
from src.train import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DG cancer subtype classifier on multi-cohort expression data")
    parser.add_argument("--expression-path", required=True, help="Path to expression matrix CSV (samples x genes)")
    parser.add_argument("--subtype-labels-path", required=True, help="Path to subtype labels CSV")
    parser.add_argument("--domain-labels-path", required=True, help="Path to domain/cohort labels CSV")
    parser.add_argument("--gene-names-path", default=None, help="Optional gene names file")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for metrics")
    parser.add_argument("--method", choices=["erm", "dann", "domain_std"], default="domain_std",
                        help="DG method: erm (baseline), dann, domain_std (per-domain standardization)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for p in [args.expression_path, args.subtype_labels_path, args.domain_labels_path]:
        if not Path(p).exists():
            print(f"File not found: {p}", file=sys.stderr)
            print("For real TCGA/cBioPortal/GDC data, download via:", file=sys.stderr)
            print("  cBioPortal: https://www.cbioportal.org/datasets", file=sys.stderr)
            print("  GDC: https://portal.gdc.cancer.gov/  (use gdc-client or TCGAbiolinks)", file=sys.stderr)
            sys.exit(1)

    dataset = load_dataset(args.expression_path, args.subtype_labels_path, args.domain_labels_path, args.gene_names_path)
    print(f"Loaded dataset: {dataset.n_samples} samples, {dataset.n_genes} genes, "
          f"{dataset.n_classes} subtypes, domains={dataset.unique_domains}")

    results = train_and_evaluate(dataset, method=args.method, seed=args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps(results, indent=2, default=float))
    print(f"Metrics written to {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
