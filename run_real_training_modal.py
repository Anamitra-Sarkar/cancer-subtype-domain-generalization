"""Run cancer-subtype-domain-generalization on real METABRIC data from cBioPortal.

Real, public, no-auth source that docs/data_sources.md itself lists:
  cBioPortal REST API (the S3 datahub tarball now returns 403).

Why METABRIC is the right fit here: it ships both the PAM50 subtype call and a real
COHORT field (METABRIC was assembled from several patient cohorts), so leave-one-domain-out
uses genuine acquisition batches rather than an invented split. That is exactly the
protocol the repo implements.

The repository's own pipeline (data_pipeline.cli / train_and_evaluate) is used unchanged;
this script only prepares the three flat files its loader documents.

Run: modal run train_csdg.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy==1.26.4", "pandas==2.2.3", "scikit-learn==1.5.2",
        "scipy==1.14.1", "requests==2.32.3", "torch==2.4.1",
    )
)

app = modal.App("csdg-metabric-real-run", image=image)
vol = modal.Volume.from_name("csdg-artifacts", create_if_missing=True)

REPO = "https://github.com/Anamitra-Sarkar/cancer-subtype-domain-generalization.git"
API = "https://www.cbioportal.org/api"
STUDY = "brca_metabric"
PROFILE = "brca_metabric_mrna"          # Illumina HT-12 v3 microarray
SAMPLE_LIST = "brca_metabric_all"

# PAM50 gene set (Parker et al. 2009) - the classifier's intended feature space.
PAM50 = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6",
    "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1",
    "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T",
]


@app.function(timeout=7200, cpu=8.0, memory=32768, volumes={"/art": vol})
def train() -> dict:
    import io, json, subprocess, sys, tarfile
    from pathlib import Path
    import pandas as pd
    import requests

    subprocess.run(["git", "clone", "--depth", "1", REPO, "/repo"], check=True)
    sys.path.insert(0, "/repo")

    # cBioPortal's S3 datahub tarball now returns 403, so the official REST API is
    # used instead. Same study, same public data, no authentication.
    def api_get(path, **kw):
        r = requests.get(f"{API}/{path}", timeout=600, **kw); r.raise_for_status(); return r.json()

    def api_post(path, payload, **kw):
        r = requests.post(f"{API}/{path}", json=payload, timeout=900, **kw); r.raise_for_status(); return r.json()

    print("resolving PAM50 gene ids...", flush=True)
    genes = api_post("genes/fetch?geneIdType=HUGO_GENE_SYMBOL", PAM50)
    entrez = [g["entrezGeneId"] for g in genes]
    sym_by_id = {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in genes}
    print(f"  resolved {len(entrez)}/50 PAM50 genes", flush=True)

    print("fetching expression matrix...", flush=True)
    md = api_post(
        f"molecular-profiles/{PROFILE}/molecular-data/fetch?projection=SUMMARY",
        {"entrezGeneIds": entrez, "sampleListId": SAMPLE_LIST},
    )
    print(f"  {len(md)} expression values", flush=True)
    edf = pd.DataFrame(
        [{"sample_id": r["sampleId"], "gene": sym_by_id.get(r["entrezGeneId"]), "v": r.get("value")} for r in md]
    ).dropna(subset=["gene"])
    X = edf.pivot_table(index="sample_id", columns="gene", values="v", aggfunc="first")

    print("fetching clinical data...", flush=True)
    cd = api_get(f"studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY&pageSize=10000000")
    clin = pd.DataFrame(cd).pivot_table(
        index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first"
    ).reset_index()
    print(f"  expression {X.shape}, clinical {clin.shape}", flush=True)
    print(f"  clinical columns: {list(clin.columns)[:25]}", flush=True)

    present = [g for g in PAM50 if g in X.columns]
    print(f"  PAM50 genes present: {len(present)}/50", flush=True)
    X = X[present]
    X.index.name = "sample_id"

    sub_col = next((c for c in clin.columns if "PAM50" in c.upper() or "CLAUDIN_SUBTYPE" in c.upper()), None)
    dom_col = next((c for c in clin.columns if c.upper() == "COHORT"), None)
    id_col = "patientId"
    print(f"  subtype col={sub_col} domain col={dom_col} id col={id_col}", flush=True)
    if sub_col is None or dom_col is None:
        raise RuntimeError(f"expected PAM50 + COHORT columns; got {list(clin.columns)}")

    meta = clin[[id_col, sub_col, dom_col]].dropna()
    meta.columns = ["sample_id", "subtype", "domain"]
    # Drop unusable subtype calls rather than silently recoding them.
    meta = meta[~meta["subtype"].isin(["NC", "Normal", "nan"])]

    common = X.index.intersection(meta["sample_id"])
    X = X.loc[common]
    meta = meta.set_index("sample_id").loc[common].reset_index()
    # Genes are on a microarray scale with a few missing values; fill per gene.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())

    print(f"  matched samples: {len(X)}", flush=True)
    print(f"  subtype counts:\n{meta['subtype'].value_counts()}", flush=True)
    print(f"  domain counts:\n{meta['domain'].value_counts()}", flush=True)

    d = Path("/art/metabric"); d.mkdir(parents=True, exist_ok=True)
    X.to_csv(d / "expression.csv")
    meta[["sample_id", "subtype"]].to_csv(d / "subtypes.csv", index=False)
    meta[["sample_id", "domain"]].to_csv(d / "domains.csv", index=False)

    from data_pipeline.dataset import load_dataset
    from src.train import train_and_evaluate

    ds = load_dataset(str(d / "expression.csv"), str(d / "subtypes.csv"), str(d / "domains.csv"))
    print(f"Loaded: {ds.n_samples} samples, {ds.n_genes} genes, {ds.n_classes} subtypes, "
          f"domains={ds.unique_domains}", flush=True)

    out = {}
    for method in ("erm", "domain_std", "dann"):
        print(f"--- {method} ---", flush=True)
        try:
            out[method] = train_and_evaluate(ds, method=method, seed=0)
        except Exception as e:
            out[method] = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(out[method], default=float)[:600], flush=True)

    (Path("/art") / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    vol.commit()
    print("RESULT:", json.dumps(out, indent=2, default=float)[:2500], flush=True)
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(train.remote(), indent=2, default=str))
