"""Parser hardening tests: real-format quirks for data_pipeline.dataset.load_dataset."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.dataset import load_dataset, _load_label_file, _detect_delimiter


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _basic_expression(tmp_path, n_samples=6, n_genes=4, header=True, sample_id=False, delimiter=",", with_comments=False, bom=False):
    """Helper to write a valid expression matrix."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, size=(n_samples, n_genes))
    lines = []
    if with_comments:
        lines.append("# This is a comment line from GDC manifest")
        lines.append("")
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    if header:
        if sample_id:
            lines.append(delimiter.join(["sample_id"] + gene_names))
        else:
            lines.append(delimiter.join(gene_names))
    for i in range(n_samples):
        row = [f"{v:.4f}" for v in data[i]]
        if sample_id:
            row = [f"SAMPLE_{i:03d}"] + row
        lines.append(delimiter.join(row))
    if with_comments:
        lines.append("")
        lines.append("# trailing comment")
    content = "\n".join(lines) + "\n"
    if bom:
        content = "\ufeff" + content
    return data, content


def test_csv_with_header_and_sample_id(tmp_path):
    data, content = _basic_expression(tmp_path, n_samples=6, n_genes=4, header=True, sample_id=True, delimiter=",")
    exp = _write(tmp_path, "exp.csv", content)
    subtypes = _write(tmp_path, "sub.csv", "sample_id,subtype\n" + "\n".join(f"SAMPLE_{i:03d},Luminal A" for i in range(6)))
    domains = _write(tmp_path, "dom.csv", "sample_id,domain\n" + "\n".join(f"SAMPLE_{i:03d},TCGA" for i in range(6)))
    ds = load_dataset(exp, subtypes, domains)
    assert ds.n_samples == 6
    assert ds.n_genes == 4
    assert ds.gene_names == ["GENE0", "GENE1", "GENE2", "GENE3"]
    assert ds.sample_ids is not None
    assert len(ds.sample_ids) == 6
    np.testing.assert_allclose(ds.X, data, atol=1e-3)


def test_csv_without_header(tmp_path):
    data, content = _basic_expression(tmp_path, n_samples=5, n_genes=3, header=False, sample_id=False, delimiter=",")
    exp = _write(tmp_path, "exp.csv", content)
    subtypes = _write(tmp_path, "sub.csv", "\n".join(["Basal-like"] * 5))
    domains = _write(tmp_path, "dom.csv", "\n".join(["cohort_A"] * 5))
    ds = load_dataset(exp, subtypes, domains)
    assert ds.n_genes == 3
    assert ds.gene_names == ["GENE_0", "GENE_1", "GENE_2"]
    np.testing.assert_allclose(ds.X, data, atol=1e-3)


def test_tsv_with_header_no_sample_id(tmp_path):
    data, content = _basic_expression(tmp_path, n_samples=4, n_genes=5, header=True, sample_id=False, delimiter="\t")
    exp = _write(tmp_path, "exp.tsv", content)
    subtypes = _write(tmp_path, "sub.csv", "subtype\n" + "\n".join(["HER2-enriched"]*4))
    domains = _write(tmp_path, "dom.csv", "domain\n" + "\n".join(["METABRIC"]*4))
    ds = load_dataset(exp, subtypes, domains)
    assert ds.n_genes == 5
    assert ds.gene_names[0] == "GENE0"


def test_tsv_with_sample_id_and_comments_and_blank_lines(tmp_path):
    data, content = _basic_expression(tmp_path, n_samples=6, n_genes=4, header=True, sample_id=True, delimiter="\t", with_comments=True)
    exp = _write(tmp_path, "exp.tsv", content)
    # subtype/domain with comments and blank lines
    sub_content = "# comment\nsample_id,subtype\n\nSAMPLE_000,Luminal A\nSAMPLE_001,Luminal B\n\nSAMPLE_002,Basal-like\nSAMPLE_003,HER2-enriched\nSAMPLE_004,Luminal A\nSAMPLE_005,Normal-like\n# end\n"
    dom_content = "\n# domains\nsample_id,domain\nSAMPLE_000,TCGA\nSAMPLE_001,TCGA\nSAMPLE_002,METABRIC\nSAMPLE_003,METABRIC\nSAMPLE_004,GEO\nSAMPLE_005,GEO\n\n"
    subtypes = _write(tmp_path, "sub.csv", sub_content)
    domains = _write(tmp_path, "dom.csv", dom_content)
    ds = load_dataset(exp, subtypes, domains)
    assert ds.n_samples == 6
    assert ds.n_genes == 4
    # Domains should be preserved
    assert set(ds.unique_domains) == {"TCGA", "METABRIC", "GEO"}


def test_single_column_subtype_without_header(tmp_path):
    data, content = _basic_expression(tmp_path, n_samples=4, n_genes=3, header=False)
    exp = _write(tmp_path, "exp.csv", content)
    subtypes = _write(tmp_path, "sub.csv", "Luminal A\nLuminal B\nBasal-like\nHER2-enriched\n")
    domains = _write(tmp_path, "dom.csv", "TCGA\nTCGA\nMETABRIC\nMETABRIC\n")
    ds = load_dataset(exp, subtypes, domains)
    assert ds.n_samples == 4
    assert ds.subtype_names == ["Basal-like", "HER2-enriched", "Luminal A", "Luminal B"] or set(ds.subtype_names) == {"Luminal A", "Luminal B", "Basal-like", "HER2-enriched"}


def test_two_col_header_variants_case_insensitive(tmp_path):
    data, content = _basic_expression(tmp_path, n_samples=3, n_genes=2, header=True, sample_id=True)
    exp = _write(tmp_path, "exp.csv", content)
    # Use "Sample" and "cohort" as header variants
    subtypes = _write(tmp_path, "sub.csv", "Sample,Label\nSAMPLE_000,Luminal A\nSAMPLE_001,Luminal B\nSAMPLE_002,Basal-like\n")
    domains = _write(tmp_path, "dom.csv", "ID,Batch\nSAMPLE_000,TCGA\nSAMPLE_001,TCGA\nSAMPLE_002,METABRIC\n")
    ds = load_dataset(exp, subtypes, domains)
    assert ds.n_samples == 3


def test_gene_names_path_with_comments(tmp_path):
    data, content = _basic_expression(tmp_path, n_samples=3, n_genes=3, header=False)
    exp = _write(tmp_path, "exp.csv", content)
    gene_path = _write(tmp_path, "genes.txt", "# PAM50 subset\nBRCA1\n# comment\nBRCA2\nTP53\n")
    subtypes = _write(tmp_path, "sub.csv", "Luminal A\nLuminal B\nBasal-like\n")
    domains = _write(tmp_path, "dom.csv", "A\nA\nB\n")
    ds = load_dataset(exp, subtypes, domains, gene_names_path=gene_path)
    assert ds.gene_names == ["BRCA1", "BRCA2", "TP53"]


def test_bom_handling(tmp_path):
    data, content = _basic_expression(tmp_path, n_samples=3, n_genes=2, header=True, sample_id=True, bom=True)
    exp = _write(tmp_path, "exp.csv", content)
    subtypes = _write(tmp_path, "sub.csv", "\ufeffsample_id,subtype\nSAMPLE_000,Luminal A\nSAMPLE_001,Luminal B\nSAMPLE_002,Basal-like\n")
    domains = _write(tmp_path, "dom.csv", "sample_id,domain\nSAMPLE_000,A\nSAMPLE_001,A\nSAMPLE_002,B\n")
    ds = load_dataset(exp, subtypes, domains)
    assert ds.gene_names[0] == "GENE0"  # BOM stripped


def test_quoted_fields_and_whitespace(tmp_path):
    content = ' "sample_id" , "GENE_A" , "GENE_B" \n "SAMPLE_000" , " 1.0 " , "2.0"\n"SAMPLE_001","3.0","4.0"\n'
    exp = _write(tmp_path, "exp.csv", content)
    subtypes = _write(tmp_path, "sub.csv", ' "sample_id" , "subtype" \n "SAMPLE_000" , " Luminal A " \n"SAMPLE_001","Basal-like"\n')
    domains = _write(tmp_path, "dom.csv", "sample_id,domain\nSAMPLE_000,TCGA\nSAMPLE_001,METABRIC\n")
    ds = load_dataset(exp, subtypes, domains)
    assert ds.n_samples == 2
    assert ds.n_genes == 2
    np.testing.assert_allclose(ds.X[0], [1.0, 2.0])
    assert ds.gene_names == ["GENE_A", "GENE_B"]


def test_trailing_comma_and_extra_whitespace(tmp_path):
    content = "GENE1,GENE2,\n1.0,2.0,\n3.0,4.0,\n"
    exp = _write(tmp_path, "exp.csv", content)
    subtypes = _write(tmp_path, "sub.csv", "Luminal A\nBasal-like\n")
    domains = _write(tmp_path, "dom.csv", "A\nB\n")
    # Should handle trailing comma gracefully (ignore empty last col)
    ds = load_dataset(exp, subtypes, domains)
    # trailing comma is treated as no extra gene after filtering empties -> 2 genes
    assert ds.n_genes == 2


def test_delimiter_sniffing_tsv_with_csv_suffix(tmp_path):
    # File named .csv but actually tab-delimited - sniff should detect
    data, content = _basic_expression(tmp_path, n_samples=3, n_genes=2, header=True, sample_id=False, delimiter="\t")
    exp = _write(tmp_path, "exp.csv", content)  # mislabeled suffix
    subtypes = _write(tmp_path, "sub.csv", "Luminal A\nBasal-like\nLuminal A\n")
    domains = _write(tmp_path, "dom.csv", "A\nB\nA\n")
    ds = load_dataset(exp, subtypes, domains)
    assert ds.n_genes == 2


def test_mismatched_sample_ids_raises(tmp_path):
    data, content = _basic_expression(tmp_path, n_samples=3, n_genes=2, header=True, sample_id=True)
    exp = _write(tmp_path, "exp.csv", content)
    subtypes = _write(tmp_path, "sub.csv", "sample_id,subtype\nSAMPLE_000,Luminal A\nSAMPLE_001,Luminal B\nOTHER_999,Basal-like\n")
    domains = _write(tmp_path, "dom.csv", "sample_id,domain\nSAMPLE_000,A\nSAMPLE_001,A\nSAMPLE_002,B\n")
    with pytest.raises(ValueError, match="differ"):
        load_dataset(exp, subtypes, domains)


def test_inconsistent_gene_count_raises(tmp_path):
    content = "GENE1,GENE2,GENE3\n1.0,2.0,3.0\n4.0,5.0\n"
    exp = _write(tmp_path, "exp.csv", content)
    subtypes = _write(tmp_path, "sub.csv", "Luminal A\nBasal-like\n")
    domains = _write(tmp_path, "dom.csv", "A\nB\n")
    with pytest.raises(ValueError, match="Inconsistent gene count"):
        load_dataset(exp, subtypes, domains)


def test_nonfinite_expression_raises(tmp_path):
    content = "GENE1,GENE2\n1.0,NaN\n3.0,4.0\n"
    exp = _write(tmp_path, "exp.csv", content)
    subtypes = _write(tmp_path, "sub.csv", "Luminal A\nBasal-like\n")
    domains = _write(tmp_path, "dom.csv", "A\nB\n")
    with pytest.raises(ValueError, match="Non-finite"):
        load_dataset(exp, subtypes, domains)


def test_reorder_by_sample_id(tmp_path):
    # Expression order is 000,001,002 but labels are 002,000,001 - should reorder X correctly
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    content = "sample_id,GENE1,GENE2\nSAMPLE_000,1.0,2.0\nSAMPLE_001,3.0,4.0\nSAMPLE_002,5.0,6.0\n"
    exp = _write(tmp_path, "exp.csv", content)
    subtypes = _write(tmp_path, "sub.csv", "sample_id,subtype\nSAMPLE_002,Basal-like\nSAMPLE_000,Luminal A\nSAMPLE_001,Luminal B\n")
    domains = _write(tmp_path, "dom.csv", "sample_id,domain\nSAMPLE_002,B\nSAMPLE_000,A\nSAMPLE_001,A\n")
    ds = load_dataset(exp, subtypes, domains)
    # After reorder, first sample should be SAMPLE_002's expression [5,6]
    np.testing.assert_allclose(ds.X[0], [5.0, 6.0])
    assert ds.sample_ids[0] == "SAMPLE_002"
    assert ds.y[0] == ds.subtype_names.index("Basal-like")


def test_detect_delimiter(tmp_path):
    p = _write(tmp_path, "a.csv", "a,b,c\n1,2,3\n")
    assert _detect_delimiter(p) == ","
    p2 = _write(tmp_path, "b.tsv", "a\tb\tc\n1\t2\t3\n")
    assert _detect_delimiter(p2) == "\t"
    p3 = _write(tmp_path, "c.txt", "# comment\n\n a;b;c\n1;2;3\n")
    # Should detect semicolon even through comment
    assert _detect_delimiter(p3) == ";"


def test_label_file_comment_and_blank_handling(tmp_path):
    p = _write(tmp_path, "labels.csv", "# header comment\n\nsample_id,subtype\n\nSAMPLE_000, Luminal A \n# mid comment\nSAMPLE_001,Basal-like\n\n")
    labels, sids = _load_label_file(p)
    assert labels == ["Luminal A", "Basal-like"]
    assert sids == ["SAMPLE_000", "SAMPLE_001"]


def test_single_col_with_header(tmp_path):
    p = _write(tmp_path, "labels.csv", "subtype\nLuminal A\nBasal-like\n")
    labels, sids = _load_label_file(p)
    assert labels == ["Luminal A", "Basal-like"]
    assert sids is None


def test_missing_files_raise(tmp_path):
    exp = tmp_path / "nonexistent.csv"
    sub = _write(tmp_path, "sub.csv", "Luminal A\n")
    dom = _write(tmp_path, "dom.csv", "A\n")
    with pytest.raises(FileNotFoundError):
        load_dataset(exp, sub, dom)
