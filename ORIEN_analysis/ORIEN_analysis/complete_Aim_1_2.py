'''
Aim 1.2 — Evaluate sex differences in phenotypes of T cell populations (CD8 TILs)

Pipeline:
  1) Load RNA-seq expression matrix (genes x samples) and metadata with 'sex'.
  2) Pre-rank genes by point-biserial correlation (Pearson vs. binary sex).
  3) Run GSEA using fgsea (via rpy2) on Sade-Feldman CD8 TIL signatures
     (especially partially exhausted cytotoxic T cells; CD8_1..CD8_6).
  4) Compute sample-level module scores for each fine cluster (CD8_1..CD8_6)
     and for combined groups CD8_B = {CD8_1,2,3}, CD8_G = {CD8_4,5,6}.
  5) Compare CD8_G vs CD8_B by sex (Mann-Whitney U, Cliff's delta, BH-FDR).
  6) Save tidy CSV outputs and figures.

Notes:
  - Include all tumor specimens with RNA-seq data. If your metadata contains a
    tumor indicator, pass --tumor-only with column/value filters; otherwise all
    samples in the expression/metadata intersection are used.
  - Expression should be gene symbols (case-insensitive). Gene sets are matched
    case-insensitively.

Gene sets are derived from Table S4 from https://pmc.ncbi.nlm.nih.gov/articles/PMC6641984/#SD5 .
'''

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# stats / FDR
from statsmodels.stats.multitest import multipletests

# rpy2 / fgsea
from rpy2.robjects import r, pandas2ri, ListVector, FloatVector, StrVector
from rpy2.robjects.packages import importr

from ORIEN_analysis.config import paths

from rpy2.robjects.conversion import localconverter
from rpy2 import robjects as ro
from rpy2.robjects import vectors

# -------------------------- IO helpers -------------------------- #

def load_expression(expr_path: str) -> pd.DataFrame:
    """
    Load expression matrix.
    Expected shape: rows=genes (index), cols=samples (column names).
    Accepts CSV/TSV. Auto-detect delimiter.
    """
    expr_path = str(expr_path)
    if expr_path.endswith(".tsv") or expr_path.endswith(".txt"):
        df = pd.read_csv(expr_path, sep="\t", index_col=0)
    else:
        df = pd.read_csv(expr_path, sep=None, engine="python", index_col=0)
    # normalize index to upper-case symbols for consistent matching
    df.index = df.index.astype(str).str.upper()
    return df


def load_metadata(meta_path: str) -> pd.DataFrame:
    """
    Load metadata table with at least:
      - 'sample_id'
      - sex column (name provided via --sex-col)
    """
    if meta_path.endswith(".tsv") or meta_path.endswith(".txt"):
        df = pd.read_csv(meta_path, sep="\t")
    else:
        df = pd.read_csv(meta_path, sep=None, engine="python")
    # enforce str for sample ids
    if "sample_id" not in df.columns:
        raise ValueError("Metadata must contain a 'sample_id' column.")
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def load_gene_sets() -> Dict[str, List[str]]:
    '''
    Load gene sets from .json.
    Returns dict: {set_name: [GENE1, GENE2, ...]} with genes upper-cased.
    '''
    obj = json.loads(paths.gene_sets.read_text(encoding = "utf-8"))
    return {k: [g.upper() for g in v] for k, v in obj.items()}


# -------------------- Pre-ranked stats (sex) -------------------- #

def map_sex_to_binary(
    meta: pd.DataFrame,
    sex_col: str,
    male_label: str,
    female_label: str
) -> pd.Series:
    """
    Map sex column to binary series: female=1, male=0 (default).
    Accepts case-insensitive labels.
    """
    s = meta[sex_col].astype(str)
    m = male_label.lower()
    f = female_label.lower()
    vals = s.str.lower().map({m: 0, f: 1})
    if vals.isna().any():
        bad = meta.loc[vals.isna(), [ "sample_id", sex_col ]]
        raise ValueError(
            f"Unmapped sex labels in '{sex_col}'. Offending rows:\n{bad.head(10)}"
        )
    return vals.astype(int)


def point_biserial_by_gene(
    expr: pd.DataFrame,       # genes x samples
    sex01: pd.Series          # indexed by sample_id, values 0/1
) -> pd.DataFrame:
    """
    Compute point-biserial correlation for each gene (Pearson vs. binary sex),
    returning DataFrame with 'gene', 'r', 'n', and 'pseudovar' for diagnostics.

    Implementation: Pearson correlation x ~ sex01 (0/1).
    """
    # Align columns/samples
    common = expr.columns.intersection(sex01.index)
    if len(common) < 2:
        raise ValueError("Need at least 2 aligned samples to compute correlations.")
    
    X = expr.loc[:, common]
    y = sex01.loc[common].astype(float)

    # center y
    y_c = y - y.mean()
    y_var = y_c.pow(2).sum()

    r_vals = []
    n = float(len(common))
    # vectorized: corr(x, y) = cov(x,y)/(sd(x) sd(y))
    # We'll compute covariance via dot of centered vectors.
    y_c_np = y_c.to_numpy()
    y_sd = y_c_np.std(ddof=1)
    if not np.isfinite(y_sd) or y_sd == 0:
        raise ValueError("Sex column has zero variance or is misaligned (check labels/indexing).")

    # center genes
    Xc = X.subtract(X.mean(axis=1), axis=0)
    # covariance with y for each gene: sum(xc * yc) / (n-1)
    cov = (Xc.values @ y_c_np) / (len(common) - 1)
    x_sd = Xc.std(axis=1, ddof=1).to_numpy()

    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        r = cov / (x_sd * y_sd)
    r = np.where(np.isfinite(r), r, 0.0)

    out = pd.DataFrame({
        "gene": X.index.values,
        "r": r,
        "n": int(n),
        "x_sd": x_sd,
        "y_sd": y_sd
    }).set_index("gene")

    return out


def build_preranked_vector(stats_df: pd.DataFrame, break_ties: bool = True) -> pd.Series:
    """
    Build a pre-ranked Series for fgsea from 'r' values.
    Name = gene, value = correlation coefficient.

    If break_ties=True, add a tiny deterministic tie-breaker
    based on the "first" rank to avoid fgsea tie warnings while preserving order.
    """
    s = stats_df["r"].astype(float).copy()
    if break_ties:
        s = s + (s.rank(method="first") * 1e-12)
    s = s.sort_values(ascending=False)
    s.name = "stat"
    return s


def run_fgsea(
    preranked: pd.Series,
    gene_sets: dict[str, list[str]],
    nperm: int | None = None,
    min_size: int = 10,
    max_size: int = 1_000,
    seed: int = 0
) -> pd.DataFrame:
    '''
    Run fgsea on a pre-ranked vector and Python dict of gene sets.
    
    - If nperm is None (default), run fgseaMultilevel (recommended).
    - If nperm is an int, run fgsea (Simple) with nperm permutations.
    '''
    base = importr("base")
    fgsea = importr("fgsea")

    stats_r = vectors.FloatVector(preranked.values.astype(float))
    stats_r.names = vectors.StrVector(preranked.index.tolist())
    pathways_r = ro.ListVector({
        k: vectors.StrVector(sorted(set(v)))
        for k, v in gene_sets.items() if v
    })
    ro.r(f"set.seed({int(seed)})")
    
    if nperm is None:
        # Recommended method
        res_r = fgsea.fgseaMultilevel(
            pathways = pathways_r,
            stats = stats_r,
            minSize = min_size
        )
    else:
        res_r = fgsea.fgsea(
            pathways = pathways_r,
            stats = stats_r,
            nperm = int(nperm),
            minSize = min_size
        )

    sanitize_r = ro.r('''
        function(d) {
            d <- as.data.frame(d, stringsAsFactors = FALSE)
            if (nrow(d) == 0L) {
                return(data.frame(
                    pathway=character(), pvalue=double(), padj=double(),
                    ES=double(), NES=double(), size=integer(),
                    leadingEdge=character(), stringsAsFactors=FALSE
                ))
            }
            for (nm in names(d)) {
                if (is.list(d[[nm]])) {
                    d[[nm]] <- vapply(
                        d[[nm]],
                        function(x) {
                            if (is.null(x) || length(x) == 0L) return("")
                            if (is.atomic(x)) return(paste(as.character(x), collapse = ","))
                            if (is.list(x)) return(paste(as.character(unlist(x, use.names = FALSE)), collapse = ","))
                            as.character(x)
                        },
                        FUN.VALUE = character(1L)
                    )
                }
            }
            d
        }
    ''')
    res_r_sanitized = sanitize_r(res_r)
    with localconverter(ro.default_converter + pandas2ri.converter):
        res_df = ro.conversion.rpy2py(res_r_sanitized)
    res_df = res_df.rename(columns = {
        "pathway": "term"
    })
    cols = [c for c in ["term", "pval", "padj", "ES", "NES", "size", "leadingEdge"] if c in res_df.columns]
    return res_df.loc[:, cols] if cols else res_df


# ------------------ Sample-level module scoring ------------------ #

def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each gene (row) across samples (columns).
    """
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=1).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
    return z


def module_score(
    expr: pd.DataFrame,               # genes x samples
    genes: List[str]
) -> pd.Series:
    """
    Mean z-scored expression across given genes present in expr.
    """
    genes = [g for g in genes if g in expr.index]
    if len(genes) == 0:
        return pd.Series(0.0, index=expr.columns, name="score")
    z = zscore_rows(expr.loc[genes])
    return z.mean(axis=0).rename("score")


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta effect size: P(X>Y)-P(X<Y).
    Computed via ranks to avoid O(n*m).
    """
    nx, ny = len(x), len(y)
    allv = np.concatenate([x, y])
    ranks = rankdata(allv, method="average")
    rx = ranks[:nx]
    ry = ranks[nx:]
    # Equivalent formula using U statistic:
    # U = sum_{i} rank(x_i) - nx*(nx+1)/2
    U = rx.sum() - nx * (nx + 1) / 2.0
    delta = (2*U) / (nx*ny) - 1
    return float(delta)


def compare_by_sex(
    scores: pd.Series,         # indexed by sample_id
    sex01: pd.Series,          # 0=male, 1=female aligned to sample_id
    label: str
) -> pd.DataFrame:
    """
    Mann-Whitney test of score by sex, with Cliff's delta.
    """
    common = scores.index.intersection(sex01.index)
    s = scores.loc[common]
    y = sex01.loc[common]

    x = s[y == 0].to_numpy()  # males
    z = s[y == 1].to_numpy()  # females
    if len(x) == 0 or len(z) == 0:
        raise ValueError("One of the sex groups is empty after alignment.")
    u_stat, pval = mannwhitneyu(x, z, alternative="two-sided")
    delta = cliff_delta(z, x)  # positive -> higher in females
    return pd.DataFrame({
        "contrast": [label],
        "n_male": [len(x)],
        "n_female": [len(z)],
        "mw_u": [u_stat],
        "pval": [pval],
        "cliffs_delta_female_minus_male": [delta],
        "mean_male": [float(np.mean(x))],
        "mean_female": [float(np.mean(z))]
    }).set_index("contrast")


# ----------------------------- Plots ----------------------------- #

def volcano_from_fgsea(res_df: pd.DataFrame, out_png: str):
    '''
    Volcano-like plot: NES vs -log10(FDR) using 'padj' from fgsea.
    '''
    df = res_df.copy()
    if df.empty:
        print("[WARN] fgsea result is empty; skipping volcano plot.")
        return
    if "padj" not in df.columns:
        raise ValueError("fgsea results are missing 'padj' (FDR) column.")
    df["neglog10FDR"] = -np.log10(df["padj"].replace(0, np.nextafter(0, 1)))
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(
        data=df, x="NES", y="neglog10FDR",
        hue=(df["padj"] < 0.1), style=(df["padj"] < 0.05),
        s=60
    )
    ax.axhline(-np.log10(0.1), ls="--", lw=1)
    ax.axvline(0, ls=":", lw=1)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("fgsea results (sex pre-ranked)")
    if ax.get_legend() is not None:
        ax.legend(title = "FDR thresholds")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def box_by_sex(scores: pd.Series, sex01: pd.Series, title: str, out_png: str):
    common = scores.index.intersection(sex01.index)
    df = pd.DataFrame({
        "score": scores.loc[common].values,
        "sex01": sex01.loc[common].values
    })
    df["sex"] = np.where(df["sex01"] == 1, "Female", "Male")
    plt.figure(figsize=(5, 5))
    ax = sns.boxplot(data=df, x="sex", y="score")
    sns.stripplot(data=df, x="sex", y="score", color="black", alpha=0.5)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------- Main ------------------------------ #

def main(args: argparse.Namespace):
    #outdir = Path(args.outdir)
    outdir = Path(paths.outputs_of_completing_Aim_1_2)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    #expr = load_expression(args.expr)
    expr = load_expression(paths.expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest)
    #meta = load_metadata(args.meta)
    list_of_sample_IDs = expr.columns.tolist()
    meta = pd.DataFrame(list_of_sample_IDs, columns = ["sample_id"])
    clinical_molecular_linkage_data = pd.read_csv(paths.clinical_molecular_linkage_data)
    patient_data = pd.read_csv(paths.patient_data)
    meta = (
        meta
        .merge(
            clinical_molecular_linkage_data[["RNASeq", "ORIENAvatarKey"]],
            how = "left",
            left_on = "sample_id",
            right_on = "RNASeq"
        )
        .drop(columns = "RNASeq")
        .merge(
            patient_data[["AvatarKey", "Sex"]],
            how = "left",
            left_on = "ORIENAvatarKey",
            right_on = "AvatarKey"
        )
        .drop(columns = ["ORIENAvatarKey", "AvatarKey"])
    )

    '''
    # (Optional) tumor filtering; by default include all samples present
    if args.tumor_only_col and args.tumor_only_values:
        vals = [v.strip().lower() for v in args.tumor_only_values.split(",")]
        if args.tumor_only_col not in meta.columns:
            raise ValueError(f"--tumor-only-col '{args.tumor_only_col}' not in metadata.")
        mask = meta[args.tumor_only_col].astype(str).str.lower().isin(vals)
        meta = meta.loc[mask].copy()
    '''

    # Harmonize samples
    sample_intersection = pd.Index(expr.columns.astype(str)).intersection(meta["sample_id"].astype(str))
    if sample_intersection.empty:
        raise ValueError("No overlapping samples between expression and metadata.")
    meta = meta.assign(sample_id = meta["sample_id"].astype(str)).set_index("sample_id").loc[sample_intersection]
    expr = expr.loc[:, sample_intersection]

    meta["Sex"] = (
        meta["Sex"].astype(str).str.strip().str.upper()
        .map({"M": "Male", "MALE": "Male", "F": "Female", "FEMALE": "Female"})
        .fillna(meta["Sex"])
    )
    
    sex01 = map_sex_to_binary(meta, "Sex", "Male", "Female").astype(int)

    if sex01.isna().any():
        bad = sex01[sex01.isna()]
        raise ValueError(f"Sex is missing for {len(bad)} samples; e.g., {bad.index.tolist()[:10]}")

    # 3) Pre-ranked stats
    stats_df = point_biserial_by_gene(expr, sex01)
    preranked = build_preranked_vector(stats_df, break_ties=True)
    stats_df.assign(stat=stats_df["r"]).to_csv(outdir / "preranked_stats_point_biserial.csv")

    # 4) Gene sets
    #gene_sets = load_gene_sets(args.gene_sets)
    gene_sets = load_gene_sets()

    # 5) fgsea
    fgsea_res = run_fgsea(
        preranked,
        gene_sets,
        nperm = None,
        min_size = 10,
        max_size = 1_000,
        seed = 0
    )
    fgsea_res.to_csv(outdir / "fgsea_results.csv", index=False)

    # volcano-like plot
    try:
        volcano_from_fgsea(fgsea_res, str(outdir / "fgsea_volcano.png"))
    except Exception as e:
        print(f"[WARN] Volcano plot failed: {e}")

    # 6) Sample-level module scores: CD8_1..CD8_6
    zscores = zscore_rows(expr)  # genes x samples (z per gene)
    fine_labels = [f"CD8_{i}" for i in range(1, 7)]
    missing = [lb for lb in fine_labels if lb not in gene_sets]
    if missing:
        print(f"[WARN] Missing expected fine clusters in gene sets: {missing}")

    fine_scores = {}
    for lb in fine_labels:
        genes = gene_sets.get(lb, [])
        fine_scores[lb] = module_score(expr, genes)

    fine_df = pd.DataFrame(fine_scores)
    fine_df.index.name = "sample_id"
    fine_df.to_csv(outdir / "sample_module_scores_CD8_1_to_6.csv")

    # Combined groups
    cd8_b_names = [f"CD8_{i}" for i in (1, 2, 3)]
    cd8_g_names = [f"CD8_{i}" for i in (4, 5, 6)]
    cd8_b_genes = sorted(set(sum((gene_sets.get(n, []) for n in cd8_b_names), [])))
    cd8_g_genes = sorted(set(sum((gene_sets.get(n, []) for n in cd8_g_names), [])))

    score_b = module_score(expr, cd8_b_genes).rename("CD8_B_score")
    score_g = module_score(expr, cd8_g_genes).rename("CD8_G_score")
    # Difference and log-ratio
    diff = (score_g - score_b).rename("CD8_G_minus_CD8_B")
    eps = 1e-6
    min_val = float(pd.concat([score_b, score_g]).min())
    shift = (-min_val + eps) if min_val <= 0 else 0.0
    ratio = np.log2((score_g + shift) / (score_b + shift)).rename("log_2_CD8_G_over_CD8_B")

    combined = pd.concat([score_b, score_g, diff, ratio], axis=1)
    combined.index.name = "sample_id"
    combined.to_csv(outdir / "sample_module_scores_CD8_B_vs_CD8_G.csv")

    # 7) Stats by sex for key contrasts
    tests = []
    tests.append(compare_by_sex(diff, sex01, "CD8_G_minus_CD8_B"))
    tests.append(compare_by_sex(ratio, sex01, "log2_CD8_G_over_CD8_B"))

    # Add each fine cluster as well
    for lb in fine_labels:
        tests.append(compare_by_sex(fine_df[lb], sex01, lb))

    stats_tbl = pd.concat(tests, axis=0)
    # BH-FDR across all tests
    stats_tbl["fdr"] = multipletests(stats_tbl["pval"].values, method="fdr_bh")[1]
    stats_tbl.to_csv(outdir / "by_sex_tests_CD8_scores.csv")

    # 8) Plots: box by sex for CD8_G-CD8_B and log2 ratio
    try:
        box_by_sex(diff, sex01, "CD8_G - CD8_B (module score) by sex",
                   str(outdir / "box_CD8_G_minus_CD8_B_by_sex.png"))
        box_by_sex(ratio, sex01, "log2(CD8_G / CD8_B) (module score) by sex",
                   str(outdir / "box_log2_CD8G_over_CD8B_by_sex.png"))
    except Exception as e:
        print(f"[WARN] Boxplot failed: {e}")

    # 9) Write a compact README
    with open(outdir / "README_Aim1_2.txt", "w", encoding="utf-8") as fh:
        fh.write(
            "Aim 1.2 — CD8 TIL phenotypes by sex\n"
            "-----------------------------------\n"
            "Files:\n"
            "  - preranked_stats_point_biserial.csv: gene-level r vs sex (female=1, male=0).\n"
            "  - fgsea_results.csv: fgsea on pre-ranked stats (NES, ES, pval, padj [BH-FDR], leading edge).\n"
            "  - fgsea_volcano.png: NES vs -log10(FDR) scatter (FDR = padj).\n"
            "  - sample_module_scores_CD8_1_to_6.csv: per-sample module scores for fine clusters.\n"
            "  - sample_module_scores_CD8_B_vs_CD8_G.csv: per-sample CD8_B, CD8_G, diff, log2 ratio.\n"
            "  - by_sex_tests_CD8_scores.csv: Mann-Whitney + Cliff's delta + FDR across contrasts.\n"
            "  - box_*.png: box/strip plots by sex for key contrasts.\n"
            "\n"
            "Methods:\n"
            "  - Pre-ranking uses point-biserial (Pearson vs binary sex), with tiny deterministic tie-breaking.\n"
            "  - fgsea: Multilevel by default (recommended); min/max size filter applied.\n"
            "  - Module scores = mean z-scored expression across genes in the set.\n"
            "  - CD8_B = union of CD8_1,2,3; CD8_G = union of CD8_4,5,6.\n"
        )

    print(f"[DONE] Outputs written to: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aim 1.2 CD8 TILs by sex (GSEA + module scores)")
    '''
    parser.add_argument("--expr", required=True, help="Expression matrix (genes x samples). CSV/TSV; index=gene Ensembl IDs.")
    parser.add_argument("--meta", required=True, help="Metadata table with sample_id and sex.")
    parser.add_argument("--tumor-only-col", default=None, help="(Optional) Metadata column to filter tumor specimens.")
    parser.add_argument("--tumor-only-values", default=None, help="(Optional) Comma-separated allowed values indicating tumor.")
    parser.add_argument("--sex-col", required=True, help="Column in metadata with sex labels.")
    parser.add_argument("--male-label", default="Male", help="Label for male in metadata (default: Male).")
    parser.add_argument("--female-label", default="Female", help="Label for female in metadata (default: Female).")
    parser.add_argument("--gene-sets", required=True, help="Gene set file (.gmt/.json/.yaml) including CD8_1..CD8_6.")
    parser.add_argument("--nperm", type=int, default=10000, help="fgsea permutations (default: 10000).")
    parser.add_argument("--min-size", type=int, default=10, help="Minimum gene set size (default: 10).")
    parser.add_argument("--max-size", type=int, default=1000, help="Maximum gene set size (default: 1000).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for fgsea.")
    parser.add_argument("--cd8-prefix", default="CD8_", help="Prefix for fine clusters (default: CD8_).")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    '''
    args = parser.parse_args()
    main(args)