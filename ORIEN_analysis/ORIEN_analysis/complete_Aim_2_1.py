"""
Aim 2.1 — Evaluate sex differences in immune signaling pathways (Hallmark, BioCarta, custom T cell–inflamed GEP)

Pipeline:
  1) Load RNA-seq expression (genes x samples) and metadata with 'sex' (+ optional stage/ICI outcome).
  2) Gene-level pre-ranking by point-biserial correlation vs sex (female=0, male=1).
  3) Build pathway collections: MSigDB Hallmark (H), BioCarta (C2:CP:BIOCARTA), and custom T cell–inflamed GEP.
  4) Run fgsea on the pre-ranked vector for pathway-level enrichment by sex.
  5) Compute ssGSEA sample-by-pathway scores (GSVA::gsva, method='ssgsea', ssgsea.norm=TRUE).
  6) Compare ssGSEA scores by sex overall (Mann-Whitney U, Cliff's delta, BH-FDR),
     and (if stage available) within Early vs Metastatic strata.
  7) (Optional) Associate ssGSEA with outcomes: ICI response if present (binary Mann-Whitney),
     OS/PFS if present (Spearman rho with time or event-coded z—not fitting survival models here).
  8) Save CSVs and figures (volcano plot for fgsea; boxplots for highlighted pathways).

Notes:
  - Expression index must be gene symbols (case-insensitive). We uppercase to match gene sets.
  - Hallmark and BioCarta sets are pulled via msigdbr (Homo sapiens) through rpy2.
  - Provide a custom T cell–inflamed gene set via paths.tcell_inflamed_geneset_json (list or {name: [genes]}).
    If missing, a small placeholder list is used (please replace with your validated set).

Outputs (under paths.outputs_of_completing_Aim_2_1):
  - preranked_stats_point_biserial.csv
  - fgsea_results_all_pathways.csv (Hallmark + BioCarta + custom)
  - ssGSEA_scores_samples_x_pathways.csv
  - by_sex_tests_ssGSEA_overall.csv
  - by_sex_tests_ssGSEA_by_stage.csv (if stage available)
  - box_*.png and fgsea_volcano.png
  - README_Aim2_1.txt
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, rankdata

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# stats / FDR
from statsmodels.stats.multitest import multipletests

# rpy2 & R packages
from rpy2 import robjects as ro
from rpy2.robjects import vectors
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


def _to_r_matrix(expr_df: pd.DataFrame):
    """
    Convert pandas DataFrame (genes x samples) to an R 'double' matrix with
    explicit dimnames (rownames=genes, colnames=samples). Ensures unique,
    non-empty rownames to satisfy GSVA's .check_rownames().
    """
    import numpy as np
    import rpy2.robjects as ro
    from rpy2.robjects import vectors

    # sanitize in pandas
    expr = expr_df.copy()
    expr.index = expr.index.astype(str)
    expr.columns = expr.columns.astype(str)

    # drop empty and duplicate gene names (keep first)
    expr = expr.loc[expr.index.str.len() > 0]
    expr = expr[~expr.index.duplicated(keep="first")]

    # build numeric matrix (genes x samples), column-major
    arr = np.asarray(expr.values, dtype=float, order="F")
    r_mat = ro.r.matrix(vectors.FloatVector(arr.ravel(order="F")),
                        nrow=arr.shape[0], ncol=arr.shape[1])

    # set dimnames in one shot and REASSIGN (use an UNNAMED R list)
    set_dimnames = ro.r("`dimnames<-`")
    rn = vectors.StrVector(expr.index.tolist())
    cn = vectors.StrVector(expr.columns.tolist())
    r_list = ro.r.list(rn, cn)          # <-- unnamed list, not ListVector([...])
    r_mat = set_dimnames(r_mat, r_list) # reassign the returned object

    return r_mat


# project paths (adjust to your repo; follows Aim 1.2 convention)
from ORIEN_analysis.config import paths

# ============================== IO ============================== #

def load_expression(expr_path: str) -> pd.DataFrame:
    expr_path = str(expr_path)
    if expr_path.endswith((".tsv", ".txt")):
        df = pd.read_csv(expr_path, sep="\t", index_col=0)
    else:
        df = pd.read_csv(expr_path, sep=None, engine="python", index_col=0)
    df.index = df.index.astype(str).str.upper()
    return df


def load_metadata_for_sex(expr_cols: List[str]) -> pd.DataFrame:
    """
    Reconstructs metadata similarly to Aim 1.2 using ORIEN linkages.
    Requires:
      - paths.clinical_molecular_linkage_data (with 'RNASeq', 'ORIENAvatarKey')
      - paths.patient_data (with 'AvatarKey', 'Sex')
    Returns DataFrame indexed by sample_id with at least 'Sex'.
    Also passes through any stage/outcome columns if found in patient_data or linkage.
    """
    meta = pd.DataFrame({"sample_id": [str(x) for x in expr_cols]})
    link = pd.read_csv(paths.clinical_molecular_linkage_data)
    pat = pd.read_csv(paths.patient_data)

    # Grab a few likely stage/outcome columns if present
    pass_cols = [
        # stage-like columns
        "Stage", "stage", "PathologicStage", "pathologic_stage", "AJCC_Pathologic_Tumor_Stage",
        "ajcc_pathologic_tumor_stage", "Overall_Stage", "Clinical_Stage",
        # outcomes
        "ICI_Response", "ICI_response", "Response", "BestOverallResponse",
        "OS_months", "OS_event", "PFS_months", "PFS_event",
    ]
    keep_link = [c for c in pass_cols if c in link.columns]
    keep_pat = [c for c in pass_cols if c in pat.columns]

    meta = (
        meta.merge(link[["RNASeq", "ORIENAvatarKey"] + keep_link], how="left",
                   left_on="sample_id", right_on="RNASeq")
            .drop(columns=["RNASeq"])
            .merge(pat[["AvatarKey", "Sex"] + keep_pat], how="left",
                   left_on="ORIENAvatarKey", right_on="AvatarKey")
            .drop(columns=["ORIENAvatarKey", "AvatarKey"])
    )
    # Normalize sex labels
    meta["Sex"] = (
        meta["Sex"].astype(str).str.strip().str.upper()
        .map({"M": "Male", "MALE": "Male", "F": "Female", "FEMALE": "Female"})
        .fillna(meta["Sex"])
    )
    meta = meta.set_index("sample_id")
    return meta


def load_custom_tcell_inflamed_geneset() -> Dict[str, List[str]]:
    """
    Tries to read a JSON file with either:
      - {"Tcell_inflamed_GEP":[ "CXCL9", "CXCL10", ... ]}  OR
      - [ "CXCL9", "CXCL10", ... ]  (will wrap with default name)
    If not available, returns a small placeholder list you should replace.
    """
    default_name = "Tcell_inflamed_GEP_custom"
    try:
        obj = json.loads(Path(paths.tcell_inflamed_geneset_json).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return {k: [g.upper() for g in v] for k, v in obj.items()}
        elif isinstance(obj, list):
            return {default_name: [str(g).upper() for g in obj]}
    except Exception:
        pass
    # --- Placeholder (replace with validated GEP) ---
    placeholder = [
        "CXCL9", "CXCL10", "IDO1", "HLA-DRA", "STAT1", "IFNG",
        "GZMA", "GZMB", "PRF1", "CD8A", "PSMB10", "NKG7",
        "CCL5", "CXCL11", "CD27", "PDCD1LG2", "TIGIT", "LAG3"
    ]
    return {default_name: [g.upper() for g in placeholder]}

# ========================= Gene-level stats ====================== #

def map_sex_to_binary(meta: pd.DataFrame, sex_col: str) -> pd.Series:
    s = meta[sex_col].astype(str)
    vals = s.str.lower().map({"female": 0, "male": 1})
    if vals.isna().any():
        bad = meta.loc[vals.isna(), [sex_col]]
        raise ValueError(f"Unmapped sex labels in '{sex_col}'; e.g. {bad.head(8)}")
    return vals.astype(int)


def point_biserial_by_gene(expr: pd.DataFrame, sex01: pd.Series) -> pd.DataFrame:
    common = expr.columns.intersection(sex01.index)
    if len(common) < 2:
        raise ValueError("Need ≥2 aligned samples.")
    X = expr.loc[:, common]
    y = sex01.loc[common].astype(float)

    y_c = y - y.mean()
    y_sd = y_c.std(ddof=1)
    if not np.isfinite(y_sd) or y_sd == 0:
        raise ValueError("Sex has zero variance or misaligned labels.")

    Xc = X.subtract(X.mean(axis=1), axis=0)
    cov = (Xc.values @ y_c.values) / (len(common) - 1)
    x_sd = Xc.std(axis=1, ddof=1).to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / (x_sd * y_sd)
    r = np.where(np.isfinite(r), r, 0.0)

    return pd.DataFrame({
        "gene": X.index.values,
        "r": r,
        "n": int(len(common)),
        "x_sd": x_sd,
        "y_sd": y_sd
    }).set_index("gene")


def build_preranked_vector(stats_df: pd.DataFrame) -> pd.Series:
    s = stats_df["r"].astype(float).copy()
    s = s + (s.rank(method="first") * 1e-12)  # tiny tie-breaker
    s = s.sort_values(ascending=False)
    s.name = "stat"
    return s

# =========================== R helpers =========================== #

def get_msigdb_hallmark_and_biocarta() -> Dict[str, List[str]]:
    """
    Use msigdbr (R) to fetch Hallmark (collection H) and BioCarta (C2:CP:BIOCARTA).
    Returns dict { 'HALLMARK_...' or 'BIOCARTA_...': [GENES...] } with UPPERCASE gene symbols.
    """
    msigdbr = importr("msigdbr")
    base = importr("base")
    dplyr = importr("dplyr")

    # Hallmark
    df_h = msigdbr.msigdbr(species="Homo sapiens", collection="H")
    # BioCarta
    df_b = msigdbr.msigdbr(species="Homo sapiens", collection="C2", subcollection="CP:BIOCARTA")

    with localconverter(ro.default_converter + pandas2ri.converter):
        h = ro.conversion.rpy2py(df_h)
        b = ro.conversion.rpy2py(df_b)

    def collapse(df: pd.DataFrame) -> Dict[str, List[str]]:
        # Columns: gs_name, gene_symbol
        out = {}
        for term, sub in df.groupby("gs_name"):
            genes = sorted(set(sub["gene_symbol"].astype(str).str.upper()))
            out[str(term)] = genes
        return out

    gs = {}
    gs.update(collapse(h))
    gs.update(collapse(b))
    return gs


def run_fgsea(preranked: pd.Series, pathways: Dict[str, List[str]],
              min_size: int = 10, max_size: int = 1000, seed: int = 0) -> pd.DataFrame:
    fgsea = importr("fgsea")
    stats_r = vectors.FloatVector(preranked.values.astype(float))
    stats_r.names = vectors.StrVector(preranked.index.tolist())
    pathways_r = ro.ListVector({k: vectors.StrVector(v) for k, v in pathways.items() if v})

    ro.r(f"set.seed({int(seed)})")
    res_r = fgsea.fgseaMultilevel(pathways=pathways_r, stats=stats_r,
                                  minSize=min_size, maxSize=max_size)
    # sanitize list-cols
    sanitize = ro.r('''
        function(d){
          d <- as.data.frame(d, stringsAsFactors=FALSE)
          for(nm in names(d)){
            if(is.list(d[[nm]])){
              d[[nm]] <- vapply(d[[nm]], function(x){
                if (is.null(x) || length(x)==0L) return("")
                paste(as.character(unlist(x, use.names=FALSE)), collapse=",")
              }, FUN.VALUE=character(1L))
            }
          }
          d
        }
    ''')
    res_r = sanitize(res_r)
    with localconverter(ro.default_converter + pandas2ri.converter):
        res_df = ro.conversion.rpy2py(res_r)
    # standardize colnames
    res_df = res_df.rename(columns={"pathway": "term", "padj": "FDR", "pval": "pval"})
    cols = [c for c in ["term", "NES", "ES", "size", "pval", "FDR", "leadingEdge"] if c in res_df.columns]
    return res_df.loc[:, cols]


def run_ssgsea(expr: pd.DataFrame, gene_sets,
               min_size: int = 10, max_size: int = 1000, parallel_sz: int = 0) -> pd.DataFrame:
    """
    GSVA >= 2.x S4 workflow:
      se    <- SummarizedExperiment(assays=SimpleList(exprs=matrix_with_rownames))
      param <- GSVA::ssgseaParam(exprData=se, geneSets=..., minSize=..., maxSize=..., alpha=0.25, normalize=TRUE)
      res   <- GSVA::gsva(param, BPPARAM=BiocParallel::SerialParam())
    Returns pandas DataFrame (samples x pathways).
    """
    import numpy as np
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.vectors import ListVector, StrVector
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    # --- Build strict R matrix (genes x samples) with explicit dimnames ---
    # Also enforce non-empty, unique rownames before converting.
    expr2 = expr.copy()
    expr2.index = expr2.index.astype(str)
    expr2 = expr2[~expr2.index.duplicated(keep="first")]
    expr2 = expr2.loc[expr2.index.str.len() > 0]  # no empty names
    r_expr = _to_r_matrix(expr2)

    # --- Named list of character vectors for gene sets ---
    if isinstance(gene_sets, dict):
        r_gs = ListVector({
            str(k): StrVector(sorted({str(g).upper() for g in v if pd.notna(g)}))
            for k, v in gene_sets.items() if v
        })
    else:
        r_gs = gene_sets

    # --- Import R packages we need ---
    importr("GSVA")
    importr("BiocParallel")
    importr("SummarizedExperiment")
    importr("S4Vectors")

    # --- R helper: wrap matrix in SummarizedExperiment with assay name 'exprs' ---
    ro.r("""
      run_ssgsea_newapi <- function(expr, gset, min_sz, max_sz, do_norm=TRUE, verbose=FALSE) {
        # pin base symbols
        rownames  <- base::rownames; colnames <- base::colnames
        dimnames  <- base::dimnames; as.vector <- base::as.vector

        stopifnot(!is.null(base::rownames(expr)), !is.null(base::colnames(expr)))

        # Build SE with a recognized assay name for GSVA ("exprs")
        se <- SummarizedExperiment::SummarizedExperiment(
          assays = S4Vectors::SimpleList(exprs = expr)
        )

        # Serial backend (deterministic from Python)
        BPPARAM <- BiocParallel::SerialParam(progressbar = verbose)

        # Method-specific parameters with exprData carried inside the S4 object
        param <- GSVA::ssgseaParam(
          exprData  = se,
          geneSets  = gset,
          minSize   = as.numeric(min_sz),
          maxSize   = as.numeric(max_sz),
          alpha     = 0.25,
          normalize = isTRUE(do_norm)
        )

        # Run GSVA
        GSVA::gsva(param, verbose = verbose, BPPARAM = BPPARAM)
      }
    """)
    run_ssgsea_newapi = ro.globalenv["run_ssgsea_newapi"]

    # --- Execute ---
    res_r = run_ssgsea_newapi(r_expr, r_gs, float(min_size), float(max_size), True, False)

    # --- If GSVA returned a SummarizedExperiment, extract the scores assay as a base matrix ---
    extract_mat = ro.r("""
      function(x){
        if (methods::is(x, "SummarizedExperiment")) {
          nms <- SummarizedExperiment::assayNames(x)
          if (length(nms) && "scores" %in% nms) {
            m <- SummarizedExperiment::assay(x, "scores")
          } else {
            m <- SummarizedExperiment::assay(x, 1L)
          }
          return(base::as.matrix(m))
        } else if (base::is.matrix(x)) {
          return(x)
        } else {
          # try to coerce DelayedMatrix / Matrix, etc., to base matrix
          return(base::as.matrix(x))
        }
      }
    """)
    mat_r = extract_mat(res_r)

    # --- Convert R matrix (pathways x samples) -> pandas (samples x pathways) ---
    with localconverter(ro.default_converter + pandas2ri.converter):
        res_obj = ro.conversion.rpy2py(mat_r)

    # Build a proper DataFrame even if rpy2 returned a NumPy array
    if isinstance(res_obj, pd.DataFrame):
        mat_df = res_obj
    else:
        # fetch dimnames from the R matrix to label rows/cols
        rn = list(ro.r("base::rownames")(mat_r))
        cn = list(ro.r("base::colnames")(mat_r))
        mat_df = pd.DataFrame(np.asarray(res_obj), index=rn, columns=cn)

    # R returns pathways x samples; flip to samples x pathways
    scores = mat_df.T.copy()
    scores.index.name = None
    scores.columns.name = None
    return scores

# ====================== Effect sizes & tests ===================== #

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    allv = np.concatenate([x, y])
    ranks = rankdata(allv, method="average")
    rx = ranks[:nx]
    U = rx.sum() - nx * (nx + 1) / 2.0
    return float((2 * U) / (nx * ny) - 1)


def mw_by_sex(scores: pd.Series, sex01: pd.Series, label: str) -> pd.DataFrame:
    common = scores.index.intersection(sex01.index)
    s = scores.loc[common]
    y = sex01.loc[common]
    f = s[y == 0].to_numpy()
    m = s[y == 1].to_numpy()
    if len(f) == 0 or len(m) == 0:
        return pd.DataFrame(columns=[
            "term", "n_female", "n_male", "mw_u", "pval",
            "cliffs_delta_male_minus_female", "mean_female", "mean_male"
        ])
    u, p = mannwhitneyu(f, m, alternative="two-sided")
    delta = cliffs_delta(m, f)  # + => higher in males
    return pd.DataFrame({
        "term": [label], "n_female": [len(f)], "n_male": [len(m)],
        "mw_u": [u], "pval": [p],
        "cliffs_delta_male_minus_female": [delta],
        "mean_female": [float(np.mean(f))], "mean_male": [float(np.mean(m))]
    })


def stage_bucket(s: pd.Series) -> pd.Series:
    """
    Map free-text stage to {Early, Metastatic}. Heuristics:
      Early: contains I or II (and not 'metastatic'/'iv')
      Metastatic: contains III, IV, M1, or 'metastatic'
    Unknown -> NaN
    """
    x = s.astype(str).str.lower()
    is_meta = (
        x.str.contains(r"\b(iii|iv|m1|metastatic)\b", regex=True) |
        x.str.contains("stage iv") | x.str.contains("stage iii")
    )
    is_early = (
        x.str.contains(r"\b(i|ii)\b", regex=True) |
        x.str.contains("stage i") | x.str.contains("stage ii")
    ) & (~is_meta)
    out = pd.Series(index=s.index, dtype="object")
    out[is_early] = "Early"
    out[is_meta] = "Metastatic"
    return out


# ============================== Plots ============================ #

def volcano_from_fgsea(res_df: pd.DataFrame, out_png: str, title: str):
    if res_df.empty:
        print("[WARN] fgsea result is empty; skipping volcano.")
        return
    df = res_df.copy()
    df["FDR_plot"] = df["FDR"].replace(0, np.nextafter(0, 1))
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(data=df, x="NES", y="FDR_plot",
                         hue=(df["FDR"] < 0.1), style=(df["FDR"] < 0.05), s=60)
    ax.axhline(0.1, ls="--", lw=1)
    ax.axvline(0, ls=":", lw=1)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_ylabel("FDR")
    ax.set_title(title)
    if ax.get_legend() is not None:
        ax.legend(title="FDR thresholds")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def box_by_sex(scores: pd.Series, sex01: pd.Series, term: str, out_png: str):
    common = scores.index.intersection(sex01.index)
    df = pd.DataFrame({
        "score": scores.loc[common].values,
        "sex": np.where(sex01.loc[common].values == 0, "Female", "Male")
    })
    plt.figure(figsize=(5, 5))
    ax = sns.boxplot(data=df, x="sex", y="score")
    sns.stripplot(data=df, x="sex", y="score", color="black", alpha=0.5)
    ax.set_title(term)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ============================== Main ============================ #

def main():
    outdir = Path(paths.outputs_of_completing_Aim_2_1)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    expr = load_expression(paths.expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest)
    meta = load_metadata_for_sex(expr.columns.tolist())

    # Harmonize
    common = pd.Index(expr.columns.astype(str)).intersection(meta.index.astype(str))
    if common.empty:
        raise ValueError("No overlapping samples between expression and metadata.")
    expr = expr.loc[:, common]
    meta = meta.loc[common].copy()

    # 2) Pre-ranked stats vs sex
    sex01 = map_sex_to_binary(meta, "Sex")
    stats_df = point_biserial_by_gene(expr, sex01)
    stats_df.to_csv(outdir / "preranked_stats_point_biserial.csv")
    preranked = build_preranked_vector(stats_df)

    # 3) Gene sets: Hallmark + BioCarta + custom T cell–inflamed
    msigdb_sets = get_msigdb_hallmark_and_biocarta()
    custom_sets = load_custom_tcell_inflamed_geneset()
    # keep some immune-focused labels for plotting later
    immune_keywords = ["INTERFERON", "INFLAMMATORY", "IL", "TNF", "TCR", "CYTOKINE", "ANTIGEN", "COMPLEMENT", "ALLOGRAFT", "APOPTOSIS"]
    all_sets = {}
    all_sets.update(msigdb_sets)
    all_sets.update(custom_sets)

    # 4) Pathway-level fgsea on pre-ranked vector
    fgsea_res = run_fgsea(preranked, all_sets, min_size=10, max_size=1000, seed=0)
    fgsea_res.to_csv(outdir / "fgsea_results_all_pathways.csv", index=False)
    volcano_from_fgsea(fgsea_res, str(outdir / "fgsea_volcano.png"),
                       "fgsea (sex pre-ranked) — Hallmark + BioCarta + custom")

    expr = expr.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    # 5) ssGSEA sample x pathway scores
    ssgsea_scores = run_ssgsea(expr, all_sets, min_size=10, max_size=1000, parallel_sz=0)
    ssgsea_scores.to_csv(outdir / "ssGSEA_scores_samples_x_pathways.csv")

    # 6) Compare ssGSEA by sex (overall)
    tests = []
    for term in ssgsea_scores.columns:
        tests.append(mw_by_sex(ssgsea_scores[term], sex01, term))
    overall_tbl = pd.concat(tests, axis=0, ignore_index=True)
    if not overall_tbl.empty:
        overall_tbl["FDR"] = multipletests(overall_tbl["pval"].values, method="fdr_bh")[1]
    overall_tbl.to_csv(outdir / "by_sex_tests_ssGSEA_overall.csv", index=False)

    # Boxplots for a few immune-highlighted pathways
    try:
        # choose top immune-like terms by FDR
        immune_like = overall_tbl[
            overall_tbl["term"].str.upper().str.contains("|".join(immune_keywords))
        ].sort_values("FDR", ascending=True).head(6)
        for term in immune_like["term"].tolist():
            box_by_sex(ssgsea_scores[term], sex01, f"{term} (ssGSEA) by sex",
                       str(outdir / f"box_ssGSEA_{term.replace('/', '_')}_by_sex.png"))
    except Exception as e:
        print(f"[WARN] Boxplot generation skipped: {e}")

    # 6b) Stratified by stage if available
    bystage_tbl = pd.DataFrame()
    stage_cols = [c for c in meta.columns if "stage" in c.lower()]
    stage_series = None
    if stage_cols:
        stage_series = stage_bucket(meta[stage_cols[0]])
        stage_series.name = "StageBucket"
        for bucket in ["Early", "Metastatic"]:
            idx = stage_series[stage_series == bucket].index
            if len(idx) >= 6:  # need at least modest N
                tests_b = []
                sex01_b = sex01.loc[idx]
                scores_b = ssgsea_scores.loc[idx]
                for term in scores_b.columns:
                    tests_b.append(mw_by_sex(scores_b[term], sex01_b, term))
                sub = pd.concat(tests_b, axis=0, ignore_index=True)
                if not sub.empty:
                    sub["FDR"] = multipletests(sub["pval"].values, method="fdr_bh")[1]
                sub.insert(0, "stage_bucket", bucket)
                bystage_tbl = pd.concat([bystage_tbl, sub], axis=0, ignore_index=True)
        if not bystage_tbl.empty:
            bystage_tbl.to_csv(outdir / "by_sex_tests_ssGSEA_by_stage.csv", index=False)

    # 7) Optional associations with outcomes (best-effort)
    assoc_rows = []
    # ICI response (binary)
    resp_cols = [c for c in meta.columns if "response" in c.lower()]
    if resp_cols:
        rc = resp_cols[0]
        resp = meta[rc].astype(str).str.lower()
        # Map common categories to binary responder vs non-responder
        resp_map = {
            "cr": 1, "complete response": 1,
            "pr": 1, "partial response": 1,
            "sd": 0, "stable disease": 0,
            "pd": 0, "progressive disease": 0,
            "responder": 1, "nonresponder": 0, "non-responder": 0
        }
        y = resp.replace(resp_map)
        if y.isin([0, 1]).sum() >= 6:
            y = y.where(y.isin([0, 1])).dropna().astype(int)
            common_r = ssgsea_scores.index.intersection(y.index)
            for term in ssgsea_scores.columns:
                s = ssgsea_scores.loc[common_r, term]
                # Mann-Whitney responders vs nonresponders
                r = s[y.loc[common_r] == 1].to_numpy()
                n = s[y.loc[common_r] == 0].to_numpy()
                if len(r) >= 3 and len(n) >= 3:
                    u, p = mannwhitneyu(r, n, alternative="two-sided")
                    assoc_rows.append({
                        "outcome": "ICI_response",
                        "term": term,
                        "n_resp": len(r),
                        "n_nonresp": len(n),
                        "mw_u": u, "pval": p,
                        "effect_resp_minus_nonresp": float(np.mean(r) - np.mean(n))
                    })

    # Simple Spearman with OS/PFS months if available
    for tcol in ["OS_months", "PFS_months"]:
        if tcol in meta.columns:
            y = pd.to_numeric(meta[tcol], errors="coerce")
            common_t = ssgsea_scores.index.intersection(y.dropna().index)
            if len(common_t) >= 6:
                for term in ssgsea_scores.columns:
                    rho, p = spearmanr(ssgsea_scores.loc[common_t, term], y.loc[common_t])
                    assoc_rows.append({
                        "outcome": tcol, "term": term, "spearman_rho": rho, "pval": p, "n": len(common_t)
                    })

    if assoc_rows:
        assoc_df = pd.DataFrame(assoc_rows)
        # FDR within each outcome family
        for oc in assoc_df["outcome"].unique():
            mask = assoc_df["outcome"] == oc
            assoc_df.loc[mask, "FDR"] = multipletests(assoc_df.loc[mask, "pval"].values, method="fdr_bh")[1]
        assoc_df.to_csv(outdir / "associations_ssGSEA_outcomes.csv", index=False)

    # 8) README
    with open(outdir / "README_Aim2_1.txt", "w", encoding="utf-8") as fh:
        fh.write(
            "Aim 2.1 — Immune signaling pathways by sex (Hallmark, BioCarta, custom GEP)\n"
            "----------------------------------------------------------------------------\n"
            "Key files:\n"
            "  - preranked_stats_point_biserial.csv: gene-level r vs sex.\n"
            "  - fgsea_results_all_pathways.csv: fgsea (NES, ES, pval, FDR, leadingEdge).\n"
            "  - fgsea_volcano.png: NES vs FDR scatter.\n"
            "  - ssGSEA_scores_samples_x_pathways.csv: per-sample ssGSEA scores.\n"
            "  - by_sex_tests_ssGSEA_overall.csv: Mann-Whitney + Cliff's delta + FDR (overall).\n"
            "  - by_sex_tests_ssGSEA_by_stage.csv: same, within Early/Metastatic (if stage available).\n"
            "  - associations_ssGSEA_outcomes.csv: optional ICI response and OS/PFS associations (best-effort).\n"
            "\n"
            "Methods:\n"
            "  - Pre-ranking uses point-biserial (Pearson vs binary sex) with tie-breaking.\n"
            "  - Pathways from msigdbr: Hallmark (H) and BioCarta (C2:CP:BIOCARTA), plus custom T cell–inflamed GEP.\n"
            "  - fgseaMultilevel for enrichment; GSVA::gsva(method='ssgsea', ssgsea.norm=TRUE) for sample-level scores.\n"
            "  - Sex comparisons: Mann-Whitney, Cliff's delta (positive = higher in males), BH-FDR across all pathways.\n"
            "  - Stage buckets via simple I/II vs III/IV/metastatic string heuristics (customize as needed).\n"
            "  - Outcome associations are exploratory summaries (no survival models here).\n"
        )

    print(f"[DONE] Aim 2.1 outputs written to: {outdir}")


if __name__ == "__main__":
    main()