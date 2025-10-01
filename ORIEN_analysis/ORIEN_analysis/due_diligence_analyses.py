import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

from ORIEN_analysis.config import paths
from ORIEN_analysis.complete_Aim_1_2 import (
    load_expression,
    load_metadata,
    map_sex_to_binary,
    zscore_rows,
    module_score,
    compare_by_sex,
)

# It is assumed that PyTIDE is installed in the environment.
# pip install PyTIDE
try:
    from PyTIDE import tide
except ImportError:
    print("PyTIDE is not installed. Please install it using: pip install PyTIDE")
    tide = None

# It is assumed that rpy2 and GSVA are installed.
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

# --- Utility Functions ---

def load_data_and_align(
    expr_path: Path, meta_path: Path, patient_path: Path, sex_col: str, male_label: str, female_label: str
):
    """Loads expression and metadata, aligns samples, and adds binary sex indicator."""
    expr = load_expression(expr_path)
    
    # Simplified metadata loading from complete_Aim_1_2.py
    list_of_sample_IDs = expr.columns.tolist()
    meta = pd.DataFrame(list_of_sample_IDs, columns=["sample_id"])
    clinical_molecular_linkage_data = pd.read_csv(paths.clinical_molecular_linkage_data)
    patient_data = pd.read_csv(paths.patient_data)
    staging_data = pd.read_csv(paths.output_of_pairing_clinical_data_and_stages_of_tumors)
    
    meta = (
        meta
        .merge(
            clinical_molecular_linkage_data[["RNASeq", "ORIENAvatarKey", "Age At Specimen Collection", "DeidSpecimenID"]],
            how="left", left_on="sample_id", right_on="RNASeq"
        )
        .rename(columns = {"Age At Specimen Collection": "Age_At_Specimen_Collection"})
        .drop(columns="RNASeq")
        .merge(
            patient_data[["AvatarKey", "Sex"]],
            how="left", left_on="ORIENAvatarKey", right_on="AvatarKey"
        )
        .merge(
            staging_data[["ORIENSpecimenID", "EKN Assigned Stage"]],
            how="left", left_on="DeidSpecimenID", right_on="ORIENSpecimenID"
        )
        .drop(columns=["ORIENAvatarKey", "AvatarKey", "ORIENSpecimenID"])
        .rename(columns={"EKN Assigned Stage": "Stage"})
    )

    # Harmonize samples
    sample_intersection = pd.Index(expr.columns.astype(str)).intersection(meta["sample_id"].astype(str))
    if sample_intersection.empty:
        raise ValueError("No overlapping samples between expression and metadata.")
    
    meta = meta.set_index("sample_id").loc[sample_intersection]
    expr = expr.loc[:, sample_intersection]

    meta["Sex"] = (
        meta["Sex"].astype(str).str.strip().str.upper()
        .map({"M": "Male", "MALE": "Male", "F": "Female", "FEMALE": "Female"})
        .fillna(meta["Sex"])
    )
    
    meta["sex01"] = map_sex_to_binary(meta, "Sex", "Male", "Female").astype(int)
    
    # Ensure stage is categorical for models
    meta['Stage'] = meta['Stage'].astype('category')
    
    return expr, meta


# --- Analysis Functions ---

def run_aim1_2_with_covariates(expr: pd.DataFrame, meta: pd.DataFrame, outdir: Path):
    """
    Performs Aim 1.2 analysis (module scores) adjusting for age and stage.
    Instead of direct comparison, it compares the residuals of the scores after
    regressing out covariates.
    """
    print("Running Aim 1.2 module score comparison with covariate adjustment...")
    
    from ORIEN_analysis.complete_Aim_1_2 import load_gene_sets

    gene_sets = load_gene_sets()
    
    cd8_b_names = [f"CD8_{i}" for i in (1, 2, 3)]
    cd8_g_names = [f"CD8_{i}" for i in (4, 5, 6)]
    cd8_b_genes = sorted(set(sum((gene_sets.get(n, []) for n in cd8_b_names), [])))
    cd8_g_genes = sorted(set(sum((gene_sets.get(n, []) for n in cd8_g_names), [])))

    score_b = module_score(expr, cd8_b_genes).rename("CD8_B_score")
    score_g = module_score(expr, cd8_g_genes).rename("CD8_G_score")
    diff_score = (score_g - score_b).rename("CD8_G_minus_CD8_B")

    scores_df = pd.DataFrame({"score": diff_score})
    joined_df = meta.join(scores_df)
    analysis_df = joined_df.dropna(subset=['score', 'Age_At_Specimen_Collection', 'Stage', 'sex01'])

    # Model: score ~ Age + Stage
    model = smf.ols("score ~ Age_At_Specimen_Collection + C(Stage)", data=analysis_df).fit()
    analysis_df['residuals'] = model.resid

    # Compare residuals by sex
    females = analysis_df[analysis_df['sex01'] == 0]['residuals']
    males = analysis_df[analysis_df['sex01'] == 1]['residuals']
    
    u_stat, pval = mannwhitneyu(females, males, alternative="two-sided")

    result = pd.DataFrame({
        "contrast": ["CD8_G_minus_CD8_B_adjusted"],
        "n_female": [len(females)],
        "n_male": [len(males)],
        "mw_u": [u_stat],
        "pval": [pval],
    })
    
    result.to_csv(outdir / "aim1_2_module_scores_adjusted.csv", index=False)
    print(f"Saved covariate-adjusted Aim 1.2 results to {outdir}")


def run_tide_analysis(expr: pd.DataFrame, meta: pd.DataFrame, outdir: Path):
    """
    Runs TIDE analysis and compares Dysfunction and Exclusion scores between sexes.
    """
    if tide is None:
        print("Skipping TIDE analysis as PyTIDE is not installed.")
        return
        
    print("Running TIDE analysis...")
    # TIDE expects samples in rows, genes in columns, so we transpose
    tide_results = tide.run(expr.T)
    
    # Merge with metadata
    tide_results.index.name = "sample_id"
    analysis_df = meta.join(tide_results)

    results = []
    for score in ['Dysfunction', 'Exclusion']:
        if score in analysis_df.columns:
            females = analysis_df[analysis_df['sex01'] == 0][score].dropna()
            males = analysis_df[analysis_df['sex01'] == 1][score].dropna()
            
            u_stat, pval = mannwhitneyu(females, males, alternative="two-sided")
            
            results.append({
                "score_type": score,
                "n_female": len(females),
                "n_male": len(males),
                "mean_female": females.mean(),
                "mean_male": males.mean(),
                "mw_u": u_stat,
                "pval": pval,
            })

    if results:
        pd.DataFrame(results).to_csv(outdir / "tide_results_by_sex.csv", index=False)
        print(f"Saved TIDE analysis results to {outdir}")

def run_hypoxia_hla_analysis(expr: pd.DataFrame, meta: pd.DataFrame, outdir: Path):
    """
    Calculates hypoxia scores and compares HLA gene expression between sexes.
    """
    print("Running Hypoxia and HLA analysis...")
    
    # --- Hypoxia Analysis using GSVA via rpy2 ---
    try:
        msigdbr = importr('msigdbr')
        gsva = importr('GSVA')
        
        # Get Hallmark Hypoxia gene set
        hallmark_sets = msigdbr.msigdbr(species="Homo sapiens", category="H")
        
        with (ro.default_converter + pandas2ri.converter).context():
            hallmark_r = ro.conversion.get_conversion().py2rpy(hallmark_sets)

        hallmark_list = msigdbr.msigdbr_to_list(hallmark_r)
        hypoxia_genes = hallmark_list.rx2('HALLMARK_HYPOXIA')

        # Run GSVA
        with (ro.default_converter + pandas2ri.converter).context():
            expr_r = ro.conversion.get_conversion().py2rpy(expr)
        
        gsva_results = gsva.gsva(expr_r, ro.ListVector({'HALLMARK_HYPOXIA': hypoxia_genes}), method='ssgsea', verbose=False)
        
        with (ro.default_converter + pandas2ri.converter).context():
             hypoxia_scores = ro.conversion.get_conversion().rpy2py(gsva_results).T
        
        hypoxia_scores = pd.DataFrame(hypoxia_scores, index=expr.columns, columns=['hypoxia_score'])
        
        # Compare scores
        analysis_df = meta.join(hypoxia_scores)
        females = analysis_df[analysis_df['sex01'] == 0]['hypoxia_score'].dropna()
        males = analysis_df[analysis_df['sex01'] == 1]['hypoxia_score'].dropna()
        u_stat, pval = mannwhitneyu(females, males, alternative="two-sided")
        
        hypoxia_result = pd.DataFrame([{
            "n_female": len(females), "n_male": len(males),
            "mean_female": females.mean(), "mean_male": males.mean(),
            "pval": pval
        }])
        hypoxia_result.to_csv(outdir / "hypoxia_score_by_sex.csv", index=False)
        print("Hypoxia analysis complete.")

    except Exception as e:
        print(f"Could not run Hypoxia analysis: {e}")

    # --- HLA Gene Expression Analysis ---
    hla_class_I = ['HLA-A', 'HLA-B', 'HLA-C']
    hla_class_II = ['HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRA', 'HLA-DRB1']
    
    hla_results = []
    for hla_type, hla_genes in [("Class_I", hla_class_I), ("Class_II", hla_class_II)]:
        # Find which genes are present in the expression matrix (case-insensitive)
        present_genes = [g for g in hla_genes if g.upper() in expr.index.str.upper()]
        
        if not present_genes:
            print(f"No {hla_type} genes found in expression data.")
            continue
            
        # Calculate mean expression for the gene set
        mean_expr = expr.loc[present_genes].mean(axis=0)
        mean_expr.name = "mean_hla_expr"
        
        analysis_df = meta.join(mean_expr)
        females = analysis_df[analysis_df['sex01'] == 0]['mean_hla_expr'].dropna()
        males = analysis_df[analysis_df['sex01'] == 1]['mean_hla_expr'].dropna()
        u_stat, pval = mannwhitneyu(females, males, alternative="two-sided")

        hla_results.append({
            "hla_type": hla_type,
            "n_female": len(females), "n_male": len(males),
            "mean_female": females.mean(), "mean_male": males.mean(),
            "pval": pval
        })

    if hla_results:
        pd.DataFrame(hla_results).to_csv(outdir / "hla_expression_by_sex.csv", index=False)
        print("HLA analysis complete.")


def main():
    parser = argparse.ArgumentParser(description="Run due diligence analyses for ORIEN melanoma project.")
    parser.add_argument(
        "--analysis",
        choices=['all', 'aim1.2-covariates', 'tide', 'hypoxia-hla'],
        default='all',
        help="Specify which analysis to run."
    )
    args = parser.parse_args()

    outdir = paths.output / "due_diligence_analyses"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load and Prepare Data ---
    print("Loading and preparing data...")
    expr, meta = load_data_and_align(
        expr_path=paths.expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest,
        meta_path=paths.clinical_molecular_linkage_data, # Placeholder, logic inside function uses multiple files
        patient_path=paths.patient_data,
        sex_col="Sex",
        male_label="Male",
        female_label="Female"
    )

    # --- Run Selected Analyses ---
    if args.analysis in ['all', 'aim1.2-covariates']:
        run_aim1_2_with_covariates(expr, meta, outdir)
        
    if args.analysis in ['all', 'tide']:
        run_tide_analysis(expr, meta, outdir)

    if args.analysis in ['all', 'hypoxia-hla']:
        run_hypoxia_hla_analysis(expr, meta, outdir)
        
    print("\nDue diligence script finished.")

if __name__ == "__main__":
    main()
