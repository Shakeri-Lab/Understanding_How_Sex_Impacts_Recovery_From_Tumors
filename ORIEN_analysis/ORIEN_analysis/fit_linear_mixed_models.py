import argparse
from statsmodels.stats.multitest import multipletests
import numpy as np
from ORIEN_analysis.config import paths
import pandas as pd
import statsmodels.formula.api as smf


def numericize_age(age) -> float:
    if age == "Age 90 or older":
        return 90.0
    elif age == "Age Unknown/Not Recorded":
        return np.nan
    else:
        return float(age)


def create_data_frame_of_enrichment_scores_and_clinical_and_QC_data(path_of_expression_matrix) -> tuple[pd.DataFrame, list[str]]:
    QC_data = pd.read_csv(paths.QC_data)
    QC_data["SequencingDepth"] = np.log10(QC_data["TotalReads"])
    clinical_molecular_linkage_data = pd.read_csv(paths.clinical_molecular_linkage_data)
    diagnosis_data = pd.read_csv(paths.diagnosis_data)
    diagnosis_data["AgeAtDiagnosis"] = diagnosis_data["AgeAtDiagnosis"].apply(numericize_age)
    enrichment_matrix = pd.read_csv(path_of_expression_matrix)
    medications_data = pd.read_csv(paths.medications_data)
    patient_data = pd.read_csv(paths.patient_data)

    set_of_medications_for_ICB_therapy = {
        "Pembrolizumab", "Nivolumab", # PD‑1
        "Atezolizumab", # PD‑L1
        "Ipilimumab" # CTLA‑4
    }
    medications_data["medication_is_for_ICB_therapy"] = medications_data["Medication"].isin(set_of_medications_for_ICB_therapy)
    medications_data_for_ICB_therapy = medications_data[medications_data["medication_is_for_ICB_therapy"]].copy()
    medications_data_for_ICB_therapy["age_at_start_of_medication"] = (
        medications_data_for_ICB_therapy["AgeAtMedStart"].apply(numericize_age)
    )
    data_frame_of_patient_information = (
        medications_data_for_ICB_therapy
        .groupby("AvatarKey", as_index = False)
        .agg(
            patient_has_received_ICB_therapy = ("medication_is_for_ICB_therapy", "any"),
            age_at_start_of_ICB_therapy = ("age_at_start_of_medication", "min")
        )
    )

    def determine_stage_at_start_of_ICB_therapy(row_of_patient_information):
        age_at_start_of_ICB_therapy = row_of_patient_information["age_at_start_of_ICB_therapy"]
        diagnosis_data_for_patient = diagnosis_data[diagnosis_data["AvatarKey"] == row_of_patient_information["AvatarKey"]]
        diagnosis_data_for_patient["time_between_start_of_ICB_therapy_and_diagnosis"] = (
            age_at_start_of_ICB_therapy - diagnosis_data_for_patient["AgeAtDiagnosis"]
        ).abs()
        row_of_diagnosis_data = diagnosis_data_for_patient.sort_values("time_between_start_of_ICB_therapy_and_diagnosis").iloc[0]
        stage = row_of_diagnosis_data.get("PathGroupStage")
        if pd.isna(stage):
            stage = row_of_diagnosis_data.get("ClinGroupStage")
        return stage
    data_frame_of_patient_information["stage_at_start_of_ICB_therapy"] = data_frame_of_patient_information.apply(
        determine_stage_at_start_of_ICB_therapy,
        axis = 1
    )

    data_frame_of_enrichment_scores_and_clinical_and_QC_data = (
        enrichment_matrix
        .merge(
            clinical_molecular_linkage_data[["RNASeq", "ORIENAvatarKey"]],
            how = "left",
            left_on = "SampleID",
            right_on = "RNASeq"
        )
        .drop(columns = "RNASeq")
        .merge(
            patient_data[["AvatarKey", "AgeAtClinicalRecordCreation", "Sex"]],
            how = "left",
            left_on = "ORIENAvatarKey",
            right_on = "AvatarKey"
        )
        .drop(columns = ["AvatarKey"])
        .merge(
            data_frame_of_patient_information[
                ["AvatarKey", "patient_has_received_ICB_therapy", "stage_at_start_of_ICB_therapy"]
            ],
            how = "left",
            left_on = "ORIENAvatarKey",
            right_on = "AvatarKey"
        )
        .drop(columns = ["AvatarKey"])
        .merge(
            QC_data[["SLID", "NexusBatch", "SequencingDepth"]],
            how = "left",
            left_on = "SampleID",
            right_on = "SLID"
        )
        .drop(columns = ["SLID"])
    )
    data_frame_of_enrichment_scores_and_clinical_and_QC_data["patient_has_received_ICB_therapy"] = (
        data_frame_of_enrichment_scores_and_clinical_and_QC_data["patient_has_received_ICB_therapy"]
        .astype("boolean")
        .fillna(False)
    )
    data_frame_of_enrichment_scores_and_clinical_and_QC_data["stage_at_start_of_ICB_therapy"] = (
        data_frame_of_enrichment_scores_and_clinical_and_QC_data["stage_at_start_of_ICB_therapy"].fillna("Unknown")
    )

    list_of_cell_types = [
        name_of_column
        for name_of_column in enrichment_matrix.columns
        if name_of_column not in ["SampleID", "ImmuneScore", "StromaScore", "MicroenvironmentScore"]
    ]

    return data_frame_of_enrichment_scores_and_clinical_and_QC_data, list_of_cell_types


def matrix_condition_number(exog: np.ndarray, drop_constant: bool = True) -> float:
    """
    Compute the 2-norm (spectral-norm) condition number of the fixed-effects
    design matrix.

    Parameters
    ----------
    exog : array-like (n_obs × n_params)
        The design matrix returned by `model.exog` in statsmodels.
    drop_constant : bool, default True
        If True, columns with zero variance (e.g. the intercept) are removed
        before computing the condition number—otherwise an all-ones column
        makes the result explode.

    Returns
    -------
    float
        κ(X) = σ_max / σ_min  (ratio of largest to smallest non-zero singular
        values).  Returns ``np.inf`` if the matrix is rank-deficient, and
        ``np.nan`` if no columns remain after dropping constants.
    """
    X = np.asarray(exog, dtype=float)

    if drop_constant:
        # keep only columns whose sample variance is non-zero
        keep = X.std(axis=0) > 0
        X = X[:, keep]
        if X.size == 0:
            return np.nan                          # all columns were constant

    # handle rank deficiency / numerical underflow robustly
    try:
        # full_matrices=False gives economy-size SVD (faster, less memory)
        _, sing_vals, _ = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.inf                              # SVD failed ⇒ treat as ill-conditioned

    # discard tiny singular values below numeric tolerance
    tol = np.finfo(float).eps * max(X.shape) * sing_vals[0]
    sing_vals = sing_vals[sing_vals > tol]
    if sing_vals.size == 0:
        return np.inf                              # effectively rank-deficient

    return float(sing_vals[0] / sing_vals[-1])


def fit_linear_mixed_models(
    data_frame_of_enrichment_scores_clinical_data_and_QC_data: pd.DataFrame,
    list_of_cell_types: list[str],
    models_should_be_diagnosed: bool
) -> pd.DataFrame:    
    list_of_results = []
    for cell_type in list_of_cell_types:
        data_frame = data_frame_of_enrichment_scores_clinical_data_and_QC_data[
            [
                cell_type,
                "Sex",
                "AgeAtClinicalRecordCreation",
                "stage_at_start_of_ICB_therapy",
                "patient_has_received_ICB_therapy",
                "NexusBatch",
                "SequencingDepth",
                "ORIENAvatarKey"
            ]
        ].rename(
            columns = {cell_type: "Score"}
        )
        data_frame["indicator_of_sex"] = (data_frame["Sex"] == "Male").astype(int)
        data_frame["AgeAtClinicalRecordCreation"] = data_frame["AgeAtClinicalRecordCreation"].apply(numericize_age)
        data_frame["patient_has_received_ICB_therapy"] = data_frame["patient_has_received_ICB_therapy"].astype(bool)

        if models_should_be_diagnosed:
            # 1. Within-patient replication
            replication_counts = data_frame.groupby("ORIENAvatarKey").size()
            frac_singletons = (replication_counts <= 1).mean()
            print(
                "Diagnostic - %s: %.1f%% of patients contribute only one sample.",
                cell_type, 100 * frac_singletons
            )

            # 3. Spread of the response
            score_std = data_frame["Score"].std(ddof = 0)
            print("Diagnostic - %s: response std = %.4f", cell_type, score_std)

            # 4. Categorical level sparsity
            sparsity_msgs = []
            for cat in ["stage_at_start_of_ICB_therapy", "NexusBatch"]:
                lvl_counts = data_frame[cat].value_counts()
                rare_lvls  = lvl_counts[lvl_counts < 3]
                if not rare_lvls.empty:
                    sparsity_msgs.append(f"{cat}: {len(rare_lvls)} sparse levels")
            if sparsity_msgs:
                print("Diagnostic - %s: %s", cell_type, "; ".join(sparsity_msgs))

            # 6. Scaling of numeric predictors
            scale_summary = data_frame[["AgeAtClinicalRecordCreation", "SequencingDepth"]].describe().loc[["mean", "std"]]
            print("Diagnostic - %s: numeric predictor scale\n%s", cell_type, scale_summary.to_string())
        
        formula = (
            f"Score ~ " +
            "indicator_of_sex + " +
            "AgeAtClinicalRecordCreation + " +
            "C(stage_at_start_of_ICB_therapy) + " +
            "C(patient_has_received_ICB_therapy) + " +
            "C(NexusBatch) + " +
            "SequencingDepth"
        ) # C means "make categorical".
        mixed_linear_model = smf.mixedlm(
            formula,
            data_frame,
            groups = data_frame["ORIENAvatarKey"]
        )
        mixed_linear_model_results_wrapper = mixed_linear_model.fit(reml = False)
        
        if models_should_be_diagnosed:
            # 2. Random-effect variance & boundary fit indicator
            try:
                rand_var = mixed_linear_model_results_wrapper.cov_re.iloc[0, 0]
            except Exception:
                rand_var = np.nan
            print("Diagnostic - %s: random-effect variance = %.6g", cell_type, rand_var)

            # 5. Design-matrix condition number (fixed effects only)
            cond_num = matrix_condition_number(mixed_linear_model.exog)
            print("Diagnostic - %s: design-matrix condition number = %.2e", cell_type, cond_num)

            # 7. AIC comparison with OLS (no random intercept)
            try:
                ols_res = smf.ols(formula, data_frame).fit()
                print(
                    "Diagnostic - %s: AIC (mixed) = %.1f ; AIC (OLS) = %.1f",
                    cell_type, mixed_linear_model_results_wrapper.aic, ols_res.aic
                )
            except Exception as exc:
                print("Diagnostic - %s: could not fit OLS for AIC comparison (%s)", cell_type, exc)
        
        parameter_for_Sex = mixed_linear_model_results_wrapper.params["indicator_of_sex"]
        standard_error_of_parameter_for_Sex = mixed_linear_model_results_wrapper.bse["indicator_of_sex"]
        p_value_for_Sex = mixed_linear_model_results_wrapper.pvalues["indicator_of_sex"]
        number_of_patients = data_frame["ORIENAvatarKey"].nunique()
        list_of_results.append(
            (
                cell_type,
                parameter_for_Sex,
                standard_error_of_parameter_for_Sex,
                p_value_for_Sex,
                number_of_patients
            )
        )
    data_frame_of_results = pd.DataFrame(
        list_of_results,
        columns = [
            "cell type",
            "parameter for Sex",
            "standard error of parameter for Sex",
            "p value for Sex",
            "number of patients"
        ]
    )
    return data_frame_of_results


def adjust_p_values_for_Sex(data_frame_of_results_of_fitting_LMMs: pd.DataFrame) -> pd.DataFrame:
    '''
    p values for the Sex coefficient are adjusted across all cell types
    with the Benjamini-Hochberg procedure and a False Discovery Rate of 10 percent.
    '''
    indicator_of_whether_to_reject_null_hypothesis, array_of_q_values, _, _ = multipletests(
        data_frame_of_results_of_fitting_LMMs["p value for Sex"],
        method = "fdr_bh",
        alpha = 0.10
    )
    data_frame_of_results_of_fitting_LMMs["q value for Sex"] = array_of_q_values
    data_frame_of_results_of_fitting_LMMs["significant"] = indicator_of_whether_to_reject_null_hypothesis
    return data_frame_of_results_of_fitting_LMMs


def main():
    paths.ensure_dependencies_for_fitting_LMMs_exist()

    parser = argparse.ArgumentParser(description = "Fit linear mixed models of xCell enrichment scores.")
    parser.add_argument("-d", "--diagnose", action = "store_true", help = "Run diagnostics for each cell type.")
    args = parser.parse_args()

    dictionary_of_paths_of_enrichment_data_frames_and_tuples_of_paths_of_results_of_fitting_LMMs = {
        paths.enrichment_data_frame_per_xCell: (
            paths.results_of_fitting_LMMs_per_xCell,
            paths.significant_results_of_fitting_LMMs_per_xCell
        ),
        paths.enrichment_data_frame_per_xCell2_and_Pan_Cancer: (
            paths.results_of_fitting_LMMs_per_xCell2_and_Pan_Cancer,
            paths.significant_results_of_fitting_LMMs_per_xCell2_and_Pan_Cancer
        )
    }

    for path_of_enrichment_data_frame, tuple_of_paths_of_results_of_fitting_LMMs in dictionary_of_paths_of_enrichment_data_frames_and_tuples_of_paths_of_results_of_fitting_LMMs.items():
        path_of_results_of_fitting_LMMs = tuple_of_paths_of_results_of_fitting_LMMs[0]
        path_of_significant_results_of_fitting_LMMs = tuple_of_paths_of_results_of_fitting_LMMs[1]
        data_frame_of_enrichment_scores_and_clinical_and_QC_data, list_of_cell_types = create_data_frame_of_enrichment_scores_and_clinical_and_QC_data(path_of_enrichment_data_frame)
        data_frame_of_results_of_fitting_LMMs = fit_linear_mixed_models(data_frame_of_enrichment_scores_and_clinical_and_QC_data, list_of_cell_types, args.diagnose)
        data_frame_of_results_of_fitting_LMMs = adjust_p_values_for_Sex(data_frame_of_results_of_fitting_LMMs)
        data_frame_of_results_of_fitting_LMMs.sort_values("q value for Sex").to_csv(path_of_results_of_fitting_LMMs, index = False)
        data_frame_of_results_of_fitting_LMMs.query("significant").sort_values("q value for Sex").to_csv(path_of_significant_results_of_fitting_LMMs, index = False)


if __name__ == "__main__":
    main()