from statsmodels.stats.multitest import multipletests
import numpy as np
from ORIEN_analysis.config import paths
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


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


def try_to_fit_different_models(formula: str, data_frame: pd.DataFrame):
    series_of_unique_patient_IDs_and_numbers_of_occurrences = data_frame["ORIENAvatarKey"].value_counts()
    series_of_unique_batch_IDs_and_numbers_of_occurrences = data_frame["NexusBatch"].value_counts()
    use_patient_re = (series_of_unique_patient_IDs_and_numbers_of_occurrences >= 2).sum() >= 2
    use_batch_vc = (series_of_unique_batch_IDs_and_numbers_of_occurrences >= 2).sum() >= 1 and data_frame["NexusBatch"].nunique() >= 2
    if use_patient_re and use_batch_vc:
        md = smf.mixedlm(
            formula,
            data_frame,
            groups = data_frame["ORIENAvatarKey"],
            re_formula = "1",
            vc_formula = {"batch": "0 + C(NexusBatch)"}
        )
        return md.fit(reml = False), "mixed_patient_batch"
    if use_patient_re:
        md = smf.mixedlm(
            formula,
            data_frame,
            groups = data_frame["ORIENAvatarKey"],
            re_formula = "1"
        )
        return md.fit(reml = False), "mixed_patient"
    if use_batch_vc:
        ols = smf.ols(formula, data_frame).fit(
            cov_type = "cluster",
            cov_kwds = {"groups": data_frame["NexusBatch"]}
        )
        return ols, "ols_cluster_batch"
    if data_frame["ORIENAvatarKey"].nunique() >= 2:
        ols = smf.ols(formula, data_frame).fit(
            cov_type = "cluster",
            cov_kwds = {"groups": data_frame["ORIENAvatarKey"]}
        )
        return ols, "ols_cluster_patient"
    ols = smf.ols(formula, data_frame).fit()
    return ols, "ols"


def get_statistics(regression_results_wrapper, variable):
    list_of_names_of_fixed_effects = regression_results_wrapper.model.exog_names
    series_of_parameters_of_fixed_effects = regression_results_wrapper.params.loc[list_of_names_of_fixed_effects]
    parameter = series_of_parameters_of_fixed_effects[variable]
    
    series_of_standard_errors_of_parameters = regression_results_wrapper.bse
    series_of_standard_errors_of_parameters_of_fixed_effects = series_of_standard_errors_of_parameters.loc[
        list_of_names_of_fixed_effects
    ]
    standard_error = series_of_standard_errors_of_parameters_of_fixed_effects[variable]

    series_of_p_values_of_parameters = regression_results_wrapper.pvalues
    series_of_p_values_of_parameters_of_fixed_effects = series_of_p_values_of_parameters.loc[list_of_names_of_fixed_effects]
    p_value = series_of_p_values_of_parameters_of_fixed_effects[variable]

    return parameter, standard_error, p_value


def fit_models(
    data_frame_of_enrichment_scores_clinical_data_and_QC_data: pd.DataFrame,
    list_of_cell_types: list[str]
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
        data_frame["patient_has_received_ICB_therapy"] = data_frame["patient_has_received_ICB_therapy"].astype(int)
        formula = (
            f"Score ~ " +
            "indicator_of_sex + " +
            "AgeAtClinicalRecordCreation + " +
            "C(stage_at_start_of_ICB_therapy, Treatment(reference='Unknown')) + " +
            "patient_has_received_ICB_therapy + " +
            "SequencingDepth"
        ) # C means "make categorical".
        regression_results_wrapper, model_kind = try_to_fit_different_models(formula, data_frame)
        parameter_for_Sex, standard_error_of_parameter_for_Sex, p_value_for_Sex = get_statistics(
            regression_results_wrapper,
            "indicator_of_sex"
        )
        number_of_patients = data_frame["ORIENAvatarKey"].nunique()

        '''
        We create a matrix of fixed effects rows corresponding to samples and columns corresponding to fixed effects.
        This matrix has columns corresponding to
        - intercept,
        - indicator of sex,
        - age at clinical record creation,
        - each non-reference level of stage at start of ICB therapy,
        - each non-reference level of indicator that patient has received ICB therapy (i.e., level True), and
        - sequencing depth.
        '''        
        matrix_of_fixed_effects = pd.DataFrame(
            regression_results_wrapper.model.exog,
            columns = regression_results_wrapper.model.exog_names
        )
        series_of_values_of_parameters = getattr(
            regression_results_wrapper,
            "fe_params",
            regression_results_wrapper.params
        )
        matrix_of_fixed_effects_with_indicator_of_sex_0 = matrix_of_fixed_effects.copy()
        matrix_of_fixed_effects_with_indicator_of_sex_0["indicator_of_sex"] = 0
        series_of_predicted_enrichment_scores_for_females = (
            matrix_of_fixed_effects_with_indicator_of_sex_0 @ series_of_values_of_parameters
        )
        expected_enrichment_score_given_patient_is_female_and_other_objects_have_certain_values = (
            series_of_predicted_enrichment_scores_for_females.mean()
        )
        matrix_of_fixed_effects_with_indicator_of_sex_1 = matrix_of_fixed_effects.copy()
        matrix_of_fixed_effects_with_indicator_of_sex_1["indicator_of_sex"] = 1
        series_of_predicted_enrichment_scores_for_males = (
            matrix_of_fixed_effects_with_indicator_of_sex_1 @ series_of_values_of_parameters
        )
        expected_enrichment_score_given_patient_is_male_and_other_objects_have_certain_values = (
            series_of_predicted_enrichment_scores_for_males.mean()
        )
        difference_between_expected_values = (
            expected_enrichment_score_given_patient_is_male_and_other_objects_have_certain_values -
            expected_enrichment_score_given_patient_is_female_and_other_objects_have_certain_values
        )
        percent_difference_between_expected_values = (
            difference_between_expected_values /
            expected_enrichment_score_given_patient_is_female_and_other_objects_have_certain_values *
            100.0
        )
        log_fold_change = np.log2(
            expected_enrichment_score_given_patient_is_male_and_other_objects_have_certain_values /
            expected_enrichment_score_given_patient_is_female_and_other_objects_have_certain_values
        )
        standard_deviation_of_enrichment_score = data_frame["Score"].std()
        parameter_for_Sex_standardized_by_standard_deviation_of_enrichment_score = (
            parameter_for_Sex / standard_deviation_of_enrichment_score
        )
        confidence_level = 0.95
        critical_value = stats.norm.ppf((confidence_level + 1) / 2)
        lower_bound_of_confidence_interval_for_parameter_for_Sex = (
            parameter_for_Sex - critical_value * standard_error_of_parameter_for_Sex
        )
        upper_bound_of_confidence_interval_for_parameter_for_Sex = (
            parameter_for_Sex + critical_value * standard_error_of_parameter_for_Sex
        )
        variance_of_residuals = getattr(
            regression_results_wrapper,
            "scale",
            np.nan
        )
        standard_deviation_of_residuals = np.sqrt(variance_of_residuals)
        parameter_for_Sex_standardized_by_standard_deviation_of_residuals = (
            parameter_for_Sex / standard_deviation_of_residuals
        )
        list_of_results.append(
            (
                cell_type,
                model_kind,
                parameter_for_Sex,
                standard_error_of_parameter_for_Sex,
                p_value_for_Sex,
                number_of_patients,
                expected_enrichment_score_given_patient_is_female_and_other_objects_have_certain_values,
                expected_enrichment_score_given_patient_is_male_and_other_objects_have_certain_values,
                difference_between_expected_values,
                percent_difference_between_expected_values,
                log_fold_change,
                parameter_for_Sex_standardized_by_standard_deviation_of_enrichment_score,
                lower_bound_of_confidence_interval_for_parameter_for_Sex,
                upper_bound_of_confidence_interval_for_parameter_for_Sex,
                standard_deviation_of_residuals,
                parameter_for_Sex_standardized_by_standard_deviation_of_residuals
            )
        )
    data_frame_of_results = pd.DataFrame(
        list_of_results,
        columns = [
            "cell type",
            "type of model",
            "parameter for Sex",
            "standard error of parameter for Sex",
            "p value for Sex",
            "number of patients",
            "expected enrichment score given patient is female and other objects have certain values",
            "expected enrichment score given patient is male and other objects have certain values",
            "difference between expected values",
            "percent difference between expected values",
            "log fold change",
            "parameter for Sex standardized by standard deviation of enrichment score",
            "lower bound of confidence interval for parameter for Sex",
            "upper bound of confidence interval for parameter for Sex",
            "standard deviation of residuals",
            "parameter for Sex standardized by standard deviation of residuals"
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
        data_frame_of_results_of_fitting_LMMs = fit_models(data_frame_of_enrichment_scores_and_clinical_and_QC_data, list_of_cell_types)
        data_frame_of_results_of_fitting_LMMs = adjust_p_values_for_Sex(data_frame_of_results_of_fitting_LMMs)
        data_frame_of_results_of_fitting_LMMs.sort_values("q value for Sex").to_csv(path_of_results_of_fitting_LMMs, index = False)
        data_frame_of_results_of_fitting_LMMs.query("significant").sort_values("q value for Sex").to_csv(path_of_significant_results_of_fitting_LMMs, index = False)


if __name__ == "__main__":
    main()