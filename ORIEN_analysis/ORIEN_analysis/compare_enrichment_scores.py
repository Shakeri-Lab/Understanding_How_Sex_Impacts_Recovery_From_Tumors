#!/usr/bin/env python3
'''
This script compares enrichment scores produced by xCell or xCell2 for
- tumors of females (0) and males (1),
- ICB-naive tumors (0) and ICB-experienced (1) tumors of females, and
- ICB-naive tumors and ICB-experienced tumors of males
across various cell types.
This script for each cell type and category of tumors
performs a 2 sided Mann-Whitney U Test / Wilcoxon Ranak Sum Test.

This script optionally removes variability due to age at clinical record creation and stage at start of ICB therapy before comparing.

Enrichment scores from group A (e.g., samples of females) and group B (e.g., samples of males)
may be combined in a list.
The list may be sorted from smallest to largest.
A rank is the position of an enrichment score in the sorted list.
If multiple enrichment scores are equal, their ranks are averaged and
assigned to each equal enrichment score.
When an enrichment score precedes another enrichment score,
the former occurs earlier in the sorted list.

A Mann-Whitney U Test / Wilcoxon Rank Sum Test tests a null hypothesis that every enrichment score in either group A or B is drawn from the same population distribution of enrichment scores.
The U test statistic of a Mann-Whitney U Test / Wilcoxon Rank Sum Test is
the number of pairs of enrichment scores in which
the enrichment score for group A precedes the enrichment score for group B.
For a given U test statistic, the p value associated with that U statistic is
the probability, when assuming the null hypothesis is true,
of observing a U statistic at least as extreme as the calculated U statistic.
Smaller p values provide stronger evidence against the null hypothesis.

This script adjusts p values into False Discovery Rates (FDRs) with the Benjamini-Hochberg procedure.
p values p_i and any associated information are sorted in ascending order.
Each p value p_i is multiplied by the number of p values, divided by i, and clipped to 1.
The resulting series is corrected so that each value is at least as large as the preceding value.
The resulting series contains FDRs and may be added as a column next to the p values.

A cell type is significant if the FDR associated with that cell type is less than or equal to 0.05.
A cell type being significant indicates that enrichment scores for group A and that cell type
were not drawn from the same population distribution as
enrichment scores for group B and that cell type.
A cell type is suggestive if the FDR associated with that cell type is greater than 0.05 and
less than or equal to 0.20.
A cell type being suggestive suggests that enrichment scores for group A and that cell type
were not drawn from the same population distribution as
enrichment scores for group B and that cell type.


Usage
-----
conda activate ici_sex
./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.compare_enrichment_scores --adjust-covariates

Pass flag `--adjust-covariates` to remove variability due to age at clinical record creation and stage at start of ICB therapy before comparing.
Otherwise raw enrichment scores are used.

Outputs
-------
icb_wilcoxon_female.csv
icb_wilcoxon_male.csv
sex_wilcoxon.csv

Each file contains one row per cell type with columns
    - n_F and n_M or n_Naive and n_Exp containing
      numbers of samples corresponding to female and male patients or
      numbers of ICB-naive and ICB-experienced samples,
    - U-stat containing U statistics,
    - p value containing p values,
    - FDR containing False Discovery Rates,
    - significant containing indicators of significance, and
    - suggestive containing indicators of suggestiveness.
'''

from pathlib import Path
import argparse
from ORIEN_analysis.fit_linear_models import (
    create_data_frame_of_enrichment_scores_and_clinical_and_QC_data
)
import logging
from statsmodels.stats.multitest import multipletests
import numpy as np
from ORIEN_analysis.config import paths
import pandas as pd
import matplotlib.pyplot as plt
import re
import statsmodels.formula.api as smf
import scipy.stats as ss


def determine_FDRs(series_of_p_values: pd.Series) -> pd.Series:
    '''
    Provide a series of False Discovery Rates based on a series of p values.
    '''
    _, q, _, _ = multipletests(series_of_p_values, method = "fdr_bh")
    return pd.Series(q, index = series_of_p_values.index)


def determine_direction(value_of_cliffs_delta: float, label_0: str, label_1: str) -> str:
    if value_of_cliffs_delta > 0:
        return f"P(ES of {label_1} > ES of {label_0}) > P(ES of {label_1} < ES of {label_0})"
    if value_of_cliffs_delta < 0:
        return f"P(ES of {label_1} < ES of {label_0}) > P(ES of {label_1} > ES of {label_0})"
    return f"P(ES of {label_1} > ES of {label_0}) = P(ES of {label_1} < ES of {label_0})"


def save_data_frame_of_cell_types_and_statistics(
    data_frame_of_cell_types_and_statistics: pd.DataFrame, category: str, path: Path
) -> None:
    '''
    Save a data frame of cell types and statistics with added columns FDR, significant, and suggestive.
    Log numbers of cell types that are significant / suggestive for each category of tests.
    '''
    data_frame_of_cell_types_and_statistics["FDR"] = determine_FDRs(data_frame_of_cell_types_and_statistics["p_value"])
    data_frame_of_cell_types_and_statistics["significant"] = data_frame_of_cell_types_and_statistics["FDR"] <= 0.05
    data_frame_of_cell_types_and_statistics["suggestive"] = (
        (data_frame_of_cell_types_and_statistics["FDR"] > 0.05) & (data_frame_of_cell_types_and_statistics["FDR"] <= 0.20)
    )
    data_frame_of_cell_types_and_statistics.sort_values(["FDR", "p_value"]).to_csv(path, index = False)


def create_series_of_enrichment_scores_or_residuals(
    data_frame_of_enrichment_scores_and_clinical_and_QC_data: pd.DataFrame,
    cell_type: str,
    category: str,
    covariates_will_be_adjusted: bool
) -> tuple[pd.Series, pd.Series]:
    data_frame_of_enrichment_scores_and_indicators = data_frame_of_enrichment_scores_and_clinical_and_QC_data[
        [cell_type, category]
    ].copy()
    if covariates_will_be_adjusted:
        data_frame_of_enrichment_scores_ages_and_stages = data_frame_of_enrichment_scores_and_clinical_and_QC_data[
            [cell_type, "Age_At_Specimen_Collection", "EKN_Assigned_Stage"]
        ]
        OLS_linear_regression_model = smf.ols(
            f"{cell_type} ~ Age_At_Specimen_Collection + C(EKN_Assigned_Stage, Treatment(reference='II'))",
            data = data_frame_of_enrichment_scores_ages_and_stages
        )
        regression_results_wrapper = OLS_linear_regression_model.fit()
        series_of_residuals = regression_results_wrapper.resid
        data_frame_of_enrichment_scores_and_indicators["residual"] = series_of_residuals.loc[
            data_frame_of_enrichment_scores_and_indicators.index
        ]
        name_of_column_of_enrichment_scores_or_residuals = "residual"
    else:
        name_of_column_of_enrichment_scores_or_residuals = cell_type
    series_of_values_for_indicator_0 = data_frame_of_enrichment_scores_and_indicators.loc[
        data_frame_of_enrichment_scores_and_indicators[category] == 0,
        name_of_column_of_enrichment_scores_or_residuals
    ]
    series_of_values_for_indicator_1 = data_frame_of_enrichment_scores_and_indicators.loc[
        data_frame_of_enrichment_scores_and_indicators[category] == 1,
        name_of_column_of_enrichment_scores_or_residuals
    ]
    return series_of_values_for_indicator_0, series_of_values_for_indicator_1


from cliffs_delta import cliffs_delta


def create_data_frame_of_cell_types_and_statistics(
    data_frame_of_enrichment_scores_and_clinical_and_QC_data: pd.DataFrame,
    list_cell_types: list[str],
    category: str,
    covariates_will_be_adjusted: bool
) -> pd.DataFrame:
    '''
    Perform Mann-Whitney U Tests / Wilcoxon Rank Sum Tests for every cell type.
    If covariates will be adjusted, residuals of a linear model of enrichment scores
    vs. age at clinical record creation and stage at start of ICB therapy will be compared
    instead of raw enrichment scores.
    
    Returns
    -------
    a data frame with 
    - cell types,
    - numbers of samples corresponding to female and male patients or
    - numbers of ICB-naive and ICB-experienced samples,
    - U statistics,
    - p values,
    - differences between medians,
    - values of Cliff's delta, and
    - directions.

    For each cell type,
    Choose either the raw enrichment scores for that cell type or the residuals of a linear model of
    enrichment score vs. age at clinical record creation and stage at start of ICB therapy.
    In the first case, split enrichment scores by sex or experience of ICB therapy.
    In the second case, split residuals by sex or experience of ICB therapy.
    Find a U statistic and a p value using the Mann-Whitney U Test / Wilcoxon Rank Sum Test.
    Record in a row corresponding to a cell type of a data frame
    - numbers of samples corresponding to female and male patients or
    - numbers of ICB-naive and ICB-experienced samples,
    - U statistic,
    - p value,
    - difference between medians,
    - value of Cliff's delta, and
    - direction.
    '''
    list_of_dictionaries_of_cell_types_and_statistics = []
    for cell_type in list_cell_types:
        series_of_values_for_indicator_0, series_of_values_for_indicator_1 = create_series_of_enrichment_scores_or_residuals(
            data_frame_of_enrichment_scores_and_clinical_and_QC_data = data_frame_of_enrichment_scores_and_clinical_and_QC_data,
            cell_type = cell_type,
            category = category,
            covariates_will_be_adjusted = covariates_will_be_adjusted
        )
        U_statistic, p_value = ss.mannwhitneyu(
            series_of_values_for_indicator_0,
            series_of_values_for_indicator_1,
            alternative = "two-sided"
        )
        dictionary_of_cell_types_and_statistics = dict(
            cell_type = cell_type,
            number_of_samples_for_indicator_0 = len(series_of_values_for_indicator_0),
            number_of_samples_for_indicator_1 = len(series_of_values_for_indicator_1),
            U_statistic = U_statistic,
            p_value = p_value
        )
        difference_between_medians = (
            np.median(series_of_values_for_indicator_1) - np.median(series_of_values_for_indicator_0)
        )
        value_of_cliffs_delta = cliffs_delta(series_of_values_for_indicator_1, series_of_values_for_indicator_0)[0]
        dictionary_of_cell_types_and_statistics["difference_between_medians"] = difference_between_medians
        dictionary_of_cell_types_and_statistics["cliffs_delta"] = value_of_cliffs_delta
        dictionary_of_categories_and_lists_of_labels = {
            "indicator_of_sex": ["Females", "Males"],
            "integer_indicating_that_patient_received_ICB_therapy_at_or_before_age_of_specimen_collection": ["ICB naive", "ICB experienced"]
        }
        dictionary_of_cell_types_and_statistics["direction"] = determine_direction(
            value_of_cliffs_delta,
            dictionary_of_categories_and_lists_of_labels[category][0],
            dictionary_of_categories_and_lists_of_labels[category][1]
        )
        list_of_dictionaries_of_cell_types_and_statistics.append(dictionary_of_cell_types_and_statistics)
    data_frame_of_cell_types_and_statistics = pd.DataFrame(list_of_dictionaries_of_cell_types_and_statistics)
    return data_frame_of_cell_types_and_statistics


def create_plot(
    series_of_values_for_indicator_0: pd.Series,
    series_of_values_for_indicator_1: pd.Series,
    cell_type: str,
    title: str,
    subtitle: str,
    path_to_which_to_save_plot: Path,
    response: str,
    list_of_labels: tuple[str, str]
) -> None:
    list_of_arrays = [series_of_values_for_indicator_0.to_numpy(), series_of_values_for_indicator_1.to_numpy()]
    figure = plt.figure()
    ax_1 = figure.add_subplot(1, 2, 1)
    ax_1.violinplot(list_of_arrays, showmeans = True, showmedians = True)
    ax_1.set_xticks([1, 2], list_of_labels)
    ax_1.set_ylabel(response)
    ax_2 = figure.add_subplot(1, 2, 2)
    ax_2.boxplot(list_of_arrays, tick_labels = list_of_labels, showmeans = True, notch = True)
    figure.suptitle(title)
    figure.text(0.5, 0.7, subtitle, ha = "center")
    figure.tight_layout(rect = [0, 0.04, 1, 0.93])
    path_to_which_to_save_plot.parent.mkdir(parents = True, exist_ok = True)
    figure.savefig(path_to_which_to_save_plot)
    plt.close(figure)


def create_plots_for_significant_cell_types_within_sex(
    data_frame_of_enrichment_scores_and_clinical_and_QC_data: pd.DataFrame,
    list_of_cell_types: list[str],
    indicator_of_sex: int,
    path_to_comparisons_for_ICB_naive_and_experienced_samples: Path,
    covariates_will_be_adjusted: bool
) -> None:
    data_frame_of_comparisons_for_ICB_naive_and_experienced_samples = pd.read_csv(
        path_to_comparisons_for_ICB_naive_and_experienced_samples
    )
    data_frame_of_significant_comparisons = data_frame_of_comparisons_for_ICB_naive_and_experienced_samples.loc[
        data_frame_of_comparisons_for_ICB_naive_and_experienced_samples["FDR"] <= 0.05
    ]
    data_frame_of_enrichment_scores_and_clinical_and_QC_data_for_sex = data_frame_of_enrichment_scores_and_clinical_and_QC_data.loc[
        data_frame_of_enrichment_scores_and_clinical_and_QC_data["indicator_of_sex"] == indicator_of_sex
    ]
    for _, row_of_significant_comparisons in data_frame_of_significant_comparisons.iterrows():
        cell_type = row_of_significant_comparisons["cell_type"]
        series_of_values_for_indicator_0, series_of_values_for_indicator_1 = create_series_of_enrichment_scores_or_residuals(
            data_frame_of_enrichment_scores_and_clinical_and_QC_data = data_frame_of_enrichment_scores_and_clinical_and_QC_data_for_sex,
            cell_type = cell_type,
            category = "integer_indicating_that_patient_received_ICB_therapy_at_or_before_age_of_specimen_collection",
            covariates_will_be_adjusted = covariates_will_be_adjusted
        )
        response = "Residuals" if covariates_will_be_adjusted else "Enrichment Scores"
        sex = "Females" if indicator_of_sex == 0 else "Males"
        title = f"Violin and Box Plots of {response} of {cell_type}\nof ICB Naive and Experienced Samples of {sex}"
        subtitle = (
            f"Numbers of naive and experienced samples are {len(series_of_values_for_indicator_0)} and {len(series_of_values_for_indicator_1)}.\n" +
            f"U statistic is {row_of_significant_comparisons.get('U_statistic')}.\n" +
            f"p value is {row_of_significant_comparisons.get('p_value')}.\n" +
            f"FDR is {row_of_significant_comparisons.get('FDR')}.\n" +
            f"Difference between medians is {row_of_significant_comparisons.get('difference_between_medians')}.\n" +
            f"Cliff's delta is {row_of_significant_comparisons.get('cliffs_delta')}."
        )
        response = "residuals" if covariates_will_be_adjusted else "enrichment_scores"
        sex = "females" if indicator_of_sex == 0 else "males"
        filename = (
            paths.plots_for_comparing_enrichment_scores /
            ("covariates_were_adjusted" if covariates_will_be_adjusted else "covariates_were_not_adjusted") /
            sex /
            f"violin_and_box_plots_of_{response}_of_{cell_type}_of_ICB_naive_and_experienced_samples_of_{sex}.png"
        )
        create_plot(
            series_of_values_for_indicator_0 = series_of_values_for_indicator_0,
            series_of_values_for_indicator_1 = series_of_values_for_indicator_1,
            cell_type = cell_type,
            title = title,
            subtitle = subtitle,
            path_to_which_to_save_plot = filename,
            response = response,
            list_of_labels = ("ICB naive", "ICB experienced")
        )


def create_plots_for_significant_cell_types_by_sex_within_subset(
    data_frame: pd.DataFrame,
    path_to_comparisons: Path,
    covariates_will_be_adjusted: bool,
    subset: str
) -> None:
    data_frame_of_comparisons_for_female_and_male_samples = pd.read_csv(path_to_comparisons)
    data_frame_of_significant_comparisons_for_female_and_male_samples = (
        data_frame_of_comparisons_for_female_and_male_samples.loc[
            data_frame_of_comparisons_for_female_and_male_samples["FDR"] <= 0.05
        ]
    )
    for _, row_of_significant_comparisons in data_frame_of_significant_comparisons_for_female_and_male_samples.iterrows():
        cell_type = row_of_significant_comparisons["cell_type"]
        series_of_values_for_indicator_0, series_of_values_for_indicator_1 = create_series_of_enrichment_scores_or_residuals(
            data_frame_of_enrichment_scores_and_clinical_and_QC_data = data_frame,
            cell_type = cell_type,
            category = "indicator_of_sex",
            covariates_will_be_adjusted = covariates_will_be_adjusted
        )
        response = "Residuals" if covariates_will_be_adjusted else "Enrichment Scores"
        description_of_samples = "Naive" if subset == "naive_samples" else "Experienced"
        title = f"Violin and Box Plots of {response} of {cell_type}\nof Female and Male ICB {description_of_samples} Samples"
        subtitle = (
            f"Numbers of female and male samples are {len(series_of_values_for_indicator_0)} and {len(series_of_values_for_indicator_1)}.\n" +
            f"U statistic is {row_of_significant_comparisons.get('U_statistic')}.\n" +
            f"p value is {row_of_significant_comparisons.get('p_value')}.\n" +
            f"FDR is {row_of_significant_comparisons.get('FDR')}.\n" +
            f"Difference between medians is {row_of_significant_comparisons.get('difference_between_medians')}.\n" +
            f"Cliff's delta is {row_of_significant_comparisons.get('cliffs_delta')}."
        )
        response = "residuals" if covariates_will_be_adjusted else "enrichment_scores"
        description_of_samples = "naive" if subset == "naive_samples" else "Experienced"
        filename = (
            paths.plots_for_comparing_enrichment_scores /
            ("covariates_were_adjusted" if covariates_will_be_adjusted else "covariates_were_not_adjusted") /
            subset /
            f"violin_and_box_plots_of_{response}_of_{cell_type}_of_female_and_male_ICB_{description_of_samples}_samples.png"
        )
        create_plot(
            series_of_values_for_indicator_0 = series_of_values_for_indicator_0,
            series_of_values_for_indicator_1 = series_of_values_for_indicator_1,
            cell_type = cell_type,
            title = title,
            subtitle = subtitle,
            path_to_which_to_save_plot = filename,
            response = response,
            list_of_labels = ("Females", "Males")
        )


def main():
    paths.ensure_dependencies_for_comparing_enrichment_scores_exist()
    
    parser = argparse.ArgumentParser(description = "Compare enrichment scores by category and cell type.")
    parser.add_argument("--adjust_covariates", action = "store_true", help = "Regress out Age and Stage before rank tests.")
    args = parser.parse_args()

    dictionary_of_paths = {
        paths.enrichment_data_frame_per_xCell: (
            paths.comparisons_for_females_and_males_and_xCell,
            paths.comparisons_for_female_and_male_naive_samples_and_xCell,
            paths.comparisons_for_female_and_male_experienced_samples_and_xCell,
            paths.comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell,
            paths.comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell
        ),
        paths.enrichment_data_frame_per_xCell2_and_Pan_Cancer: (
            paths.comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer,
            paths.comparisons_for_female_and_male_naive_samples_and_xCell2_and_Pan_Cancer,
            paths.comparisons_for_female_and_male_experienced_samples_and_xCell2_and_Pan_Cancer,
            paths.comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer,
            paths.comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_Pan_Cancer
        )
    }
    
    for path_to_enrichment_data, tuple_of_paths_to_comparisons in dictionary_of_paths.items():
        path_to_comparisons_for_females_and_males = tuple_of_paths_to_comparisons[0]
        path_to_comparisons_for_female_and_male_naive_samples = tuple_of_paths_to_comparisons[1]
        path_to_comparisons_for_female_and_male_experienced_samples = tuple_of_paths_to_comparisons[2]
        path_to_comparisons_for_ICB_naive_and_experienced_samples_of_females = tuple_of_paths_to_comparisons[3]
        path_to_comparisons_for_ICB_naive_and_experienced_samples_of_males = tuple_of_paths_to_comparisons[4]
    
        data_frame, list_of_cell_types = create_data_frame_of_enrichment_scores_and_clinical_and_QC_data(
            path_to_enrichment_data
        )
        data_frame["indicator_of_sex"] = (data_frame["Sex"] == "Male").astype(int)
        data_frame["integer_indicating_that_patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"] = (
            data_frame["patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"].astype(int)
        )
        data_frame = data_frame.rename(
            columns = lambda cell_type: cell_type.replace(' ', '_').replace('-', '_').replace('+', "plus").replace(',', '')
        )
        list_of_cell_types = [
            cell_type.replace(' ', '_').replace('-', '_').replace('+', "plus").replace(',', '')
            for cell_type in list_of_cell_types
        ]

        data_frame_of_cell_types_and_statistics = create_data_frame_of_cell_types_and_statistics(
            data_frame,
            list_of_cell_types,
            category = "indicator_of_sex",
            covariates_will_be_adjusted = args.adjust_covariates
        ).rename(
            columns = {
                "number_of_samples_for_indicator_0": "number_of_samples_for_females",
                "number_of_samples_for_indicator_1": "number_of_samples_for_males"
            }
        )
        save_data_frame_of_cell_types_and_statistics(
            data_frame_of_cell_types_and_statistics,
            "female / male",
            path_to_comparisons_for_females_and_males
        )

        dictionary_of_indicators_of_sex_and_paths_to_comparisons_for_ICB_naive_and_experienced_samples = {
            0: path_to_comparisons_for_ICB_naive_and_experienced_samples_of_females,
            1: path_to_comparisons_for_ICB_naive_and_experienced_samples_of_males,
        }
        for indicator_of_sex, path in dictionary_of_indicators_of_sex_and_paths_to_comparisons_for_ICB_naive_and_experienced_samples.items():
            data_frame_for_indicator = data_frame[data_frame["indicator_of_sex"] == indicator_of_sex]
            data_frame_of_cell_types_and_statistics = create_data_frame_of_cell_types_and_statistics(
                data_frame_for_indicator,
                list_of_cell_types,
                category = "integer_indicating_that_patient_received_ICB_therapy_at_or_before_age_of_specimen_collection",
                covariates_will_be_adjusted = args.adjust_covariates
            ).rename(
                columns = {
                    "number_of_samples_for_indicator_0": "number_of_naive_samples",
                    "number_of_samples_for_indicator_1": "number_of_experienced_samples"
                }
            )
            group = "females" if indicator_of_sex == 0 else "males"
            save_data_frame_of_cell_types_and_statistics(
                data_frame_of_cell_types_and_statistics,
                f"naive / experienced for {group}",
                path
            )
            create_plots_for_significant_cell_types_within_sex(
                data_frame_of_enrichment_scores_and_clinical_and_QC_data = data_frame,
                list_of_cell_types = list_of_cell_types,
                indicator_of_sex = indicator_of_sex,
                path_to_comparisons_for_ICB_naive_and_experienced_samples = path,
                covariates_will_be_adjusted = args.adjust_covariates
            )

        data_frame_for_naive_samples = data_frame[
            data_frame["integer_indicating_that_patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"] == 0
        ]
        data_frame_of_cell_types_and_statistics_for_naive_samples = create_data_frame_of_cell_types_and_statistics(
            data_frame_for_naive_samples,
            list_of_cell_types,
            category = "indicator_of_sex",
            covariates_will_be_adjusted = args.adjust_covariates
        ).rename(
            columns = {
                "number_of_samples_for_indicator_0": "number_of_samples_for_female_naive_samples",
                "number_of_samples_for_indicator_1": "number_of_samples_for_male_naive_samples"
            }
        )
        save_data_frame_of_cell_types_and_statistics(
            data_frame_of_cell_types_and_statistics_for_naive_samples,
            "female / male",
            path_to_comparisons_for_female_and_male_naive_samples
        )
        create_plots_for_significant_cell_types_by_sex_within_subset(
            data_frame = data_frame_for_naive_samples,
            path_to_comparisons = path_to_comparisons_for_female_and_male_naive_samples,
            covariates_will_be_adjusted = args.adjust_covariates,
            subset = "naive_samples"
        )
        data_frame_for_experienced_samples = data_frame[
            data_frame["integer_indicating_that_patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"] == 1
        ]
        data_frame_of_cell_types_and_statistics_for_experienced_samples = create_data_frame_of_cell_types_and_statistics(
            data_frame_for_experienced_samples,
            list_of_cell_types,
            category = "indicator_of_sex",
            covariates_will_be_adjusted = args.adjust_covariates
        ).rename(
            columns = {
                "number_of_samples_for_indicator_0": "number_of_samples_for_female_experienced_samples",
                "number_of_samples_for_indicator_1": "number_of_samples_for_male_experienced_samples"
            }
        )
        save_data_frame_of_cell_types_and_statistics(
            data_frame_of_cell_types_and_statistics_for_experienced_samples,
            "female / male",
            path_to_comparisons_for_female_and_male_experienced_samples
        )
        create_plots_for_significant_cell_types_by_sex_within_subset(
            data_frame = data_frame_for_experienced_samples,
            path_to_comparisons = path_to_comparisons_for_female_and_male_experienced_samples,
            covariates_will_be_adjusted = args.adjust_covariates,
            subset = "experienced_samples"
        )


if __name__ == "__main__":
    main()