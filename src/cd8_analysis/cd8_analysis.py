'''
cd8_analysis.py

CD8 analysis produces a data frame of CD8 signatures and statistics including
mean signature scores by sex, numbers of patients with each sex, and
test statistics and p values relating to comparing these means using Welch's t test.

CD8 analysis produces a data frame of patient IDs and CD8 signature scores.

CD8 analysis produces a data frame of sexes, CD8 signatures, and statistics.

CD8 analysis produces a data frame of diagnoses with at least 20 patients, CD8 signatures, and statistics.

CD8 analysis plots for each CD8 signature mean CD8 signature score vs. sex.

CD8 analysis plots for each CD8 signature mean CD8 signature score vs. diagnosis for diagnoses with at least 20 patients.

CD8 analysis plots for each CD8 signature 2 Kaplan-Meier survival curves
of survival probabilities of patients in each group for that CD8 signature vs. time in months.
A survival probability is the probability that a patient will be alive after a certain number of months.

CD8 analysis produces results of a log rank test for each of many CD8 signatures.
The log rank test is a hypothesis test used in survival analysis
to compare the distributions of times to events for 2 independent groups.
A time to event is the time between a patient's death and the age at diagnosis of the patient.
For a given CD8 signature, Group 1 is a group of patients (e.g., 150)
with scores for that CD8 signature above the median score for all patients (e.g., 300)
involved in plotting survival curves and performing log rank tests.
Group 2 is a group of patients (e.g., 150) with scores for that CD8 below the median score for all patients.
The log rank test may be used to evaluate whether
there is a significant difference in times to events between Group 1 and Group 2.
The null hypothesis of a log rank test is that Group 1 and Group 2 have the same distributions.
The alternative hypothesis is that Group 1 and Group 2 have different distributions.
If the p value of the log rank test is greater than or equal to a significance level (e.g., 0.05),
we fail to reject the null hypothesis and note that
there is not sufficient evidence to conclude that Group 1 and Group 2 have different distributions.
If the p value of the log rank test is less than the significance level, we reject the null hypothesis and note that
there is sufficient evidence to conclude that Group 1 and Group 2 have different distributions.

Usage
./miniconda3/envs/ici_sex/bin/python -m src.cd8_analysis.cd8_analysis
'''

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import textwrap
import traceback
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from src.immune_analysis.immune_analysis import ImmuneAnalysis
from src.utils.shared_functions import calculate_survival_months, filter_by_primary_diagnosis_site, load_expression_matrix, map_sample_IDs_to_patient_IDs
from src.config import paths


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s – %(levelname)s – %(message)s"
)
logger = logging.getLogger(__name__)


class CD8Analysis(ImmuneAnalysis):
    '''
    Class CD8Analysis is a template for an object for analyzing CD8+ T cell signatures.
    '''
    
    def __init__(self):
        super().__init__()
        
        self.dictionary_of_CD8_signatures_and_lists_of_IDs_of_genes = {
            "CD8_A": [
                "ENSG00000153563", # HGNC gene symbol is CD8A.
            ],
            "CD8_B": [
                "ENSG00000172116", # CD8B
            ],
            "CD8_cytotoxic": [
                "ENSG00000145649", # GZMA
                "ENSG00000100453", # GZMB
                "ENSG00000100450", # GZMH
                "ENSG00000113088", # GZMK
                "ENSG00000180644", # PRF1
                "ENSG00000105374", # NKG7
                "ENSG00000115523" # GNLY
            ],
            "CD8_activation": [
                "ENSG00000110848", # CD69
                "ENSG00000204287", # HLA-DRA
                "ENSG00000196126", # HLA-DRB1
                "ENSG00000004468", # CD38
                "ENSG00000163600", # ICOS
                "ENSG00000163599", # CTLA4
                "ENSG00000089692", # LAG3
                "ENSG00000188389", # PDCD1
            ],
            "CD8_exhaustion": [
                "ENSG00000188389", # PDCD1
                "ENSG00000163599", # CTLA4
                "ENSG00000089692", # LAG3
                "ENSG00000135077", # HAVCR2
                "ENSG00000181847", # TIGIT
                "ENSG00000186265", # BTLA
                "ENSG00000107738", # VSIR
            ],
            "CD8_memory": [
                "ENSG00000168685", # IL7R
                "ENSG00000126353", # CCR7
                "ENSG00000188404", # SELL
                "ENSG00000139193", # CD27
                "ENSG00000178562", # CD28
                "ENSG00000049249", # TNFRSF9
                "ENSG00000163508" # EOMES
            ],
            "CD8_naive": [
                "ENSG00000126353", # CCR7
                "ENSG00000188404", # SELL
                "ENSG00000138795", # LEF1
                "ENSG00000081059", # TCF7
                "ENSG00000168685", # IL7R
            ],
            "CD8_effector": [
                "ENSG00000168329", # CX3CR1
                "ENSG00000137441", # FGFBP2
                "ENSG00000203747", # FCGR3A
                "ENSG00000139187", # KLRG1
                "ENSG00000100453", # GZMB
                "ENSG00000180644", # PRF1
            ]
        }
    
    
    def create_data_frame_of_patient_IDs_CD8_signatures_and_scores(self, expression_matrix: pd.DataFrame) -> pd.DataFrame:

        logger.info("Data frame of patient IDs, CD8 signatures, and scores will be created.")

        data_frame_of_genes_and_samples = expression_matrix.copy()
        # We assume that each element of the index is an Ensembl gene ID.
        data_frame_of_genes_and_samples.index = data_frame_of_genes_and_samples.index.astype(str).str.split(".").str[0]
        data_frame_of_genes_and_samples = data_frame_of_genes_and_samples.groupby(data_frame_of_genes_and_samples.index).mean()
        data_frame_of_sample_IDs_CD8_signatures_and_scores = pd.DataFrame(index = data_frame_of_genes_and_samples.columns)
        for CD8_signature, list_of_genes_of_CD8_signature in self.dictionary_of_CD8_signatures_and_lists_of_IDs_of_genes.items():
            list_of_genes_of_CD8_signature_in_expression_matrix = [
                gene
                for gene in list_of_genes_of_CD8_signature
                if gene in data_frame_of_genes_and_samples.index
            ]
            
            logger.info(f"Scores for column {CD8_signature} will be determined.")

            data_frame_of_sample_IDs_CD8_signatures_and_scores[CD8_signature] = data_frame_of_genes_and_samples.loc[
                list_of_genes_of_CD8_signature_in_expression_matrix
            ].mean()
        data_frame_of_patient_IDs_CD8_signatures_and_scores = map_sample_IDs_to_patient_IDs(
            data_frame_of_sample_IDs_CD8_signatures_and_scores
        )
        data_frame_of_patient_IDs_CD8_signatures_and_scores.to_csv(paths.data_frame_of_patient_IDs_and_CD8_signature_scores)

        logger.info(f"Data frame of patient IDs and CD8 signature scores has shape {data_frame_of_patient_IDs_CD8_signatures_and_scores.shape}.")

        return data_frame_of_patient_IDs_CD8_signatures_and_scores
    
    
    def analyze_signatures_by_sex(self, data_frame_of_patient_IDs_CD8_signatures_and_scores, clinical_data):
        
        logger.info("CD8 signatures by sex will be analyzed.")
        logger.info(f"The shape of data frame of patient IDs, CD8 signatures, and scores is {data_frame_of_patient_IDs_CD8_signatures_and_scores.shape}.")
        logger.info(f"The shape of clinical data is {clinical_data.shape}.")

        list_of_CD8_signatures = data_frame_of_patient_IDs_CD8_signatures_and_scores.columns.tolist()
        list_of_columns = ["PATIENT_ID", "Sex", "PrimaryDiagnosisSite"] + list_of_CD8_signatures
        data_frame_of_clinical_data_and_CD8_signature_scores = clinical_data.merge(
            data_frame_of_patient_IDs_CD8_signatures_and_scores,
            left_on = "PATIENT_ID",
            right_index = True,
            how = "inner"
        )[list_of_columns]

        logger.info(f"Data frame of clinical data and CD8 signature scores is {data_frame_of_clinical_data_and_CD8_signature_scores.shape}.")

        data_frame_of_clinical_data_and_CD8_signature_scores = filter_by_primary_diagnosis_site(data_frame_of_clinical_data_and_CD8_signature_scores)
        list_of_dictionaries_of_sexes_CD8_signatures_and_statistics = []
        for sex in ["Male", "Female"]:
            data_frame_of_clinical_data_and_CD8_signature_scores_for_sex = data_frame_of_clinical_data_and_CD8_signature_scores[
                data_frame_of_clinical_data_and_CD8_signature_scores["Sex"] == sex
            ]
            for CD8_signature in self.dictionary_of_CD8_signatures_and_lists_of_IDs_of_genes.keys():
                if CD8_signature in data_frame_of_clinical_data_and_CD8_signature_scores_for_sex.columns:
                    list_of_dictionaries_of_sexes_CD8_signatures_and_statistics.append(
                        {
                            'sex': sex,
                            'signature': CD8_signature,
                            'mean': data_frame_of_clinical_data_and_CD8_signature_scores_for_sex[CD8_signature].mean(),
                            'median': data_frame_of_clinical_data_and_CD8_signature_scores_for_sex[CD8_signature].median(),
                            'std': data_frame_of_clinical_data_and_CD8_signature_scores_for_sex[CD8_signature].std(),
                            'count': len(data_frame_of_clinical_data_and_CD8_signature_scores_for_sex)
                        }
                    )
        data_frame_of_sexes_CD8_signatures_and_statistics = pd.DataFrame(list_of_dictionaries_of_sexes_CD8_signatures_and_statistics)
        
        logger.info("Data frame of sexes, CD8 signatures, and statistics will be saved.")
        
        data_frame_of_sexes_CD8_signatures_and_statistics.to_csv(paths.data_frame_of_sexes_CD8_signatures_and_statistics, index = False)
        self.plot_mean_CD8_signature_scores_by_sex(data_frame_of_sexes_CD8_signatures_and_statistics)
        self.perform_statistical_tests_for_CD8_signatures_by_sex(data_frame_of_clinical_data_and_CD8_signature_scores)

        logger.info(f"CD8 signatures by sex for {len(data_frame_of_clinical_data_and_CD8_signature_scores)} patients were analyzed.")

        return data_frame_of_clinical_data_and_CD8_signature_scores
    
    
    def plot_mean_CD8_signature_scores_by_sex(self, data_frame_of_sexes_CD8_signatures_and_statistics):
        graph = sns.catplot(
            data = data_frame_of_sexes_CD8_signatures_and_statistics,
            x = "sex",
            y = "mean",
            col = "signature",
            kind = "bar"
        )
        graph.set_titles("{col_name}")
        graph.set_xlabels("Sex")
        graph.set_ylabels("Mean CD8 Signature Score")
        graph.figure.savefig(paths.plot_of_mean_CD8_signature_scores_by_sex)
        plt.close(graph.figure)
        logger.info("Plot of mean CD8 signature scores by sex was saved.")
    
    
    def perform_statistical_tests_for_CD8_signatures_by_sex(self, data_frame_of_clinical_data_and_CD8_signature_scores):
        list_of_dictionaries_of_CD8_signatures_and_statistics = []
        for CD8_signature in self.dictionary_of_CD8_signatures_and_lists_of_IDs_of_genes.keys():
            series_of_CD8_signature_scores_of_males = data_frame_of_clinical_data_and_CD8_signature_scores[
                data_frame_of_clinical_data_and_CD8_signature_scores["Sex"] == "Male"
            ][CD8_signature]
            series_of_CD8_signature_scores_of_females = data_frame_of_clinical_data_and_CD8_signature_scores[
                data_frame_of_clinical_data_and_CD8_signature_scores["Sex"] == "Female"
            ][CD8_signature]
            if len(series_of_CD8_signature_scores_of_males) < 10 or len(series_of_CD8_signature_scores_of_females) < 10:
                raise Exception(f"There are not enough samples for signature {signature}.")
            test_statistic, p_value = stats.ttest_ind(
                series_of_CD8_signature_scores_of_males,
                series_of_CD8_signature_scores_of_females,
                equal_var = False
            )
            list_of_dictionaries_of_CD8_signatures_and_statistics.append(
                {
                    'signature': CD8_signature,
                    'male_mean': series_of_CD8_signature_scores_of_males.mean(),
                    'female_mean': series_of_CD8_signature_scores_of_females.mean(),
                    'male_count': len(series_of_CD8_signature_scores_of_males),
                    'female_count': len(series_of_CD8_signature_scores_of_females),
                    't_stat': test_statistic,
                    'p_value': p_value
                }
            )
        data_frame_of_CD8_signatures_and_statistics = pd.DataFrame(list_of_dictionaries_of_CD8_signatures_and_statistics)
        data_frame_of_CD8_signatures_and_statistics['significant'] = data_frame_of_CD8_signatures_and_statistics['p_value'] < 0.05
        data_frame_of_CD8_signatures_and_statistics.to_csv(paths.data_frame_of_CD8_signatures_and_statistics, index = False)
        
        logger.info("Statistical tests for CD8 signatures by sex were performed.")
        
        return data_frame_of_CD8_signatures_and_statistics
    
    
    def analyze_CD8_signatures_by_diagnosis(self, data_frame_of_patient_IDs_CD8_signatures_and_scores, clinical_data):

        logger.info("CD8 signatures by diagnosis will be analyzed.")

        data_frame_of_clinical_data_and_CD8_signature_scores = clinical_data.merge(
            data_frame_of_patient_IDs_CD8_signatures_and_scores,
            left_on = "PATIENT_ID",
            right_index = True,
            how = "inner"
        )
        series_of_diagnoses_and_numbers_of_patients = data_frame_of_clinical_data_and_CD8_signature_scores["PrimaryDiagnosisSite"].value_counts()
        series_of_diagnoses_with_at_least_20_patients = series_of_diagnoses_and_numbers_of_patients[series_of_diagnoses_and_numbers_of_patients >= 20].index.tolist()
        data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnoses_with_at_least_20_patients = data_frame_of_clinical_data_and_CD8_signature_scores[
            data_frame_of_clinical_data_and_CD8_signature_scores["PrimaryDiagnosisSite"].isin(
                series_of_diagnoses_with_at_least_20_patients
            )
        ]
        list_of_diagnoses = (
            data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnoses_with_at_least_20_patients[
                "PrimaryDiagnosisSite"
            ].value_counts()
            .sort_values(ascending = False)
            .index
            .tolist()
        )
        list_of_dictionaries_of_diagnoses_CD8_signatures_and_statistics = []
        for diagnosis in list_of_diagnoses:
            data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnosis = data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnoses_with_at_least_20_patients[
                data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnoses_with_at_least_20_patients[
                    "PrimaryDiagnosisSite"
                ] == diagnosis
            ]
            for CD8_signature in self.dictionary_of_CD8_signatures_and_lists_of_IDs_of_genes.keys():
                if CD8_signature in data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnosis.columns:
                    list_of_dictionaries_of_diagnoses_CD8_signatures_and_statistics.append(
                        {
                            "diagnosis": diagnosis,
                            "signature": CD8_signature,
                            "mean": data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnosis[CD8_signature].mean(),
                            "median": data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnosis[CD8_signature].median(),
                            "std": data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnosis[CD8_signature].std(),
                            "count": len(data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnosis)
                        }
                    )
        data_frame_of_diagnoses_CD8_signatures_and_statistics = pd.DataFrame(
            list_of_dictionaries_of_diagnoses_CD8_signatures_and_statistics
        )
        data_frame_of_diagnoses_CD8_signatures_and_statistics.to_csv(
            paths.data_frame_of_top_diagnoses_CD8_signatures_and_statistics,
            index = False
        )
        list_of_wrapped_diagnoses = [textwrap.fill(str(diagnosis), width = 10) for diagnosis in list_of_diagnoses]
        dictionary_of_diagnoses_and_wrapped_diagnoses = dict(zip(list_of_diagnoses, list_of_wrapped_diagnoses))
        data_frame_of_diagnoses_CD8_signatures_and_statistics["diagnosis_wrapped"] = data_frame_of_diagnoses_CD8_signatures_and_statistics["diagnosis"].map(dictionary_of_diagnoses_and_wrapped_diagnoses)
        data_frame_of_diagnoses_CD8_signatures_and_statistics["diagnosis_wrapped"] = pd.Categorical(
            data_frame_of_diagnoses_CD8_signatures_and_statistics["diagnosis_wrapped"],
            categories = list_of_wrapped_diagnoses,
            ordered = True
        )
        self.plot_CD8_signature_scores_by_diagnosis(data_frame_of_diagnoses_CD8_signatures_and_statistics)

        logger.info(f"CD8 signatures by diagnosis for {len(data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnoses_with_at_least_20_patients)} patients with diagnoses with at least 20 patients were analyzed.")

        return data_frame_of_clinical_data_and_CD8_signature_scores_of_patients_with_diagnoses_with_at_least_20_patients
    
    
    def plot_CD8_signature_scores_by_diagnosis(self, data_frame_of_diagnoses_CD8_signatures_and_statistics):
        graph = sns.catplot(
            data = data_frame_of_diagnoses_CD8_signatures_and_statistics,
            x = "diagnosis_wrapped",
            y = "mean",
            col = "signature",
            kind = "bar",
            order = list(data_frame_of_diagnoses_CD8_signatures_and_statistics["diagnosis_wrapped"].cat.categories)
        )
        graph.set_titles("{col_name}")
        for axis in graph.axes.flatten():
            axis.set_xlabel("Diagnosis")
            axis.set_ylabel("Mean Score")
            for label in axis.get_xticklabels():
                label.set_rotation(0)
                label.set_ha("center")
        graph.figure.subplots_adjust(bottom = 0.25, top = 0.9)
        graph.figure.savefig(paths.plot_of_mean_CD8_signature_scores_by_diagnosis)
        plt.close(graph.figure)
        
        logger.info("Plot of mean CD8 signature scores by diagnosis was saved.")
    
    
    def analyze_survival_by_CD8_signature(self, data_frame_of_patient_IDs_CD8_signatures_and_scores, clinical_data):
        
        logger.info("Survival by CD8 signature will be analyzed.")
        
        data_frame_of_clinical_data_and_CD8_signature_scores = clinical_data.merge(
            data_frame_of_patient_IDs_CD8_signatures_and_scores,
            left_on = "PATIENT_ID",
            right_index = True,
            how = "inner"
        )
        data_frame_of_clinical_data_and_CD8_signature_scores = filter_by_primary_diagnosis_site(data_frame_of_clinical_data_and_CD8_signature_scores)
        vital_status_data = pd.read_csv(paths.vital_status_data)
        data_frame_of_clinical_data_and_CD8_signature_scores = data_frame_of_clinical_data_and_CD8_signature_scores.merge(
            vital_status_data,
            how = "left",
            left_on = "PATIENT_ID",
            right_on = "AvatarKey"
        )
        data_frame_of_clinical_data_and_CD8_signature_scores = calculate_survival_months(data_frame_of_clinical_data_and_CD8_signature_scores)
        data_frame_of_clinical_data_and_CD8_signature_scores = data_frame_of_clinical_data_and_CD8_signature_scores.rename(columns = {"survival_months": "OS_MONTHS"})
        data_frame_of_clinical_data_and_CD8_signature_scores["OS_STATUS"] = data_frame_of_clinical_data_and_CD8_signature_scores["event"].map({1: "DECEASED", 0: "ALIVE"})
        series_of_indicators_that_survival_data_is_valid = (
            data_frame_of_clinical_data_and_CD8_signature_scores["OS_MONTHS"].notna() &
            np.isfinite(data_frame_of_clinical_data_and_CD8_signature_scores["OS_MONTHS"])
        )
        data_frame_of_clinical_data_and_CD8_signature_scores = data_frame_of_clinical_data_and_CD8_signature_scores.loc[series_of_indicators_that_survival_data_is_valid]
        number_of_rows_of_survival_data_dropped = (~series_of_indicators_that_survival_data_is_valid).sum()
        
        logger.info(f"{number_of_rows_of_survival_data_dropped} rows of survival data were dropped.")
        
        for CD8_signature in self.dictionary_of_CD8_signatures_and_lists_of_IDs_of_genes.keys():
            median_CD8_signature_score = data_frame_of_clinical_data_and_CD8_signature_scores[CD8_signature].median()
            data_frame_of_clinical_data_and_CD8_signature_scores[f"group_for_CD8_signature_{CD8_signature}"] = (data_frame_of_clinical_data_and_CD8_signature_scores[CD8_signature] > median_CD8_signature_score).map({True: "High", False: "Low"})
            self.plot_survival_curves(data_frame_of_clinical_data_and_CD8_signature_scores, f"group_for_CD8_signature_{CD8_signature}", CD8_signature)
        
        logger.info(f"Survival by CD8 signature for {len(data_frame_of_clinical_data_and_CD8_signature_scores)} patients was analyzed.")
        
        return data_frame_of_clinical_data_and_CD8_signature_scores
    
    
    def plot_survival_curves(self, data_frame_of_clinical_data_and_CD8_signature_scores: pd.DataFrame, group_for_CD8_signature: str, title: str):
        '''
        Plot Kaplan-Meier survival curves by group.
        '''            
        df = data_frame_of_clinical_data_and_CD8_signature_scores[
            data_frame_of_clinical_data_and_CD8_signature_scores[group_for_CD8_signature].notna()
        ].copy()
        df = df.dropna(subset = ["event"])
        list_of_groups_for_CD8_signature = sorted(df[group_for_CD8_signature].unique().tolist())
        Kaplan_Meier_fitter = KaplanMeierFitter()
        for group in list_of_groups_for_CD8_signature:
            slice_of_data_frame_corresponding_to_group = df[df[group_for_CD8_signature] == group]
            Kaplan_Meier_fitter.fit(
                durations = slice_of_data_frame_corresponding_to_group["OS_MONTHS"],
                event_observed = slice_of_data_frame_corresponding_to_group["event"],
                label = f"{group} (n = {len(slice_of_data_frame_corresponding_to_group)})"
            )
            Kaplan_Meier_fitter.plot()
        plt.xlabel("Months")
        plt.ylabel("Survival Probability")
        plt.title(f"Kaplan-Meier Survival Curves by {title}")
        plt.grid(alpha = 0.3)
        out_name = f"survival_by_{title}.png"
        plt.savefig(paths.cd8_analysis_plots / out_name, dpi = 300, bbox_inches = "tight")
        plt.close()
        
        logger.info(f"Plot of survival curves by {title} was saved.")
        
        if len(list_of_groups_for_CD8_signature) == 2:
            g1, g2 = list_of_groups_for_CD8_signature
            slice_of_data_frame_corresponding_to_group_1 = df[df[group_for_CD8_signature] == g1]
            slice_of_data_frame_corresponding_to_group_2 = df[df[group_for_CD8_signature] == g2]
            if len(slice_of_data_frame_corresponding_to_group_1) > 0 and len(slice_of_data_frame_corresponding_to_group_2) > 0:
                results = logrank_test(
                    slice_of_data_frame_corresponding_to_group_1["OS_MONTHS"],
                    slice_of_data_frame_corresponding_to_group_2["OS_MONTHS"],
                    event_observed_for_patient_in_group_1 = slice_of_data_frame_corresponding_to_group_1["event"],
                    event_observed_for_patient_in_group_2 = slice_of_data_frame_corresponding_to_group_2["event"]
                )
                
                logger.info(f"p value for log rank test for CD8 signature {title} is {results.p_value:.4f}.")

                path_to_results_of_log_rank_test = os.path.join(paths.outputs_of_cd8_analysis, f"results_of_log_rank_test_for_{title}.txt")
                with open(path_to_results_of_log_rank_test, 'w') as file:
                    file.write(f"results of log rank test for CD8 signature {title}:\n")
                    file.write(f"Group 1: {g1} (n = {len(slice_of_data_frame_corresponding_to_group_1)})\n")
                    file.write(f"Group 2: {g2} (n = {len(slice_of_data_frame_corresponding_to_group_2)})\n")
                    file.write(f"p value: {results.p_value:.4f}")
    
    
    def run(self):
        
        logger.info("CD8 analysis will be run.")

        expression_matrix = load_expression_matrix()
        data_frame_of_patient_IDs_CD8_signatures_and_scores = self.create_data_frame_of_patient_IDs_CD8_signatures_and_scores(expression_matrix)
        clinical_data = pd.read_csv(paths.melanoma_clinical_data)
        clinical_data = clinical_data[~clinical_data["Primary/Met"].str.contains("germline", case = False, na = False)]
        self.analyze_signatures_by_sex(data_frame_of_patient_IDs_CD8_signatures_and_scores, clinical_data)
        self.analyze_CD8_signatures_by_diagnosis(data_frame_of_patient_IDs_CD8_signatures_and_scores, clinical_data)
        self.analyze_survival_by_CD8_signature(data_frame_of_patient_IDs_CD8_signatures_and_scores, clinical_data)

        print("CD8 analysis is complete.")
        print(f"Results were saved to `{paths.outputs_of_cd8_analysis}`.")
        print(f"Plots were saved to `{paths.cd8_analysis_plots}`.")

        return data_frame_of_patient_IDs_CD8_signatures_and_scores


if __name__ == "__main__":
    
    paths.ensure_dependencies_for_cd8_analysis_exist()
    
    analysis = CD8Analysis()
    analysis.run()