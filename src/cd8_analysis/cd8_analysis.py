'''
CD8 Analysis
Analyzes CD8+ T cell signatures

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
    
    
    def analyze_survival_by_signature(self, scores, clinical_data):
        '''
        Analyze survival by CD8 signature.
        '''
        print("Analyzing survival by CD8 signature...")
        
        merged = clinical_data.merge(
            scores,
            left_on = 'PATIENT_ID',
            right_index = True,
            how = 'inner'
        )
        
        merged = filter_by_primary_diagnosis_site(merged)
        
        vital_status_data = pd.read_csv(paths.vital_status_data)
        merged = merged.merge(
            vital_status_data,
            how = "left",
            left_on = "PATIENT_ID",
            right_on = "AvatarKey"
        )
        
        merged = calculate_survival_months(
            merged,
            age_at_diagnosis_col = "AgeAtDiagnosis",
            age_at_last_contact_col = "AgeAtLastContact",
            age_at_death_col = "AgeAtDeath",
            vital_status_col = "VitalStatus"
        )
        
        merged = merged.rename(columns = {"survival_months": "OS_MONTHS"})
        merged["OS_STATUS"] = merged["event"].map({1: "DECEASED", 0: "ALIVE"})
        
        merged["OS_MONTHS"] = pd.to_numeric(merged["OS_MONTHS"], errors = "raise")
        merged["event"] = pd.to_numeric(merged["event"], errors = "raise")
        
        valid = merged["OS_MONTHS"].notna() & np.isfinite(merged["OS_MONTHS"]) & (merged["OS_MONTHS"] >= 0) & merged["event"].isin([0, 1])
        dropped = (~valid).sum()
        merged = merged.loc[valid].copy()
        
        print(
            f"Calculated survival months for {len(merged)} patients" +
            (f" (dropped {dropped} invalid rows)" if dropped else "")
        )
        
        # Analyze each signature
        for signature in self.dictionary_of_CD8_signatures_and_lists_of_IDs_of_genes.keys():
            if signature not in merged.columns:
                continue
            
            # Group by median
            median = merged[signature].median()
            merged[f'{signature}_group'] = (merged[signature] > median).map({True: 'High', False: 'Low'})
            
            # Plot Kaplan-Meier curves
            self.plot_survival_curves(merged, f'{signature}_group', signature)
        
        print(f"Analyzed survival by CD8 signature for {len(merged)} patients")
        
        return merged
    
    
    def plot_survival_curves(self, merged: pd.DataFrame, group_col: str, title: str | None = None):
        '''
        Plot Kaplan-Meier survival curves by group.
        '''
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
            
            df = merged.copy()
            
            df = df[df[group_col].notna()].copy()
            df = df.dropna(subset = ["OS_MONTHS", "event"])
            df = df[np.isfinite(df["OS_MONTHS"])]
            df = df[df["OS_MONTHS"] >= 0]
            df = df[df["event"].isin([0, 1])]
            
            groups = sorted(df[group_col].unique().tolist())
            if len(groups) == 0:
                print(f"[{title or group_col}] No groups after cleaning; skipping plot.")
                return
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Initialize Kaplan-Meier fitter
            kmf = KaplanMeierFitter()
            
            # Plot survival curve for each group.
            plotted_any = False
            for group in groups:
                group_data = df[df[group_col] == group]
                
                # Skip if not enough samples
                if len(group_data) < 10:
                    print(f"[{title or group_col}] Skipping group '{group}' (n={len(group_data)} < 10).")
                    continue
                
                # Fit survival curve
                kmf.fit(
                    durations = group_data['OS_MONTHS'],
                    event_observed = group_data['event'],
                    label = f'{group} (n={len(group_data)})'
                )
                
                # Plot survival curve
                kmf.plot()
                plotted_any = True
            
            if not plotted_any:
                print(f"[{title or group_col}] No groups met minimum size; skipping plot.")
                plt.close()
                return
            
            # Add labels and title
            plt.xlabel('Months')
            plt.ylabel('Survival Probability')
            plt.title(f"Kaplan-Meier Survival Curves by {title or group_col}")
            plt.grid(alpha=0.3)
            
            # Save plot
            plt.tight_layout()
            out_name = f'survival_by_{title or group_col}.png'
            plt.savefig(paths.cd8_analysis_plots / out_name, dpi = 300, bbox_inches = "tight")
            plt.close()
            print(f"Saved survival curves by {title or group_col}")
            
            # Perform log-rank test if there are exactly 2 groups
            if len(groups) == 2:
                g1, g2 = groups
                group1_data = df[df[group_col] == g1]
                group2_data = df[df[group_col] == g2]
                if len(group1_data) > 0 and len(group2_data) > 0:
                    results = logrank_test(
                        group1_data['OS_MONTHS'],
                        group2_data['OS_MONTHS'],
                        event_observed_A = group1_data['event'],
                        event_observed_B = group2_data['event']
                    )
                    print(f"[{title or group_col}] Log-rank test p-value: {results.p_value:.4f}")
                
                    output_file = os.path.join(paths.outputs_of_cd8_analysis, f'logrank_{title or group_col}.txt')
                    
                    with open(output_file, 'w') as f:
                        f.write(f"Log-rank test results for {title or group_col}:\n")
                        f.write(f"Group 1: {g1} (n={len(group1_data)})\n")
                        f.write(f"Group 2: {g2} (n={len(group2_data)})\n")
                        f.write(f"p-value: {results.p_value:.4f}\n")
        
        except Exception as e:
            print(f"Error plotting survival curves: {e}")
            print(traceback.format_exc())
    
    
    def run(self):
        
        logger.info("CD8 analysis will be run.")

        expression_matrix = load_expression_matrix()
        data_frame_of_patient_IDs_CD8_signatures_and_scores = self.create_data_frame_of_patient_IDs_CD8_signatures_and_scores(expression_matrix)
        clinical_data = pd.read_csv(paths.melanoma_clinical_data)
        clinical_data = clinical_data[~clinical_data["Primary/Met"].str.contains("germline", case = False, na = False)]
        self.analyze_signatures_by_sex(data_frame_of_patient_IDs_CD8_signatures_and_scores, clinical_data)
        self.analyze_CD8_signatures_by_diagnosis(data_frame_of_patient_IDs_CD8_signatures_and_scores, clinical_data)
        self.analyze_survival_by_signature(data_frame_of_patient_IDs_CD8_signatures_and_scores, clinical_data)

        print("\nCD8 analysis complete!")
        print(f"Results were saved to {paths.outputs_of_cd8_analysis}")
        print(f"Plots were saved to {paths.cd8_analysis_plots}")

        return data_frame_of_patient_IDs_CD8_signatures_and_scores


if __name__ == "__main__":
    
    paths.ensure_dependencies_for_cd8_analysis_exist()
    
    analysis = CD8Analysis()
    analysis.run()