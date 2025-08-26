'''
cd8_groups_analysis.py

`cd8_groups_analysis.py` turns bulk RNA sequencing data for samples of tumors of patients
into abundance scores for 7 different states of CD8+ T cells for each sample.
The script clusters samples by scores and relates clusters to clinical data.

The script loads RNA sequencing data into a matrix where each of 17,120 filtered rows corresponds to a gene.
Each of 333 columns corresponds to a physical library of fragments of RNA from a sample of a tumor of patient.
Each value represents a number of fragments.

The script creates a 333 x 7 matrix of scores / means of values for a sample and genes related to a state.

The script clusters samples into 6 clusters based on states B (exhausted) and G (proliferative).

The script relates clusters to sexes and diagnoses and relates survival to clusters.

`CD8_B_by_diagnosis.png`
A bar chart of mean `CD8_B` score by diagnosis summarizes how `CD8_B` score varies across primary diagnosis sites.
Patients with high `CD8_B` scores typically have exhausted T cells and don't respond to ICB therapy.
`CD8_B` score is highest for tumors of skin of upper limb and shoulder.
`CD8_B` scores vary highly.
Differences in means may not be statistically significant.

`CD8_G_by_diagnosis.png`
A bar chart of mean `CD8_G` score by diagnosis summarizes how `CD8_G` score varies across primary diagnosis sites.
Patients with high `CD8_G` scores typically have proliferative T cells and respond to ICB therapy.
`CD8_G` score is highest for tumors of skin Not Otherwise Specified.
`CD8_G` scores vary highly.
Differences in means may not be statistically significant.

`CD8_GtoB_log_by_diagnosis.png`
A bar chart of mean log of ratio of `CD8_G` score to `CD8_B` score by diagnosis
summarizes how log of ratio varies across primary diagnosis sites.
Patients with high logs of ratios typically have mostly proliferative T cells with little exhaustion and respond to ICB therapy.
Log of ratio is highest for tumors of skin of trunk.
Log of ratios vary highly.
Differences in means may not be statistically significant.

`CD8_GtoB_ratio_by_diagnosis.png`
A bar chart of mean ratio of `CD8_G` score to `CD8_B` score by diagnosis summarizes how ratio varies across diagnosis sites.
Patients with high ratios typically have mostly proliferative T cells with little exhaustion and respond to ICB therapy.
Ratio is highest for tumors of skin of trunk.
Ratios vary highly.
Differences in means may not be statistically significant.

`cd8_pca.png`
The script creates a scatter plot of samples--
defined by `CD8_B` score, `CD8_G` score, ratio of `CD8_G` score to `CD8_B` score, and log of ratio --
projected onto Principal Components 1 and 2.
PC1 may represent a spectrum of samples with few CD8+ T cells to samples with many CD8+ T cells, mostly proliferative.
PC2 may represent a spectrum of low ratio / exhausted to high ratio / proliferative.
The cluster of red points represents mostly proliferative T cells with little exhaustion.
The cluster of green points represents tumors without many CD8+ T cells.

`cluster_by_diagnosis.png`
The script creates a bar chart of percentage of patients with a given diagnosis by cluster.
The x axis represents clusters 0 through 5.
Each bar has a color that corresponds to a diagnosis and gives the percentage of of patients with that diagnosis in that cluster.
Percentages sum to 100 across clusters with each diagnosis.
Diagnoses with at least 20 patients are included.
Percentages of patients with diagnoses are fairly constant for patients with samples in cluster 0.
About 29 percent of samples with primary diagnosis site skin of upper limb and shoulder in cluster 1
have high CD8_B scores / exhausted T cells.
Modest percentages of samples with other primary diagnosis sites in cluster 1 have high CD8_B scores / exhausted T cells.
About 32 percent of samples with primary diagnosis site skin of trunk in cluster 2 have few CD8+ T cells.
About 20 percent of samples with other primary diagnosis sites in cluster 2 have few CD8+ T cells.
About 5 percent of samples in cluster 3 have high `CD8_G` scores and moderate `CD8_B` scores.
About 30 percent of samples with primary diagnosis sites in cluster 4 have moderate `CD8_G` scores and low `CD8_B` scores.
About 15 percent of samples with primary diagnosis sites in cluster 4 have moderate `CD8_G` scores and low `CD8_B` scores.
About 21 percent of samples with primary diagnosis sites in cluster 5 have moderate `CD8_G` scores and moderate `CD8_B` scores.
About 15 percent of samples with primary diagnosis sites in cluster 5 have moderate `CD8_G` scores and moderate `CD8_B` scores.

`plot_of_CD8_clusters.png`
A scatter plot of `CD8_G` score vs. `CD8_B` score for each sample allows visualizing the results of K means clusters where K is 6.
The cluster of red points represents mostly proliferative T cells with little exhaustion.
The cluster of green points represents tumors without many CD8+ T cells.

`plots_of_mean_CD8_group_scores_by_sex.png`
The script creates bar charts of
mean `CD8_B` score by sex,
mean `CD8_G` score by sex,
mean ratio of `CD8_G` score to `CD8_B` score, and
mean log of ratio by sex.
Patients with high `CD8_B` scores typically have exhausted T cells and don't respond to ICB therapy.
Patients with high `CD8_G` scores typically have proliferative T cells and respond to ICB therapy.
Patients with high ratios typically have mostly proliferative T cells with little exhaustion and respond to ICB therapy.
Differences in means may not be statistically significant.

`plot_of_distributions_of_clusters_by_sex.png`
A bar chart of percentage of patients in a given cluster by sex shows that about 25 percent of patients have low `CD8_G` and moderate `CD8_B` scores. This cluster is more common for females.

`survival_by_CD8_GtoB_group.png`
Survival curves by groups of ratios of `CD8_G` score to CD8_B` score suggest that patients with high ratios are less likely to survive over time. `CD8_B` score being equal, high ratio means high `CD8_G` score, high levels of T cells, and high levels of reproducing tumor cells.

`survival_by_cluster.png`
Survival curves by cluster, when compared with the scatter plot of `CD8_G` score vs. `CD8_B` score for each sample, suggest that patients with low ratios (orange, blue, brown) are more likely to survive over time. `CD8_B` score being equal, high ratio means high `CD8_G` score, high levels of T cells, and high levels of reproducing tumor cells. Patients with low `CD8_G` scores, low `CD8_B` scores, and low general immune response are less likely to survive over time.


Usage
./miniconda3/envs/ici_sex/bin/python -m src.cd8_analysis.cd8_groups_analysis
'''


from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from src.cd8_analysis.cd8_analysis import CD8Analysis
from src.utils.shared_functions import calculate_survival_months, filter_by_primary_diagnosis_site, load_expression_matrix, map_sample_IDs_to_patient_IDs
from src.config import paths


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s – %(levelname)s – %(message)s"
)
logger = logging.getLogger(__name__)


class CD8GroupAnalysis(CD8Analysis):
    '''
   CD8GroupAnalysis is a template for an object for analyzing CD8+ T cell signatures and groups.
    '''
    
    def __init__(self):        
        '''
        Define CD8 T cell groups and their marker genes
        CD8_A is a classical cytotoxic / effector described in Sade-Feldman et al. 2018 and Li e al. 2023.
        TODO: Reword "classical cytotoxic / effector".
        CD8_B is an exhausted / inhibitory-receptor-hi described in Sade-Feldman et al. 2018.
        TODO: Reword "exhausted / inhibitory-receptor-hi".
        CD8_C is a TCM / TPEX (memory-like, IL-7R-hi) described in Beltra et al. 2020.
        TODO: Reword "TCM / TPEX (memory-like, IL-7R-hi)".
        CD8_D is an NR4A-driven exhaustion described in Chen et al. 2019.
        TODO: Reword "NR4A-driven exhaustion".
        CD8_E is a transitional / tissue-resident described in Wu et al. 2021.
        TODO: Reword "transitional / tissue-resident".
        CD8_F is a cytotoxic NK-like described in Guo et al. 2018.
        TODO: Reword "cytotoxic NK-like".
        CD8_G is a cycling / proliferative described in Li et al. 2023.
        TODO: Reword "cycling / proliferative".
        '''
        self.cd8_groups = {
            "CD8_A": [
                "ENSG00000153563", # HGNC gene symbol is CD8A.
                "ENSG00000172116", # CD8B
                "ENSG00000145649", # GZMA
                "ENSG00000100453", # GZMB
                "ENSG00000180644", # PRF1
                "ENSG00000111537" # IFNG
            ],
            "CD8_B": [
                "ENSG00000138185", # ENTPD1
                "ENSG00000188389", # PDCD1
                "ENSG00000163599", # CTLA4
                "ENSG00000089692", # LAG3
                "ENSG00000135077", # HAVCR2
                "ENSG00000181847" # TIGIT
            ],
            "CD8_C": [
                "ENSG00000168685", # IL7R
                "ENSG00000081059", # TCF7
                "ENSG00000126353", # CCR7
                "ENSG00000188404", # SELL
                "ENSG00000138795", # LEF1
                "ENSG00000139193" # CD27
            ],
            "CD8_D": [
                "ENSG00000123358", # NR4A1
                "ENSG00000153234", # NR4A2
                "ENSG00000119508", # NR4A3
                "ENSG00000198846", # TOX
                "ENSG00000124191" # TOX2
            ],
            "CD8_E": [
                "ENSG00000113088", # GZMK
                "ENSG00000197540", # GZMM
                "ENSG00000019582", # CD74
                "ENSG00000139193", # CD27
                "ENSG00000143184", # XCL1
                "ENSG00000143185" # XCL2
            ],
            "CD8_F": [
                "ENSG00000137441", # FGFBP2
                "ENSG00000203747", # FCGR3A
                "ENSG00000168329", # CX3CR1
                "ENSG00000139187", # KLRG1
                "ENSG00000115523", # GNLY
                "ENSG00000105374" # NKG7
            ],
            "CD8_G": [
                "ENSG00000148773", # MKI67
                "ENSG00000131747", # TOP2A
                "ENSG00000170312", # CDK1
                "ENSG00000132646", # PCNA
                "ENSG00000176890" # TYMS
            ]
        }
        
        self.dictionary_of_names_of_features_and_labels = {
            "CD8_B": "Non-responder enriched (Clusters 1-3)",
            "CD8_G": "Responder enriched (Clusters 4-6)",
            "CD8_GtoB_ratio": "Responder/Non-responder ratio",
            "CD8_GtoB_log": "Log2(Responder/Non-responder ratio)"
        }
    
    
    def calculate_CD8_group_scores(self, expression_matrix: pd.DataFrame) -> pd.DataFrame:
        '''
        Calculate CD8 group scores.
        '''
        
        logger.info("CD8 group scores will be calculated.")
        
        condensed_expression_matrix = expression_matrix.copy()
        condensed_expression_matrix.index = condensed_expression_matrix.index.str.split('.').str[0]
        condensed_expression_matrix = condensed_expression_matrix.groupby(condensed_expression_matrix.index).mean()
        data_frame_of_scores = pd.DataFrame(
            {
                group: condensed_expression_matrix.loc[
                    [gene for gene in list_of_genes if gene in condensed_expression_matrix.index]
                ].mean()
                for group, list_of_genes in self.cd8_groups.items()
            }
        )
        if "CD8_B" in data_frame_of_scores.columns and "CD8_G" in data_frame_of_scores.columns:
            constant_to_avoid_division_by_0 = 0.01
            data_frame_of_scores["CD8_GtoB_ratio"] = (
                data_frame_of_scores["CD8_G"] / (data_frame_of_scores["CD8_B"] + constant_to_avoid_division_by_0)
            )
            data_frame_of_scores["CD8_GtoB_log"] = np.log2(data_frame_of_scores["CD8_GtoB_ratio"])
        data_frame_of_scores.to_csv(paths.data_frame_of_CD8_group_scores)
        
        logger.info(f"CD8 group scores for {len(data_frame_of_scores)} samples were calculated.")
        
        return data_frame_of_scores

    
    def cluster_samples(self, data_frame_of_scores: pd.DataFrame) -> pd.DataFrame:
        '''
        Cluster samples based on CD8 group scores.
        '''
        number_of_clusters = 6
        
        logger.info(f"Samples will be clustered into {number_of_clusters} clusters.")
        
        list_of_features = ["CD8_B", "CD8_G"]
        array_of_CD8_B_and_CD8_G_scores = data_frame_of_scores[list_of_features].values
        standard_scaler = StandardScaler()
        scaled_array_of_CD8_B_and_CD8_G_scores = standard_scaler.fit_transform(array_of_CD8_B_and_CD8_G_scores)
        kmeans = KMeans(n_clusters = number_of_clusters, random_state = 0, n_init = 10)
        array_of_indices_of_clusters = kmeans.fit_predict(scaled_array_of_CD8_B_and_CD8_G_scores)
        data_frame_of_scores_and_indices_of_clusters = data_frame_of_scores.copy()
        data_frame_of_scores_and_indices_of_clusters["cluster"] = array_of_indices_of_clusters
        silhouette = silhouette_score(scaled_array_of_CD8_B_and_CD8_G_scores, array_of_indices_of_clusters)
        
        logger.info(f"Silhouette score is {silhouette:.3f}.")
        
        data_frame_of_scores_and_indices_of_clusters.to_csv(paths.data_frame_of_scores_and_indices_of_clusters)
        self.plot_clusters(data_frame_of_scores_and_indices_of_clusters, list_of_features)
        
        logger.info(f"{len(data_frame_of_scores)} samples were clustered into {number_of_clusters} clusters.")
        
        return data_frame_of_scores_and_indices_of_clusters

    
    def plot_clusters(self, data_frame_of_scores_and_indices_of_clusters: pd.DataFrame, list_of_features: list[str]):
        '''
        Plot clusters.
        '''
        for cluster in sorted(data_frame_of_scores_and_indices_of_clusters["cluster"].unique()):
            data_frame_of_scores_for_cluster = data_frame_of_scores_and_indices_of_clusters[
                data_frame_of_scores_and_indices_of_clusters["cluster"] == cluster
            ]
            plt.scatter(
                data_frame_of_scores_for_cluster[list_of_features[0]],
                data_frame_of_scores_for_cluster[list_of_features[1]],
                label = f"Cluster {cluster}",
                alpha = 0.7,
                s = 50
            )
        plt.xlabel(f"{list_of_features[0]} ({self.dictionary_of_names_of_features_and_labels.get(list_of_features[0], "Unknown")})")
        plt.ylabel(f"{list_of_features[1]} ({self.dictionary_of_names_of_features_and_labels.get(list_of_features[1], "Unknown")})")
        plt.title("CD8 Group Clusters")
        plt.legend()
        plt.savefig(paths.plot_of_CD8_clusters)
        plt.close()
        
        self.plot_PCA(data_frame_of_scores_and_indices_of_clusters)
        
        logger.info("Plots of CD8 clusters were saved.")
    

    def plot_PCA(self, data_frame_of_scores_and_indices_of_clusters: pd.DataFrame):
        '''
        Plot PCA of CD8 group scores.
        '''
        list_of_features = ["CD8_B", "CD8_G", "CD8_GtoB_ratio", "CD8_GtoB_log"]
        list_of_features = [feature for feature in list_of_features if feature in data_frame_of_scores_and_indices_of_clusters.columns]
        array_of_features = data_frame_of_scores_and_indices_of_clusters[list_of_features].values
        standard_scaler = StandardScaler()
        scaled_array_of_features = standard_scaler.fit_transform(array_of_features)
        pca = PCA(n_components = 2)
        projected_array_of_features = pca.fit_transform(scaled_array_of_features)
        for cluster in sorted(data_frame_of_scores_and_indices_of_clusters["cluster"].unique()):
            plt.scatter(
                projected_array_of_features[data_frame_of_scores_and_indices_of_clusters["cluster"] == cluster, 0],
                projected_array_of_features[data_frame_of_scores_and_indices_of_clusters["cluster"] == cluster, 1],
                label = f"Cluster {cluster}",
                alpha = 0.7,
                s = 50
            )
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        plt.title("PCA of CD8 Group Scores")
        plt.legend()
        plt.savefig(paths.plot_of_CD8_PCA)
        plt.close()
    

    def analyze_clusters_by_sex(self, data_frame_of_scores_and_indices_of_clusters: pd.DataFrame, clinical_data: pd.DataFrame):
        '''
        Analyze clusters by sex.
        '''
        
        logger.info("Clusters will be analyzed by sex.")
        
        data_frame_of_clinical_data_scores_and_indices_of_clusters = clinical_data.merge(
            data_frame_of_scores_and_indices_of_clusters,
            left_on = "PATIENT_ID",
            right_index = True,
            how = "inner"
        )
        data_frame_of_clinical_data_scores_and_indices_of_clusters = filter_by_primary_diagnosis_site(
            data_frame_of_clinical_data_scores_and_indices_of_clusters
        )
        contigency_data_frame_of_cluster_and_sex = pd.crosstab(
            data_frame_of_clinical_data_scores_and_indices_of_clusters["cluster"],
            data_frame_of_clinical_data_scores_and_indices_of_clusters["Sex"],
            normalize = "columns"
        ) * 100
        contigency_data_frame_of_cluster_and_sex.to_csv(paths.contigency_data_frame_of_cluster_and_sex)
        contigency_data_frame_of_cluster_and_sex.plot(kind = "bar")
        plt.xlabel("Cluster")
        plt.ylabel("Percentage (%)")
        plt.title("Cluster Distribution by Sex")
        plt.xticks(rotation = 0)
        plt.legend(title = "Sex")
        plt.savefig(paths.plot_of_distributions_of_clusters_by_sex)
        plt.close()
        list_of_dictionaries_of_sexes_features_and_statistics = []
        for sex in ["Male", "Female"]:
            sex_data = data_frame_of_clinical_data_scores_and_indices_of_clusters[
                data_frame_of_clinical_data_scores_and_indices_of_clusters["Sex"] == sex
            ]
            for feature in ["CD8_B", "CD8_G", "CD8_GtoB_ratio", "CD8_GtoB_log"]:
                if feature in sex_data.columns:
                    list_of_dictionaries_of_sexes_features_and_statistics.append(
                        {
                            "sex": sex,
                            "feature": feature,
                            "mean": sex_data[feature].mean(),
                            "median": sex_data[feature].median(),
                            "std": sex_data[feature].std(),
                            "count": len(sex_data)
                        }
                    )
        data_frame_of_sexes_features_and_statistics = pd.DataFrame(list_of_dictionaries_of_sexes_features_and_statistics)
        data_frame_of_sexes_features_and_statistics.to_csv(paths.data_frame_of_sexes_features_and_statistics, index = False)
        self.plot_mean_CD8_group_scores_by_sex(data_frame_of_sexes_features_and_statistics)
        self.perform_t_tests_for_mean_CD8_group_scores_by_sex(data_frame_of_clinical_data_scores_and_indices_of_clusters)
        
        logger.info(f"Clusters by sex for {len(data_frame_of_clinical_data_scores_and_indices_of_clusters)} patients were analyzed.")
        
        return data_frame_of_clinical_data_scores_and_indices_of_clusters

    
    def plot_mean_CD8_group_scores_by_sex(self, data_frame_of_sexes_features_and_statistics: pd.DataFrame):
        '''
        Plot mean CD8 group scores by sex.
        '''
        list_of_features = ["CD8_B", "CD8_G", "CD8_GtoB_ratio", "CD8_GtoB_log"]
        list_of_sexes = ["Female", "Male"]
        figure, array_of_axes = plt.subplots(1, 4, figsize = (14, 5))#, constrained_layout = True)
        for axes, feature in zip(array_of_axes, list_of_features):
            feature_data = data_frame_of_sexes_features_and_statistics[
                data_frame_of_sexes_features_and_statistics["feature"] == feature
            ].copy()
            sns.barplot(
                ax = axes,
                data = feature_data,
                errorbar = None,
                order = list_of_sexes,
                x = "sex",
                y = "mean"
            )
            for horizontal_position, sex in enumerate(list_of_sexes):
                row = feature_data.loc[feature_data["sex"] == sex]
                mean = row["mean"].iloc[0]
                standard_deviation = row["std"].iloc[0]
                axes.errorbar(
                    horizontal_position,
                    mean,
                    yerr = standard_deviation
                )
            axes.set_xlabel("Sex")
            axes.set_ylabel("Mean Score")
            axes.set_title(f"{feature}")
            #ymax = max((feature_data["mean"] + feature_data["std"].fillna(0)).max(), 0)
            #ax.set_ylim(top = ymax * 1.1 if ymax > 0 else None)
        figure.savefig(paths.plots_of_mean_CD8_group_scores_by_sex)
        plt.close(figure)
        
        logger.info("Plots of mean CD8 group scores by sex were saved.")

    
    def perform_t_tests_for_mean_CD8_group_scores_by_sex(self, data_frame_of_clinical_data_scores_and_indices_of_clusters):
        list_of_dictionaries_of_features_and_statistics = []
        for feature in ["CD8_B", "CD8_G", "CD8_GtoB_ratio", "CD8_GtoB_log"]:
            feature_data_for_males = data_frame_of_clinical_data_scores_and_indices_of_clusters[
                data_frame_of_clinical_data_scores_and_indices_of_clusters["Sex"] == "Male"
            ][feature]
            feature_data_for_females = data_frame_of_clinical_data_scores_and_indices_of_clusters[
                data_frame_of_clinical_data_scores_and_indices_of_clusters["Sex"] == "Female"
            ][feature]
            test_statistic, p_value = stats.ttest_ind(feature_data_for_males, feature_data_for_females, equal_var = False)
            list_of_dictionaries_of_features_and_statistics.append(
                {
                    "feature": feature,
                    "male_mean": feature_data_for_males.mean(),
                    "female_mean": feature_data_for_females.mean(),
                    "male_count": len(feature_data_for_males),
                    "female_count": len(feature_data_for_females),
                    "t_stat": test_statistic,
                    "p_value": p_value
                }
            )
        data_frame_of_features_and_statistics = pd.DataFrame(list_of_dictionaries_of_features_and_statistics)
        data_frame_of_features_and_statistics["significant"] = data_frame_of_features_and_statistics["p_value"] < 0.05
        data_frame_of_features_and_statistics.to_csv(paths.data_frame_of_features_and_statistics, index = False)
        
        logger.info("T tests for mean CD8 group scores by sex were performed.")
        
        return data_frame_of_features_and_statistics

    
    def analyze_clusters_by_diagnosis(self, data_frame_of_scores_and_indices_of_clusters: pd.DataFrame, clinical_data: pd.DataFrame):
        '''
        Analyze clusters by diagnosis.
        '''
        
        logger.info("Clusters by diagnosis will be analyzed.")
        
        data_frame_of_clinical_data_scores_and_indices_of_clusters = clinical_data.merge(
            data_frame_of_scores_and_indices_of_clusters,
            left_on = "PATIENT_ID",
            right_index = True,
            how = "inner"
        )
        series_of_diagnoses_and_numbers_of_patients = data_frame_of_clinical_data_scores_and_indices_of_clusters["PrimaryDiagnosisSite"].value_counts()
        list_of_diagnoses_with_at_least_20_patients = series_of_diagnoses_and_numbers_of_patients[
            series_of_diagnoses_and_numbers_of_patients >= 20
        ].index.tolist()
        if len(list_of_diagnoses_with_at_least_20_patients) == 0:
            raise Exception("No diagnoses with at least 20 patients")
        data_frame_of_clinical_data_scores_and_indicators_of_clusters_for_patients_with_diagnoses_with_at_least_20_patients = data_frame_of_clinical_data_scores_and_indices_of_clusters[
            data_frame_of_clinical_data_scores_and_indices_of_clusters["PrimaryDiagnosisSite"].isin(
                list_of_diagnoses_with_at_least_20_patients
            )
        ]
        contingency_data_frame_of_cluster_and_diagnosis = pd.crosstab(
            data_frame_of_clinical_data_scores_and_indicators_of_clusters_for_patients_with_diagnoses_with_at_least_20_patients["cluster"],
            data_frame_of_clinical_data_scores_and_indicators_of_clusters_for_patients_with_diagnoses_with_at_least_20_patients["PrimaryDiagnosisSite"],
            normalize = "columns"
        ) * 100
        contingency_data_frame_of_cluster_and_diagnosis.to_csv(paths.contingency_data_frame_of_cluster_and_diagnosis)
        contingency_data_frame_of_cluster_and_diagnosis.plot(kind = "bar")
        plt.xlabel("Cluster")
        plt.ylabel("Percentage (%)")
        plt.title("Cluster Distribution by Diagnosis")
        #plt.xticks(rotation = 0)
        #plt.legend(title = "Diagnosis", bbox_to_anchor = (1.05, 1), loc = "upper left")
        plt.legend(title = "Diagnosis")
        plt.savefig(paths.plot_of_cluster_and_diagnosis)
        plt.close()
        list_of_dictionaries_of_diagnoses_features_and_statistics = []
        for diagnosis in list_of_diagnoses_with_at_least_20_patients:
            diagnosis_data = data_frame_of_clinical_data_scores_and_indices_of_clusters[
                data_frame_of_clinical_data_scores_and_indices_of_clusters["PrimaryDiagnosisSite"] == diagnosis
            ]
            for feature in ["CD8_B", "CD8_G", "CD8_GtoB_ratio", "CD8_GtoB_log"]:
                if feature in diagnosis_data.columns:
                    list_of_dictionaries_of_diagnoses_features_and_statistics.append(
                        {
                            "diagnosis": diagnosis,
                            "feature": feature,
                            "mean": diagnosis_data[feature].mean(),
                            "median": diagnosis_data[feature].median(),
                            "std": diagnosis_data[feature].std(),
                            "count": len(diagnosis_data)
                        }
                    )
        data_frame_of_diagnoses_features_and_statistics = pd.DataFrame(list_of_dictionaries_of_diagnoses_features_and_statistics)
        data_frame_of_diagnoses_features_and_statistics.to_csv(paths.data_frame_of_diagnoses_features_and_statistics, index = False)
        self.plot_CD8_group_scores_by_diagnosis(data_frame_of_diagnoses_features_and_statistics)
        
        logger.info(f"Clusters by diagnosis for {len(data_frame_of_clinical_data_scores_and_indicators_of_clusters_for_patients_with_diagnoses_with_at_least_20_patients)} patients with diagnoses with at least 20 patients were analyzed.")
        
        return data_frame_of_clinical_data_scores_and_indicators_of_clusters_for_patients_with_diagnoses_with_at_least_20_patients

    
    def plot_CD8_group_scores_by_diagnosis(self, data_frame_of_diagnoses_features_and_statistics):
        for feature in ["CD8_B", "CD8_G", "CD8_GtoB_ratio", "CD8_GtoB_log"]:
            feature_data = data_frame_of_diagnoses_features_and_statistics[
                data_frame_of_diagnoses_features_and_statistics['feature'] == feature
            ]
            if len(feature_data) == 0:
                raise Exception("")
            sns.barplot(x = "diagnosis", y = "mean", data = feature_data)
            ax = sns.barplot(x = "diagnosis", y = "mean", data = feature_data, errorbar = None)
            for x, (_, row) in enumerate(feature_data.iterrows()):
                ax.errorbar(
                    x,
                    row["mean"],
                    yerr = row["std"]
                )
            plt.xlabel("Diagnosis")
            plt.ylabel("Mean Score")
            plt.title(f"{feature} ({self.dictionary_of_names_of_features_and_labels.get(feature, "Unknown")}) by Diagnosis")
            plt.xticks(rotation = 45, ha = "right")
            plt.tight_layout()
            plt.savefig(paths.outputs_of_CD8_groups_analysis / f"{feature}_by_diagnosis.png")
            plt.close()
        
        logger.info("Plots of CD8 group scores by diagnosis were saved.")

    
    def analyze_survival_by_cluster(self, data_frame_of_scores_and_indices_of_clusters, clinical_data):
        
        logger.info("Survival by cluster will be analyzed.")
        
        data_frame_of_clinical_data_scores_and_indices_of_clusters = clinical_data.merge(
            data_frame_of_scores_and_indices_of_clusters,
            left_on = "PATIENT_ID",
            right_index = True,
            how = "inner"
        )
        data_frame_of_clinical_data_scores_and_indices_of_clusters = filter_by_primary_diagnosis_site(data_frame_of_clinical_data_scores_and_indices_of_clusters)
        if "OS_MONTHS" not in data_frame_of_clinical_data_scores_and_indices_of_clusters.columns or "OS_STATUS" not in data_frame_of_clinical_data_scores_and_indices_of_clusters.columns:
            raise Exception("Survival data not available")
        data_frame_of_clinical_data_scores_and_indices_of_clusters["event"] = (
            data_frame_of_clinical_data_scores_and_indices_of_clusters["OS_STATUS"] == "DECEASED"
        ).astype(int)
        if "CD8_GtoB_ratio" in data_frame_of_clinical_data_scores_and_indices_of_clusters.columns:
            median_ratio = data_frame_of_clinical_data_scores_and_indices_of_clusters["CD8_GtoB_ratio"].median()
            data_frame_of_clinical_data_scores_and_indices_of_clusters["CD8_GtoB_group"] = (
                data_frame_of_clinical_data_scores_and_indices_of_clusters["CD8_GtoB_ratio"] > median_ratio
            ).map({True: 'High', False: 'Low'})
            self.plot_survival_curves(data_frame_of_clinical_data_scores_and_indices_of_clusters, "CD8_GtoB_group")
        self.plot_survival_curves(data_frame_of_clinical_data_scores_and_indices_of_clusters, "cluster")
        
        logger.info(f"Survival by cluster for {len(data_frame_of_clinical_data_scores_and_indices_of_clusters)} patients was analyzed.")
        
        return data_frame_of_clinical_data_scores_and_indices_of_clusters
            
    
    def plot_survival_curves(self, data_frame_of_clinical_data_scores_and_indices_of_clusters, feature):
        Kaplan_Meier_fitter = KaplanMeierFitter()
        data_frame_of_survival_times_events_and_indices_of_clusters = (
            data_frame_of_clinical_data_scores_and_indices_of_clusters.loc[
                :,
                ["OS_MONTHS", "event", feature]
            ]
            .dropna(subset = ["OS_MONTHS", "event", feature])
            .query("OS_MONTHS > 0")
        )
        for value in sorted(data_frame_of_survival_times_events_and_indices_of_clusters[feature].unique()):
            feature_data = data_frame_of_survival_times_events_and_indices_of_clusters[
                data_frame_of_survival_times_events_and_indices_of_clusters[feature] == value
            ]
            Kaplan_Meier_fitter.fit(
                durations = feature_data["OS_MONTHS"].astype(float),
                event_observed = feature_data["event"].astype(int),
                label = f"{value} (n = {len(feature_data)})"
            )
            Kaplan_Meier_fitter.plot()
        plt.xlabel("Months")
        plt.ylabel("Survival Probability")
        plt.title(f"Kaplan Meier Survival Curves by Cluster")
        plt.grid(alpha = 0.3)
        plt.savefig(paths.outputs_of_CD8_groups_analysis / f"survival_by_{feature}.png")
        
        logger.info(f"Plot of survival curves by {feature} was saved.")
        
        plt.close()
        if len(data_frame_of_clinical_data_scores_and_indices_of_clusters[feature].unique()) == 2:
            list_of_values = sorted(data_frame_of_clinical_data_scores_and_indices_of_clusters[feature].unique())
            group1_data = data_frame_of_clinical_data_scores_and_indices_of_clusters[
                data_frame_of_clinical_data_scores_and_indices_of_clusters[feature] == list_of_values[0]
            ]
            group2_data = data_frame_of_clinical_data_scores_and_indices_of_clusters[
                data_frame_of_clinical_data_scores_and_indices_of_clusters[feature] == list_of_values[1]
            ]
            results = logrank_test(
                group1_data['OS_MONTHS'],
                group2_data['OS_MONTHS'],
                group1_data['event'],
                group2_data['event']
            )
            with open(paths.outputs_of_CD8_groups_analysis / f"results_of_log_rank_test_for_{feature}.txt", 'w') as file:
                file.write(f"Log-rank test results for {feature}:\n")
                file.write(f"Group 1: {list_of_values[0]} (n = {len(group1_data)})\n")
                file.write(f"Group 2: {list_of_values[1]} (n = {len(group2_data)})\n")
                file.write(f"p value: {results.p_value:.4f}\n")
                
            logger.info(f"Results of log rank test for {feature} were saved.")

    
    def run(self):
        '''
        Run full CD8 groups analysis.
        '''
        
        logger.info("Full CD8 group analysis will be run.")
        
        expression_matrix = load_expression_matrix()
        data_frame_of_scores = self.calculate_CD8_group_scores(expression_matrix)
        data_frame_of_scores_and_indices_of_clusters = self.cluster_samples(data_frame_of_scores)
        data_frame_of_scores_and_indices_of_clusters = map_sample_IDs_to_patient_IDs(data_frame_of_scores_and_indices_of_clusters)
        clinical_data = pd.read_csv(paths.melanoma_clinical_data)
        clinical_data = clinical_data[~clinical_data["Primary/Met"].str.contains("germline")]
        vital_status_data = pd.read_csv(paths.vital_status_data)
        clinical_data = clinical_data.merge(
            vital_status_data,
            how = "left",
            left_on = "PATIENT_ID",
            right_on = "AvatarKey"
        )
        clinical_data = calculate_survival_months(clinical_data)
        clinical_data = clinical_data.rename(columns = {"survival_months": "OS_MONTHS"})
        clinical_data["OS_STATUS"] = clinical_data["event"].map({1: "DECEASED", 0: "ALIVE"})
        self.analyze_clusters_by_sex(data_frame_of_scores_and_indices_of_clusters, clinical_data)
        self.analyze_clusters_by_diagnosis(data_frame_of_scores_and_indices_of_clusters, clinical_data)
        clinical_data.to_csv(paths.clinical_data_massaged_by_CD8_groups_analysis)
        self.analyze_survival_by_cluster(data_frame_of_scores_and_indices_of_clusters, clinical_data)
        
        logger.info("CD8 group analysis is complete.")


if __name__ == "__main__":    
    paths.ensure_dependencies_for_CD8_groups_analysis_exist()
    analysis = CD8GroupAnalysis()
    analysis.run() 