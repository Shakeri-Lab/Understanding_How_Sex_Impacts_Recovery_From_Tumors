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

from src.immune_analysis.immune_analysis import ImmuneAnalysis
from src.utils.shared_functions import load_rnaseq_data, filter_by_diagnosis, map_sample_ids
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
        
        self.cd8_groups = {
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
            ],
        }
    
    
    def calculate_signature_scores(self, rnaseq_data):
        '''
        Calculate CD8 signature scores.
        '''
        print("CD8 signature scores will be calculated.")

        rnaseq = rnaseq_data.copy()
        if rnaseq.index.astype(str).str.startswith("ENSG").any():
            rnaseq.index = rnaseq.index.astype(str).str.split(".").str[0]
            rnaseq = rnaseq.groupby(rnaseq.index).mean()
        
        scores = pd.DataFrame(index = rnaseq.columns)
        
        # Calculate scores for each signature.
        for signature, genes in self.cd8_groups.items():
            # Filter to genes in the signature that are present in the data
            signature_genes = [gene for gene in genes if gene in rnaseq.index]

            if not signature_genes:
                print(f"Warning: No genes found for signature {signature}")
                continue
            
            print(f"Calculating {signature} score using {len(signature_genes)} genes")

            # Calculate mean expression across genes
            scores[signature] = rnaseq.loc[signature_genes].mean()

        scores = map_sample_ids(scores)
            
        # Save scores
        scores.to_csv(paths.data_frame_of_patient_IDs_and_CD8_signature_scores)

        print(f"Calculated CD8 signature scores for {len(scores)} samples")

        return scores
    
    
    def analyze_signatures_by_sex(self, scores, clinical_data):
        print(
            "CD8 signatures will be analyzed by sex.\n" +
            f"The shape of scores is {scores.shape}.\n" +
            f"The shape of clinical data is {clinical_data.shape}."
        )

        # Merge with clinical data
        merged = clinical_data.merge(
            scores,
            left_on = "PATIENT_ID",
            right_index = True,
            how = "inner"
        )

        print(f"The shape of clinical data inner joined with scores is {merged.shape}.")

        merged = filter_by_diagnosis(merged)

        # Create summary statistics by sex
        summary = []

        for sex in ['Male', 'Female']:
            sex_data = merged[merged['Sex'] == sex]

            for signature in self.cd8_groups.keys():
                if signature in sex_data.columns:
                    summary.append({
                        'sex': sex,
                        'signature': signature,
                        'mean': sex_data[signature].mean(),
                        'median': sex_data[signature].median(),
                        'std': sex_data[signature].std(),
                        'count': len(sex_data)
                    })

        summary_df = pd.DataFrame(summary)
        print("Data frame of sexes, CD8 signatures, and statistics will be saved.")
        summary_df.to_csv(paths.data_frame_of_sexes_CD8_signatures_and_statistics, index = False)

        self.plot_signatures_by_sex(summary_df)
        self.test_signatures_by_sex(merged)

        print(f"Analyzed CD8 signatures by sex for {len(merged)} patients")

        return merged
    
    
    def plot_signatures_by_sex(self, summary_df):
        '''
        Plot mean CD8 signatures by sex.
        '''
        g = sns.catplot(
            data = summary_df,
            x = "sex",
            y = "mean",
            col = "signature",
            kind = "bar",
            col_wrap = 3,
            height = 3,
            aspect = 1
        )
        g.set_titles('{col_name}')
        g.set_xlabels('Sex')
        g.set_ylabels('Mean Score')
        g.figure.savefig(paths.plot_of_mean_CD8_signature_scores_by_sex, dpi = 300, bbox_inches = "tight")
        plt.close(g.figure)
        logger.info("Plot of mean CD8 signature scores by sex was saved.")
    
    
    def test_signatures_by_sex(self, merged):
        '''
        Perform statistical tests for CD8 signatures by sex.
        '''
        test_results = []
        
        # Test each signature.
        for signature in self.cd8_groups.keys():
            if signature not in merged.columns:
                raise Exception(f"Signature {signature} is not a label of a column.")
            
            # Get data by sex.
            male = merged[merged['Sex'] == 'Male'][signature]
            female = merged[merged['Sex'] == 'Female'][signature]
            
            if len(male) < 10 or len(female) < 10:
                raise Exception(f"There are not enough samples for signature {signature}.")
            
            # Perform t-test
            t_stat, p_val = stats.ttest_ind(male, female, equal_var = False)
            
            # Add to results
            test_results.append(
                {
                    'signature': signature,
                    'male_mean': male.mean(),
                    'female_mean': female.mean(),
                    'male_count': len(male),
                    'female_count': len(female),
                    't_stat': t_stat,
                    'p_value': p_val
                }
            )
        
        test_df = pd.DataFrame(test_results)
        test_df['significant'] = test_df['p_value'] < 0.05
        test_df.to_csv(paths.data_frame_of_CD8_signatures_and_statistics, index = False)
        
        print("Statistical tests for CD8 signatures by sex were performed.")
        
        return test_df
    
    
    def analyze_signatures_by_diagnosis(self, scores, clinical_data):
        '''
        Analyze CD8 signatures by diagnosis.
        '''
        print("\nAnalyzing CD8 signatures by diagnosis...")

        merged = clinical_data.merge(
            scores,
            left_on = 'PATIENT_ID',
            right_index = True,
            how = 'inner'
        )

        diagnosis_counts = merged['PrimaryDiagnosisSite'].value_counts()
        top_diagnoses = diagnosis_counts[diagnosis_counts >= 20].index.tolist()

        if len(top_diagnoses) == 0:
            raise Exception("There are no diagnoses with at least 20 patients.")

        merged_top = merged[merged['PrimaryDiagnosisSite'].isin(top_diagnoses)].copy()
        diagnosis_order = merged_top["PrimaryDiagnosisSite"].value_counts().sort_values(ascending = False).index.tolist()

        summary_rows = []

        for diagnosis in diagnosis_order:
            diag_data = merged_top[merged_top["PrimaryDiagnosisSite"] == diagnosis]
            for signature in self.cd8_groups.keys():
                if signature in diag_data.columns:
                    summary_rows.append(
                        {
                            'diagnosis': diagnosis,
                            'signature': signature,
                            'mean': diag_data[signature].mean(),
                            'median': diag_data[signature].median(),
                            'std': diag_data[signature].std(),
                            'count': len(diag_data)
                        }
                    )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(paths.data_frame_of_top_diagnoses_CD8_signatures_and_statistics, index = False)

        wrapped = [textwrap.fill(str(d), width = 18) for d in diagnosis_order]
        wrap_map = dict(zip(diagnosis_order, wrapped))
        summary_df["diagnosis_wrapped"] = summary_df["diagnosis"].map(wrap_map)
        summary_df["diagnosis_wrapped"] = pd.Categorical(summary_df["diagnosis_wrapped"], categories = wrapped, ordered = True)
        self.plot_signatures_by_diagnosis(summary_df)

        print(f"CD8 signatures by diagnosis for {len(merged_top)} patients with top diagnoses were analyzed.")

        return merged_top
    
    
    def plot_signatures_by_diagnosis(self, summary_df):
        '''
        Plot CD8 signatures by diagnosis.
        '''
        g = sns.catplot(
            data = summary_df,
            x = "diagnosis_wrapped",
            y = "mean",
            col = "signature",
            kind = "bar",
            order = list(summary_df["diagnosis_wrapped"].cat.categories),
            col_wrap = 3,
            height = 3.2,
            aspect = 1.5,
            errorbar = None,
            sharex = False
        )
        g.set_titles("{col_name}")
        for ax in g.axes.flatten():
            ax.set_xlabel("Diagnosis")
            ax.set_ylabel("Mean Score")
            for label in ax.get_xticklabels():
                label.set_rotation(0)
                label.set_ha("center")
        g.figure.subplots_adjust(bottom = 0.25, top = 0.9)
        g.figure.savefig(paths.plot_of_mean_CD8_signature_scores_by_diagnosis, dpi = 300, bbox_inches = "tight")
        plt.close(g.figure)
        logger.info('Plot of mean CD8 signature scores by diagnosis was saved.')
    
    
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
        
        merged = filter_by_diagnosis(merged)
        
        # Check if survival data is available
        if 'OS_MONTHS' not in merged.columns or 'OS_STATUS' not in merged.columns:
            print("Warning: Survival data not available")
            return None
        
        # Create survival status indicator (1 for death, 0 for censored)
        merged['event'] = (merged['OS_STATUS'] == 'DECEASED').astype(int)
        
        # Analyze each signature
        for signature in self.cd8_groups.keys():
            if signature not in merged.columns:
                continue
            
            # Group by median
            median = merged[signature].median()
            merged[f'{signature}_group'] = (merged[signature] > median).map({True: 'High', False: 'Low'})
            
            # Plot Kaplan-Meier curves
            self.plot_survival_curves(merged, f'{signature}_group', signature)
        
        print(f"Analyzed survival by CD8 signature for {len(merged)} patients")
        
        return merged
    
    
    def plot_survival_curves(self, merged, group_col, title = None):
        '''
        Plot Kaplan-Meier survival curves by group
        '''
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Initialize Kaplan-Meier fitter
            kmf = KaplanMeierFitter()
            
            # Plot survival curve for each group
            for group in sorted(merged[group_col].unique()):
                group_data = merged[merged[group_col] == group]
                
                # Skip if not enough samples
                if len(group_data) < 10:
                    continue
                
                # Fit survival curve
                kmf.fit(
                    group_data['OS_MONTHS'],
                    group_data['event'],
                    label=f'{group} (n={len(group_data)})'
                )
                
                # Plot survival curve
                kmf.plot()
            
            # Add labels and title
            plt.xlabel('Months')
            plt.ylabel('Survival Probability')
            if title:
                plt.title(f'Kaplan-Meier Survival Curves by {title}')
            else:
                plt.title(f'Kaplan-Meier Survival Curves by {group_col}')
            
            # Add grid
            plt.grid(alpha=0.3)
            
            # Save plot
            plt.tight_layout()
            out_name = f'survival_by_{title or group_col}.png'
            plt.savefig(paths.cd8_analysis_plots / out_name, dpi = 300, bbox_inches = "tight")
            
            # Perform log-rank test if there are exactly 2 groups
            if len(merged[group_col].unique()) == 2:
                groups = sorted(merged[group_col].unique())
                group1_data = merged[merged[group_col] == groups[0]]
                group2_data = merged[merged[group_col] == groups[1]]
                
                results = logrank_test(
                    group1_data['OS_MONTHS'],
                    group2_data['OS_MONTHS'],
                    group1_data['event'],
                    group2_data['event']
                )
                
                print(f"Log-rank test p-value: {results.p_value:.4f}")
                
                # Save results
                output_file = os.path.join(paths.outputs_of_cd8_analysis, f'logrank_{group_col}.txt')
                if title:
                    output_file = os.path.join(paths.outputs_of_cd8_analysis, f'logrank_{title}.txt')
                    
                with open(output_file, 'w') as f:
                    f.write(f"Log-rank test results for {group_col}:\n")
                    f.write(f"Group 1: {groups[0]} (n={len(group1_data)})\n")
                    f.write(f"Group 2: {groups[1]} (n={len(group2_data)})\n")
                    f.write(f"p-value: {results.p_value:.4f}\n")
            
            if title:
                print(f"Saved survival curves by {title}")
            else:
                print(f"Saved survival curves by {group_col}")
            
        except Exception as e:
            print(f"Error plotting survival curves: {e}")
            print(traceback.format_exc())
    
    
    def run_full_analysis(self):
        print("\nFull CD8 analysis will be run.")

        rnaseq_data = load_rnaseq_data()
        scores = self.calculate_signature_scores(rnaseq_data)

        clinical_data = pd.read_csv(paths.melanoma_clinical_data)
        clinical_data = clinical_data[~clinical_data["Primary/Met"].str.contains("germline")]
        
        self.analyze_signatures_by_sex(scores, clinical_data)
        self.analyze_signatures_by_diagnosis(scores, clinical_data)
        self.analyze_survival_by_signature(scores, clinical_data)

        print("\nCD8 analysis complete!")
        print(f"Results were saved to {paths.outputs_of_cd8_analysis}")
        print(f"Plots were saved to {paths.cd8_analysis_plots}")

        return scores


if __name__ == "__main__":
    
    paths.ensure_dependencies_for_cd8_analysis_exist()
    
    analysis = CD8Analysis()
    analysis.run_full_analysis()