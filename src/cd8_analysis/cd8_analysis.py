"""
CD8 Analysis
Analyzes CD8+ T cell signatures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import traceback
from datetime import datetime
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from immune_analysis.immune_analysis import ImmuneAnalysis
from utils.shared_functions import load_rnaseq_data, load_clinical_data, filter_by_diagnosis

class CD8Analysis(ImmuneAnalysis):
    """Analyzes CD8+ T cell signatures"""
    
    def __init__(self, base_path):
        """Initialize the CD8 analysis"""
        super().__init__(base_path)
        
        # Define output directories
        self.cd8_dir = os.path.join(self.output_dir, 'cd8_analysis')
        self.cd8_plots_dir = os.path.join(self.cd8_dir, 'plots')
        
        # Create output directories
        os.makedirs(self.cd8_dir, exist_ok=True)
        os.makedirs(self.cd8_plots_dir, exist_ok=True)
        
        # Define CD8 signatures
        self.signatures = {
            'CD8A': ['CD8A'],
            'CD8B': ['CD8B'],
            'CD8_cytotoxic': ['GZMA', 'GZMB', 'GZMH', 'GZMK', 'PRF1', 'NKG7', 'GNLY'],
            'CD8_activation': ['CD69', 'HLA-DRA', 'HLA-DRB1', 'CD38', 'ICOS', 'CTLA4', 'LAG3', 'PDCD1'],
            'CD8_exhaustion': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'BTLA', 'VSIR'],
            'CD8_memory': ['IL7R', 'CCR7', 'SELL', 'CD27', 'CD28', 'TNFRSF9', 'EOMES'],
            'CD8_naive': ['CCR7', 'SELL', 'LEF1', 'TCF7', 'IL7R'],
            'CD8_effector': ['CX3CR1', 'FGFBP2', 'FCGR3A', 'KLRG1', 'GZMB', 'PRF1']
        }
    
    def calculate_signature_scores(self, rnaseq_data):
        """Calculate CD8 signature scores"""
        try:
            print("\nCalculating CD8 signature scores...")
            
            # Initialize scores DataFrame
            scores = pd.DataFrame(index=rnaseq_data.columns)
            
            # Calculate scores for each signature
            for signature, genes in self.signatures.items():
                # Filter to genes in the signature that are present in the data
                signature_genes = [gene for gene in genes if gene in rnaseq_data.index]
                
                if len(signature_genes) == 0:
                    print(f"Warning: No genes found for signature {signature}")
                    continue
                
                print(f"Calculating {signature} score using {len(signature_genes)} genes")
                
                # Calculate mean expression across genes
                scores[signature] = rnaseq_data.loc[signature_genes].mean()
            
            # Save scores
            scores.to_csv(os.path.join(self.cd8_dir, 'cd8_signature_scores.csv'))
            
            print(f"Calculated CD8 signature scores for {len(scores)} samples")
            
            return scores
            
        except Exception as e:
            print(f"Error calculating signature scores: {e}")
            print(traceback.format_exc())
            return None
    
    def analyze_signatures_by_sex(self, scores, clinical_data):
        """Analyze CD8 signatures by sex"""
        try:
            print("\nAnalyzing CD8 signatures by sex...")
            
            # Merge with clinical data
            merged = clinical_data.merge(
                scores,
                left_on='PATIENT_ID',
                right_index=True,
                how='inner'
            )
            
            # Filter by diagnosis
            merged = filter_by_diagnosis(merged)
            
            # Create summary statistics by sex
            summary = []
            
            for sex in ['Male', 'Female']:
                sex_data = merged[merged['SEX'] == sex]
                
                for signature in self.signatures.keys():
                    if signature in sex_data.columns:
                        summary.append({
                            'sex': sex,
                            'signature': signature,
                            'mean': sex_data[signature].mean(),
                            'median': sex_data[signature].median(),
                            'std': sex_data[signature].std(),
                            'count': len(sex_data)
                        })
            
            # Create DataFrame
            summary_df = pd.DataFrame(summary)
            
            # Save summary
            summary_df.to_csv(os.path.join(self.cd8_dir, 'cd8_by_sex.csv'), index=False)
            
            # Plot CD8 signatures by sex
            self.plot_signatures_by_sex(summary_df)
            
            # Perform statistical tests
            self.test_signatures_by_sex(merged)
            
            print(f"Analyzed CD8 signatures by sex for {len(merged)} patients")
            
            return merged
            
        except Exception as e:
            print(f"Error analyzing signatures by sex: {e}")
            print(traceback.format_exc())
            return None
    
    def plot_signatures_by_sex(self, summary_df):
        """Plot CD8 signatures by sex"""
        try:
            # Create bar plot
            plt.figure(figsize=(12, 8))
            
            # Get unique signatures
            signatures = summary_df['signature'].unique()
            n_signatures = len(signatures)
            
            # Calculate grid dimensions
            n_cols = min(3, n_signatures)
            n_rows = (n_signatures + n_cols - 1) // n_cols
            
            # Plot each signature
            for i, signature in enumerate(signatures):
                signature_data = summary_df[summary_df['signature'] == signature]
                
                if len(signature_data) == 0:
                    continue
                
                # Create subplot
                plt.subplot(n_rows, n_cols, i+1)
                
                # Create bar plot
                sns.barplot(x='sex', y='mean', data=signature_data)
                
                # Add error bars
                for j, row in signature_data.iterrows():
                    plt.errorbar(
                        x=j % 2,  # Position based on sex
                        y=row['mean'],
                        yerr=row['std'],
                        fmt='none',
                        color='black',
                        capsize=5
                    )
                
                # Add labels and title
                plt.xlabel('Sex')
                plt.ylabel('Mean Score')
                plt.title(signature)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.cd8_plots_dir, 'cd8_by_sex.png'), dpi=300)
            plt.close()
            
            print("Saved CD8 by sex plots")
            
        except Exception as e:
            print(f"Error plotting signatures by sex: {e}")
            print(traceback.format_exc())
    
    def test_signatures_by_sex(self, merged):
        """Perform statistical tests for CD8 signatures by sex"""
        try:
            # Create results list
            test_results = []
            
            # Test each signature
            for signature in self.signatures.keys():
                if signature not in merged.columns:
                    continue
                
                # Get data by sex
                male = merged[merged['SEX'] == 'Male'][signature]
                female = merged[merged['SEX'] == 'Female'][signature]
                
                # Skip if not enough samples
                if len(male) < 10 or len(female) < 10:
                    continue
                
                # Perform t-test
                t_stat, p_val = stats.ttest_ind(male, female, equal_var=False)
                
                # Add to results
                test_results.append({
                    'signature': signature,
                    'male_mean': male.mean(),
                    'female_mean': female.mean(),
                    'male_count': len(male),
                    'female_count': len(female),
                    't_stat': t_stat,
                    'p_value': p_val
                })
            
            # Create DataFrame
            test_df = pd.DataFrame(test_results)
            
            # Add significance indicator
            test_df['significant'] = test_df['p_value'] < 0.05
            
            # Save results
            test_df.to_csv(os.path.join(self.cd8_dir, 'cd8_by_sex_tests.csv'), index=False)
            
            print("Performed statistical tests for CD8 signatures by sex")
            
            return test_df
            
        except Exception as e:
            print(f"Error testing signatures by sex: {e}")
            print(traceback.format_exc())
            return None
    
    def analyze_signatures_by_diagnosis(self, scores, clinical_data):
        """Analyze CD8 signatures by diagnosis"""
        try:
            print("\nAnalyzing CD8 signatures by diagnosis...")
            
            # Merge with clinical data
            merged = clinical_data.merge(
                scores,
                left_on='PATIENT_ID',
                right_index=True,
                how='inner'
            )
            
            # Get top diagnoses
            diagnosis_counts = merged['DIAGNOSIS'].value_counts()
            top_diagnoses = diagnosis_counts[diagnosis_counts >= 20].index.tolist()
            
            if len(top_diagnoses) == 0:
                print("Warning: No diagnoses with at least 20 patients")
                return None
            
            # Filter to top diagnoses
            merged_top = merged[merged['DIAGNOSIS'].isin(top_diagnoses)]
            
            # Create summary statistics by diagnosis
            summary = []
            
            for diagnosis in top_diagnoses:
                diag_data = merged[merged['DIAGNOSIS'] == diagnosis]
                
                for signature in self.signatures.keys():
                    if signature in diag_data.columns:
                        summary.append({
                            'diagnosis': diagnosis,
                            'signature': signature,
                            'mean': diag_data[signature].mean(),
                            'median': diag_data[signature].median(),
                            'std': diag_data[signature].std(),
                            'count': len(diag_data)
                        })
            
            # Create DataFrame
            summary_df = pd.DataFrame(summary)
            
            # Save summary
            summary_df.to_csv(os.path.join(self.cd8_dir, 'cd8_by_diagnosis.csv'), index=False)
            
            # Plot CD8 signatures by diagnosis
            self.plot_signatures_by_diagnosis(summary_df)
            
            print(f"Analyzed CD8 signatures by diagnosis for {len(merged_top)} patients with top diagnoses")
            
            return merged_top
            
        except Exception as e:
            print(f"Error analyzing signatures by diagnosis: {e}")
            print(traceback.format_exc())
            return None
    
    def plot_signatures_by_diagnosis(self, summary_df):
        """Plot CD8 signatures by diagnosis"""
        try:
            # Get unique signatures
            signatures = summary_df['signature'].unique()
            
            # Plot each signature
            for signature in signatures:
                signature_data = summary_df[summary_df['signature'] == signature]
                
                if len(signature_data) == 0:
                    continue
                
                # Create bar plot
                plt.figure(figsize=(12, 6))
                sns.barplot(x='diagnosis', y='mean', data=signature_data)
                
                # Add error bars
                for i, row in signature_data.iterrows():
                    plt.errorbar(
                        x=i % len(signature_data['diagnosis'].unique()),  # Position based on diagnosis
                        y=row['mean'],
                        yerr=row['std'],
                        fmt='none',
                        color='black',
                        capsize=5
                    )
                
                # Add labels and title
                plt.xlabel('Diagnosis')
                plt.ylabel('Mean Score')
                plt.title(f'{signature} by Diagnosis')
                plt.xticks(rotation=45, ha='right')
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(self.cd8_plots_dir, f'{signature}_by_diagnosis.png'), dpi=300)
                plt.close()
            
            print("Saved CD8 by diagnosis plots")
            
        except Exception as e:
            print(f"Error plotting signatures by diagnosis: {e}")
            print(traceback.format_exc())
    
    def analyze_survival_by_signature(self, scores, clinical_data):
        """Analyze survival by CD8 signature"""
        try:
            print("\nAnalyzing survival by CD8 signature...")
            
            # Merge with clinical data
            merged = clinical_data.merge(
                scores,
                left_on='PATIENT_ID',
                right_index=True,
                how='inner'
            )
            
            # Filter by diagnosis
            merged = filter_by_diagnosis(merged)
            
            # Check if survival data is available
            if 'OS_MONTHS' not in merged.columns or 'OS_STATUS' not in merged.columns:
                print("Warning: Survival data not available")
                return None
            
            # Create survival status indicator (1 for death, 0 for censored)
            merged['event'] = (merged['OS_STATUS'] == 'DECEASED').astype(int)
            
            # Analyze each signature
            for signature in self.signatures.keys():
                if signature not in merged.columns:
                    continue
                
                # Group by median
                median = merged[signature].median()
                merged[f'{signature}_group'] = (merged[signature] > median).map({True: 'High', False: 'Low'})
                
                # Plot Kaplan-Meier curves
                self.plot_survival_curves(merged, f'{signature}_group', signature)
            
            print(f"Analyzed survival by CD8 signature for {len(merged)} patients")
            
            return merged
            
        except Exception as e:
            print(f"Error analyzing survival by signature: {e}")
            print(traceback.format_exc())
            return None
    
    def plot_survival_curves(self, merged, group_col, title=None):
        """Plot Kaplan-Meier survival curves by group"""
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
            if title:
                plt.savefig(os.path.join(self.cd8_plots_dir, f'survival_by_{title}.png'), dpi=300)
            else:
                plt.savefig(os.path.join(self.cd8_plots_dir, f'survival_by_{group_col}.png'), dpi=300)
            plt.close()
            
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
                output_file = os.path.join(self.cd8_dir, f'logrank_{group_col}.txt')
                if title:
                    output_file = os.path.join(self.cd8_dir, f'logrank_{title}.txt')
                    
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
        """Run full CD8 analysis"""
        try:
            print("\nRunning full CD8 analysis...")
            
            # Load RNA-seq data
            rnaseq_data = load_rnaseq_data(self.base_path)
            if rnaseq_data is None:
                return None
            
            # Calculate CD8 signature scores
            scores = self.calculate_signature_scores(rnaseq_data)
            if scores is None:
                return None
            
            # Load clinical data
            clinical_data = load_clinical_data(self.base_path)
            if clinical_data is None:
                return None
            
            # Analyze signatures by sex
            self.analyze_signatures_by_sex(scores, clinical_data)
            
            # Analyze signatures by diagnosis
            self.analyze_signatures_by_diagnosis(scores, clinical_data)
            
            # Analyze survival by signature
            self.analyze_survival_by_signature(scores, clinical_data)
            
            print("\nCD8 analysis complete!")
            print(f"Results saved to {self.cd8_dir}")
            print(f"Plots saved to {self.cd8_plots_dir}")
            
            return scores
            
        except Exception as e:
            print(f"Error running full analysis: {e}")
            print(traceback.format_exc())
            return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CD8 Analysis')
    parser.add_argument('--base-path', type=str, default='/project/orien/data/aws/24PRJ217UVA_IORIG',
                        help='Base path for data files')
    args = parser.parse_args()
    
    analysis = CD8Analysis(args.base_path)
    analysis.run_full_analysis() 