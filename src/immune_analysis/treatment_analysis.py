import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from lifelines import KaplanMeierFitter
from scipy import stats
from immune_analysis import ImmuneAnalysis
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.api import OLS
from lifelines import CoxPHFitter
import warnings
import traceback

class TreatmentResponseAnalysis(ImmuneAnalysis):
    """Analyze immune composition vs treatment outcomes"""
    
    def __init__(self, base_path="/project/orien/data/aws/24PRJ217UVA_IORIG"):
        # Initialize parent class
        super().__init__()
        
        # Store base path
        self.base_path = base_path
        
        # Create plots directory
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        print(f"Plots will be saved to: {self.plots_dir}")
        
        # Load additional clinical data
        clinical_path = os.path.join(base_path, 'Clinical_Data/24PRJ217UVA_NormalizedFiles')
        self.medications = pd.read_csv(os.path.join(clinical_path, '24PRJ217UVA_20241112_Medications_V4.csv'))
        self.outcomes = pd.read_csv(os.path.join(clinical_path, '24PRJ217UVA_20241112_Outcomes_V4.csv'))
        self.vital = pd.read_csv(os.path.join(clinical_path, '24PRJ217UVA_20241112_VitalStatus_V4.csv'))
        
        # Process treatment data
        self._process_treatment_data()
    
    def _process_treatment_data(self):
        """Process and merge treatment-related data"""
        def convert_age(age_str):
            """Convert age string to numeric value"""
            if pd.isna(age_str):
                return np.nan
            if isinstance(age_str, (int, float)):
                return float(age_str)
            if isinstance(age_str, str):
                if 'Age 90 or older' in age_str:
                    return 90.0
                try:
                    return float(age_str)
                except ValueError:
                    return np.nan
            return np.nan
        
        # Process medications
        immunotherapy_keywords = [
            'pembrolizumab', 'nivolumab', 'atezolizumab', 'durvalumab',
            'ipilimumab', 'avelumab', 'cemiplimab'
        ]
        
        # Flag immunotherapy treatments
        self.medications['is_immunotherapy'] = self.medications['Medication'].str.lower().apply(
            lambda x: any(drug in str(x).lower() for drug in immunotherapy_keywords)
        )
        
        # Get first immunotherapy date per patient
        immuno_starts = (self.medications[self.medications['is_immunotherapy']]
                        .groupby('AvatarKey')
                        .agg({'YearOfMedStart': 'min'})
                        .rename(columns={'YearOfMedStart': 'immunotherapy_start'}))
        
        # Merge with main data
        self.data = self.data.merge(
            immuno_starts, 
            left_on='PATIENT_ID', 
            right_index=True, 
            how='left'
        )
        
        # Process outcomes and map responses
        response_map = {
            'Complete Response': 'CR',
            'Partial Response': 'PR',
            'Stable Disease': 'SD',
            'Progressive Disease': 'PD'
        }
        
        # Get best response
        best_response = self.outcomes.groupby('AvatarKey')['SolidTumorResponse'].first()
        
        # Print response distribution before merge
        print("\nResponse distribution before merge:")
        print(best_response.value_counts())
        
        self.data = self.data.merge(
            best_response.to_frame('best_response'),
            left_on='PATIENT_ID',
            right_index=True,
            how='left'
        )
        
        # Print response distribution after merge
        print("\nResponse distribution after merge:")
        print(self.data['best_response'].value_counts())
        print("Missing responses:", self.data['best_response'].isna().sum())
        
        # Process survival data
        vital_data = self.vital[['AvatarKey', 'VitalStatus', 'AgeAtLastContact', 'AgeAtDeath']].copy()
        # Convert age columns to numeric
        vital_data['AgeAtLastContact'] = vital_data['AgeAtLastContact'].apply(convert_age)
        vital_data['AgeAtDeath'] = vital_data['AgeAtDeath'].apply(convert_age)
        
        self.data = self.data.merge(
            vital_data,
            left_on='PATIENT_ID',
            right_on='AvatarKey',
            how='left'
        )
        
        # Print data quality check
        print("\nData quality check:")
        print("Number of patients:", len(self.data))
        print("Missing age at last contact:", self.data['AgeAtLastContact'].isna().sum())
        print("Missing age at death:", self.data['AgeAtDeath'].isna().sum())
        print("Number of death events:", (self.data['VitalStatus'] == 'Dead').sum())
        
        # Calculate survival time
        self.data['survival_time'] = self.data.apply(
            lambda x: x['AgeAtDeath'] if pd.notna(x['AgeAtDeath']) 
            else x['AgeAtLastContact'], axis=1
        ) - self.data['AGE']
        
        self.data['death_event'] = self.data['VitalStatus'] == 'Dead'
        
        # Print survival data summary
        print("\nSurvival data summary:")
        print(self.data[['survival_time', 'death_event']].describe())
        print("\nNumber of events:", self.data['death_event'].sum())
    
    def analyze_response_patterns(self):
        """Analyze immune patterns by treatment response"""
        response_groups = self.data.groupby('best_response')
        
        results = []
        for cell_type in self.immune_cols:
            # Kruskal-Wallis test across response groups
            cell_values = [group[cell_type].values for name, group in response_groups]
            h_stat, p_val = stats.kruskal(*cell_values)
            
            results.append({
                'cell_type': cell_type,
                'cell_type_clean': self._clean_cell_type_name(cell_type),
                'h_statistic': h_stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'bonferroni_significant': p_val < self.bonferroni_threshold
            })
        
        return pd.DataFrame(results).sort_values('p_value')
    
    def plot_response_patterns(self, cell_type):
        """Plot immune cell levels by response category"""
        print(f"\nPlotting response patterns for {cell_type}")
        
        # Define response label mapping and colors
        response_info = {
            'Complete Response/Remission (CR)': {'label': 'CR', 'color': '#2ecc71'},
            'Partial Response (PR)': {'label': 'PR', 'color': '#3498db'},
            'No Response (NR)/Stable Disease (SD)': {'label': 'SD', 'color': '#f1c40f'},
            'Progressive Disease (Prog/PD)': {'label': 'PD', 'color': '#e74c3c'},
            'Unknown/Not Applicable': {'label': 'NA', 'color': '#95a5a6'},
            'Not Assessed/Unknown': {'label': 'NA', 'color': '#95a5a6'},
            'Objective Response (OR)': {'label': 'OR', 'color': '#9b59b6'}
        }
        
        # Create a copy of data with mapped responses
        plot_data = self.data.copy()
        # Map to simplified categories
        plot_data['response_category'] = plot_data['best_response'].map(
            {k: v['label'] for k, v in response_info.items()}
        ).fillna('NA')
        
        # Get response counts
        response_counts = plot_data['response_category'].value_counts()
        
        if len(response_counts) < 2:
            print("Warning: Not enough response categories to create plot")
            return None
        
        # Print response groups for this cell type
        print("\nResponse group sizes:")
        for response, count in response_counts.items():
            print(f"{response}: {count} patients")
        
        plt.figure(figsize=(12, 7))
        
        # Fix: Create color mapping for present categories
        category_colors = {}
        for response_cat in response_counts.index:
            for orig, info in response_info.items():
                if info['label'] == response_cat:
                    category_colors[response_cat] = info['color']
                    break
        
        # Fix seaborn warning by explicitly setting hue
        ax = sns.boxplot(data=plot_data, x='response_category', y=cell_type,
                        hue='response_category', palette=category_colors, legend=False)
        
        clean_name = self._clean_cell_type_name(cell_type)
        plt.title(f'{clean_name} by Treatment Response')
        
        # Update x-axis labels with counts
        current_labels = ax.get_xticklabels()
        new_labels = [f"{label.get_text()}\n(n={response_counts[label.get_text()]})"
                     for label in current_labels]
        
        # Fix ticklabels warning by setting ticks first
        ax.set_xticks(range(len(new_labels)))
        ax.set_xticklabels(new_labels, rotation=45, ha='right')
        
        plt.xlabel('Treatment Response')
        plt.ylabel('Cell Score')
        
        # Perform Kruskal-Wallis test
        h_stat, p_val = stats.kruskal(*[
            group[cell_type].values
            for name, group in plot_data.groupby('response_category')
        ])
        
        # Add multiple testing correction note
        n_groups = len(response_counts)
        adjusted_alpha = 0.05 / (n_groups * (n_groups-1)/2)
        
        # Fix: Create legend elements using category_colors directly
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=color, 
                         label=response_cat)
            for response_cat, color in category_colors.items()
        ]
        
        ax.legend(handles=legend_elements, title='Response Categories',
                 loc='lower right', bbox_to_anchor=(0.98, 0.02))
        
        # Add statistical annotation
        plt.text(0.5, 0.95, 
                f'Kruskal-Wallis p = {p_val:.2e}\n' +
                f'(Bonferroni threshold: {adjusted_alpha:.2e})',
                transform=plt.gca().transAxes,
                horizontalalignment='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Adjust layout
        plt.tight_layout()
        
        output_file = os.path.join(self.plots_dir, f'{cell_type}_response_pattern.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved plot to: {output_file}")
        return output_file
    
    def analyze_survival(self, cell_type, cutpoint='median'):
        """Analyze survival based on immune cell levels"""
        if cutpoint == 'median':
            threshold = self.data[cell_type].median()
        
        # Split patients by cell level
        high_group = self.data[self.data[cell_type] > threshold]
        low_group = self.data[self.data[cell_type] <= threshold]
        
        # Fit KM curves
        kmf = KaplanMeierFitter()
        
        plt.figure(figsize=(10, 6))
        
        # Plot high group
        kmf.fit(high_group['survival_time'], 
                high_group['death_event'],
                label=f'High {self._clean_cell_type_name(cell_type)}')
        kmf.plot()
        
        # Plot low group
        kmf.fit(low_group['survival_time'],
                low_group['death_event'],
                label=f'Low {self._clean_cell_type_name(cell_type)}')
        kmf.plot()
        
        # Add log-rank test
        from lifelines.statistics import logrank_test
        log_rank = logrank_test(high_group['survival_time'], low_group['survival_time'],
                              high_group['death_event'], low_group['death_event'])
        
        plt.text(0.05, 0.95, f'Log-rank p = {log_rank.p_value:.2e}',
                transform=plt.gca().transAxes)
        
        plt.title(f'Survival by {self._clean_cell_type_name(cell_type)} Level')
        plt.xlabel('Time (years)')
        plt.ylabel('Survival Probability')
        
        output_file = os.path.join(self.plots_dir, f'{cell_type}_survival.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        return log_rank.p_value

    def analyze_drug_specific_patterns(self):
        """Analyze immune patterns by specific immunotherapy drug"""
        # Get drug-specific treatment groups
        drug_groups = self.medications[self.medications['is_immunotherapy']].groupby('Medication')
        
        results = []
        for drug, drug_data in drug_groups:
            # Get patients on this drug
            drug_patients = drug_data['AvatarKey'].unique()
            
            # Compare immune profiles
            for cell_type in self.immune_cols:
                # Compare treated vs untreated
                treated = self.data[self.data['PATIENT_ID'].isin(drug_patients)][cell_type]
                untreated = self.data[~self.data['PATIENT_ID'].isin(drug_patients)][cell_type]
                
                # Statistical test
                stat, pval = stats.mannwhitneyu(treated, untreated)
                
                results.append({
                    'drug': drug,
                    'cell_type': cell_type,
                    'p_value': pval,
                    'effect_size': (treated.median() - untreated.median()) / untreated.std()
                })
        
        return pd.DataFrame(results)

    def analyze_combination_therapy(self):
        """Analyze impact of combination therapy vs monotherapy"""
        # Identify patients on combination therapy
        patient_drugs = self.medications.groupby('AvatarKey')['Medication'].nunique()
        combo_patients = patient_drugs[patient_drugs > 1].index
        
        results = []
        for cell_type in self.immune_cols:
            # Compare mono vs combo therapy
            combo = self.data[self.data['PATIENT_ID'].isin(combo_patients)][cell_type]
            mono = self.data[~self.data['PATIENT_ID'].isin(combo_patients)][cell_type]
            
            stat, pval = stats.mannwhitneyu(combo, mono)
            results.append({
                'cell_type': cell_type,
                'p_value': pval,
                'effect_size': (combo.median() - mono.median()) / mono.std()
            })
        
        return pd.DataFrame(results)

    def analyze_temporal_patterns(self):
        """Analyze changes in immune composition over time"""
        # Get treatment timeline
        treatment_timeline = self.medications[['AvatarKey', 'YearOfMedStart', 'YearOfMedStop']]
        
        # Compare pre vs post treatment immune profiles
        # This would require temporal immune data
        # Could be implemented if we have multiple timepoints

    def _load_clinical_data(self):
        """Load and process clinical data"""
        try:
            clinical_path = os.path.join(self.base_path, 'Clinical_Data/24PRJ217UVA_NormalizedFiles')
            
            # Load clinical files
            self.diagnosis = pd.read_csv(os.path.join(clinical_path, '24PRJ217UVA_20241112_Diagnosis_V4.csv'))
            self.medications = pd.read_csv(os.path.join(clinical_path, '24PRJ217UVA_20241112_Medications_V4.csv'))
            self.outcomes = pd.read_csv(os.path.join(clinical_path, '24PRJ217UVA_20241112_Outcomes_V4.csv'))
            self.vital = pd.read_csv(os.path.join(clinical_path, '24PRJ217UVA_20241112_VitalStatus_V4.csv'))
            
            # Print sex distribution before processing
            print("\nSex distribution before processing:")
            print(self.data['SEX'].value_counts())
            
            # Clean sex coding
            sex_map = {
                'Male': 'M',
                'Female': 'F',
                'MALE': 'M',
                'FEMALE': 'F',
                'male': 'M',
                'female': 'F',
                'M': 'M',
                'F': 'F'
            }
            self.data['SEX'] = self.data['SEX'].map(sex_map)
            
            # Print sex distribution after processing
            print("\nSex distribution after processing:")
            print(self.data['SEX'].value_counts())
            
            # Validate sex coding
            invalid_sex = self.data[~self.data['SEX'].isin(['M', 'F'])]
            if len(invalid_sex) > 0:
                print("\nWarning: Invalid sex values found:")
                print(invalid_sex['SEX'].value_counts())
                
                # Drop invalid sex values
                self.data = self.data[self.data['SEX'].isin(['M', 'F'])]
                print("\nRows with invalid sex values dropped")
                print("Final sex distribution:")
                print(self.data['SEX'].value_counts())
            
            # Rest of the clinical data processing...
            # Process stage information from diagnosis
            stage_data = self.diagnosis[[
                'AvatarKey',
                'ClinTStage',  # Clinical T stage
                'ClinNStage',  # Clinical N stage
                'ClinMStage',  # Clinical M stage
                'ClinGroupStage',  # Clinical group stage
                'PathTStage',  # Pathological T stage
                'PathNStage',  # Pathological N stage
                'PathMStage',  # Pathological M stage
                'PathGroupStage',  # Pathological group stage
                'AgeAtDiagnosis',
                'YearOfDiagnosis'
            ]].copy()
            
            # Use pathological stage if available, otherwise clinical stage
            stage_data['stage'] = stage_data['PathGroupStage'].fillna(stage_data['ClinGroupStage'])
            
            # Process medication/treatment information
            med_data = self.medications.groupby('AvatarKey').agg({
                'MedLineRegimen': lambda x: min(x.astype(str).fillna('999')),  # Get earliest line
                'Medication': lambda x: ', '.join(x.dropna().unique()),  # All medications
                'AgeAtMedStart': 'min',  # Age at first treatment
                'YearOfMedStart': 'min'  # Year of first treatment
            })
            
            # Create prior treatment flag (before current line)
            med_data['prior_treatment'] = med_data['MedLineRegimen'].apply(
                lambda x: 'Yes' if str(x).strip() != '1' and str(x).strip() != '999' else 'No'
            )
            
            # Process outcome information using correct column names
            outcome_data = self.outcomes.groupby('AvatarKey').agg({
                'SolidTumorResponse': 'first',  # Best response
                'ProgRecurInd': 'max',  # Any progression/recurrence
                'CurrentDiseaseStatus': 'last',  # Latest disease status
                'PerformStatusMostRecent': 'last'  # Latest performance status
            })
            
            # Process vital status
            vital_data = self.vital[[
                'AvatarKey', 
                'VitalStatus',
                'AgeAtLastContact',
                'AgeAtDeath',
                'CauseOfDeath'
            ]].copy()
            
            # Calculate survival time
            vital_data['survival_time'] = vital_data.apply(
                lambda x: x['AgeAtDeath'] if pd.notna(x['AgeAtDeath']) 
                else x['AgeAtLastContact'], axis=1
            )
            
            # Print column names for debugging
            print("\nAvailable columns in diagnosis data:")
            print(self.diagnosis.columns.tolist())
            print("\nAvailable columns in medications data:")
            print(self.medications.columns.tolist())
            
            # Merge all clinical data
            clinical_merged = (
                stage_data
                .merge(med_data, left_on='AvatarKey', right_index=True, how='left')
                .merge(outcome_data, left_on='AvatarKey', right_index=True, how='left')
                .merge(vital_data, on='AvatarKey', how='left')
            )
            
            # Merge with existing data
            self.data = self.data.merge(
                clinical_merged, 
                left_on='PATIENT_ID',
                right_on='AvatarKey',
                how='left'
            )
            
            # Fill missing values
            self.data['prior_treatment'] = self.data['prior_treatment'].fillna('No')
            self.data['stage'] = self.data['stage'].fillna('Unknown')
            
            # Print summary statistics
            print("\nClinical Data Summary:")
            print(f"Total patients: {len(self.data)}")
            print("\nStage distribution:")
            print(self.data['stage'].value_counts())
            print("\nResponse distribution:")
            print(self.data['SolidTumorResponse'].value_counts())
            print("\nVital status distribution:")
            print(self.data['VitalStatus'].value_counts())
            print("\nPrior treatment distribution:")
            print(self.data['prior_treatment'].value_counts())
            
            # Validate key variables
            missing_rates = {
                'stage': self.data['stage'].isna().mean(),
                'survival_time': self.data['survival_time'].isna().mean(),
                'SolidTumorResponse': self.data['SolidTumorResponse'].isna().mean(),
                'prior_treatment': self.data['prior_treatment'].isna().mean()
            }
            
            print("\nMissing data rates:")
            for var, rate in missing_rates.items():
                print(f"{var}: {rate:.1%}")
            
            # Print final data quality check
            print("\nFinal Data Quality Check:")
            print(f"Total patients: {len(self.data)}")
            print("\nSex distribution:")
            print(self.data['SEX'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
            
        except Exception as e:
            print(f"Error loading clinical data: {e}")
            print(traceback.format_exc())
            
    def plot_drug_specific_patterns(self, significant_only=True):
        """Visualize immune patterns by drug"""
        
    def plot_temporal_changes(self, cell_type):
        """Plot immune cell changes over treatment timeline"""

    def test_response_prediction(self, features='immune'):
        """Test predictive value of immune features for response"""

    def analyze_sex_survival_mediation(self):
        """Analyze mediation of sex effects through immune features"""
        try:
            print("\nPerforming mediation analysis...")
            
            # Get immune features
            immune_cols = [col for col in self.data.columns if '%' in col]
            
            if not immune_cols:
                print("No immune features found for mediation analysis")
                return None
            
            # Prepare outcome
            # Use time-to-event for survival
            self.data['event'] = (self.data['VitalStatus'] == 'Dead').astype(int)
            
            results = []
            
            for mediator in immune_cols:
                try:
                    # Create clean column name
                    clean_mediator = 'immune_score'
                    temp_df = self.data.copy()
                    temp_df[clean_mediator] = temp_df[mediator]
                    temp_df['sex_code'] = (temp_df['SEX'] == 'F').astype(int)
                    
                    # Fit mediation models
                    # 1. Sex -> Mediator
                    med_model = smf.ols(
                        f"{clean_mediator} ~ sex_code + AGE + C(stage)", 
                        data=temp_df
                    ).fit()
                    
                    # 2. Sex + Mediator -> Outcome
                    from lifelines import CoxPHFitter
                    cox = CoxPHFitter()
                    cox.fit(
                        temp_df,
                        duration_col='survival_time',
                        event_col='event',
                        covariates=['sex_code', clean_mediator, 'AGE', 'stage']
                    )
                    
                    # Calculate mediation effect
                    indirect_effect = (
                        med_model.params['sex_code'] * 
                        cox.print_summary()['coef'][clean_mediator]
                    )
                    
                    results.append({
                        'mediator': mediator,
                        'total_effect': cox.print_summary()['coef']['sex_code'],
                        'indirect_effect': indirect_effect,
                        'proportion_mediated': indirect_effect / cox.print_summary()['coef']['sex_code'],
                        'p_value': med_model.pvalues['sex_code']
                    })
                    
                except Exception as e:
                    print(f"Error in mediation analysis for {mediator}: {str(e)}")
                    continue
            
            if not results:
                print("No successful mediation analyses")
                return None
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Calculate FDR
            results_df['fdr'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
            
            # Sort by mediation effect
            results_df = results_df.sort_values('proportion_mediated', ascending=False)
            
            # Print summary
            print("\nTop mediators (FDR < 0.1):")
            sig_results = results_df[results_df['fdr'] < 0.1]
            if len(sig_results) > 0:
                print(sig_results[['mediator', 'proportion_mediated', 'p_value', 'fdr']].to_string())
            else:
                print("No significant mediators found")
    
    # Save results
            results_df.to_csv(os.path.join(self.output_dir, 'mediation_results.csv'), index=False)
            
            return results_df
            
        except Exception as e:
            print(f"Error in mediation analysis: {e}")
            print(traceback.format_exc())
            return None

    def plot_sex_immune_differences(self, results_df, fdr_threshold=0.1):
        """Visualize significant sex differences in immune cells"""
        try:
            # Filter significant results
            sig_cells = results_df[results_df['fdr'] < fdr_threshold].copy()
            
            if len(sig_cells) == 0:
                print("No significant differences to plot")
                return
            
            # Create plot
            plt.figure(figsize=(12, max(6, len(sig_cells) * 0.3)))
            
            # Create forest plot
            y_pos = np.arange(len(sig_cells))
            
            plt.errorbar(
                sig_cells['effect_size'],
                y_pos,
                xerr=[
                    sig_cells['effect_size'] - sig_cells['ci_lower'],
                    sig_cells['ci_upper'] - sig_cells['effect_size']
                ],
                fmt='o',
                capsize=5
            )
            
            # Add reference line
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Customize plot
            plt.yticks(y_pos, sig_cells['cell_type'])
            plt.xlabel('Effect Size (Female vs Male)')
            plt.title(f'Sex Differences in Immune Cell Infiltration\n(FDR < {fdr_threshold})')
            
            # Add FDR values
            for i, row in sig_cells.iterrows():
                plt.text(
                    max(sig_cells['ci_upper']) + 0.1,
                    i,
                    f"FDR = {row['fdr']:.2e}",
                    va='center'
                )
            
            # Save plot
            plt.tight_layout()
            output_file = os.path.join(self.plots_dir, 'immune_cell_differences.png')
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Plot saved to: {output_file}")
            
        except Exception as e:
            print(f"Error plotting immune differences: {e}")
            print(traceback.format_exc())

    def plot_tcell_phenotypes(self, tcell_results):
        """Plot T-cell phenotype differences by sex"""
        try:
            plt.figure(figsize=(8, 6))
            
            # Create forest plot
            y_pos = np.arange(len(tcell_results))
            
            plt.errorbar(
                tcell_results['effect_size'],
                y_pos,
                xerr=[
                    tcell_results['effect_size'] - tcell_results['ci_lower'],
                    tcell_results['ci_upper'] - tcell_results['effect_size']
                ],
                fmt='o',
                capsize=5
            )
            
            plt.axvline(x=0, color='gray', linestyle='--')
            plt.yticks(y_pos, tcell_results['signature'])
            
            plt.title('Sex Differences in T-cell Phenotypes')
            plt.xlabel('Effect Size (Female vs Male)')
            
            # Add p-values
            for i, row in tcell_results.iterrows():
                plt.text(
                    max(tcell_results['ci_upper']) + 0.1,
                    i,
                    f"p = {row['p_value']:.2e}",
                    va='center'
                )
            
            plt.tight_layout()
            output_file = os.path.join(self.plots_dir, 'tcell_phenotypes.png')
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Saved T-cell phenotype plot to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error plotting T-cell phenotypes: {e}")
            print(traceback.format_exc())
            return None

    def plot_mediation_results(self, mediation_results):
        """Visualize mediation analysis results"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Create bar plot for total and indirect effects
            x = np.arange(len(mediation_results))
            width = 0.35
            
            plt.bar(x - width/2, mediation_results['total_effect'], 
                   width, label='Total Effect')
            plt.bar(x + width/2, mediation_results['indirect_effect'], 
                   width, label='Indirect Effect')
            
            plt.xticks(x, mediation_results['mediator'])
            plt.ylabel('Effect Size')
            plt.title('Mediation Analysis: Sex Effect on Survival')
            plt.legend()
            
            # Add proportion mediated
            for i, row in mediation_results.iterrows():
                plt.text(
                    i, 
                    max(row['total_effect'], row['indirect_effect']),
                    f"{row['proportion_mediated']:.1%}\nmediated",
                    ha='center',
                    va='bottom'
                )
            
            plt.tight_layout()
            output_file = os.path.join(self.plots_dir, 'mediation_analysis.png')
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Saved mediation analysis plot to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error plotting mediation results: {e}")
            print(traceback.format_exc())
            return None

    def _validate_data(self):
        """Validate required columns and data types"""
        required_cols = ['SEX', 'AGE', 'stage', 'prior_treatment', 'PATIENT_ID']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            print("\nData columns available:")
            print(self.data.columns.tolist())
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Validate sex coding
        invalid_sex = self.data[~self.data['SEX'].isin(['M', 'F'])]
        if len(invalid_sex) > 0:
            print("\nInvalid sex values found:")
            print(invalid_sex['SEX'].value_counts())
            raise ValueError("SEX column contains invalid values. Please check sex coding.")
        
        # Check sample sizes
        print(f"\nSample sizes:")
        print(f"Total samples: {len(self.data)}")
        print(f"Males: {sum(self.data['SEX'] == 'M')}")
        print(f"Females: {sum(self.data['SEX'] == 'F')}")

    def analyze_survival_with_zscore(self, scores, clinical_data):
        """Analyze survival using z-score normalization"""
        try:
            print("\nAnalyzing survival with z-score normalization...")
            
            # Merge data
            merged = clinical_data.merge(
                scores,
                left_on='PATIENT_ID', 
                right_index=True,
                how='inner'
            )
            
            # Map survival columns
            merged['survival_time'] = merged['OS_TIME']
            merged['death_event'] = merged['OS_EVENT']
            
            # Remove missing survival data
            merged = merged.dropna(subset=['survival_time', 'death_event'])
            
            # Split by sex
            male_data = merged[merged['SEX'] == 'Male']
            female_data = merged[merged['SEX'] == 'Female']
            
            # Results containers
            male_results = []
            female_results = []
            
            # Analyze each signature
            for sig in self.signatures.keys():
                # Calculate Z-scores based on entire cohort
                mean = merged[sig].mean()
                std = merged[sig].std()
                merged[f'{sig}_zscore'] = (merged[sig] - mean) / std
                
                # Define high/low based on z-score > 0 (above mean)
                merged[f'{sig}_high'] = (merged[f'{sig}_zscore'] > 0).astype(int)
                
                # MALE analysis
                if len(male_data) >= 20:
                    # Use common index for subset
                    male_cox = male_data.loc[male_data.index.intersection(merged.index)]
                    male_cox[f'{sig}_high'] = merged.loc[male_cox.index, f'{sig}_high']
                    male_cox = male_cox[['survival_time', 'death_event', f'{sig}_high', 'AGE']].dropna()
                    
                    male_high = male_cox[male_cox[f'{sig}_high'] == 1]
                    male_low = male_cox[male_cox[f'{sig}_high'] == 0]
                    
                    if len(male_cox) >= 10 and len(male_high) > 5 and len(male_low) > 5:
                        cph = CoxPHFitter()
                        cph.fit(
                            male_cox,
                            duration_col='survival_time',
                            event_col='death_event'
                        )
                        
                        male_results.append({
                            'signature': sig,
                            'hazard_ratio': np.exp(cph.params_[f'{sig}_high']),
                            'p_value': cph.summary.loc[f'{sig}_high', 'p'],
                            'ci_lower': np.exp(cph.confidence_intervals_.loc[f'{sig}_high'].iloc[0]),
                            'ci_upper': np.exp(cph.confidence_intervals_.loc[f'{sig}_high'].iloc[1]),
                            'n_patients': len(male_cox),
                            'n_high': len(male_high),
                            'n_low': len(male_low)
                        })
                        
                        # Create KM plot
                        self.plot_survival_curves(
                            male_cox, 
                            f"{sig}_male_zscore", 
                            f'{sig}_high',
                            title=f'Male Survival by {sig} Level (Z-score > 0)'
                        )
                
                # FEMALE analysis
                if len(female_data) >= 20:
                    # Use common index for subset
                    female_cox = female_data.loc[female_data.index.intersection(merged.index)]
                    female_cox[f'{sig}_high'] = merged.loc[female_cox.index, f'{sig}_high']
                    female_cox = female_cox[['survival_time', 'death_event', f'{sig}_high', 'AGE']].dropna()
                    
                    female_high = female_cox[female_cox[f'{sig}_high'] == 1]
                    female_low = female_cox[female_cox[f'{sig}_high'] == 0]
                    
                    if len(female_cox) >= 10 and len(female_high) > 5 and len(female_low) > 5:
                        cph = CoxPHFitter()
                        cph.fit(
                            female_cox,
                            duration_col='survival_time',
                            event_col='death_event'
                        )
                        
                        female_results.append({
                            'signature': sig,
                            'hazard_ratio': np.exp(cph.params_[f'{sig}_high']),
                            'p_value': cph.summary.loc[f'{sig}_high', 'p'],
                            'ci_lower': np.exp(cph.confidence_intervals_.loc[f'{sig}_high'].iloc[0]),
                            'ci_upper': np.exp(cph.confidence_intervals_.loc[f'{sig}_high'].iloc[1]),
                            'n_patients': len(female_cox),
                            'n_high': len(female_high),
                            'n_low': len(female_low)
                        })
                        
                        # Create KM plot
                        self.plot_survival_curves(
                            female_cox, 
                            f"{sig}_female_zscore", 
                            f'{sig}_high',
                            title=f'Female Survival by {sig} Level (Z-score > 0)'
                        )
            
            # Process results and create plots
            # ... similar to previous method ...
            
            return male_results, female_results
            
        except Exception as e:
            print(f"Error in z-score survival analysis: {e}")
            print(traceback.format_exc())
            return None, None

class SexStratifiedAnalysis(ImmuneAnalysis):
    """Analyze sex differences in immune microenvironment and outcomes"""
    
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path
        
        # Create output directories
        self.output_dir = os.path.join(base_path, 'codes/output/sex_analysis')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print(f"Sex analysis output directory: {self.output_dir}")
        print(f"Sex analysis plots directory: {self.plots_dir}")
        
        # Load clinical data first
        self._load_clinical_data()
        self._validate_data()
        
    def _load_clinical_data(self):
        """Load and process clinical data"""
        try:
            clinical_path = os.path.join(self.base_path, 'Clinical_Data/24PRJ217UVA_NormalizedFiles')
            
            # Load clinical files
            self.diagnosis = pd.read_csv(os.path.join(clinical_path, '24PRJ217UVA_20241112_Diagnosis_V4.csv'))
            self.medications = pd.read_csv(os.path.join(clinical_path, '24PRJ217UVA_20241112_Medications_V4.csv'))
            self.outcomes = pd.read_csv(os.path.join(clinical_path, '24PRJ217UVA_20241112_Outcomes_V4.csv'))
            self.vital = pd.read_csv(os.path.join(clinical_path, '24PRJ217UVA_20241112_VitalStatus_V4.csv'))
            
            # Process stage information from diagnosis
            stage_data = self.diagnosis[[
                'AvatarKey',
                'ClinTStage',  # Clinical T stage
                'ClinNStage',  # Clinical N stage
                'ClinMStage',  # Clinical M stage
                'ClinGroupStage',  # Clinical group stage
                'PathTStage',  # Pathological T stage
                'PathNStage',  # Pathological N stage
                'PathMStage',  # Pathological M stage
                'PathGroupStage',  # Pathological group stage
                'AgeAtDiagnosis',
                'YearOfDiagnosis'
            ]].copy()
            
            # Use pathological stage if available, otherwise clinical stage
            stage_data['stage'] = stage_data['PathGroupStage'].fillna(stage_data['ClinGroupStage'])
            
            # Process medication/treatment information
            med_data = self.medications.groupby('AvatarKey').agg({
                'MedLineRegimen': lambda x: min(x.astype(str).fillna('999')),  # Get earliest line
                'Medication': lambda x: ', '.join(x.dropna().unique()),  # All medications
                'AgeAtMedStart': 'min',  # Age at first treatment
                'YearOfMedStart': 'min'  # Year of first treatment
            })
            
            # Create prior treatment flag (before current line)
            med_data['prior_treatment'] = med_data['MedLineRegimen'].apply(
                lambda x: 'Yes' if str(x).strip() != '1' and str(x).strip() != '999' else 'No'
            )
            
            # Process outcome information
            outcome_data = self.outcomes.groupby('AvatarKey').agg({
                'SolidTumorResponse': 'first',  # Best response
                'ProgRecurInd': 'max',  # Any progression/recurrence
                'CurrentDiseaseStatus': 'last',  # Latest disease status
                'PerformStatusMostRecent': 'last'  # Latest performance status
            })
            
            # Process vital status
            vital_data = self.vital[[
                'AvatarKey', 
                'VitalStatus',
                'AgeAtLastContact',
                'AgeAtDeath',
                'CauseOfDeath'
            ]].copy()
            
            # Calculate survival time
            vital_data['survival_time'] = vital_data.apply(
                lambda x: x['AgeAtDeath'] if pd.notna(x['AgeAtDeath']) 
                else x['AgeAtLastContact'], axis=1
            )
            
            # Merge all clinical data
            clinical_merged = (
                stage_data
                .merge(med_data, left_on='AvatarKey', right_index=True, how='left')
                .merge(outcome_data, left_on='AvatarKey', right_index=True, how='left')
                .merge(vital_data, on='AvatarKey', how='left')
            )
            
            # Initialize self.data if not already present
            if not hasattr(self, 'data'):
                self.data = pd.DataFrame()
            
            # Merge with existing data
            self.data = self.data.merge(
                clinical_merged, 
                left_on='PATIENT_ID',
                right_on='AvatarKey',
                how='left'
            )
            
            # Fill missing values
            self.data['prior_treatment'] = self.data['prior_treatment'].fillna('No')
            self.data['stage'] = self.data['stage'].fillna('Unknown')
            
            # Print summary statistics
            print("\nClinical Data Summary:")
            print(f"Total patients: {len(self.data)}")
            print("\nStage distribution:")
            print(self.data['stage'].value_counts())
            print("\nResponse distribution:")
            print(self.data['SolidTumorResponse'].value_counts())
            print("\nVital status distribution:")
            print(self.data['VitalStatus'].value_counts())
            print("\nPrior treatment distribution:")
            print(self.data['prior_treatment'].value_counts())
            
            # Validate key variables
            missing_rates = {
                'stage': self.data['stage'].isna().mean(),
                'survival_time': self.data['survival_time'].isna().mean(),
                'SolidTumorResponse': self.data['SolidTumorResponse'].isna().mean(),
                'prior_treatment': self.data['prior_treatment'].isna().mean()
            }
            
            print("\nMissing data rates:")
            for var, rate in missing_rates.items():
                print(f"{var}: {rate:.1%}")
            
        except Exception as e:
            print(f"Error loading clinical data: {e}")
            print(traceback.format_exc())

    def analyze_sex_differences_immune(self):
        """Analyze sex differences in immune cell composition"""
        try:
            # Get immune cell columns
            immune_cols = [col for col in self.data.columns if '%' in col]
            print(f"\nAnalyzing {len(immune_cols)} immune cell types")
            
            results = []
            
            for cell in immune_cols:
                try:
                    # Create a clean column name for the formula
                    clean_cell = 'immune_score'
                    
                    # Create temporary dataframe with renamed column
                    temp_df = self.data.copy()
                    temp_df[clean_cell] = temp_df[cell]
                    
                    # Recode sex as 0/1 for modeling
                    temp_df['sex_code'] = (temp_df['SEX'] == 'F').astype(int)
                    
                    # Fit linear model
                    model = smf.ols(
                        formula=f"{clean_cell} ~ sex_code + AGE + C(stage) + prior_treatment",
                        data=temp_df
                    ).fit()
                    
                    # Store results
                    results.append({
                        'cell_type': cell,
                        'effect_size': model.params['sex_code'],  # Effect for females vs males
                        'std_error': model.bse['sex_code'],
                        'p_value': model.pvalues['sex_code'],
                        'ci_lower': model.conf_int().loc['sex_code', 0],
                        'ci_upper': model.conf_int().loc['sex_code', 1]
                    })
                    
                except Exception as e:
                    print(f"Error analyzing {cell}: {str(e)}")
                    continue
            
            # Convert to DataFrame
            if not results:
                print("No successful analyses to report")
                return None
            
            results_df = pd.DataFrame(results)
            
            # Calculate FDR
            results_df['fdr'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
            
            # Sort by p-value
            results_df = results_df.sort_values('p_value')
            
            # Print summary
            print("\nTop significant differences (FDR < 0.1):")
            sig_results = results_df[results_df['fdr'] < 0.1]
            if len(sig_results) > 0:
                print(sig_results[['cell_type', 'effect_size', 'p_value', 'fdr']].to_string())
    else:
                print("No significant differences found")
            
            # Save results
            results_df.to_csv(os.path.join(self.output_dir, 'sex_differences_immune.csv'), index=False)
            
            # Create visualization
            self.plot_sex_immune_differences(results_df)
            
            return results_df
            
        except Exception as e:
            print(f"Error in sex difference analysis: {e}")
            print(traceback.format_exc())
            return None
            
    def analyze_tcell_phenotypes(self):
        """Analyze sex differences in T-cell phenotypes"""
        try:
            # Get T-cell related columns
            tcell_cols = [col for col in self.data.columns if any(x in col.lower() for x in ['cd4', 'cd8', 't-cell', 'tcell'])]
            
            if not tcell_cols:
                print("No T-cell phenotype columns found")
                return None
                
            print(f"\nAnalyzing {len(tcell_cols)} T-cell phenotypes")
            
            results = []
            
            for col in tcell_cols:
                try:
                    # Create temporary dataframe
                    temp_df = self.data.copy()
                    temp_df['sex_code'] = (temp_df['SEX'] == 'F').astype(int)
                    
                    # Create clean column name for formula
                    clean_col = 'tcell_score'
                    temp_df[clean_col] = temp_df[col]
                    
                    # Fit model
                    model = smf.ols(
                        formula=f"{clean_col} ~ sex_code + AGE + C(stage) + prior_treatment",
                        data=temp_df
                    ).fit()
                    
                    # Store results
                    results.append({
                        'phenotype': col,
                        'effect_size': model.params['sex_code'],
                        'std_error': model.bse['sex_code'],
                        'p_value': model.pvalues['sex_code'],
                        'ci_lower': model.conf_int().loc['sex_code', 0],
                        'ci_upper': model.conf_int().loc['sex_code', 1]
                    })
                    
                except Exception as e:
                    print(f"Error analyzing {col}: {str(e)}")
                    continue
                    
            if not results:
                print("No successful T-cell analyses")
                return None
                
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Calculate FDR
            results_df['fdr'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
            
            # Sort by significance
            results_df = results_df.sort_values('p_value')
            
            # Print summary
            print("\nSignificant T-cell differences (FDR < 0.1):")
            sig_results = results_df[results_df['fdr'] < 0.1]
            if len(sig_results) > 0:
                print(sig_results[['phenotype', 'effect_size', 'p_value', 'fdr']].to_string())
            else:
                print("No significant T-cell differences found")
                
            # Save results
            results_df.to_csv(os.path.join(self.output_dir, 'tcell_sex_differences.csv'), index=False)
            
            return results_df
            
        except Exception as e:
            print(f"Error in T-cell analysis: {e}")
            print(traceback.format_exc())
            return None

    def plot_sex_immune_differences(self, results_df, fdr_threshold=0.1):
        """Visualize significant sex differences in immune cells"""
        try:
            # Filter significant results
            sig_cells = results_df[results_df['fdr'] < fdr_threshold].copy()
            
            if len(sig_cells) == 0:
                print("No significant differences to plot")
                return
            
            # Create plot
            plt.figure(figsize=(12, max(6, len(sig_cells) * 0.3)))
            
            # Create forest plot
            y_pos = np.arange(len(sig_cells))
            
            plt.errorbar(
                sig_cells['effect_size'],
                y_pos,
                xerr=[
                    sig_cells['effect_size'] - sig_cells['ci_lower'],
                    sig_cells['ci_upper'] - sig_cells['effect_size']
                ],
                fmt='o',
                capsize=5
            )
            
            # Add reference line
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Customize plot
            plt.yticks(y_pos, sig_cells['cell_type'])
            plt.xlabel('Effect Size (Female vs Male)')
            plt.title(f'Sex Differences in Immune Cell Infiltration\n(FDR < {fdr_threshold})')
            
            # Add FDR values
            for i, row in sig_cells.iterrows():
                plt.text(
                    max(sig_cells['ci_upper']) + 0.1,
                    i,
                    f"FDR = {row['fdr']:.2e}",
                    va='center'
                )
            
            # Save plot
            plt.tight_layout()
            output_file = os.path.join(self.plots_dir, 'immune_cell_differences.png')
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Plot saved to: {output_file}")
            
        except Exception as e:
            print(f"Error plotting immune differences: {e}")
            print(traceback.format_exc())

    def _validate_data(self):
        """Validate required columns and data types"""
        # Check required columns
        required_cols = ['SEX', 'AGE', 'stage', 'prior_treatment', 'PATIENT_ID']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            print("\nData columns available:")
            print(self.data.columns.tolist())
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Clean and validate sex coding
        sex_map = {
            'Male': 'M',
            'Female': 'F',
            'MALE': 'M',
            'FEMALE': 'F',
            'male': 'M',
            'female': 'F',
            'M': 'M',
            'F': 'F'
        }
        
        # Print sex distribution before cleaning
        print("\nSex distribution before cleaning:")
        print(self.data['SEX'].value_counts())
        
        # Clean sex values
        self.data['SEX'] = self.data['SEX'].map(sex_map)
        
        # Print sex distribution after cleaning
        print("\nSex distribution after cleaning:")
        print(self.data['SEX'].value_counts())
        
        # Check for invalid sex values
        invalid_sex = self.data[~self.data['SEX'].isin(['M', 'F'])]
        if len(invalid_sex) > 0:
            print("\nInvalid sex values found:")
            print(invalid_sex['SEX'].value_counts())
            raise ValueError("SEX column contains invalid values")
            
        # Validate age
        if not self.data['AGE'].between(0, 100).all():
            raise ValueError("AGE values outside expected range (0-100)")
            
        # Validate stage
        if self.data['stage'].isna().any():
            print("\nWarning: Missing stage values found")
            print(self.data['stage'].value_counts(dropna=False))
            
        # Validate prior treatment
        if not self.data['prior_treatment'].isin(['Yes', 'No']).all():
            raise ValueError("prior_treatment should only contain 'Yes' or 'No'")
            
        # Print sample sizes
        print("\nSample sizes:")
        print(f"Total samples: {len(self.data)}")
        print(f"Males: {sum(self.data['SEX'] == 'M')}")
        print(f"Females: {sum(self.data['SEX'] == 'F')}")
        
        # Print sex distribution percentages
        sex_dist = self.data['SEX'].value_counts(normalize=True).mul(100).round(1)
        print("\nSex distribution (%):")
        print(sex_dist.apply(lambda x: f"{x}%"))
        
        return True

def main():
    """Run comprehensive analysis"""
    # Define base path
    base_path = "/project/orien/data/aws/24PRJ217UVA_IORIG"
    
    # Initialize both analysis classes with explicit base_path
    treatment_analysis = TreatmentResponseAnalysis(base_path=base_path)
    sex_analysis = SexStratifiedAnalysis(base_path=base_path)
    
    # Create output directories
    os.makedirs(treatment_analysis.plots_dir, exist_ok=True)
    os.makedirs(treatment_analysis.output_dir, exist_ok=True)
    
    # First run standard treatment analysis
    print("\nAnalyzing treatment response patterns...")
    response_results = treatment_analysis.analyze_response_patterns()
    if response_results is not None:
        response_results.to_csv(
            os.path.join(treatment_analysis.output_dir, 'response_patterns.csv'),
        index=False
    )
    
    # Then run sex-stratified analyses
    print("\nAnalyzing sex differences in immune cells...")
    sex_results = sex_analysis.analyze_sex_differences_immune()
    if sex_results is not None and not sex_results.empty:
        sex_analysis.plot_sex_immune_differences(sex_results)
        sex_results.to_csv(
            os.path.join(sex_analysis.output_dir, 'sex_differences.csv'),
            index=False
        )
    
    print("\nAnalyzing T-cell phenotypes...")
    tcell_results = sex_analysis.analyze_tcell_phenotypes()
    if tcell_results is not None and not tcell_results.empty:
        sex_analysis.plot_tcell_phenotypes(tcell_results)
        tcell_results.to_csv(
            os.path.join(sex_analysis.output_dir, 'tcell_phenotypes.csv'),
            index=False
        )
    else:
        print("No significant T-cell results to plot")
    
    print("\nPerforming mediation analysis...")
    mediation_results = sex_analysis.analyze_sex_survival_mediation()
    if mediation_results is not None and not mediation_results.empty:
        sex_analysis.plot_mediation_results(mediation_results)
        mediation_results.to_csv(
            os.path.join(sex_analysis.output_dir, 'mediation_results.csv'),
            index=False
        )
    else:
        print("No significant mediation results to plot")
    
    print("\nAnalysis complete!")
    print(f"Treatment results saved to: {treatment_analysis.output_dir}")
    print(f"Sex analysis results saved to: {sex_analysis.output_dir}")
    print(f"Plots saved to: {sex_analysis.plots_dir}")

if __name__ == "__main__":
    main() 