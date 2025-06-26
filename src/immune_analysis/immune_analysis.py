'''
Usage:
./miniconda3/envs/ici_sex/bin/python src/immune_analysis/immune_analysis.py
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import os
import sys
import logging

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logger
# Basic config if not already set elsewhere
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the Focused Panel (must match the one used in microenv.py)
FOCUSED_XCELL_PANEL = [
    'CD8+ T-cells',
    'CD4+ memory T-cells',
    'Tgd cells', 
    'Macrophages M2',
    'Tregs',
    'cDC',
    'pDC',
    'Memory B-cells',
    'Plasma cells',
    'Endothelial cells',
    'Fibroblasts',
    'ImmuneScore',
    'StromaScore',
    'MicroenvironmentScore'
]

class ImmuneAnalysis:
    '''
    Class for analyzing immune cell composition data
    '''
    
    def __init__(self, data_path = None):
        '''
        Initialize with merged immune and clinical data
        '''
        # Set default base path
        BASE_PATH = "/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors"
        # Set correct data path to the output of microenv.py
        if data_path is None:
            data_path = os.path.join(BASE_PATH, "output/melanoma_analysis", "melanoma_sample_immune_clinical.csv") # Corrected path
        
        # Load the data
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data from: {data_path}")
            
            # Check for specimen site information
            self.has_site_info = False
            site_columns = ['SpecimenSite', 'SpecimenSiteOfOrigin', 'SequencingSites']
            for col in site_columns:
                if col in self.data.columns:
                    logger.info(f"Found specimen site information in column '{col}'")
                    self.site_column = col
                    self.has_site_info = True
                    
                    # Create metastatic flag if not already present
                    if 'IsMetastatic' not in self.data.columns:
                        self._create_metastatic_flag(col)
                    break
            
            if not self.has_site_info:
                logger.info("No specimen site information found in data")
                
        except FileNotFoundError:
             logger.error(f"Input data file not found: {data_path}. Make sure microenv.py has been run successfully.")
             raise
        except Exception as e:
             logger.error(f"Error loading data from {data_path}: {e}")
             raise
        
        # --- Corrected Immune Column Identification ---
        # Find which panel columns are actually in the loaded data
        self.immune_cols = [col for col in FOCUSED_XCELL_PANEL if col in self.data.columns]
        if not self.immune_cols:
            logger.error("No columns from the FOCUSED_XCELL_PANEL found in the loaded data. Check input file.")
            raise ValueError("Input data does not contain expected immune score columns.")
        else:
            logger.info(f"Identified {len(self.immune_cols)} immune score columns based on the focused panel: {self.immune_cols}")
        # --- End Correction ---
        
        # Create output directory
        self.output_dir = os.path.join(BASE_PATH, "codes/output/immune_analysis") # Keep separate output dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")
        
        # Define cell type categories (Optional refinement: filter based on available cols)
        # For now, keep the full dict but analysis loops will use self.immune_cols
        self.cell_categories = {
            'Innate': {
                'aDC': 'Activated Dendritic Cells',
                'cDC': 'Conventional Dendritic Cells', # Added based on panel
                'DC': 'Dendritic Cells',
                'NK': 'Natural Killer cells',
                'NKT': 'Natural Killer T cells',
                'Neutrophils': 'Neutrophils',
                'Macrophages': 'Macrophages', 
                'Macrophages M2': 'M2 Macrophages', # Added based on panel
                'Monocytes': 'Monocytes',
                'pDC': 'Plasmacytoid Dendritic Cells', # Added based on panel
                'Eosinophils': 'Eosinophils',
                'Basophils': 'Basophils',
                'Mast': 'Mast Cells'
            },
            'Adaptive': {
                'CD4+ memory T-cells': 'CD4+ Memory T cells', # Added based on panel
                'CD8+ T-cells': 'CD8+ T cells', # Added based on panel
                'Tcm': 'Central Memory T cells',
                'Tem': 'Effector Memory T cells',
                'Th1': 'T Helper 1 cells',
                'Th2': 'T Helper 2 cells',
                'Tgd cells': 'Gamma Delta T cells', # Added based on panel
                'Tregs': 'Regulatory T cells', # Added based on panel
                'B-cells': 'B cells',
                'Memory B-cells': 'Memory B cells', # Added based on panel
                'Plasma cells': 'Plasma cells' # Added based on panel
            },
            'Stromal': {
                'Endothelial cells': 'Endothelial cells', # Added based on panel
                'Fibroblasts': 'Fibroblasts', # Added based on panel
                'MSC': 'Mesenchymal Stem Cells',
                'Adipocytes': 'Adipocytes'
            },
            'Summary Scores': {
                'ImmuneScore': 'Immune Score', # Added based on panel
                'StromaScore': 'Stroma Score', # Added based on panel
                'MicroenvironmentScore': 'Microenvironment Score' # Added based on panel
            }
        }
        
        # Create clean names and category mappings (primarily for _clean_cell_type_name)
        self.clean_names = {}
        self.cell_category = {}
        for category, cells in self.cell_categories.items():
            for short_name, full_name in cells.items():
                 # Use the exact column name from FOCUSED_XCELL_PANEL as the key
                 if short_name in FOCUSED_XCELL_PANEL:
                      self.clean_names[short_name] = full_name
                      self.cell_category[short_name] = category

        # Calculate Bonferroni correction threshold based on actual tests
        self.n_tests = len(self.immune_cols)
        if self.n_tests > 0:
            self.bonferroni_threshold = 0.05 / self.n_tests
            logger.info(f"Bonferroni-corrected p-value threshold ({self.n_tests} tests): {self.bonferroni_threshold:.3e}")
        else:
             self.bonferroni_threshold = 0.05 # Avoid division by zero, error already raised
    
    def _clean_cell_type_name(self, col_name):
        """Convert column names (immune scores) to readable format for plots."""
        # Use the pre-generated clean names if available, otherwise use the column name itself
        clean_name = self.clean_names.get(col_name, col_name)
        category = self.cell_category.get(col_name, 'Other') # Get category if mapped
        return f"{clean_name} ({category})" if category != 'Other' else clean_name
    
    def get_cell_types(self): # This now returns the focused list
        """Return list of immune cell types being analyzed"""
        return self.immune_cols
    
    def compare_groups(self, cell_type, group_col='SEX', test='mann-whitney'):
        """
        Compare cell type abundance between groups.
        Ensures cell_type exists in the data.
        """
        if cell_type not in self.data.columns:
            logger.error(f"Column '{cell_type}' not found in data for comparison.")
            return None # Return None instead of raising error directly
            
        groups = self.data[group_col].dropna().unique() # Drop NA in grouping column
        if len(groups) != 2:
            logger.warning(f"Grouping column '{group_col}' does not have exactly 2 unique non-NA values ({groups}). Skipping comparison for {cell_type}.")
            return None
            
        group1 = self.data[self.data[group_col] == groups[0]][cell_type].dropna()
        group2 = self.data[self.data[group_col] == groups[1]][cell_type].dropna()

        if len(group1) < 3 or len(group2) < 3: # Require minimum samples per group
             logger.warning(f"Not enough data points (n<{3}) for comparison of '{cell_type}' between groups {groups[0]} (n={len(group1)}) and {groups[1]} (n={len(group2)}) in '{group_col}'.")
             return None
        
        try:
            if test == 'mann-whitney':
                stat, pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            elif test == 't-test':
                stat, pval = stats.ttest_ind(group1, group2)
            else:
                 logger.error(f"Unsupported test type '{test}'")
                 return None
        except ValueError as ve:
             # Handle cases like all values being identical
             logger.warning(f"Statistical test failed for {cell_type} by {group_col} (groups: {groups}). Reason: {ve}")
             return None

        return {
            'groups': groups,
            'means': [group1.mean(), group2.mean()],
            'medians': [group1.median(), group2.median()],
            'statistic': stat,
            'pvalue': pval
        }
    
    def plot_cell_distribution(self, cell_type, group_col='SEX', plot_type='violin'):
        """
        Plot distribution of cell type abundance by group.
        Handles cases where comparison is not possible.
        """
        if cell_type not in self.data.columns:
            logger.error(f"Cannot plot: Column '{cell_type}' not found in data.")
            return None
            
        plt.figure(figsize=(10, 6))
        
        # Create plot
        try:
            if plot_type == 'violin':
                ax = sns.violinplot(data=self.data, x=group_col, y=cell_type)
            elif plot_type == 'box':
                ax = sns.boxplot(data=self.data, x=group_col, y=cell_type)
            elif plot_type == 'both':
                ax = sns.violinplot(data=self.data, x=group_col, y=cell_type)
                sns.boxplot(data=self.data, x=group_col, y=cell_type, width=0.2, color='white')
            else:
                 logger.error(f"Unsupported plot type: {plot_type}")
                 plt.close()
                 return None
        except Exception as plot_err:
             logger.error(f"Error creating plot for {cell_type} vs {group_col}: {plot_err}")
             plt.close()
             return None

        clean_name = self._clean_cell_type_name(cell_type)
        plt.title(clean_name)
        
        # Add statistical annotation inside plot
        stats_result = self.compare_groups(cell_type, group_col)
        if stats_result is not None:
             pval = stats_result['pvalue']
             # Add significance stars
             if pval < self.bonferroni_threshold:
                 sig_symbol = '***'  # Bonferroni-significant
             elif pval < 0.05:
                 sig_symbol = '*'    # Nominally significant
             else:
                 sig_symbol = 'ns'   # Not significant
             
             # Add p-value text inside plot
             ymin, ymax = plt.ylim()
             # Adjust position slightly if ymax is very small
             text_y_pos = max(ymax * 0.95, ymin + (ymax - ymin) * 0.1) 
             plt.text(0.5, text_y_pos, 
                     f'p = {pval:.2e} {sig_symbol}',
                     transform=ax.transAxes, # Use axes coordinates for positioning
                     horizontalalignment='center',
                     verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        else:
             logger.info(f"No statistical comparison result available for {cell_type} vs {group_col} to annotate plot.")
             
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, f'{cell_type.replace(" ", "_").replace("+", "")}_{group_col}_dist.png') # Sanitize filename
        try:
             plt.savefig(output_file, bbox_inches='tight', dpi=300)
             logger.info(f"Saved plot to {output_file}")
        except Exception as save_err:
             logger.error(f"Failed to save plot {output_file}: {save_err}")
        plt.close()
        
        return output_file
    
    def plot_correlation_matrix(self, group_col='SEX'): # Simplified to only plot all immune_cols
        """
        Plot correlation matrix of the identified immune cell types.
        """
        logger.info(f"Generating correlation matrix for {len(self.immune_cols)} immune scores...")
        data = self.data[self.immune_cols].copy() # Use copy to avoid modifying original data
        title = 'Immune Cell Score Correlations - Focused Panel'
            
        # Clean column names for plotting using the refined method
        data.columns = [self._clean_cell_type_name(col) for col in data.columns]
        
        # Calculate correlation matrix
        corr = data.corr()
        
        # Check if correlation matrix is empty or all NaN
        if corr.isnull().all().all() or corr.empty:
            logger.warning("Correlation matrix is empty or all NaN. Skipping plot.")
            return None
        
        # Plot
        # Adjust figsize based on number of features for better readability
        num_features = len(self.immune_cols)
        fig_size = max(8, num_features * 0.6) # Basic heuristic
        plt.figure(figsize=(fig_size, fig_size * 0.8))
        
        sns.heatmap(corr, cmap='RdBu_r', center=0,
                   xticklabels=True, yticklabels=True,
                   square=False, annot=True, fmt='.1f', # Adjust fmt for readability
                   annot_kws={"size": 8}, # Adjust font size
                   linewidths=.5, # Add lines between cells
                   cbar_kws={'shrink': .8, 'label': 'Correlation'}) # Adjust color bar
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.title(title)
        
        # Save plot
        output_file = os.path.join(self.output_dir, f'correlation_matrix_focused.png')
        try:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            logger.info(f"Saved correlation matrix to {output_file}")
        except Exception as save_err:
             logger.error(f"Failed to save correlation matrix {output_file}: {save_err}")
        plt.close()
        
        return output_file

    def _create_metastatic_flag(self, site_column):
        """
        Create IsMetastatic flag based on specimen site information.
        
        Args:
            site_column (str): Name of the column containing specimen site information
        """
        try:
            # Define keywords indicating metastatic sites
            metastatic_keywords = ['lymph node', 'metastatic', 'metastasis', 'lymph_node', 'brain', 'liver', 'lung', 'distant']
            
            # Handle lists in SequencingSites column
            if site_column == 'SequencingSites':
                # For list-type columns, try to eval the string representation
                def check_metastatic_sites(sites_str):
                    if pd.isna(sites_str):
                        return np.nan
                    try:
                        sites = eval(sites_str) if isinstance(sites_str, str) else sites_str
                        if isinstance(sites, list):
                            # Check if any site in the list matches metastatic keywords
                            return any(
                                any(keyword in str(site).lower() for keyword in metastatic_keywords)
                                for site in sites if pd.notna(site)
                            )
                        return False
                    except:
                        return np.nan
                
                self.data['IsMetastatic'] = self.data[site_column].apply(check_metastatic_sites)
            else:
                # For regular columns, directly check against keywords
                self.data['IsMetastatic'] = self.data[site_column].apply(
                    lambda x: any(keyword in str(x).lower() for keyword in metastatic_keywords) 
                            if pd.notna(x) else np.nan
                )
            
            # Log the results
            metastatic_count = self.data['IsMetastatic'].sum()
            non_metastatic_count = (self.data['IsMetastatic'] == False).sum()
            na_count = self.data['IsMetastatic'].isna().sum()
            
            logger.info(f"Classified {metastatic_count} samples as metastatic, {non_metastatic_count} as primary, and {na_count} unknown")
            
        except Exception as e:
            logger.error(f"Error creating metastatic flag: {e}")
            # Create empty flag to avoid errors in downstream analysis
            self.data['IsMetastatic'] = np.nan

    def compare_metastatic_vs_primary(self):
        """
        Compare immune infiltration between metastatic and primary sites.
        Requires specimen site information to be available.
        
        Returns:
            pd.DataFrame or None: Results of comparison, or None if site information is not available
        """
        if not self.has_site_info or 'IsMetastatic' not in self.data.columns:
            logger.warning("Cannot compare metastatic vs primary: No specimen site information available")
            return None
            
        logger.info("Comparing immune cell infiltration between metastatic and primary sites...")
        
        # Check if we have both metastatic and primary samples
        metastatic_samples = self.data[self.data['IsMetastatic'] == True]
        primary_samples = self.data[self.data['IsMetastatic'] == False]
        
        if len(metastatic_samples) < 3 or len(primary_samples) < 3:
            logger.warning(f"Not enough samples for comparison: Metastatic={len(metastatic_samples)}, Primary={len(primary_samples)}")
            return None
            
        logger.info(f"Comparing {len(metastatic_samples)} metastatic samples vs {len(primary_samples)} primary samples")
        
        # Generate a detailed report of specimen sites
        if self.site_column in self.data.columns:
            # Count specimens by site 
            site_counts = self.data.groupby(['IsMetastatic', self.site_column]).size().reset_index(name='Count')
            site_counts = site_counts.sort_values('Count', ascending=False)
            
            # Save site counts to CSV
            site_counts_file = os.path.join(self.output_dir, 'specimen_sites_by_metastatic_status.csv')
            site_counts.to_csv(site_counts_file, index=False)
            logger.info(f"Saved specimen site counts to {site_counts_file}")
            
            # Create visualization of specimen sites
            try:
                plt.figure(figsize=(14, 8))
                # Filter to top 10 sites for readability
                top_sites = site_counts.groupby(self.site_column)['Count'].sum().nlargest(10).index
                filtered_counts = site_counts[site_counts[self.site_column].isin(top_sites)]
                
                # Create plot with site on x-axis, metastatic status as hue
                site_plot = sns.barplot(
                    data=filtered_counts,
                    x=self.site_column,
                    y='Count',
                    hue='IsMetastatic',
                    palette={'True': 'red', 'False': 'blue'},
                    alpha=0.7
                )
                
                plt.title('Sample Counts by Specimen Site and Metastatic Status')
                plt.xlabel('Specimen Site')
                plt.ylabel('Number of Samples')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Metastatic Status', labels=['Primary', 'Metastatic'])
                plt.tight_layout()
                
                site_plot_file = os.path.join(self.output_dir, 'specimen_sites_plot.png')
                plt.savefig(site_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Generated specimen site visualization: {site_plot_file}")
            except Exception as e:
                logger.error(f"Error creating specimen site visualization: {e}")
        
        # Run comparison for each immune cell type
        all_results = []
        for cell_type in self.immune_cols:
            # Compare groups using 'IsMetastatic' as the grouping column
            result = self.compare_groups(cell_type, group_col='IsMetastatic')
            if result:
                # Ensure we're associating the right groups with metastatic/primary
                # result['groups'] contains [False, True] or [True, False] 
                is_metastatic_first = result['groups'][0] == True
                
                all_results.append({
                    'cell_type': self._clean_cell_type_name(cell_type),
                    'category': self.cell_category.get(cell_type, 'Other'),
                    'p_value': result['pvalue'],
                    'significant': result['pvalue'] < 0.05,
                    'bonferroni_significant': result['pvalue'] < self.bonferroni_threshold,
                    'mean_primary': result['means'][0] if not is_metastatic_first else result['means'][1],
                    'mean_metastatic': result['means'][1] if is_metastatic_first else result['means'][0],
                    'median_primary': result['medians'][0] if not is_metastatic_first else result['medians'][1],
                    'median_metastatic': result['medians'][1] if is_metastatic_first else result['medians'][0]
                })
                
                # Calculate fold change, handling potential division by zero
                primary_mean = result['means'][0] if not is_metastatic_first else result['means'][1]
                metastatic_mean = result['means'][1] if is_metastatic_first else result['means'][0]
                
                if primary_mean > 0:
                    all_results[-1]['fold_change'] = metastatic_mean / primary_mean
                else:
                    all_results[-1]['fold_change'] = float('inf') if metastatic_mean > 0 else np.nan
                
                # Create plot for this comparison
                self.plot_cell_distribution(cell_type, group_col='IsMetastatic')
            
        # Create summary DataFrame
        if all_results:
            results_df = pd.DataFrame(all_results).sort_values('p_value')
            
            # Save results
            output_file = os.path.join(self.output_dir, 'metastatic_vs_primary_results.csv')
            results_df.to_csv(output_file, index=False)
            logger.info(f"Saved metastatic vs primary comparison results to {output_file}")
            
            # Generate heatmap of significant differences
            try:
                # Filter to significant results
                sig_results = results_df[results_df['significant']].copy()
                if len(sig_results) > 0:
                    # Create a pivot table for the heatmap
                    pivot_data = pd.DataFrame({
                        'cell_type': sig_results['cell_type'],
                        'fold_change': sig_results['fold_change'],
                        'p_value': sig_results['p_value'],
                        'sig_level': sig_results['p_value'].apply(
                            lambda p: '***' if p < self.bonferroni_threshold else 
                                    ('**' if p < 0.01 else '*')
                        )
                    })
                    
                    # Sort by fold change
                    pivot_data = pivot_data.sort_values('fold_change', ascending=False)
                    
                    # Create heatmap
                    plt.figure(figsize=(10, len(pivot_data) * 0.4 + 2))
                    ax = sns.heatmap(
                        pd.DataFrame(pivot_data['fold_change']).T,
                        annot=pd.DataFrame(pivot_data['sig_level']).T,
                        cmap='RdBu_r',
                        center=1,
                        fmt='',
                        cbar_kws={'label': 'Fold Change (Metastatic/Primary)'}
                    )
                    ax.set_xticklabels(pivot_data['cell_type'], rotation=45, ha='right')
                    plt.title('Significant Differences in Immune Cell Infiltration\n(Metastatic vs Primary Sites)')
                    plt.tight_layout()
                    
                    # Save heatmap
                    heatmap_file = os.path.join(self.output_dir, 'metastatic_vs_primary_heatmap.png')
                    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Generated heatmap of significant differences: {heatmap_file}")
            except Exception as e:
                logger.error(f"Error creating heatmap of metastatic vs primary differences: {e}")
            
            # Log summary statistics
            n_sig = results_df['significant'].sum()
            n_bon_sig = results_df['bonferroni_significant'].sum()
            logger.info(f"Metastatic vs primary comparison results:")
            logger.info(f"Total tests: {len(results_df)}")
            logger.info(f"Nominally significant (p < 0.05): {n_sig}")
            logger.info(f"Bonferroni significant (p < {self.bonferroni_threshold:.3e}): {n_bon_sig}")
            
            # Log top significant results
            if n_sig > 0:
                top_sig = results_df[results_df['significant']].head(5)
                logger.info("Top significant differences (metastatic vs primary):")
                for _, row in top_sig.iterrows():
                    logger.info(f"  {row['cell_type']}: fold change = {row['fold_change']:.2f}, p = {row['p_value']:.2e}")
            
            return results_df
        else:
            logger.warning("No results generated for metastatic vs primary comparison")
            return None

def main():
    """Main analysis workflow"""
    try:
        # Initialize analysis
        analysis = ImmuneAnalysis()
        
        # Define the grouping column based on the actual data
        grouping_column = 'Sex' # Changed from 'SEX'
        logger.info(f"\nAnalyzing {len(analysis.immune_cols)} focused immune cell types by {grouping_column}:")
        
        all_results = []
        # Analyze each identified immune column directly
        for cell_type in analysis.immune_cols:
            logger.info(f"-- Processing: {cell_type} --")
            # Use the corrected grouping column name
            result = analysis.compare_groups(cell_type, group_col=grouping_column)
            if result:
                all_results.append({
                    'cell_type': analysis._clean_cell_type_name(cell_type),
                    'p_value': result['pvalue'],
                    'significant': result['pvalue'] < 0.05,
                    'bonferroni_significant': result['pvalue'] < analysis.bonferroni_threshold,
                    'mean_group0': result['means'][0], # Assuming groups[0] is first category
                    'mean_group1': result['means'][1],
                    'median_group0': result['medians'][0],
                    'median_group1': result['medians'][1]
                })
            else:
                 logger.warning(f"Comparison skipped for {cell_type}.")

            # Create plot (even if comparison failed, plot might still be informative)
            # Use the corrected grouping column name
            analysis.plot_cell_distribution(cell_type, group_col=grouping_column)
            
        # Save combined results
        if all_results:
             results_df = pd.DataFrame(all_results).sort_values('p_value')
             output_file = os.path.join(analysis.output_dir, f'all_focused_results_by_{grouping_column}.csv') # Use variable in filename
             results_df.to_csv(output_file, index=False)
             logger.info(f"Saved all comparison results to {output_file}")
             # Print summary
             n_sig = results_df['significant'].sum()
             n_bon_sig = results_df['bonferroni_significant'].sum()
             logger.info(f"\nSummary for {grouping_column} comparison:") # Use variable in log
             logger.info(f"Total tests: {len(results_df)}")
             logger.info(f"Nominally significant (p < 0.05): {n_sig}")
             logger.info(f"Bonferroni significant (p < {analysis.bonferroni_threshold:.3e}): {n_bon_sig}")
             logger.info("\nTop significant cells (nominal):")
             logger.info(results_df[results_df['significant']].head())
        else:
             logger.warning("No statistical comparison results were generated.")

        # Create overall correlation matrix
        logger.info("\nGenerating overall correlation matrix for focused panel...")
        analysis.plot_correlation_matrix() # Correlation plot doesn't need group_col here
        
        # Run the metastatic vs primary comparison if possible
        if hasattr(analysis, 'has_site_info') and analysis.has_site_info:
            logger.info("\nComparing metastatic vs primary sites...")
            metastatic_results = analysis.compare_metastatic_vs_primary()
            if metastatic_results is not None and not metastatic_results.empty:
                logger.info(f"Completed metastatic vs primary site comparison with {len(metastatic_results)} immune features")
        
        logger.info("\nAnalysis complete! Check output directory for results and plots.")

    except Exception as e:
        logger.error(f"An error occurred during the main analysis workflow: {e}", exc_info=True)

if __name__ == "__main__":
    main() 