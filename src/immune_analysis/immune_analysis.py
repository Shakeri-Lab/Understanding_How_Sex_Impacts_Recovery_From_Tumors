'''
Usage:
./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.immune_analysis
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

from src.config import FOCUSED_XCELL_PANEL
from src.config import paths


# Configure logger
# Basic config if not already set elsewhere
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImmuneAnalysis:
    '''
    Class `ImmuneAnalysis` is a template for an object for analyzing immune cell composition data.
    '''
    
    def __init__(self):

        # Initialize data frames of melanoma sample immune clinical data and focused enrichment scores by sample and cell type.
        self.melanoma_sample_immune_clinical_data = pd.read_csv(paths.melanoma_sample_immune_clinical_data)
        
        self.focused_enrichment_data_frame = pd.read_csv(paths.focused_enrichment_data_frame)
        
        self.melanoma_sample_immune_clinical_data_and_scores = self.melanoma_sample_immune_clinical_data.merge(
            self.focused_enrichment_data_frame,
            left_on = "SLID",
            right_on = "SampleID",
            how = "left"
        )

        # Calculate Bonferroni correction threshold based on actual tests.
        self.n_tests = len(FOCUSED_XCELL_PANEL)
        if self.n_tests > 0:
            self.bonferroni_threshold = 0.05 / self.n_tests
        else:
             self.bonferroni_threshold = 0.05
        
        logger.info(f"The Bonferroni-corrected p-value threshold for {self.n_tests} tests is {self.bonferroni_threshold:.3e}.")
        self.dictionary_of_categories_of_cell_types_and_dictionaries_of_short_cell_types_and_long_cell_types = {
            'Innate': {
                'cDC': 'Conventional Dendritic Cells',
                'Macrophages M2': 'M2 Macrophages',
                'pDC': 'Plasmacytoid Dendritic Cells',
            },
            'Adaptive': {
                'CD4+ memory T-cells': 'CD4+ Memory T cells',
                'CD8+ T-cells': 'CD8+ T cells',
                'Tgd cells': 'Gamma Delta T cells',
                'Tregs': 'Regulatory T cells',
                'Memory B-cells': 'Memory B cells',
                'Plasma cells': 'Plasma cells'
            },
            'Stromal': {
                'Endothelial cells': 'Endothelial cells',
                'Fibroblasts': 'Fibroblasts',
            },
            'Summary Scores': {
                'ImmuneScore': 'Immune Score',
                'StromaScore': 'Stroma Score',
                'MicroenvironmentScore': 'Microenvironment Score'
            }
        }
        
        # Create readable cell types and category mappings.
        self.dictionary_of_short_cell_types_and_long_cell_types = {}
        self.dictionary_of_short_cell_types_and_categories_of_cell_types = {}
        for category, dictionary_of_short_cell_types_and_long_cell_types in self.dictionary_of_categories_of_cell_types_and_dictionaries_of_short_cell_types_and_long_cell_types.items():
            for short_cell_type, long_cell_type in dictionary_of_short_cell_types_and_long_cell_types.items():
                 if short_cell_type in FOCUSED_XCELL_PANEL:
                    self.dictionary_of_short_cell_types_and_long_cell_types[short_cell_type] = long_cell_type
                    self.dictionary_of_short_cell_types_and_categories_of_cell_types[short_cell_type] = category
    
    
    def make_cell_type_readable(self, cell_type):
        readable_cell_type = self.dictionary_of_short_cell_types_and_long_cell_types.get(cell_type, None)
        if readable_cell_type is None:
            raise Exception("Readable cell type is None.")
        category = self.dictionary_of_short_cell_types_and_categories_of_cell_types.get(cell_type, None)
        if category is None:
            raise Exception("Category of cell type is None.")
        return readable_cell_type, category
    
    
    def compare_abundance_of_cells_of_type_between_groups(self, cell_type, group_col, test = "mann-whitney"):
        '''
        Compare abundance of cells of a provided type between groups
        (e.g., Female and Male as values of Sex).
        '''
        groups = self.melanoma_sample_immune_clinical_data_and_scores[group_col].dropna().unique()
        if len(groups) != 2:
            raise Exception(f"Grouping column {group_col} does not have exactly 2 unique non-NA values. This column has groups {groups}.")
        series_of_enrichment_scores_for_cell_type_and_group_1 = self.melanoma_sample_immune_clinical_data_and_scores[
            self.melanoma_sample_immune_clinical_data_and_scores[group_col] == groups[0]
        ][cell_type].dropna()
        series_of_enrichment_scores_for_cell_type_and_group_2 = self.melanoma_sample_immune_clinical_data_and_scores[
            self.melanoma_sample_immune_clinical_data_and_scores[group_col] == groups[1]
        ][cell_type].dropna()
        if len(series_of_enrichment_scores_for_cell_type_and_group_1) < 3 or len(series_of_enrichment_scores_for_cell_type_and_group_2) < 3:
             raise Exception(f"There are not enough enrichment scores for cells of type {cell_type} between group {groups[0]} with {len(group1)} scores and group {groups[1]} with {len(group2)} scores.")  
        if test == 'mann-whitney':
            stat, pval = stats.mannwhitneyu(series_of_enrichment_scores_for_cell_type_and_group_1, series_of_enrichment_scores_for_cell_type_and_group_2, alternative = "two-sided")
        elif test == 't-test':
            stat, pval = stats.ttest_ind(series_of_enrichment_scores_for_cell_type_and_group_1, series_of_enrichment_scores_for_cell_type_and_group_2)
        else:
             raise Exception(f"Test type {test} is unsupported.")
        return {
            'groups': groups,
            'means': [
                series_of_enrichment_scores_for_cell_type_and_group_1.mean(),
                series_of_enrichment_scores_for_cell_type_and_group_1.mean()
            ],
            'medians': [
                series_of_enrichment_scores_for_cell_type_and_group_1.median(),
                series_of_enrichment_scores_for_cell_type_and_group_1.median()
            ],
            'statistic': stat,
            'pvalue': pval
        }
    
    
    def plot_distribution_of_abundance_of_cells_of_type_by_group(self, cell_type, group_col, plot_type = "both"):
        '''
        Plot distribution of abundance of cells of a provided type by group.
        ''' 
        plt.figure(figsize = (10, 6))
        
        if plot_type == 'violin':
            ax = sns.violinplot(
                data = self.melanoma_sample_immune_clinical_data_and_scores,
                x = group_col,
                y = cell_type
            )
        elif plot_type == 'box':
            ax = sns.boxplot(
                data = self.melanoma_sample_immune_clinical_data_and_scores,
                x = group_col,
                y = cell_type
            )
        elif plot_type == 'both':
            ax = sns.violinplot(
                data = self.melanoma_sample_immune_clinical_data_and_scores,
                x = group_col,
                y = cell_type
            )
            sns.boxplot(
                data = self.melanoma_sample_immune_clinical_data_and_scores,
                x = group_col,
                y = cell_type,
                width = 0.2,
                color = 'white'
            )
        else:
             raise Exception(f"Plot type {plot_type} is unsupported.")

        readable_cell_type, category = self.make_cell_type_readable(cell_type)
        plt.title(f"Distribution of Abundances of Cells of Type {readable_cell_type} in Category {category}")
        
        # Add a statistical annotation including p value and a significance symbol inside plot.
        dictionary_of_statistics = self.compare_abundance_of_cells_of_type_between_groups(cell_type, group_col)
        if dictionary_of_statistics is not None:
            pval = dictionary_of_statistics['pvalue']
            if pval < self.bonferroni_threshold:
                sig_symbol = "Bonferroni significant"
            elif pval < 0.05:
                sig_symbol = "nominally significant"
            else:
                sig_symbol = "not significant"
            ymin, ymax = plt.ylim()
            text_y_pos = max(ymax * 0.95, ymin + (ymax - ymin) * 0.1) 
            plt.text(
                0.5,
                text_y_pos, 
                f"p = {pval:.3e}.\np is {sig_symbol}.",
                transform = ax.transAxes,
                horizontalalignment = "center",
                verticalalignment = "top",
                bbox = {
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none"
                }
            )
        else:
             logger.info(f"No statistical comparison result is available for {cell_type} vs. {group_col} to annotate plot.")
        
        plt.tight_layout()
        output_file = os.path.join(
            paths.distributions_of_abundance_of_cells_of_type_by_group,
            f'distribution_of_abundances_of_{cell_type.replace(" ", "_").replace("+", "plus")}_by_{group_col}.png'
        )
        plt.savefig(output_file, bbox_inches = "tight", dpi = 300)
        logger.info(f"Plot was saved to {output_file}.")
        plt.close()
        
        return output_file
    
    
    def plot_correlation_matrix(self):
        '''
        Plot correlation matrix of the identified immune cell types.
        '''
        logger.info(f"A correlation matrix for all {len(FOCUSED_XCELL_PANEL)} cell types in focused data frame of xCell scores by sample and cell type will be generated.")
        data_frame_of_sample_IDs_cell_types_and_abundances = self.melanoma_sample_immune_clinical_data_and_scores[FOCUSED_XCELL_PANEL].copy()
        title = "Matrix Of Correlations Of Abundances Of Cells Types"
        
        list_of_readable_cell_types = []
        for cell_type in data_frame_of_sample_IDs_cell_types_and_abundances.columns:
            readable_cell_type, category = self.make_cell_type_readable(cell_type)
            list_of_readable_cell_types.append(f"{readable_cell_type} in {category}")
        data_frame_of_sample_IDs_cell_types_and_abundances.columns = list_of_readable_cell_types
        
        correlation_matrix = data_frame_of_sample_IDs_cell_types_and_abundances.corr()
        
        if correlation_matrix.isnull().all().all() or correlation_matrix.empty:
            raise Exception("Correlation matrix is empty or all NaN.")
        
        num_features = len(FOCUSED_XCELL_PANEL)
        fig_size = max(8, num_features * 0.6)
        plt.figure(figsize = (fig_size, fig_size * 0.8))
        
        sns.heatmap(
            correlation_matrix,
            cmap = 'RdBu_r',
            center = 0,
            xticklabels = True,
            yticklabels = True,
            square = False,
            annot = True,
            fmt = '.1f',
            annot_kws = {"size": 8},
            linewidths = .5,
            cbar_kws = {
                'shrink': .8,
                'label': 'Correlation'
            }
        )
        plt.xticks(rotation = 45, ha = 'right', fontsize = 8)
        plt.yticks(rotation = 0, fontsize = 8)
        plt.title(title)
        
        plt.savefig(
            paths.matrix_of_correlations_of_abundances_of_cell_types,
            bbox_inches = "tight",
            dpi = 300
        )
        logger.info(f"Correlation matrix was saved to {paths.matrix_of_correlations_of_abundances_of_cell_types}.")
        plt.close()
        
        return paths.matrix_of_correlations_of_abundances_of_cell_types

    
    def compare_metastatic_vs_primary(self):
        """
        Compare immune infiltration between metastatic and primary sites.
        Requires specimen site information to be available.
        
        Returns:
            pd.DataFrame or None: Results of comparison, or None if site information is not available
        """
        if "IsMetastatic" not in self.melanoma_sample_immune_clinical_data_and_scores.columns:
            logger.warning("Cannot compare metastatic vs primary: No specimen site information available")
            return None
            
        logger.info("Comparing immune cell infiltration between metastatic and primary sites...")
        
        # Check if we have both metastatic and primary samples
        metastatic_samples = self.melanoma_sample_immune_clinical_data_and_scores[self.melanoma_sample_immune_clinical_data_and_scores['IsMetastatic'] == True]
        primary_samples = self.melanoma_sample_immune_clinical_data_and_scores[self.melanoma_sample_immune_clinical_data_and_scores['IsMetastatic'] == False]
        
        if len(metastatic_samples) < 3 or len(primary_samples) < 3:
            logger.warning(f"Not enough samples for comparison: Metastatic={len(metastatic_samples)}, Primary={len(primary_samples)}")
            return None
            
        logger.info(f"Comparing {len(metastatic_samples)} metastatic samples vs {len(primary_samples)} primary samples")
        
        # Generate a detailed report of specimen sites.
        # Count specimens by site.
        site_counts = self.melanoma_sample_immune_clinical_data_and_scores.groupby(['IsMetastatic', "SpecimenSite"]).size().reset_index(name='Count')
        site_counts = site_counts.sort_values('Count', ascending=False)

        # Save site counts to CSV.
        site_counts.to_csv(paths.data_frame_of_specimen_sites_by_metastatic_status, index = False)
        logger.info(f"Specimen site counts were saved to {paths.data_frame_of_specimen_sites_by_metastatic_status}.")

        # Create visualization of specimen sites.
        plt.figure(figsize=(14, 8))
        # Filter to top 10 sites for readability
        top_sites = site_counts.groupby("SpecimenSite")['Count'].sum().nlargest(10).index
        filtered_counts = site_counts[site_counts["SpecimenSite"].isin(top_sites)]

        # Create plot with site on x-axis, metastatic status as hue
        site_plot = sns.barplot(
            data=filtered_counts,
            x="SpecimenSite",
            y='Count',
            hue='IsMetastatic',
            palette={np.True_: 'red', np.False_: 'blue'},
            alpha=0.7
        )

        plt.title('Sample Counts by Specimen Site and Metastatic Status')
        plt.xlabel('Specimen Site')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metastatic Status', labels=['Primary', 'Metastatic'])
        plt.tight_layout()

        plt.savefig(paths.specimen_sites_plot, dpi = 300, bbox_inches = "tight")
        plt.close()
        logger.info(f"Specimen site visualization was saved to {paths.specimen_sites_plot}.")
        
        # Run comparison for each immune cell type
        all_results = []
        for cell_type in FOCUSED_XCELL_PANEL:
            # Compare groups using 'IsMetastatic' as the grouping column
            result = self.compare_abundance_of_cells_of_type_between_groups(cell_type, group_col = "IsMetastatic")
            if result:
                # Ensure we're associating the right groups with metastatic/primary
                # result['groups'] contains [False, True] or [True, False] 
                is_metastatic_first = result['groups'][0] == True
                
                readable_cell_type, category = self.make_cell_type_readable(cell_type)
                all_results.append({
                    'cell_type': readable_cell_type,
                    'category': category,
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
                self.plot_distribution_of_abundance_of_cells_of_type_by_group(cell_type, group_col='IsMetastatic')
            
        # Create summary DataFrame
        if all_results:
            results_df = pd.DataFrame(all_results).sort_values('p_value')
            
            # Save results.
            results_df.to_csv(paths.data_frame_of_metastatic_vs_primary_results, index = False)
            logger.info(f"Saved metastatic vs primary comparison results to {paths.data_frame_of_metastatic_vs_primary_results}")
            
            # Generate heatmap of significant differences

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

                # Save heatmap.
                plt.savefig(paths.metastatic_vs_primary_heatmap, dpi = 300, bbox_inches = "tight")
                plt.close()
                logger.info(f"Heatmap of significant differences was saved to {paths.metastatic_vs_primary_heatmap}.")
            
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
    paths.ensure_dependencies_for_immune_analysis_exist()
    analysis = ImmuneAnalysis()
    grouping_column = 'Sex'
    
    logger.info(f"Abundances of cells of {len(FOCUSED_XCELL_PANEL)} types will be compared by {grouping_column}.")

    list_of_dictionaries_of_statistics_for_cell_types_and_sexes = []
    for cell_type in FOCUSED_XCELL_PANEL:
        
        logger.info(f"Abundances of cells of type {cell_type} will be compared by {grouping_column}.")
        
        result = analysis.compare_abundance_of_cells_of_type_between_groups(cell_type, group_col = grouping_column)
        if result:
            readable_cell_type, category = analysis.make_cell_type_readable(cell_type)
            list_of_dictionaries_of_statistics_for_cell_types_and_sexes.append(
                {
                    'cell_type': f"{readable_cell_type} in {category}",
                    'p_value': result['pvalue'],
                    'significant': result['pvalue'] < 0.05,
                    'bonferroni_significant': result['pvalue'] < analysis.bonferroni_threshold,
                    'mean_enrichment_score_for_cell_type_and_group_0': result['means'][0],
                    'mean_enrichment_score_for_cell_type_and_group_1': result['means'][1],
                    'median_enrichment_score_for_cell_type_and_group_0': result['medians'][0],
                    'median_enrichment_score_for_cell_type_and_group_0': result['medians'][1]
                }
            )
        else:
             raise Exception(f"Comparing abundances of cells of type {cell_type} yielded no result.")

        analysis.plot_distribution_of_abundance_of_cells_of_type_by_group(cell_type, group_col = grouping_column)

    if list_of_dictionaries_of_statistics_for_cell_types_and_sexes:
        data_frame_of_statistics_for_cell_types_and_sexes = pd.DataFrame(
            list_of_dictionaries_of_statistics_for_cell_types_and_sexes
        ).sort_values('p_value')
        data_frame_of_statistics_for_cell_types_and_sexes.to_csv(
            paths.data_frame_of_statistics_for_cell_types_and_sexes,
            index = False
        )
        logger.info(f"A data frame of statistics for cell types and sexes was saved to {paths.data_frame_of_statistics_for_cell_types_and_sexes}.")
         
        # Print summary.
        n_sig = data_frame_of_statistics_for_cell_types_and_sexes['significant'].sum()
        n_bon_sig = data_frame_of_statistics_for_cell_types_and_sexes['bonferroni_significant'].sum()
        logger.info(f"Summary for {grouping_column} comparison:")
        logger.info(f"Total tests: {len(data_frame_of_statistics_for_cell_types_and_sexes)}")
        logger.info(f"Nominally significant (p < 0.05): {n_sig}")
        logger.info(f"Bonferroni significant (p < {analysis.bonferroni_threshold:.3e}): {n_bon_sig}")
        if n_sig > 0:
            logger.info("First rows with nominal significance:")
            logger.info(
                data_frame_of_statistics_for_cell_types_and_sexes[
                    data_frame_of_statistics_for_cell_types_and_sexes['significant']
                ].head()
            )
    else:
         logger.warning("No statistical comparison results were generated.")

    analysis.plot_correlation_matrix()

    # Compare metastatic and primary sites.
    logger.info("Metastatic and primary sites will be compared.")
    metastatic_results = analysis.compare_metastatic_vs_primary()
    if metastatic_results is not None and not metastatic_results.empty:
        logger.info(f"Metastatic and primary sites were compared with {len(metastatic_results)} immune features.")

    logger.info("Analysis complete! Check output directory for results and plots.")


if __name__ == "__main__":
    main() 