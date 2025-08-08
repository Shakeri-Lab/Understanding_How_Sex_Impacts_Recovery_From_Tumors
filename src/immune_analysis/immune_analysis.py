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
import matplotlib.patches as mpatches

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
        col = self.melanoma_sample_immune_clinical_data_and_scores[group_col]
        group_values = pd.unique(col.dropna())
        if len(group_values) != 2:
            raise ValueError(f"2 groups in column {group_col} were expected.")
        group_order = [False, True] if pd.api.types.is_bool_dtype(col) else sorted(group_values)

        g1, g2 = group_order
        series_of_enrichment_scores_for_cell_type_and_group_1 = self.melanoma_sample_immune_clinical_data_and_scores.loc[col == g1, cell_type].dropna()
        series_of_enrichment_scores_for_cell_type_and_group_2 = self.melanoma_sample_immune_clinical_data_and_scores.loc[col == g2, cell_type].dropna()
        if len(series_of_enrichment_scores_for_cell_type_and_group_1) < 3 or len(series_of_enrichment_scores_for_cell_type_and_group_2) < 3:
             raise Exception(f"There are not enough enrichment scores for cells of type {cell_type} between group {g1} with {len(series_of_enrichment_scores_for_cell_type_and_group_1)} scores and group {g2} with {len(series_of_enrichment_scores_for_cell_type_and_group_2)} scores.")  
        if test == 'mann-whitney':
            stat, pval = stats.mannwhitneyu(series_of_enrichment_scores_for_cell_type_and_group_1, series_of_enrichment_scores_for_cell_type_and_group_2, alternative = "two-sided")
        elif test == 't-test':
            stat, pval = stats.ttest_ind(series_of_enrichment_scores_for_cell_type_and_group_1, series_of_enrichment_scores_for_cell_type_and_group_2)
        else:
             raise Exception(f"Test type {test} is unsupported.")
        return {
            'groups': [g1, g2],
            'means': [
                series_of_enrichment_scores_for_cell_type_and_group_1.mean(),
                series_of_enrichment_scores_for_cell_type_and_group_2.mean()
            ],
            'medians': [
                series_of_enrichment_scores_for_cell_type_and_group_1.median(),
                series_of_enrichment_scores_for_cell_type_and_group_2.median()
            ],
            'statistic': stat,
            'pvalue': pval
        }
    
    
    def plot_distribution_of_abundance_of_cells_of_type_by_group(self, cell_type, group_col, plot_type = "both"):
        '''
        Plot distribution of abundance of cells of a provided type by group.
        '''
        is_bool = pd.api.types.is_bool_dtype(
            self.melanoma_sample_immune_clinical_data_and_scores[group_col]
        )
        order = [False, True] if is_bool else None
        
        plt.figure(figsize = (10, 6))
        
        if plot_type == 'violin':
            ax = sns.violinplot(
                data = self.melanoma_sample_immune_clinical_data_and_scores,
                x = group_col,
                y = cell_type,
                order = order
            )
        elif plot_type == 'box':
            ax = sns.boxplot(
                data = self.melanoma_sample_immune_clinical_data_and_scores,
                x = group_col,
                y = cell_type,
                order = order
            )
        elif plot_type == 'both':
            ax = sns.violinplot(
                data = self.melanoma_sample_immune_clinical_data_and_scores,
                x = group_col,
                y = cell_type,
                order = order
            )
            sns.boxplot(
                data = self.melanoma_sample_immune_clinical_data_and_scores,
                x = group_col,
                y = cell_type,
                width = 0.2,
                color = 'white',
                order = order
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
            ax.text(
                0.5,
                0.95,
                f"p = {pval:.3e}.\np is {sig_symbol}.",
                transform = ax.transAxes,
                ha = "center",
                va = "top",
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
            },
            vmin = -1,
            vmax = 1
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
        '''
        Compare immune infiltration between metastatic and primary sites.
        
        Returns
        -------
        Data frame of statistics for cell types and indicators of metastasis
        '''
        g1 = False
        g2 = True
        if "IsMetastatic" not in self.melanoma_sample_immune_clinical_data_and_scores.columns:
            raise Exception("Cannot compare metastatic vs primary: No specimen site information available")
            
        logger.info("Immune cell infiltration between metastatic and primary sites will be compared.")
        
        metastatic_samples = self.melanoma_sample_immune_clinical_data_and_scores[
            self.melanoma_sample_immune_clinical_data_and_scores['IsMetastatic'] == True
        ]
        primary_samples = self.melanoma_sample_immune_clinical_data_and_scores[
            self.melanoma_sample_immune_clinical_data_and_scores['IsMetastatic'] == False
        ]
        if len(metastatic_samples) < 3 or len(primary_samples) < 3:
            raise Exception(f"We don't have enough samples for comparison. The number of metastatic samples is {len(metastatic_samples)}. The number of primary samples is {len(primary_samples)}.")
            
        logger.info(f"{len(metastatic_samples)} metastatic samples will be compared with {len(primary_samples)} primary samples.")
        
        data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens = self.melanoma_sample_immune_clinical_data_and_scores.groupby(
            ['IsMetastatic', "SpecimenSite"]
        ).size().reset_index(name = 'Count')
        data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens = data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens.sort_values(
            "Count",
            ascending = False
        )
        data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens.to_csv(
            paths.data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens,
            index = False
        )
        logger.info(f"A data frame of indicators of metastasis, types of specimens, and numbers of specimens was saved to {paths.data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens}.")

        plt.figure(figsize = (14, 8))
        #top_sites_by_number_of_specimens = data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens.groupby("SpecimenSite")['Count'].sum().nlargest(10).index
        #filtered_data_frame = data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens[
            #data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens["SpecimenSite"].isin(top_sites_by_number_of_specimens)
        #]

        palette = {True: "red", False: "blue"}
        site_plot = sns.barplot(
            data = data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens,
            x = "SpecimenSite",
            y = "Count",
            hue = "IsMetastatic",
            palette = palette#,
            #order = list(top_sites_by_number_of_specimens)
        )
        plt.title('Numbers of Samples by Specimen Site and Metastatic Status')
        plt.xlabel('Specimen Site')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation = 45, ha = 'right')
        
        primary_patch = mpatches.Patch(color = "blue", label = "Primary")
        metastatic_patch = mpatches.Patch(color = "red",  label = "Metastatic")
        site_plot.legend(
            handles = [primary_patch, metastatic_patch],
            title = "Metastatic Status",
            loc = "best"
        )
        plt.tight_layout()

        plt.savefig(
            paths.plot_of_numbers_of_samples_by_specimen_site_and_metastatic_status,
            dpi = 300,
            bbox_inches = "tight"
        )
        plt.close()
        logger.info(f"Plot of numbers of samples by specimen site and metastatic status was saved to {paths.plot_of_numbers_of_samples_by_specimen_site_and_metastatic_status}.")
        
        list_of_dictionaries_of_statistics_for_cell_types_and_indicators_of_metastasis = []
        for cell_type in FOCUSED_XCELL_PANEL:
            dictionary_of_statistics = self.compare_abundance_of_cells_of_type_between_groups(cell_type, group_col = "IsMetastatic")
            groups = dictionary_of_statistics["groups"]
            means = dictionary_of_statistics["means"]
            medians = dictionary_of_statistics["medians"]
            mean_by_group = dict(zip(groups, means))
            median_by_group = dict(zip(groups, medians))
            primary_mean = mean_by_group[False]
            metastatic_mean = mean_by_group[True]
            primary_median = median_by_group[False]
            metastatic_median = median_by_group[True]
            readable_cell_type, category = self.make_cell_type_readable(cell_type)
            dictionary_of_statistics_for_cell_types_and_indicators_of_metastasis = {
                'cell_type': readable_cell_type,
                'category': category,
                'p_value': dictionary_of_statistics['pvalue'],
                'significant': dictionary_of_statistics['pvalue'] < 0.05,
                'bonferroni_significant': dictionary_of_statistics['pvalue'] < self.bonferroni_threshold,
                'mean_primary': primary_mean,
                'mean_metastatic': metastatic_mean,
                'median_primary': primary_median,
                'median_metastatic': metastatic_median
            }
            dictionary_of_statistics_for_cell_types_and_indicators_of_metastasis["fold_change"] = (metastatic_mean / primary_mean) if primary_mean > 0 else (float("inf") if metastatic_mean > 0 else np.nan)
            list_of_dictionaries_of_statistics_for_cell_types_and_indicators_of_metastasis.append(
                dictionary_of_statistics_for_cell_types_and_indicators_of_metastasis
            )
                
            self.plot_distribution_of_abundance_of_cells_of_type_by_group(cell_type, group_col = 'IsMetastatic')
            
        if list_of_dictionaries_of_statistics_for_cell_types_and_indicators_of_metastasis:
            data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis = pd.DataFrame(list_of_dictionaries_of_statistics_for_cell_types_and_indicators_of_metastasis).sort_values('p_value')
            data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis.to_csv(
                paths.data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis,
                index = False
            )
            logger.info(f"A data frame of statistics for cell types and indicators of metastasis was saved to {paths.data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis}.")
            
            # Generate heatmap of significant differences.
            sig_results = data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis[
                data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis['significant']
            ].copy()
            if len(sig_results) > 0:
                #sig_results = sig_results.sort_values("fold_change", ascending = False)
                sig_results = sig_results.replace([np.inf, -np.inf], np.nan).dropna(subset = ["fold_change"])
                sig_results["log2_fold_change"] = np.log2(sig_results["fold_change"])
                mat = sig_results.set_index("cell_type")[["log2_fold_change"]].T
                L = float(np.nanmax(np.abs(mat.values)))
                if not np.isfinite(L) or L == 0:
                    L = 1e-6
                annot = sig_results.apply(
                    lambda row: f"log_2(FC) = {row['log2_fold_change']:.2f}\n(p={row['p_value']:.2g})", axis = 1
                ).to_frame().T
                plt.figure(figsize = (max(10, 0.6 * len(sig_results)), 2.6))
                ax = sns.heatmap(
                    mat,
                    cmap = 'RdBu_r',
                    center = 0,
                    vmin = -L,
                    vmax = L,
                    annot = annot,
                    fmt = '',
                    cbar_kws = {'label': 'log_2(FC)\n(M/P)'},
                    linewidths = 0.5
                )
                ax.set_xticklabels(sig_results["cell_type"], rotation = 45, ha = "right")
                ax.set_yticklabels([''], rotation = 0)
                plt.title("Fold changes in immune infiltration (Metastatic vs. Primary)")
                plt.tight_layout()
                plt.savefig(paths.metastatic_vs_primary_heatmap, dpi = 300, bbox_inches = "tight")
                plt.close()
                logger.info(f"Heatmap of significant differences was saved to {paths.metastatic_vs_primary_heatmap}.")
            
            # Log summary statistics.
            n_sig = data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis['significant'].sum()
            n_bon_sig = data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis['bonferroni_significant'].sum()
            logger.info(f"Metastatic vs primary comparison results:")
            logger.info(f"Total tests: {len(data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis)}")
            logger.info(f"Nominally significant (p < 0.05): {n_sig}")
            logger.info(f"Bonferroni significant (p < {self.bonferroni_threshold:.3e}): {n_bon_sig}")
            
            # Log top significant results.
            if n_sig > 0:
                top_sig = data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis[
                    data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis['significant']
                ].head(5)
                logger.info("Top significant differences (metastatic vs primary):")
                for _, row in top_sig.iterrows():
                    logger.info(f"{row['cell_type']}: fold change = {row['fold_change']:.9f}, p = {row['p_value']:.2e}")
            
            return data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis

        
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
                    'median_enrichment_score_for_cell_type_and_group_1': result['medians'][1]
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