from rpy2.robjects import default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import py2rpy
from ORIEN_analysis.config import paths
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.conversion import rpy2py


def load_expression_data_frame():
    index_of_columns_of_expression_data_frame = pd.read_csv(
        paths.filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest,
        nrows = 0
    ).columns
    dictionary_of_names_of_columns_and_data_types = {
        name_of_column: float
        for name_of_column in index_of_columns_of_expression_data_frame
        if name_of_column != "HGNC_symbol"
    }
    dictionary_of_names_of_columns_and_data_types["HGNC_symbol"] = str
    expression_data_frame = pd.read_csv(
        paths.filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest,
        dtype = dictionary_of_names_of_columns_and_data_types,
        index_col = "HGNC_symbol"
    )
    return expression_data_frame


def main():
    paths.ensure_dependencies_for_running_xCell_analysis_exist()

    xCell = importr("xCell")
    xCell2 = importr("xCell2")
    ro.r('data(PanCancer.xCell2Ref, package = "xCell2")')
    ro.r('data(TMECompendium.xCell2Ref, package = "xCell2")')
    reference_Pan_Cancer = ro.r('PanCancer.xCell2Ref')
    reference_TME_Compendium = ro.r('TMECompendium.xCell2Ref')

    expression_pandas_data_frame = load_expression_data_frame()
    with localconverter(default_converter + pandas2ri.converter):
        expression_r_data_frame = py2rpy(expression_pandas_data_frame)

    enrichment_r_matrix_per_xCell = xCell.xCellAnalysis(expression_r_data_frame)
    enrichment_r_matrix_per_xCell2_and_Pan_Cancer = xCell2.xCell2Analysis(
        mix = expression_r_data_frame,
        xcell2object = reference_Pan_Cancer
    )
    '''
    enrichment_r_matrix_per_xCell2_and_TME_Compendium = xCell2.xCell2Analysis(
        mix = expression_r_data_frame,
        xcell2object = reference_TME_Compendium
    )
    '''
    with localconverter(default_converter + pandas2ri.converter):
        enrichment_numpy_matrix_per_xCell = rpy2py(enrichment_r_matrix_per_xCell)
        enrichment_numpy_matrix_per_xCell2_and_Pan_Cancer = rpy2py(enrichment_r_matrix_per_xCell2_and_Pan_Cancer)
    enrichment_data_frame_per_xCell = pd.DataFrame(enrichment_numpy_matrix_per_xCell)
    enrichment_data_frame_per_xCell2_and_Pan_Cancer = pd.DataFrame(enrichment_numpy_matrix_per_xCell2_and_Pan_Cancer)
    
    enrichment_data_frame_per_xCell.columns = expression_pandas_data_frame.columns
    enrichment_data_frame_per_xCell2_and_Pan_Cancer.columns = expression_pandas_data_frame.columns
    enrichment_data_frame_per_xCell = enrichment_data_frame_per_xCell.T
    enrichment_data_frame_per_xCell2_and_Pan_Cancer = enrichment_data_frame_per_xCell2_and_Pan_Cancer.T
    enrichment_data_frame_per_xCell.index.name = "SampleID"
    enrichment_data_frame_per_xCell2_and_Pan_Cancer.index.name = "SampleID"

    vector_of_cell_types = ro.r("rownames(xCell.data$spill$K)")
    list_of_cell_types_per_xCell = [
        str(numpy_string_representing_cell_type)
        for numpy_string_representing_cell_type
        in vector_of_cell_types
    ] # See https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1349-1#Sec24 .
    list_of_names_of_composite_scores = ["ImmuneScore", "StromaScore", "MicroenvironmentScore"]
    list_of_cell_types_and_names_of_composite_scores_per_xCell = list_of_cell_types_per_xCell + list_of_names_of_composite_scores # See line 219 of https://github.com/dviraran/xCell/blob/master/R/xCell.R .
    enrichment_data_frame_per_xCell.columns = list_of_cell_types_and_names_of_composite_scores_per_xCell

    function_get_signatures = ro.r('xCell2::getSignatures')
    vector_of_lists_of_genes_per_xCell2_and_Pan_Cancer = function_get_signatures(reference_Pan_Cancer)
    function_names = ro.r['names']
    vector_of_raw_cell_types_per_xCell2_and_Pan_Cancer = function_names(vector_of_lists_of_genes_per_xCell2_and_Pan_Cancer)
    with localconverter(default_converter):
        string_vector_of_raw_cell_types_per_xCell2_and_Pan_Cancer = rpy2py(vector_of_raw_cell_types_per_xCell2_and_Pan_Cancer)
        list_of_cell_types_per_xCell2_and_Pan_Cancer = []
        for raw_cell_type in string_vector_of_raw_cell_types_per_xCell2_and_Pan_Cancer:
            cell_type = raw_cell_type.split("#")[0]
            if cell_type not in list_of_cell_types_per_xCell2_and_Pan_Cancer:
                list_of_cell_types_per_xCell2_and_Pan_Cancer.append(cell_type)
    enrichment_data_frame_per_xCell2_and_Pan_Cancer.columns = list_of_cell_types_per_xCell2_and_Pan_Cancer
    
    enrichment_data_frame_per_xCell.to_csv(paths.enrichment_data_frame_per_xCell)
    enrichment_data_frame_per_xCell2_and_Pan_Cancer.to_csv(paths.enrichment_data_frame_per_xCell2_and_Pan_Cancer)

    list_of_cell_types_and_names_of_composite_scores_on_which_to_focus = [
        "CD8+ T-cells",
        "CD4+ memory T-cells",
        "Tgd cells",
        "Macrophages M2",
        "Tregs",
        "cDC",
        "pDC",
        "Memory B-cells",
        "Plasma cells",
        "Endothelial cells",
        "Fibroblasts",
        "ImmuneScore",
        "StromaScore",
        "MicroenvironmentScore"
    ]
    list_of_cell_types_and_names_of_composite_scores_on_which_to_focus_in_enrichment_data_frame_per_xCell = [
        string
        for string in list_of_cell_types_and_names_of_composite_scores_on_which_to_focus
        if string in enrichment_data_frame_per_xCell.columns
    ]
    focused_enrichment_data_frame_per_xCell = enrichment_data_frame_per_xCell[
        list_of_cell_types_and_names_of_composite_scores_on_which_to_focus_in_enrichment_data_frame_per_xCell
    ]
    focused_enrichment_data_frame_per_xCell.to_csv(paths.focused_enrichment_data_frame_per_xCell)


if __name__ == "__main__":
    main()