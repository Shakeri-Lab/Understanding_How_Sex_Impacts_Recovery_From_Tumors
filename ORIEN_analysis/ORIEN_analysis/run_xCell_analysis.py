from rpy2.robjects import default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import py2rpy
from ORIEN_analysis.config import paths
import pandas as pd
import rpy2.robjects as ro


def load_expression_matrix():
    index_of_columns_of_expression_matrix = pd.read_csv(
        paths.filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest,
        nrows = 0
    ).columns
    dictionary_of_names_of_columns_and_data_types = {
        name_of_column: float
        for name_of_column in index_of_columns_of_expression_matrix
        if name_of_column != "HGNC_symbol"
    }
    dictionary_of_names_of_columns_and_data_types["HGNC_symbol"] = str
    expression_matrix = pd.read_csv(
        paths.filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest,
        dtype = dictionary_of_names_of_columns_and_data_types,
        index_col = "HGNC_symbol"
    )
    return expression_matrix


def main():
    paths.ensure_dependencies_for_running_xCell_analysis_exist()
    expression_matrix = load_expression_matrix()
    with localconverter(default_converter + pandas2ri.converter):
        expression_data_frame = py2rpy(expression_matrix)
    ro.r('cat(R.version.string, "is working.\\n")')
    xCell = importr("xCell")
    xCell2 = importr("xCell2")


if __name__ == "__main__":
    main()