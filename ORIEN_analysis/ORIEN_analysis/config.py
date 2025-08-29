from pathlib import Path


class Paths():
    
    def __init__(self):

        # ORIEN_analysis
        # dependencies
        # <no dependencies>
        # outputs
        # <no files>

        # ORIEN_analysis/create_expression_matrix.py
        # dependencies
        self.root = Path("/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors/ORIEN_analysis")
        self.manifest_and_QC_files = self.root / "../../Manifest_and_QC_Files"
        self.QC_data = self.manifest_and_QC_files / "24PRJ217UVA_20250130_RNASeq_QCMetrics.csv"
        self.gene_and_transcript_expression_results = self.root / "../../RNAseq/gene_and_transcript_expression_results"
        # outputs
        # TODO: Expression matrix

paths = Paths()