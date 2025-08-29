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
        self.gene_and_transcript_expression_results = self.root / "../../RNAseq/gene_and_transcript_expression_results"
        self.manifest_and_QC_files = self.root / "../../Manifest_and_QC_Files"
        self.QC_data = self.manifest_and_QC_files / "24PRJ217UVA_20250130_RNASeq_QCMetrics.csv"
        # -----
        self.clinical_molecular_linkage_data = self.root / "../../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv"
        self.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors = self.root / "../pair_clinical_data_and_stages_of_tumors/output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv"
        # outputs
        # TODO: Expression matrix

paths = Paths()