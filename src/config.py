import os
from pathlib import Path


class Paths():
    
    def __init__(self):
        
        # src/immune_analysis/microenv.py
        # dependencies
        self.root = Path("/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors")
        self.map_from_sample_to_patient = self.root / "output/eda/sample_to_patient_map.csv"
        self.data_frame_of_melanoma_patient_and_sequencing_data = self.root / "output/eda/melanoma_patients_with_sequencing.csv"
        self.outputs_of_microenv = self.root / "output/microenv"
        # outputs
        self.melanoma_sample_immune_clinical_data = self.outputs_of_microenv / "melanoma_sample_immune_clinical.csv" # TODO: Rename.
        self.map_of_biopsy_locations_to_counts = self.outputs_of_microenv / "specimen_site_summary.csv"
        self.map_of_procedure_types_to_counts = self.outputs_of_microenv / "procedure_type_summary.csv"
        self.map_of_indicators_that_specimens_are_part_of_metastatic_disease_to_counts = self.outputs_of_microenv / "metastatic_status_summary.csv"
        self.data_frame_of_scores_by_sample_and_cell_type = self.outputs_of_microenv / "xcell_scores_raw.csv"
        self.focused_data_frame_of_scores_by_sample_and_cell_type = self.outputs_of_microenv / "xcell_scores_focused_panel.csv"
    
        
    def ensure_dependencies_for_microenv_exist(self):
        for path in [self.root, self.outputs_of_microenv]:
            os.makedirs(path, exist_ok = True)
        for path in [self.map_from_sample_to_patient, self.data_frame_of_melanoma_patient_and_sequencing_data]:
            if not os.path.exists(path):
                raise Exception(f"The dependency of `src/immune_analysis/microenv.py` `{path}` does not exist.")


paths = Paths()