# ICB Analysis Module

This module contains scripts for analyzing immune checkpoint blockade (ICB) medications and their relationship with CD8+ T cell signatures. It provides tools to analyze ICB treatments, their effects on survival outcomes, and how the Tumor Microenvironment (TME) influences these outcomes, with a focus on sex-specific differences.

## Files

- `__init__.py`: ICB Analysis package initialization
- `group_analysis.py`: ?
- `group_data_loader.py`: ?
- `icb_analysis.py`: Core ICB analysis functions.
- `icb_data_loader.py`: Loads and processes data for ICB analysis.
- `icb_main.py`: Main script for running ICB analysis.
- `README.md`: This document

## Usage

Run ICB analysis from directory Understanding How Sex Impacts Recovery From Tumors via

```bash
python -m src.icb_analysis.icb_main
```

### Options

- `--base-path PATH`: Base path for data files (default: `/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors`).
- `--clinical-file PATH`: Path to clinical data file (default: searches in standard locations).
- `--medication-file PATH`: Path to medication data file (default: searches in standard locations).
- `--output-dir PATH`: Directory to save output files (default: `BASE_PATH/output/icb_analysis`).
- `--by-type`: Analyze ICB treatments by type (PD-1, PD-L1, CTLA-4, combinations). Generates distribution plots and summarizes patient counts for each ICB class.
- `--duration`: Analyze ICB treatment duration patterns. Calculates treatment duration for each patient and generates histograms by ICB type.
- `--propensity-matching`: Perform Propensity Score Matching (PSM) to balance ICB-naive and ICB-experienced groups. Uses age, stage, and other clinical variables as confounders.
- `--survival`: Analyze overall survival by ICB status. Generates Kaplan-Meier curves comparing ICB-experienced vs. ICB-naive patients.
- `--tme-survival`: Analyze TME effects on ICB survival outcomes by sex. Uses PSM to balance high vs. low CD8_G groups and generates sex-specific survival analyses.
- `--pre-icb-tme`: Analyze pre-ICB TME effects on survival after ICB treatment, stratified by sex. Identifies tumor samples collected before ICB treatment for more accurate prognostic assessment.
- `--confounders LIST`: Comma-separated list of confounders for PSM (default: `AGE,STAGE,TMB`).
- `--tme-feature NAME`: TME feature to use for analysis (default: `CD8_G`; options: `CD8_B`, `CD8_G`, `CD8_GtoB_ratio`, `CD8_GtoB_log`).
- `--plot-only`: Only generate plots from existing analysis results without rerunning analyses.

### Default Behavior
If no options are specified, the script runs a default analysis pipeline. This includes:
- Identifying ICB treatments from the medication data.
- Merging ICB status with clinical data.
- Analyzing CD8 signatures by ICB status.

To perform specific analyses (e.g., survival analysis or PSM), use the corresponding options such as `--survival` or `--propensity-matching`.

## TME Features
The `--tme-feature` option lets you select a Tumor Microenvironment (TME) feature for analysis. Available options include:
- **`CD8_B`**: Represents the CD8+ T cell signature B, typically associated with baseline immune infiltration.
- **`CD8_G`**: Represents the CD8+ T cell signature G, often linked to activated immune response.
- **`CD8_GtoB_ratio`**: The ratio of CD8_G to CD8_B, indicating relative dominance of G over B signatures.
- **`CD8_GtoB_log`**: The log-transformed ratio of CD8_G to CD8_B, useful for normalizing skewed distributions.

These features are analyzed to assess their impact on survival outcomes in ICB-treated patients.

## Propensity Score Matching (PSM)
Propensity Score Matching (PSM) balances ICB-naive and ICB-experienced groups (or high vs. low TME feature groups) based on confounders like age and stage. This statistical technique reduces bias in observational data, isolating the effect of ICB treatment or TME features on survival outcomes. PSM diagnostics are generated to assess matching quality.

## Data Requirements
To run the analysis, provide the following data files:
- **Clinical Data**: A CSV file with columns such as `PATIENT_ID`, `AGE`, `SEX`, `STAGE`, `OS_TIME` (overall survival time), and `OS_EVENT` (survival event indicator).
- **Medication Data**: A CSV file with columns like `PATIENT_ID` and `Medication` (e.g., PEMBROLIZUMAB).
- **CD8 Scores**: A CSV file with patient-level CD8 scores (e.g., `CD8_B`, `CD8_G`), indexed by `PATIENT_ID`.

Ensure patient IDs are consistent across all files.

## Example Usage
- **Run default analysis**:
  ```bash
  python icb_main.py
  ```
- **Analyze TME effects on survival by sex using `CD8_B`**:
  ```bash
  python icb_main.py --tme-survival --tme-feature CD8_B
  ```
- **Perform PSM and survival analysis**:
  ```bash
  python icb_main.py --propensity-matching --survival --confounders AGE,STAGE
  ```

## Key Findings

### ICB Treatment Analysis
- Identified 1283 ICB medication records administered to 260 patients.
- After merging with clinical data: 268 ICB-experienced patients (71.3% of females, 67.6% of males).
- Found 120 ICB-naive patients.

### ICB Types Distribution
- **Anti-PD1 only**: 149 patients
- **Anti-PD1 + Anti-CTLA4**: 111 patients
- **Anti-CTLA4 only**: 20 patients
- **Anti-PD1 + Anti-PDL1**: 1 patient
- **Anti-PD1 + Anti-PDL1 + Anti-CTLA4**: 1 patient

### TME Effects on ICB Survival by Sex
Propensity score matching analysis revealed significant sex-specific effects:

**Male Patients**:
- 68 matched pairs (high vs. low CD8_G)
- High CD8_G significantly protective: HR = 0.47 (95% CI: 0.28-0.78, p=0.0037)
- Log-rank test p-value: 0.0036

**Female Patients**:
- 36 matched pairs (high vs. low CD8_G)
- Weaker protective effect: HR = 0.80 (95% CI: 0.41-1.55, p=0.5129)
- Log-rank test p-value: 0.5977

These findings suggest that the tumor microenvironment characterized by CD8_G has different prognostic value between sexes when patients receive ICB therapy, with males showing a stronger benefit from high CD8_G levels.

## Analysis Workflow

1. Load CD8 group scores or run CD8 group analysis if scores not found.
2. Load clinical data.
3. Load medication data.
4. Identify ICB treatments.
5. Merge ICB status with clinical data.
6. Analyze CD8 signatures by ICB status.
7. Run additional analyses based on options.
8. Summarize findings.

## Analysis Components

### By Type Analysis
When using the `--by-type` option:
- Categorizes ICB treatments into mechanism-based classes.
- Identifies combination therapies (e.g., PD-1+CTLA-4).
- Generates bar plots showing distribution of ICB types by sex.
- Outputs summary statistics to CSV files.

### Duration Analysis
When using the `--duration` option:
- Calculates treatment duration for each patient.
- Analyzes differences in duration by ICB type.
- Generates histograms of treatment duration.
- Compares median treatment duration between sexes.

### Propensity Score Matching
When using the `--propensity-matching` option:
- Balances ICB-naive and ICB-experienced groups.
- Matches patients based on specified confounders.
- Generates diagnostics for matching quality.
- Outputs matched patient cohorts for further analysis.

### Survival Analysis
When using the `--survival` option:
- Compares overall survival between ICB-naive and ICB-experienced patients.
- Generates Kaplan-Meier curves.
- Performs log-rank tests and Cox proportional hazards regression.
- Analyzes survival differences by sex.

### TME Survival Analysis
When using the `--tme-survival` option:
- Stratifies analysis by sex.
- Performs PSM within each sex group.
- Generates balanced cohorts of high vs. low TME feature (e.g., CD8_G).
- Compares survival outcomes using Kaplan-Meier and Cox regression.
- Tests for sex-specific effects of TME features on ICB outcomes.

### Pre-ICB TME Survival Analysis
When using the `--pre-icb-tme` option:
- Identifies tumor samples collected before ICB treatment.
- Filters analysis to patients with pre-ICB tumor samples who later received ICB.
- Stratifies by sex and performs PSM.
- Analyzes how pre-ICB TME characteristics predict survival after ICB treatment.
- Provides more accurate prognostic assessment by focusing on pre-treatment tumor microenvironment.

## Output

Results are saved to the following directories:
- `output/cd8_analysis/icb_medication_analysis/`: Analysis results.
- `output/cd8_analysis/icb_medication_analysis/plots/`: Plots and visualizations.
- `output/icb_analysis/results/`: PSM and Cox regression results.
- `output/icb_analysis/plots/`: Kaplan-Meier survival curves and hazard ratio comparison plots.

### Key Output Files
- `sex_comparison_CD8_G.csv`: Comparison of HR between sexes.
- `male_km_curves_CD8_G.png`: Kaplan-Meier curves for male patients.
- `female_km_curves_CD8_G.png`: Kaplan-Meier curves for female patients.
- `hazard_ratio_comparison_CD8_G.png`: Visual comparison of hazard ratios by sex.
- `icb_types_distribution.csv`: Distribution of ICB types across all patients.
- `icb_types_by_sex.csv`: Sex-specific distribution of ICB types.
- `icb_duration_summary.csv`: Summary statistics for ICB treatment durations.
- `icb_status_by_tumor_patient.csv`: Patient and tumor ICB status classification.
- `pre_icb_sex_comparison_CD8_G.csv`: Comparison of pre-ICB TME effects between sexes.

### Output Interpretation
Key output files include:
- **`sex_comparison_CD8_G.csv`**: Hazard ratios (HR) and p-values comparing high vs. low `CD8_G` groups by sex.
  - HR < 1: High `CD8_G` is protective.
  - P-value < 0.05: Result is statistically significant.
- **`male_km_curves_CD8_G.png`**: Kaplan-Meier survival curves for males with high vs. low `CD8_G`.
- **`hazard_ratio_comparison_CD8_G.png`**: A forest plot comparing HRs between sexes.

Check the `output/icb_analysis/` directory for full results and plots.

## Troubleshooting
- **"File Not Found" Error**: Verify that `--base-path`, `--clinical-file`, and `--medication-file` point to valid locations.
- **Missing Values**: The script fills missing confounders with medians or excludes affected patients. Review logs for details.
- **Analysis Skipped**: Some analyses (e.g., PSM) require sufficient data (e.g., ≥5 patients per group). Adjust options or data if needed.

## Dependencies
This module depends on the following modules:
- `cd8_analysis`: For CD8 group analysis.
- `utils`: For shared functions.

## ICB Medications
The following ICB medications are analyzed:
- PD-1 inhibitors: PEMBROLIZUMAB (KEYTRUDA), NIVOLUMAB (OPDIVO), CEMIPLIMAB (LIBTAYO)
- PD-L1 inhibitors: ATEZOLIZUMAB (TECENTRIQ), DURVALUMAB (IMFINZI), AVELUMAB (BAVENCIO)
- CTLA-4 inhibitors: IPILIMUMAB (YERVOY), TREMELIMUMAB

## Version Information
**Last Updated**: November 2024  
Check for updates to ensure you’re using the latest version.

## Contact Information
For support or feedback, contact [Insert email, e.g., support@example.com] or visit [Insert GitHub repo, e.g., github.com/username/icb-analysis].

## License
This project is licensed under [Insert license, e.g., MIT License].