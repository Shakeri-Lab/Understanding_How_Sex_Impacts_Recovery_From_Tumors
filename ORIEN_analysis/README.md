# ORIEN_analysis

To run `create_expression_matrices.py`, from `Understanding_How_Sex_Impacts_Recovery_From_Tumors`, run
- `module list`
- `module load miniforge` (To unload, create new terminal. TODO: Consider whether there is a command to unload.)

The following 3 commands are not necessary.
- `conda env list`
- `conda create --name conda_environment python=3.13.5` (`conda env remove --name conda_environment`)
- `conda activate conda_environment` (`conda deactivate`)

- `pip install --no-build-isolation --no-deps --editable ORIEN_analysis` (https://github.com/conda/conda-build/issues/4251) (`pip uninstall ORIEN_analysis`)
- `python -m ORIEN_analysis.create_expression_matrices`