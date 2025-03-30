
## To install the necessary packages

1. Install conda
1. Run `conda env create --name ckd_env --file env.yml`. This create a new env name 'ckd_env' with the necessary libraries.
1. Run `conda activate ckd_env` to activate this env.

## To run experiments

1. Go to root folder
1. Run `python -m pkgs.experiments.file`

## To view analysis of train/test data

1. Go to root folder
1. Run `python -m pkgs.data.model_data_store`
