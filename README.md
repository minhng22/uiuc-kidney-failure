
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

## Experiment designs
### Non-time-variant setting
1. Cox
2. WeilbulAFT
3. DeepSurv
4. GBSA
5. SRF

### Time-variants
1. Cox
2. DynamicDeepHit
3. HazardTransformer
4. RNNSurv

### Heterogenous
In this setup, instead of only EGFR, we use [EGFR, protein, albumin]
1. Cox
2. DynamicDeepHit
3. HazardTransformer
4. RNNSurv

### EGFR components
In this setup, instead of EGFR, we use components that are used to calculate EGFR, aka [serum_creatinine, gender, age]
Using the CKD-EPI 2021 formula https://www.mdcalc.com/calc/3939/ckd-epi-equations-glomerular-filtration-rate-gfr#evidence
Based on https://pubmed.ncbi.nlm.nih.gov/34554658/
1. Cox
2. DynamicDeepHit
3. HazardTransformer
4. RNNSurv

### Notes:
- The RNN models (DeepSurv, DynamicDeepHit, HazardTransformer, RNNSurv) are run on 10 trials. 50 epochs each