# Experiments:
## 4 features
- Reason: baseline traditional KFRE
- Prediction horizon: 2 years
- Features: Age, Sex, eGFR, Urine albumin-to-creatinine ratio (ACR/UACR)

## 8 features
- Reason: baseline extended KFRE
- Prediction horizon: 2 years
- Features: Age, Sex, eGFR, Urine albumin-to-creatinine ratio (ACR/UACR), Serum calcium, Serum phosphate, Serum bicarbonate, Serum albumin

## 20 features
- Reason: stress test whether increasing number of features increase prediction power
- Following reference target 20 features

1. **Machine learning for prediction of chronic kidney disease progression: validation of the Klinrisk model in the CANVAS Program and CREDENCE trial**
2. **Validation of the Klinrisk Machine Learning Model for CKD Progression in a Large Representative US Population**
3. **A simplified prediction model for end-stage kidney disease in patients with diabetes**
4. **Machine learning models to predict end-stage kidney disease in chronic kidney disease stage 4**
5. **Development and internal validation of an interpretable machine learning model for predicting dialysis risk in patients with stage 3–4 CKD**
6. **Kidney Disease: Improving Global Outcomes (KDIGO) 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease**

- Prediction horizon: 2 years

# Guide:
## Stage 1: Build train/test dataset
- There are existing scripts to get top N most common lab features. Use it for experiment "20 features"
- Each experiment should run 5 reps. Train/test data generated must be saved in "generated_data/rep<N>"
- Scripts to extract data must be run in the background. There must be log to keep track of PID. There must be status report (.md file)

## Stage 2: Run mini-experiment
- Subsample train/test data in rep 1 and run all experiments. Call that rep99
