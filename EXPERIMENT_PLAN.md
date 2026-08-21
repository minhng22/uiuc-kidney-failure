# DON'T EDIT!!

# Experiments:
## 4 features
- Reason: baseline traditional KFRE
- Prediction horizon: 2 years
- Features: Age, Sex, eGFR, Urine albumin-to-creatinine ratio (ACR/UACR)
- Format of each row of data: age, sex, egfr, uacr

## 8 features
- Reason: baseline extended KFRE
- Prediction horizon: 2 years
- Features: Age, Sex, eGFR, Urine albumin-to-creatinine ratio (ACR/UACR), Serum calcium, Serum phosphate, Serum bicarbonate, Serum albumin
- Format of each row of data: age, sex, egfr, uacr, Serum calcium, Serum phosphate, Serum bicarbonate, Serum albumi

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
- Format of each row of data: <feature 1>, <feature 1 missing (True/False)>, <feature 2>, <feature 2 missing (True/False)>

# Guide:
# Stage 0: Build details command:
- Read stage 1 to stage 3, build a detailed plan. Include which command you will run for each action in each stage.
- Save detailed plan to EXPERIMENT_PLAN_DETAILS.md. Wait for user approval.

The rest of the stages should be executed using EXPERIMENT_PLAN_DETAILS.md

## Stage 1: Build train/test dataset
- There are existing scripts to get top N most common lab features. Use it for experiment "20 features"
- Each experiment should run 5 reps. Train/test data generated must be saved in "generated_data/rep<N>"
- Scripts to extract data must be run in the background. There must be log to keep track of PID. There must be status report (.md file)

## Stage 2: Run mini-experiment
- Subsample train/test data in rep 1 and run all experiments. Call that rep99

## Stage 2.1: Cohort data analysis:
- Analyze feature importance

## Stage 3: Run full experiments
- Run all experiments in rep 1 -> rep 5
