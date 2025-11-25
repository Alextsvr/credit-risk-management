# Credit Risk Management – Default Prediction Model

This repository contains a full end-to-end implementation of a credit default prediction pipeline.  
The project follows a realistic bank-grade workflow: large-scale data processing, feature engineering, model evaluation, and pipeline creation suitable for production usage.

## 1. Project Overview
The goal is to build a supervised machine learning model that predicts whether a borrower will default on a credit product, based on credit history, payment behavior, and product-level attributes.

The project replicates a real credit-risk scoring pipeline used in financial institutions, including:
- Memory-efficient processing of multi-GB parquet datasets  
- Extensive feature engineering for behavioral and credit-product signals  
- Model experiments with classical and gradient boosting methods  
- Hyperparameter tuning  
- A reproducible inference pipeline based on sklearn.pipeline

The final model achieves performance aligned with industry expectations for credit scoring.

## 2. Repository Structure
```
credit-risk-management/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_final_dataset.ipynb
│   ├── 04_modeling_baseline.ipynb
│   ├── 05_modeling_advanced.ipynb
│   ├── 06_pipeline.ipynb
│   ├── 07_final_report.ipynb
│   └── 08_presentation_prep.ipynb
│
├── src/
│   ├── features.py
│   ├── features_extended.py
│   ├── features_extended_v2.py
│   ├── pipeline.py
│   └── utils.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 3. Dataset Description
The dataset contains product-level credit information for each borrower:
- Payment histories (enc_paym_0…N)
- Credit limits, outstanding balances, and overdue amounts
- Delinquency buckets
- Product age, status, scheduling metadata
- Zero/non-zero behavioral flags

The target variable indicates whether the product entered a 90-day+ default state.

## 4. Feature Engineering
Feature engineering is implemented in modular form:

### Base Features
- Most recent payment status  
- Last clean (on-time) streak  
- One-hot encoded categorical attributes  

### Extended Features (v1 & v2)
Implemented in features_extended.py and features_extended_v2.py.

Includes:
- Rolling payment behavior statistics  
- Late/OK streaks  
- Payment volatility  
- Decay-weighted signals  
- Ratio-based credit product indicators  
- Delinquency severity scores  
- Behavioral flags  
- Interaction features  
- log1p monetary transforms  
- Exposure/age differentials  

## 5. Modeling
Two modeling phases:

### Baseline
- Logistic Regression  
- Random Forest

### Advanced
- LightGBM  
- XGBoost (best performing)

### Evaluation
Primary metric: ROC-AUC, with acceptance threshold ≥ 0.75.  
Final tuned model exceeds this threshold.

## 6. Pipeline
The final production-style pipeline is implemented using sklearn.pipeline, including:
- Feature preparation  
- Model training  
- Prediction  
- Serialization via pickle  

## 7. How to Run
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## 8. Key Skills Demonstrated
- Processing large datasets  
- Advanced feature engineering  
- Gradient boosting modeling and tuning  
- Production-ready ML pipeline design  
- Modular code architecture  
- Strict dtype and memory control  

## 9. License
For educational and portfolio purposes.  
