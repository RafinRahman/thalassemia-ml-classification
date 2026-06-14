# Trained Model Objects

Running `thalassemia_ml_pipeline.py` will save the trained XGBoost model to:

    models/xgboost_full_features.pkl

This file is not committed to the repository (excluded via .gitignore).
To obtain it, run the pipeline with the source dataset.

The LabelEncoder and StandardScaler objects are saved to:

    data/processed/label_encoder.pkl
    data/processed/scaler_full.pkl
    data/processed/scaler_cbc.pkl
