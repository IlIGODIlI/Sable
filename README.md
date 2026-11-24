# 🛰️ Landslide Susceptibility Classification & Inference System

An end-to-end geospatial machine learning pipeline for terrain landslide susceptibility classification across diverse Indian physiographic zones (Western Himalayas & Western Ghats). 

Fusing **Copernicus Sentinel-1 (C-Band SAR texture)**, **Sentinel-2 (optical & moisture indices)**, and **SRTM/ALOS DEM topographic derivatives** across a balanced 464k+ sample corpus, the system benchmarks three pipelines:
- **Random Forest (Champion):** **91.38% Accuracy**, **0.9755 ROC-AUC**, **0.8186 MCC**, **0.0639 Brier Score**
- **XGBoost Classifier:** **90.18% Accuracy**, **0.9703 ROC-AUC**, **0.7953 MCC**
- **Logistic Regression (Baseline):** **69.23% Accuracy**, **0.7787 ROC-AUC**

### ⚡ Highlights
- 🔬 **Dynamic Tuning:** Automated continuous & discrete hyperparameter search via `scikit-learn` `RandomizedSearchCV` (Stratified 3-Fold CV).
- 📊 **Diagnostic Rigor:** Comprehensive calibration audits using Brier Skill Score, Calibration $R^2$, and majority-class binomial $Z$-score significance ($+169.3\sigma$).
- 🎛️ **Interactive Streamlit Dashboard:** Live terrain presets, 13 parameter sliders with seasonal guidance expanders (Monsoon vs. Dry season ranges), and real-time tri-model probability inference.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-118b45?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Best ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9755-10b981)](#)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-91.38%25-blue)](#)

> **Geospatial ML pipeline for binary landslide hazard classification across 464,000+ multi-state remote sensing samples using 29 topographic, spectral, and radar texture features.**
