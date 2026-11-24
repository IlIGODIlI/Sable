# Landslide Susceptibility ML

Initial ML codebase for a landslide-susceptibility prediction system.

## Goal

Predict landslide susceptibility as a 3-class problem:

- `1` = Low
- `2` = Moderate
- `3` = High

The first training dataset is the **Mahamunkar, Kiwelekar & Netak (2023) Raigad, Maharashtra dataset**, which contains seven conditioning factors:

`Curvature, Slope, Aspect, Elevation, NDVI, Precipitation, LULC`

The project also tracks larger/authoritative inventories for the next data-engineering stage.

## Dataset sources

### 1. Primary ML-ready dataset
Mendeley Data — Dataset for Landslide Susceptibility Prediction  
DOI: `10.17632/jsn8cwb9nz.1`  
https://data.mendeley.com/datasets/jsn8cwb9nz/1

The dataset contains 100 random locations in Raigad district, Maharashtra, with the seven conditioning factors and a 3-class susceptibility label.

**Important:** 100 samples is enough to establish the pipeline and compare models, but is far too small to claim production-level generalization.

### 2. India-scale inventory
ISRO/NRSC Landslide Atlas of India  
https://www.isro.gov.in/Landslide_Atlas_India.html

NRSC/ISRO reports an inventory of approximately 80,000 landslides mapped during 1998–2022 across 17 states and 2 Union Territories. This is the important source for the eventual India-focused spatial dataset.

### 3. NASA COOLR
NASA Cooperative Open Online Landslide Repository  
https://gpm.nasa.gov/applications/landslides/coolr

COOLR combines report-based and event-based landslide inventories. Its ArcGIS service can also be queried programmatically.

### 4. IndLands
Hugging Face / IndLands  
https://huggingface.co/datasets/DataUploader/IndLands

This is a much richer remote-sensing dataset containing terrain, spectral and texture features and annotated landslide regions. The repository currently lists about 1,668 rows but the full downloadable material is about 10.6 GB, so it should be treated as a later-stage dataset rather than the first thing to train locally.

## Expected local dataset

Put the downloaded Mendeley CSV at:

`data/raw/landslide_susceptibility.csv`

Then run:

```bash
pip install -r requirements.txt
python train.py
```

`train.py` executes **dynamic parameter analysis** using scikit-learn's `RandomizedSearchCV` across continuous (`loguniform`, `uniform`) and discrete (`randint`) distributions. The optimal hyperparameters are stored in `outputs/model_parameters.json` with technical definitions and domain purposes for visualization in the Streamlit inference dashboard (`app.py`).

Command-line training options:
- `--n-iter`: Number of parameter configurations sampled dynamically (default: `6`)
- `--cv`: Cross-validation folds (default: `3`)
- `--tune-samples`: Stratified tuning subset size (default: `30000`, `0` for full train set)
- `--skip-search`: Skip search and train directly with baseline hyperparameters
- `--models`: Filter models to train (default: `logistic_regression random_forest xgboost`)

The training script saves the fitted pipelines under `models/`, writes metrics to `outputs/results.json`, and outputs the parameter dictionary to `outputs/model_parameters.json`.

## Models

1. Logistic Regression — interpretable linear baseline
2. Random Forest — nonlinear bagged-tree model
3. XGBoost — nonlinear gradient-boosted-tree model

## Metrics

The evaluation reports:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Confusion matrix
- Classification report

For a landslide warning system, **recall for the high-susceptibility class matters more than raw accuracy**, because missed hazardous areas can be much more consequential than false alarms.

## Current project order

1. Dataset acquisition
2. Data validation / cleaning
3. Train three baseline models
4. Compare models
5. Feature importance / explainability
6. Expand spatial dataset
7. UI/UX
8. API inference service
9. Testing / validation
10. Deployment

## Research caution

Do not mix the ISRO inventory and the 100-row Mendeley susceptibility table as if they were interchangeable labels. They represent different data products. A proper future training dataset should spatially join landslide/non-landslide samples with environmental conditioning factors such as DEM-derived slope/elevation, precipitation, land cover and vegetation indices.
