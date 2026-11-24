from pathlib import Path
import joblib
import pandas as pd

MODEL_DIR = Path("models")

LABELS = {0: "No Landslide", 1: "Landslide"}

FEATURES = [
    "elevation", "slope", "aspect", "curvature",
    "plan_curvature", "profile_curvature",
    "twi", "tri", "spi", "fdr",
    "NDVI", "NDWI", "ARVI", "EVI", "GNDVI", "GRVI",
    "MSAVI", "SAVI", "BSI", "mNDMI", "mNDWI",
    "ASM", "Contrast", "Dissimilarity", "Energy",
    "Entropy", "GLCMCorrelation", "GLCMMean", "Homogeneity",
]


def load_model(model_name="xgboost"):
    model_path = MODEL_DIR / f"{model_name}.joblib"
    return joblib.load(model_path)


def predict(features, model_name="xgboost"):
    model = load_model(model_name)
    row = pd.DataFrame([features])[FEATURES]

    prediction = int(model.predict(row)[0])
    result = {
        "class_id": prediction,
        "prediction": LABELS.get(prediction, str(prediction)),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        result["probabilities"] = {
            LABELS.get(cls, str(cls)): round(float(prob), 4)
            for cls, prob in zip(model.classes_, probabilities)
        }

    return result


if __name__ == "__main__":
    sample = {
        "elevation": 650.0, "slope": 32.0, "aspect": 180.0,
        "curvature": 0.12, "plan_curvature": -0.015, "profile_curvature": 0.021,
        "twi": 6.8, "tri": 7.5, "spi": 56.3, "fdr": 4,
        "NDVI": 0.44, "NDWI": -0.28, "ARVI": 0.49, "EVI": 28.3,
        "GNDVI": 0.39, "GRVI": 0.05, "MSAVI": 0.61, "SAVI": 0.66,
        "BSI": -0.10, "mNDMI": 0.13, "mNDWI": -0.28,
        "ASM": 1.37, "Contrast": 1.5, "Dissimilarity": 1.5,
        "Energy": 1.17, "Entropy": 0.78, "GLCMCorrelation": 0.998,
        "GLCMMean": 24.75, "Homogeneity": 1.25,
    }

    output = predict(sample)
    print("Prediction:", output["prediction"])
    print("Class ID:  ", output["class_id"])
    if "probabilities" in output:
        print("Probabilities:", output["probabilities"])
