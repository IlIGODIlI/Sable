from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.evaluate import evaluate_model

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

DATASET_FILES = [
    DATA_DIR / "dataset.csv",
    DATA_DIR / "maharashtra_dataset.csv",
    DATA_DIR / "Himachal Pradesh_dataset.csv",
]

TARGET = "Decision"

#use library from sckit learn

FEATURES = [
    "elevation", "slope", "aspect", "curvature",
    "plan_curvature", "profile_curvature",
    "twi", "tri", "spi", "fdr",
    "NDVI", "NDWI", "ARVI", "EVI", "GNDVI", "GRVI",
    "MSAVI", "SAVI", "BSI", "mNDMI", "mNDWI",
    "ASM", "Contrast", "Dissimilarity", "Energy",
    "Entropy", "GLCMCorrelation", "GLCMMean", "Homogeneity",
]

N_PER_CLASS = 100_000


def load_and_merge():
    frames = []
    for path in DATASET_FILES:
        if not path.exists():
            continue

        print(f"Loading {path.name}...")
        df = pd.read_csv(path, usecols=FEATURES + [TARGET], low_memory=False)
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=[TARGET])
        df[TARGET] = df[TARGET].astype(int)

        pos = df[df[TARGET] == 1]
        neg = df[df[TARGET] == 0]
        n_pos = min(len(pos), N_PER_CLASS)
        n_neg = min(len(neg), N_PER_CLASS)

        sampled = pd.concat([
            pos.sample(n=n_pos, random_state=42),
            neg.sample(n=n_neg, random_state=42),
        ])
        frames.append(sampled)
        print(f"{len(df):,} rows -> sampled {len(sampled):,} (pos={n_pos:,}, neg={n_neg:,})")

    if not frames:
        raise RuntimeError("No dataset files found in data/raw.")

    combined = pd.concat(frames, ignore_index=True)
    X = combined[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = combined[TARGET]
    return X, y


def print_distribution(y):
    counts = y.value_counts().sort_index()
    total = len(y)
    print("Class distribution:")
    for cls, cnt in counts.items():
        name = "Landslide" if cls == 1 else "No Landslide"
        print(f"  Class {cls} ({name}): {cnt:,} ({cnt / total * 100:.1f}%)")
    print()


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    X, y = load_and_merge()
    print(f"\nDataset shape: {X.shape}")
    print_distribution(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}\n")

    scale_pos_weight = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=4,
            )),
        ]),
    }

    results = {}
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
            model_name=name,
            output_dir=OUTPUT_DIR,
            labels=[0, 1],
        )

        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")
        results[name] = metrics

        summary = {k: v for k, v in metrics.items() if k != "classification_report"}
        print(json.dumps(summary, indent=2))
        print()

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Training complete.")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
