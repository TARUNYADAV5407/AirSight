from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "city_day.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "aqi_model.joblib"
RESULTS_FILE = MODEL_DIR / "model_comparison.csv"

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

# IMPORTANT:
# Current AQI is deliberately NOT a model input. The live government
# feed currently supplies pollutant observations, not a verified overall
# AQI value. The model therefore learns to forecast next-day AQI from
# pollutant readings, city, and calendar features.
NUMERIC_FEATURES = ["Month", "DayOfWeek"] + POLLUTANTS
FEATURES = ["City"] + NUMERIC_FEATURES


# --------------------------------------------------
# 2. PREPROCESSING PIPELINE
# --------------------------------------------------

def make_preprocessor(use_pca=False):
    if use_pca:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full")),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    city_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("city", city_pipeline, ["City"]),
    ])


# --------------------------------------------------
# 3. LOAD AND PREPARE DATA
# --------------------------------------------------

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Make sure city_day.csv is inside the data folder."
        )

    df = pd.read_csv(DATA_FILE)
    required = ["City", "Date", "AQI"] + POLLUTANTS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    df = df.dropna(subset=["City", "Date", "AQI"]).copy()

    for col in POLLUTANTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["City", "Date"])
    df = df.drop_duplicates(subset=["City", "Date"], keep="last")

    # Label: AQI on the next calendar day for the same city.
    df["NextDate"] = df.groupby("City")["Date"].shift(-1)
    df["NextAQI"] = df.groupby("City")["AQI"].shift(-1)
    consecutive = df["NextDate"] == df["Date"] + pd.Timedelta(days=1)
    df = df.loc[consecutive].copy()

    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.weekday
    df = df.dropna(subset=["NextAQI"]).copy()

    if len(df) < 100:
        raise ValueError(f"Only {len(df)} usable records found.")
    return df


# --------------------------------------------------
# 4. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

def main():
    df = load_data()

    # Chronological split to avoid random time-series leakage.
    split_date = df["Date"].quantile(0.80)
    train = df[df["Date"] <= split_date].copy()
    test = df[df["Date"] > split_date].copy()
    if train.empty or test.empty:
        raise ValueError("Training or testing data is empty. Check the date split.")

    X_train, y_train = train[FEATURES], train["NextAQI"]
    X_test, y_test = test[FEATURES], test["NextAQI"]

    print("=" * 60)
    print("AIRSIGHT — LIVE-COMPATIBLE NEXT-DAY AQI MODEL")
    print("=" * 60)
    print("Current AQI is NOT an input feature.")
    print(f"Usable consecutive-day records: {len(df):,}")
    print(f"Training records: {len(train):,}")
    print(f"Testing records: {len(test):,}")
    print(f"Training through: {train['Date'].max().date()}")
    print(f"Testing from: {test['Date'].min().date()}")

    models = {
        "Random Forest (pollutants, no PCA)": Pipeline([
            ("preprocessing", make_preprocessor(use_pca=False)),
            ("regressor", RandomForestRegressor(
                n_estimators=150,
                max_depth=18,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Random Forest + PCA": Pipeline([
            ("preprocessing", make_preprocessor(use_pca=True)),
            ("regressor", RandomForestRegressor(
                n_estimators=150,
                max_depth=18,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Ridge Regression + PCA": Pipeline([
            ("preprocessing", make_preprocessor(use_pca=True)),
            ("regressor", Ridge(alpha=10.0)),
        ]),
    }

    results = []
    fitted_models = {}

    for name, candidate in models.items():
        print(f"\nTraining: {name}")
        candidate.fit(X_train, y_train)
        predictions = candidate.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        fitted_models[name] = candidate

        print(f"MAE:  {mae:.2f} AQI points")
        print(f"RMSE: {rmse:.2f} AQI points")
        print(f"R²:   {r2:.4f}")

        numeric_pipeline = (
            candidate.named_steps["preprocessing"].named_transformers_["numeric"]
        )
        if "pca" in numeric_pipeline.named_steps:
            pca = numeric_pipeline.named_steps["pca"]
            print(f"PCA components retained: {pca.n_components_}")
            print(f"Explained variance retained: {pca.explained_variance_ratio_.sum():.2%}")

    # Persistence baseline: tomorrow equals today's observed AQI.
    # Used for evaluation only; it is not required at live inference time.
    baseline_mae = mean_absolute_error(y_test, test["AQI"].to_numpy())
    results_df = pd.DataFrame(results).sort_values("MAE")
    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nPersistence baseline MAE: {baseline_mae:.2f} AQI points")

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_FILE, index=False)

    joblib.dump({
        "model": best_model,
        "feature_columns": FEATURES,
        "pollutant_columns": POLLUTANTS,
        "target": "Next-day AQI",
        "selected_model": best_name,
        "training_data_end": str(df["Date"].max().date()),
        "split_date": str(split_date.date()),
        "baseline_mae": float(baseline_mae),
        "uses_current_aqi_as_feature": False,
        "feature_design": "city + calendar + pollutant concentrations",
    }, MODEL_FILE)

    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved comparison: {RESULTS_FILE}")
    print("Training complete. Restart the API after training.")


if __name__ == "__main__":
    main()
