
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "city_day.csv"

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

NUMERIC = ["AQI", "Month", "DayOfWeek"] + POLLUTANTS
FEATURES = ["City"] + NUMERIC


def main():
    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

    df = df.dropna(subset=["City", "Date", "AQI"]).copy()

    for col in POLLUTANTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["City", "Date"])
    df = df.drop_duplicates(["City", "Date"], keep="last")

    # Next calendar day's AQI for the same city
    df["NextDate"] = df.groupby("City")["Date"].shift(-1)
    df["NextAQI"] = df.groupby("City")["AQI"].shift(-1)

    df = df[
        df["NextDate"] == df["Date"] + pd.Timedelta(days=1)
    ].copy()

    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.weekday
    df = df.dropna(subset=["NextAQI"]).copy()

    # Split by DATE, not randomly by rows.
    dates = np.sort(df["Date"].unique())

    train_end = dates[int(len(dates) * 0.70)]
    validation_end = dates[int(len(dates) * 0.85)]

    train = df[df["Date"] < train_end]
    validation = df[
        (df["Date"] >= train_end)
        & (df["Date"] < validation_end)
    ]
    test = df[df["Date"] >= validation_end]

    print("=" * 55)
    print("AIRSIGHT CHRONOLOGICAL VALIDATION")
    print("=" * 55)
    print(f"Training:   {train['Date'].min().date()} to "
          f"{train['Date'].max().date()} ({len(train):,} rows)")
    print(f"Validation: {validation['Date'].min().date()} to "
          f"{validation['Date'].max().date()} ({len(validation):,} rows)")
    print(f"Test:       {test['Date'].min().date()} to "
          f"{test['Date'].max().date()} ({len(test):,} rows)")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    city_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("city", city_pipeline, ["City"]),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train = train[FEATURES]
    y_train = train["NextAQI"]

    print("\nTraining Random Forest on training period...")
    model.fit(X_train, y_train)

    # Validation is for checking generalization.
    X_val = validation[FEATURES]
    y_val = validation["NextAQI"]
    val_predictions = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, val_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    val_r2 = r2_score(y_val, val_predictions)
    val_baseline = mean_absolute_error(y_val, X_val["AQI"])

    print("\n========== VALIDATION RESULTS ==========")
    print(f"Model MAE:    {val_mae:.2f}")
    print(f"Model RMSE:   {val_rmse:.2f}")
    print(f"Model R2:     {val_r2:.4f}")
    print(f"Baseline MAE: {val_baseline:.2f}")

    if val_mae < val_baseline:
        print("Validation result: model beats the baseline.")
    else:
        print("Validation result: model does not beat the baseline.")

    # Evaluate once on the final later period.
    X_test = test[FEATURES]
    y_test = test["NextAQI"]
    test_predictions = model.predict(X_test)

    print("\n========== FINAL HOLDOUT RESULTS ==========")
    print(f"Test MAE:    {mean_absolute_error(y_test, test_predictions):.2f}")
    print(f"Test RMSE:   {np.sqrt(mean_squared_error(y_test, test_predictions)):.2f}")
    print(f"Test R2:     {r2_score(y_test, test_predictions):.4f}")
    print(f"Baseline MAE: {mean_absolute_error(y_test, X_test['AQI']):.2f}")

    print("\nValidation completed. No model files were overwritten.")


if __name__ == "__main__":
    main()
