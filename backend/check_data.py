
from pathlib import Path
import pandas as pd

# Project-root/data/city_day.csv
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "city_day.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_FILE}\n"
        "Check that the file is named city_day.csv inside the data folder."
    )

df = pd.read_csv(DATA_FILE)

print("=" * 50)
print("DATASET INSPECTION")
print("=" * 50)

print("\nShape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head().to_string())

if "Date" in df.columns:
    dates = pd.to_datetime(df["Date"], errors="coerce")
    print("\nDate range:")
    print(dates.min(), "to", dates.max())
    print("Invalid dates:", dates.isna().sum())

if "City" in df.columns:
    print("\nRecords per city:")
    print(df["City"].value_counts().to_string())

print("\nMissing values by column:")
print(df.isna().sum().to_string())

print("\nAQI summary:")
if "AQI" in df.columns:
    print(df["AQI"].describe())

print("\nInspection complete.")
