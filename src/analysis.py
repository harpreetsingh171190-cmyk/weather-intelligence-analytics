"""Reusable, tested weather data preparation and analytics.
Author: Harpreet Singh
"""
import pandas as pd

REQUIRED = {"Date/Time", "Weather", "Wind Speed_km/h"}
NUMERIC = ["Temp_C", "Rel Hum_%", "Wind Speed_km/h", "Visibility_km", "Press_kPa"]

def prepare_weather(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("The CSV contains no records.")
    data = df.copy()
    data.columns = data.columns.astype(str).str.strip()
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    data["Date/Time"] = pd.to_datetime(data["Date/Time"], errors="coerce")
    for col in NUMERIC:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data["Weather"] = data["Weather"].astype("string").str.strip().fillna("Unknown")
    data = data.dropna(subset=["Date/Time"]).drop_duplicates().sort_values("Date/Time")
    if data.empty:
        raise ValueError("No valid timestamps remain after cleaning.")
    data["Month"] = data["Date/Time"].dt.month
    data["Hour"] = data["Date/Time"].dt.hour
    return data.reset_index(drop=True)

def weather_summary(data: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in NUMERIC if c in data]
    if not numeric:
        return data.groupby("Weather", dropna=False).size().rename("Observations").reset_index()
    return data.groupby("Weather", dropna=False)[numeric].mean().round(2).join(
        data.groupby("Weather", dropna=False).size().rename("Observations")
    ).reset_index()
