import pandas as pd
import pytest
from src.analysis import prepare_weather, weather_summary

def test_cleaning_and_summary():
    data = pd.DataFrame({"Date/Time": ["2026-01-01 10:00", "2026-01-01 10:00", "invalid"], "Weather": ["Clear", "Clear", "Rain"], "Wind Speed_km/h": [4, 4, 8], "Temp_C": [20, 20, 10]})
    clean = prepare_weather(data)
    assert len(clean) == 1
    assert clean.loc[0, "Month"] == 1
    assert weather_summary(clean).loc[0, "Observations"] == 1

def test_required_columns():
    with pytest.raises(ValueError, match="Missing required"):
        prepare_weather(pd.DataFrame({"x": [1]}))
