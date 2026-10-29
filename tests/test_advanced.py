import numpy as np
import pandas as pd
import pytest
from src.analysis import prepare_weather
from src.advanced import quality_report,rolling_trend,monthly_comparison,anomaly_scores,condition_transitions
from src.io_utils import read_weather_csv
@pytest.fixture
def weather():
    return prepare_weather(pd.DataFrame({'Date/Time':['2025-01-01','2025-01-02','2025-02-01'],'Weather':['Clear','Rain','Clear'],'Wind Speed_km/h':[1,2,3],'Temp_C':[10,20,30]}))
def test_quality(weather):
    assert quality_report(weather).loc[lambda x:x['column']=='Temp_C','missing'].iloc[0]==0
def test_rolling(weather):
    result=rolling_trend(weather,'Temp_C',2)
    assert result['rolling_mean'].iloc[0]==10
def test_monthly(weather):
    result=monthly_comparison(weather,'Temp_C')
    assert result.iloc[0]['mean']==15
def test_anomalies_constant(weather):
    weather['Temp_C']=10
    assert not anomaly_scores(weather,'Temp_C')['is_anomaly'].any()
def test_anomalies_invalid(weather):
    with pytest.raises(ValueError): anomaly_scores(weather,'Temp_C',0)
def test_transitions(weather):
    assert condition_transitions(weather).values.sum()==2
def test_upload_limit():
    with pytest.raises(ValueError): read_weather_csv(b'')
def test_bad_metric(weather):
    with pytest.raises(ValueError): rolling_trend(weather,'missing')
