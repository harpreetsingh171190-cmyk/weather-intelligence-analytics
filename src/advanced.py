"""Advanced weather analytics helpers. Author: Harpreet Singh."""
import numpy as np
import pandas as pd

def quality_report(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"column":data.columns,"missing":data.isna().sum().values,
                         "missing_pct":(data.isna().mean()*100).round(2).values,
                         "unique":data.nunique(dropna=True).values})

def rolling_trend(data: pd.DataFrame, metric: str, window: int = 7) -> pd.DataFrame:
    if metric not in data or not pd.api.types.is_numeric_dtype(data[metric]):
        raise ValueError("Select an available numeric metric")
    if window < 1: raise ValueError("window must be positive")
    daily=data.set_index('Date/Time')[metric].resample('D').mean().rename('daily_mean').to_frame()
    daily['rolling_mean']=daily['daily_mean'].rolling(window, min_periods=1).mean()
    return daily.reset_index()

def monthly_comparison(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in data or not pd.api.types.is_numeric_dtype(data[metric]):
        raise ValueError("Select an available numeric metric")
    result=data.copy()
    result['month']=result['Date/Time'].dt.to_period('M').astype(str)
    return result.groupby('month',as_index=False).agg(mean=(metric,'mean'),minimum=(metric,'min'),maximum=(metric,'max'),observations=(metric,'count'))

def anomaly_scores(data: pd.DataFrame, metric: str, threshold: float = 3.0) -> pd.DataFrame:
    if threshold <= 0: raise ValueError('threshold must be positive')
    if metric not in data or not pd.api.types.is_numeric_dtype(data[metric]):
        raise ValueError('Select an available numeric metric')
    result=data[['Date/Time',metric]].copy()
    values=result[metric]
    std=values.std(ddof=0)
    result['z_score']=(values-values.mean())/std if pd.notna(std) and std>0 else np.nan
    result['is_anomaly']=result['z_score'].abs().gt(threshold).fillna(False)
    return result

def condition_transitions(data: pd.DataFrame) -> pd.DataFrame:
    ordered=data.sort_values('Date/Time')['Weather'].astype(str)
    transitions=pd.crosstab(ordered.shift().rename('previous'),ordered.rename('current'))
    return transitions

def temperature_humidity_index(data: pd.DataFrame) -> pd.DataFrame:
    """Simple temperature/humidity relationship view; NOT a medical heat index."""
    needed={'Temp_C','Rel Hum_%'}
    if not needed.issubset(data.columns): raise ValueError('Temperature and humidity required')
    return data[['Date/Time','Temp_C','Rel Hum_%','Weather']].dropna().copy()
