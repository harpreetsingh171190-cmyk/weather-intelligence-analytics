"""Leakage-aware one-step-ahead regression. Author: Harpreet Singh."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

def supervised_frame(df, target='Temp_C', lags=(1,2,3,7)):
    if target not in df: raise ValueError(f'Missing target: {target}')
    if 'Date/Time' not in df: raise ValueError('Missing Date/Time')
    daily=(df[['Date/Time',target]].dropna().set_index('Date/Time').sort_index()
            .resample('D').mean().dropna())
    if daily.index.to_series().diff().dropna().gt(pd.Timedelta(days=1)).any():
        raise ValueError('Daily observations have gaps; supply a continuous daily series to avoid misleading lag features.')
    y=daily[target]
    features=pd.DataFrame(index=daily.index)
    for lag in lags:features[f'lag_{lag}']=y.shift(lag)
    features['rolling_7_prior']=y.shift(1).rolling(7).mean()
    features['day_sin']=np.sin(2*np.pi*daily.index.dayofyear/365.25)
    features['day_cos']=np.cos(2*np.pi*daily.index.dayofyear/365.25)
    features['target']=y
    return features.dropna()

def train_and_evaluate(df,target='Temp_C',model_name='Random Forest',test_fraction=0.2):
    frame=supervised_frame(df,target)
    if len(frame)<45:raise ValueError('Need at least 45 consecutive daily supervised rows for training and testing.')
    split=int(len(frame)*(1-test_fraction))
    train,test=frame.iloc[:split],frame.iloc[split:]
    cols=[c for c in frame if c!='target']
    models={'Random Forest':RandomForestRegressor(n_estimators=120,max_depth=8,min_samples_leaf=2,random_state=42,n_jobs=-1),
            'Gradient Boosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=10,random_state=42),
            'Baseline (mean)':DummyRegressor(strategy='mean')}
    if model_name not in models:raise ValueError('Unknown model')
    pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('model',models[model_name])])
    pipe.fit(train[cols],train['target'])
    prediction=pipe.predict(test[cols]);truth=test['target'].to_numpy()
    metrics={'MAE':float(mean_absolute_error(truth,prediction)),
             'RMSE':float(np.sqrt(mean_squared_error(truth,prediction))),
             'R2':float(r2_score(truth,prediction)) if len(test)>1 else float('nan')}
    comparison=pd.DataFrame({'Date':test.index,'Actual':truth,'Predicted':prediction})
    return pipe,metrics,comparison

def next_day_estimate(df,model,target='Temp_C'):
    daily=(df[['Date/Time',target]].dropna().set_index('Date/Time').sort_index().resample('D').mean().dropna())
    if len(daily)<7:raise ValueError('At least 7 days are required')
    if daily.index.to_series().diff().dropna().gt(pd.Timedelta(days=1)).any():raise ValueError('Daily observations have gaps')
    next_date=daily.index[-1]+pd.Timedelta(days=1)
    y=daily[target]
    values={f'lag_{lag}':float(y.iloc[-lag]) for lag in (1,2,3,7)}
    values.update(rolling_7_prior=float(y.iloc[-7:].mean()),day_sin=float(np.sin(2*np.pi*next_date.dayofyear/365.25)),day_cos=float(np.cos(2*np.pi*next_date.dayofyear/365.25)))
    return next_date,float(model.predict(pd.DataFrame([values]))[0])
