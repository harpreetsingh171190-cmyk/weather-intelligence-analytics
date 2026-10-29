"""Chronological walk-forward evaluation; author: Harpreet Singh."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error
from src.ml_forecasting import supervised_frame

def walk_forward_backtest(df,target='Temp_C',splits=4):
    frame=supervised_frame(df,target)
    if len(frame)<max(60,splits*10):raise ValueError('At least 60 consecutive supervised days required')
    X=frame.drop(columns='target');y=frame['target']
    candidates={'Persistence (lag 1)':None,'Mean baseline':DummyRegressor(strategy='mean'),
      'Random Forest':RandomForestRegressor(n_estimators=80,max_depth=7,min_samples_leaf=2,random_state=42,n_jobs=-1)}
    rows=[];predictions=[]
    for fold,(tr,te) in enumerate(TimeSeriesSplit(n_splits=splits).split(X),1):
        for name,estimator in candidates.items():
            if estimator is None:pred=X.iloc[te]['lag_1'].to_numpy()
            else:
                model=make_pipeline(SimpleImputer(strategy='median'),clone(estimator))
                model.fit(X.iloc[tr],y.iloc[tr]);pred=model.predict(X.iloc[te])
            actual=y.iloc[te].to_numpy()
            rows.append({'fold':fold,'model':name,'MAE':float(mean_absolute_error(actual,pred)),
                         'RMSE':float(np.sqrt(mean_squared_error(actual,pred))), 'test_rows':len(te)})
            predictions.extend({'Date':str(date.date()),'fold':fold,'model':name,'actual':float(a),'predicted':float(p)} for date,a,p in zip(y.index[te],actual,pred))
    return pd.DataFrame(rows),pd.DataFrame(predictions)
