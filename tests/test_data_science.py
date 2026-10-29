import numpy as np
import pandas as pd
import pytest
from src.data_science import descriptive_profile,bootstrap_mean_ci,segment_weather,population_stability_index,period_drift
from src.backtesting import walk_forward_backtest

def test_profile():
    x=descriptive_profile(pd.DataFrame({'x':[1,2,np.nan,4]}),['x'])
    assert x.loc['x','missing_pct']==25

def test_bootstrap_reproducible():
    a=bootstrap_mean_ci(np.arange(30),seed=3)
    assert a==bootstrap_mean_ci(np.arange(30),seed=3)
    assert a['lower']<=a['mean']<=a['upper']

def test_segmentation():
    rng=np.random.default_rng(2)
    x=pd.DataFrame({'a':np.r_[rng.normal(0,.2,30),rng.normal(5,.2,30)],'b':np.r_[rng.normal(0,.2,30),rng.normal(5,.2,30)]})
    out,score=segment_weather(x,['a','b'],2)
    assert out['Cluster'].nunique()==2 and score>0.5

def test_psi_and_drift():
    rng=np.random.default_rng(3)
    x=rng.normal(size=200)
    assert population_stability_index(x,x)<.0001
    df=pd.DataFrame({'Date/Time':pd.date_range('2020-01-01',periods=200),'x':np.r_[x[:100],x[100:]+2]})
    assert period_drift(df,'x')>0

def test_backtest():
    days=pd.date_range('2022-01-01',periods=110)
    df=pd.DataFrame({'Date/Time':days,'Temp_C':15+np.sin(np.arange(110)/8)})
    scores,preds=walk_forward_backtest(df,splits=3)
    assert len(scores)==9 and not preds.empty
    assert set(scores.model)=={'Persistence (lag 1)','Mean baseline','Random Forest'}
