import numpy as np
import pandas as pd
from src.ml_forecasting import supervised_frame,train_and_evaluate,next_day_estimate
from src.ethics import audit_dataset
from src.computer_vision import sky_image_analysis
from PIL import Image
from io import BytesIO
def data():
    rng=np.random.default_rng(4);dates=pd.date_range('2025-01-01',periods=180,freq='D')
    return pd.DataFrame({'Date/Time':dates,'Temp_C':20+np.sin(np.arange(180)/12)*5+rng.normal(0,0.5,180),'Weather':['Clear']*180})
def test_forecasting():
    d=data();frame=supervised_frame(d);assert frame.iloc[0]['lag_1']==d.iloc[6]['Temp_C']
    model,metrics,comp=train_and_evaluate(d);assert len(comp)>0 and metrics['MAE']>=0
    dt,value=next_day_estimate(d,model);assert dt==pd.Timestamp('2025-06-30') and np.isfinite(value)
def test_gaps_rejected():
    d=data().drop(index=15)
    try:supervised_frame(d);assert False
    except ValueError:pass
def test_vision():
    im=Image.new('RGB',(16,16),(40,100,230));out=BytesIO();im.save(out,format='PNG')
    result=sky_image_analysis(out.getvalue());assert result['blue_like_pct']>99
def test_audit():
    assert audit_dataset(data())['rows']==180
