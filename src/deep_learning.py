"""Optional TensorFlow LSTM, temporal holdout; illustrative not operational. Author: Harpreet Singh."""
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
def fit_lstm(series,window=14,epochs=12):
    try:
        import tensorflow as tf
    except ImportError as exc:raise ImportError('Install optional deep learning dependencies: pip install -r requirements-deep-learning.txt') from exc
    values=np.asarray(series,dtype='float32')
    if len(values)<100 or not np.isfinite(values).all():raise ValueError('Need 100+ finite, chronologically ordered daily values')
    cut=int(len(values)*0.8)
    mean=float(values[:cut].mean());std=float(values[:cut].std()) or 1.
    scaled=(values-mean)/std
    x=[];y=[];idx=[]
    for i in range(window,len(values)):
        x.append(scaled[i-window:i]);y.append(scaled[i]);idx.append(i)
    x=np.asarray(x,dtype='float32')[...,None];y=np.asarray(y,dtype='float32');idx=np.asarray(idx)
    train=idx<cut
    tf.keras.utils.set_random_seed(42)
    model=tf.keras.Sequential([tf.keras.layers.Input(shape=(window,1)),tf.keras.layers.LSTM(24),tf.keras.layers.Dense(1)])
    model.compile(optimizer='adam',loss='mse')
    model.fit(x[train],y[train],epochs=epochs,batch_size=16,shuffle=False,verbose=0)
    predictions=model.predict(x[~train],verbose=0).ravel()*std+mean
    actual=y[~train]*std+mean
    return {'MAE':float(mean_absolute_error(actual,predictions)),'RMSE':float(np.sqrt(mean_squared_error(actual,predictions))), 'actual':actual,'predicted':predictions,'model':model}
