"""Advanced exploratory data science; author: Harpreet Singh."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score

def descriptive_profile(df, columns):
    cols=[c for c in columns if c in df and pd.api.types.is_numeric_dtype(df[c])]
    if not cols: raise ValueError('No numeric columns available')
    stats=df[cols].describe(percentiles=[.05,.25,.5,.75,.95]).T
    stats['missing_pct']=df[cols].isna().mean()*100
    stats['skewness']=df[cols].skew()
    return stats.replace([np.inf,-np.inf],np.nan)

def bootstrap_mean_ci(values,confidence=.95,n_boot=1000,seed=42):
    if not 0<confidence<1 or n_boot<100: raise ValueError('Invalid bootstrap parameters')
    v=np.asarray(values,dtype=float);v=v[np.isfinite(v)]
    if len(v)<3:raise ValueError('At least three observations required')
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(v,size=len(v),replace=True).mean() for _ in range(n_boot)])
    alpha=(1-confidence)/2
    return {'mean':float(v.mean()),'lower':float(np.quantile(means,alpha)),'upper':float(np.quantile(means,1-alpha)),'n':int(len(v))}

def segment_weather(df,columns,n_clusters=3):
    cols=[c for c in columns if c in df and df[c].notna().sum()>=3 and df[c].nunique(dropna=True)>1]
    if len(cols)<2:raise ValueError('At least two varying numeric features are required')
    if not 2<=n_clusters< len(df):raise ValueError('Cluster count must be between 2 and n-1')
    clean=df[cols].replace([np.inf,-np.inf],np.nan)
    if clean.dropna(how='all').empty:raise ValueError('No valid feature rows')
    pipeline=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),KMeans(n_clusters=n_clusters,random_state=42,n_init=10))
    labels=pipeline.fit_predict(clean)
    result=df.copy();result['Cluster']=labels.astype(str)
    scaled=pipeline[:-1].transform(clean)
    score=float(silhouette_score(scaled,labels)) if len(df)>n_clusters and len(set(labels))>1 else float('nan')
    return result,score

def population_stability_index(reference,current,bins=8):
    ref=np.asarray(reference,dtype=float);cur=np.asarray(current,dtype=float)
    ref=ref[np.isfinite(ref)];cur=cur[np.isfinite(cur)]
    if len(ref)<20 or len(cur)<20:raise ValueError('At least 20 observations per period required')
    if np.ptp(ref)==0:raise ValueError('Reference period has no variation')
    edges=np.unique(np.quantile(ref,np.linspace(0,1,bins+1)))
    if len(edges)<3:raise ValueError('Insufficient unique reference quantiles')
    edges[0]=-np.inf;edges[-1]=np.inf
    a=np.histogram(ref,bins=edges)[0]/len(ref)
    b=np.histogram(cur,bins=edges)[0]/len(cur)
    a=np.maximum(a,1e-6);b=np.maximum(b,1e-6)
    return float(np.sum((b-a)*np.log(b/a)))

def period_drift(df,column,split_fraction=.5):
    if column not in df:raise ValueError('Unknown column')
    ordered=df.sort_values('Date/Time')
    cut=int(len(ordered)*split_fraction)
    if cut<20 or len(ordered)-cut<20:raise ValueError('At least 40 ordered rows required')
    return population_stability_index(ordered[column].iloc[:cut],ordered[column].iloc[cut:])
