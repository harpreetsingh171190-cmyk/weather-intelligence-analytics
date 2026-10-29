"""Interactive professional weather analytics dashboard. Author: Harpreet Singh."""
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analysis import prepare_weather,weather_summary,NUMERIC
from src.advanced import quality_report,rolling_trend,monthly_comparison,anomaly_scores,condition_transitions
from src.io_utils import read_weather_csv
from src.ml_forecasting import train_and_evaluate,next_day_estimate
from src.computer_vision import sky_image_analysis
from src.ethics import audit_dataset
from src.data_science import descriptive_profile,bootstrap_mean_ci,segment_weather,period_drift
from src.backtesting import walk_forward_backtest

st.set_page_config(page_title='Harpreet Singh | Weather Intelligence',page_icon='🌦️',layout='wide')
st.title('🌦️ Weather Intelligence | Analytics Studio')
st.caption('Professional interactive weather analytics • Author: Harpreet Singh')
st.sidebar.header('Data source')
upload=st.sidebar.file_uploader('Upload your CSV (max 20 MB)',type=['csv'])
@st.cache_data(show_spinner=False)
def load_bytes(payload):
    return prepare_weather(read_weather_csv(payload))
try:
    if upload is not None:
        data=load_bytes(upload.getvalue());st.sidebar.success('Uploaded CSV loaded')
    else:
        data=load_bytes((ROOT/'data'/'sample_weather.csv').read_bytes())
        st.sidebar.warning('SYNTHETIC demo data — not real observations')
except (ValueError,UnicodeError,pd.errors.ParserError,OSError) as exc:
    st.error(f'Data loading failed: {exc}');st.stop()

st.sidebar.header('Interactive filters')
conditions=sorted(data['Weather'].dropna().unique().tolist())
chosen=st.sidebar.multiselect('Weather conditions',conditions,default=conditions)
first,last=data['Date/Time'].min().date(),data['Date/Time'].max().date()
selection=st.sidebar.date_input('Date range',(first,last),min_value=first,max_value=last)
if len(selection)!=2: st.info('Select both start and end dates.');st.stop()
start,end=selection
if start>end: st.error('Start date must be before end date.');st.stop()
filtered=data.loc[data['Weather'].isin(chosen)&data['Date/Time'].dt.date.between(start,end)].copy()
if filtered.empty: st.warning('No records match these filters.');st.stop()
numeric=[c for c in NUMERIC if c in filtered and filtered[c].notna().any()]
if not numeric: st.error('No usable numeric weather metrics found.');st.stop()
metric=st.sidebar.selectbox('Primary metric',numeric)
window=st.sidebar.slider('Rolling average (days)',2,30,7)
threshold=st.sidebar.slider('Anomaly threshold (z-score)',1.5,5.0,3.0,0.5)

a,b,c,d=st.columns(4)
a.metric('Records',f'{len(filtered):,}')
b.metric('Weather conditions',filtered['Weather'].nunique())
c.metric('Mean temperature',f"{filtered['Temp_C'].mean():.1f} °C" if 'Temp_C' in filtered else 'N/A')
d.metric('Mean wind speed',f"{filtered['Wind Speed_km/h'].mean():.1f} km/h" if 'Wind Speed_km/h' in filtered else 'N/A')
overview,trends,compare,anomalies,quality,explorer=st.tabs(['📊 Overview','📈 Time series','⚖️ Comparisons','🚨 Anomalies','🧹 Data quality','📥 Data explorer'])
with overview:
    left,right=st.columns(2)
    with left:
        freq=filtered['Weather'].value_counts().rename_axis('Weather').reset_index(name='Observations')
        st.plotly_chart(px.bar(freq,x='Weather',y='Observations',title='Condition frequencies'),width="stretch")
    with right:
        st.plotly_chart(px.histogram(filtered,x=metric,color='Weather',nbins=35,title=f'{metric} distribution'),width="stretch")
    st.plotly_chart(px.scatter(filtered,x='Date/Time',y=metric,color='Weather',title=f'{metric} observations'),width="stretch")
with trends:
    daily=rolling_trend(filtered,metric,window)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=daily['Date/Time'],y=daily['daily_mean'],mode='lines',name='Daily mean'))
    fig.add_trace(go.Scatter(x=daily['Date/Time'],y=daily['rolling_mean'],mode='lines',name=f'{window}-day rolling mean'))
    fig.update_layout(title=f'{metric}: daily and rolling trends',xaxis_title='Date',yaxis_title=metric)
    st.plotly_chart(fig,width="stretch")
    monthly=monthly_comparison(filtered,metric)
    st.plotly_chart(px.bar(monthly,x='month',y=['mean','minimum','maximum'],barmode='group',title='Monthly comparisons'),width="stretch")
    st.dataframe(monthly,width="stretch")
with compare:
    st.plotly_chart(px.box(filtered,x='Weather',y=metric,points='outliers',title=f'{metric} by condition'),width="stretch")
    st.dataframe(weather_summary(filtered),width="stretch")
    if len(numeric)>=2:
        corr=filtered[numeric].corr()
        st.plotly_chart(px.imshow(corr,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale='RdBu_r',title='Metric correlations'),width="stretch")
        st.caption('Correlation is not evidence of causation.')
    trans=condition_transitions(filtered)
    if not trans.empty:
        st.plotly_chart(px.imshow(trans,labels={'x':'Current','y':'Previous','color':'Count'},title='Consecutive-record condition transitions'),width="stretch")
        st.caption('Transitions are between adjacent rows; observations may not be equally spaced.')
with anomalies:
    result=anomaly_scores(filtered,metric,threshold)
    flagged=result[result['is_anomaly']]
    st.metric('Flagged observations',len(flagged))
    st.plotly_chart(px.scatter(result,x='Date/Time',y=metric,color='is_anomaly',title=f'{metric}: statistical anomaly screening'),width="stretch")
    st.dataframe(flagged,width="stretch")
    st.caption('Z-score screening is exploratory, not an operational weather alert or forecast.')
with quality:
    report=quality_report(filtered)
    st.dataframe(report,width="stretch")
    st.plotly_chart(px.bar(report,x='column',y='missing_pct',title='Missing values by column (%)'),width="stretch")
    st.caption('Quality statistics reflect cleaned, filtered records. Invalid timestamps and duplicate rows were removed during preparation.')
with explorer:
    st.dataframe(filtered,width="stretch",height=420)
    st.download_button('Download filtered CSV',filtered.to_csv(index=False).encode(),'filtered_weather.csv','text/csv')
    st.download_button('Download comparison CSV',weather_summary(filtered).to_csv(index=False).encode(),'weather_comparison.csv','text/csv')
    st.download_button('Download quality report',quality_report(filtered).to_csv(index=False).encode(),'data_quality.csv','text/csv')
    st.download_button('Download anomalies',anomaly_scores(filtered,metric,threshold).to_csv(index=False).encode(),'anomaly_screening.csv','text/csv')
st.divider()
st.header('🤖 AI & Machine Learning Lab')
mltab,dltab,cvtab,ethtab=st.tabs(['ML forecasting','Deep learning (optional)','Computer vision','Responsible AI & ethics'])
with mltab:
    st.info('Next-day prediction using lagged historical values; chronological 80/20 holdout. Demo data is synthetic. NOT a meteorological forecast.')
    targets=[x for x in ['Temp_C','Rel Hum_%','Wind Speed_km/h'] if x in data and data[x].notna().any()]
    if targets:
        target=st.selectbox('Prediction target',targets)
        model_name=st.selectbox('Regression model',['Random Forest','Gradient Boosting','Baseline (mean)'])
        if st.button('Train and evaluate model'):
            try:
                with st.spinner('Training...'):
                    model,metrics,results=train_and_evaluate(data,target,model_name)
                    next_date,next_pred=next_day_estimate(data,model,target)
                x,y,z=st.columns(3)
                x.metric('Holdout MAE',f"{metrics['MAE']:.2f}")
                y.metric('Holdout RMSE',f"{metrics['RMSE']:.2f}")
                z.metric('Holdout R²',f"{metrics['R2']:.2f}")
                st.plotly_chart(px.line(results,x='Date',y=['Actual','Predicted'],title='Chronological holdout: actual vs predicted'),width="stretch")
                st.write(f'Illustrative next-day estimate for {next_date.date()}: {next_pred:.2f} ({target})')
                st.download_button('Export holdout predictions',results.to_csv(index=False),'holdout_predictions.csv',on_click='ignore')
            except ValueError as exc:st.warning(str(exc))
with dltab:
    st.warning('Optional CPU-intensive LSTM. Requires TensorFlow and at least 100 continuous daily values. Model is educational only.')
    if st.button('Train LSTM (optional)'):
        try:
            from src.deep_learning import fit_lstm
            series=(data[['Date/Time','Temp_C']].dropna().set_index('Date/Time').sort_index().resample('D').mean().dropna())
            if series.index.to_series().diff().dropna().gt(pd.Timedelta(days=1)).any():raise ValueError('Gaps in daily time series')
            with st.spinner('Training LSTM...'):
                result=fit_lstm(series['Temp_C'].values)
            st.write({'MAE':result['MAE'],'RMSE':result['RMSE']})
            st.plotly_chart(px.line(pd.DataFrame({'Actual':result['actual'],'Predicted':result['predicted']}),title='LSTM holdout comparison'),width="stretch")
        except (ImportError,ValueError,KeyError) as exc:st.warning(str(exc))
with cvtab:
    st.info('Upload a sky photograph. Pixel-color heuristics highlight blue-like and bright-gray-like regions. This is NOT a validated cloud classifier.')
    photo=st.file_uploader('Sky image (PNG/JPG, up to 8 MB)',type=['png','jpg','jpeg'],key='sky_image')
    if photo:
        try:
            out=sky_image_analysis(photo.getvalue())
            st.image(out['preview'],caption='Uploaded sky photo')
            st.metric('Blue-like pixels',f"{out['blue_like_pct']:.1f}%")
            st.metric('Bright gray-like pixels',f"{out['bright_gray_like_pct']:.1f}%")
            st.caption('Colors depend on camera, lighting, scene content, and editing; percentages are not cloud-cover measurements.')
        except (ValueError,OSError) as exc:st.error(str(exc))
with ethtab:
    st.subheader('AI Ethics & Model Governance')
    st.json(audit_dataset(data))
    st.markdown("""**Responsible use checklist:**\n- Verify dataset origin, consent, license, time zone and representativeness.\n- Synthetic demonstration data must never be represented as observed weather.\n- Compare against a simple baseline; report chronological holdout metrics and uncertainty.\n- Weather can vary across seasons and regions; evaluate each subgroup before deployment.\n- Avoid emergency, aviation, agriculture, or public-safety decisions based on this prototype.\n- Image color heuristics are not scientifically validated cloud-cover measurements.\n- No photos or CSVs are transmitted by this code to a third-party API; review hosting platform policies if deploying online.""")
    st.caption('See docs/MODEL_CARD.md and docs/AI_ETHICS.md for limitations and governance.')
st.caption('Synthetic demo data. No live feeds, official warnings, operational forecasts, or validated image models.')

st.divider()
st.header('🔬 Professional Data Science Studio')
prof,seg,drift,backtest=st.tabs(['Statistical profiling','Unsupervised segmentation','Data drift monitoring','Walk-forward validation'])
with prof:
    st.subheader('Distribution and uncertainty')
    st.dataframe(descriptive_profile(filtered,numeric),width="stretch")
    try:
        ci=bootstrap_mean_ci(filtered[metric].dropna(),confidence=.95)
        st.metric('Sample mean',f"{ci['mean']:.2f}")
        st.write(f"Exploratory bootstrap 95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}], n={ci['n']}")
        st.caption('IID bootstrap assumes independent observations; serially correlated weather observations can make this interval too narrow. Not a forecast interval.')
    except ValueError as exc:st.warning(str(exc))
    st.plotly_chart(px.violin(filtered,x='Weather',y=metric,box=True,title='Weather distribution by condition'),width="stretch")
with seg:
    st.caption('K-means discovers exploratory feature-based segments; cluster numbers are not meteorological classifications.')
    cols=st.multiselect('Segmentation features',numeric,default=numeric[:min(3,len(numeric))])
    count=st.slider('Number of segments',2,6,3)
    if st.button('Run clustering'):
        try:
            clusters,score=segment_weather(filtered,cols,count)
            st.write(f'Silhouette score: {score:.3f}' if pd.notna(score) else 'Silhouette score unavailable')
            st.dataframe(clusters.groupby('Cluster')[cols].mean(),width="stretch")
            if len(cols)>=2:st.plotly_chart(px.scatter(clusters,x=cols[0],y=cols[1],color='Cluster',title='Exploratory weather segments'),width="stretch")
            st.download_button('Download segment assignments',clusters.to_csv(index=False),'weather_segments.csv',on_click='ignore')
        except ValueError as exc:st.warning(str(exc))
with drift:
    st.caption('Population Stability Index (PSI) compares the first and second halves of the chronologically ordered data; this is a heuristic, not a significance test.')
    try:
        psi=period_drift(filtered,metric)
        st.metric('PSI (early vs late)',f'{psi:.3f}')
        st.warning('A shift can reflect seasonality, geography, or changing measurement practices. Investigate before retraining.')
    except ValueError as exc:st.warning(str(exc))
with backtest:
    st.caption('Expanding-window time-series cross-validation compares persistence, mean baseline and Random Forest. It is historical validation, not a real-world forecast guarantee.')
    bt_targets=[x for x in ['Temp_C','Rel Hum_%','Wind Speed_km/h'] if x in data and data[x].notna().any()]
    if not bt_targets:st.info('No suitable target column (Temp_C, Rel Hum_%, Wind Speed_km/h) found for backtesting.')
    bt_target=st.selectbox('Backtesting target',bt_targets,key='bt_target') if bt_targets else None
    if bt_target and st.button('Run walk-forward evaluation'):
        try:
            with st.spinner('Evaluating chronological folds...'):
                scores,preds=walk_forward_backtest(data,bt_target)
            st.dataframe(scores,width="stretch")
            st.plotly_chart(px.box(scores,x='model',y='MAE',points='all',title='MAE across temporal folds'),width="stretch")
            st.download_button('Download fold metrics',scores.to_csv(index=False),'walk_forward_metrics.csv',on_click='ignore')
            st.download_button('Download predictions',preds.to_csv(index=False),'walk_forward_predictions.csv',on_click='ignore')
        except ValueError as exc:st.warning(str(exc))
