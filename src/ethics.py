"""Document data limitations and simple dataset auditing. Author: Harpreet Singh."""
import pandas as pd
def audit_dataset(df):
    result={'rows':len(df),'start':str(df['Date/Time'].min()),'end':str(df['Date/Time'].max()),'duplicate_timestamps':int(df['Date/Time'].duplicated().sum()),'synthetic_or_unknown_provenance':'Verify externally; the app cannot authenticate CSV provenance'}
    if 'Weather' in df:result['condition_distribution']=df['Weather'].value_counts().to_dict()
    result['missing_by_column']=df.isna().mean().round(3).to_dict()
    return result
