"""Safe CSV ingestion. Author: Harpreet Singh."""
import io
import pandas as pd
MAX_UPLOAD_BYTES=20*1024*1024
def read_weather_csv(file_or_bytes):
    if isinstance(file_or_bytes,bytes):
        payload=file_or_bytes
    else:
        payload=file_or_bytes.read()
    if len(payload)>MAX_UPLOAD_BYTES:
        raise ValueError('CSV exceeds the 20 MB upload limit')
    if not payload: raise ValueError('CSV is empty')
    return pd.read_csv(io.BytesIO(payload),low_memory=False)
