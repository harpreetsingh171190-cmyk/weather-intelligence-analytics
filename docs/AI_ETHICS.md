# Responsible AI & Ethics — Harpreet Singh
- **Transparency:** disclose synthetic samples and experimental status prominently.
- **Reliability:** compare with baseline, use temporal holdout, assess drift and out-of-region performance.
- **Fairness:** evaluate across seasons, geographies and rare weather conditions before deployment; sample may underrepresent extremes.
- **Privacy:** avoid embedding GPS/EXIF or personal information in shared datasets and sky photos. Hosting providers may process uploaded content; follow their policies.
- **Safety:** not for emergency alerts, flood/fire predictions, or public safety decisions.
- **Reproducibility:** fixed random seed for ML and LSTM; pin exact versions in deployment lockfiles.
- **Human oversight:** expert review before any operational use; document failure modes and model changes.
