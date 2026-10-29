# Changelog

## 2.0.1 — Error fixes
- Replaced deprecated `use_container_width` with `width="stretch"` (Streamlit >= 1.50).
- Download buttons inside result blocks no longer reset the page (`on_click="ignore"`).
- Guarded the backtesting target selector against missing columns.
- Added `pytest.ini` (`pythonpath = .`) so `pytest` works without `python -m`.
- Added `ipython` to requirements for the notebook; allowed pandas 3.
- Moved dashboard imports to the top of the file.

## 2.0 — Data Science Portfolio Upgrade
- Added advanced descriptive statistics, bootstrap uncertainty, clustering, drift monitoring and chronological backtesting.
- Added dashboard Data Science Studio and tests.
- Preserved existing ML, deep learning, computer vision and ethics modules.

Author: Harpreet Singh.
