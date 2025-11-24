# Gaussian Mixture Model (GMM) Baseline

## 1. Overview

This module applies a **Gaussian Mixture Model (GMM)** to (yearly detrended weather) data on a **per-site basis**.  
Each weather station (`site_id`) is clustered independently using GMM with a configurable number of components.  
The model outputs a **state** (cluster label) for each site-year pair.

This serves as a baseline for comparison with HMM-based models.

---

## 2. Input Data Requirements

The input CSV must contain the following columns:

- `site_id`
- `date` (YYYY-MM-DD)
- **Six meteorological features used for GMM:**
  - `mean_temp`
  - `max_temp`
  - `min_temp`
  - `sea_level_pressure`
  - `wind_speed`
  - `precipitation`

The GMM script will:

- Group data by `site_id`
- Sort records by `date`
- Extract `year` from the `date` column
- Standardize numerical features
- Cluster samples for each site

---

## 3. Running the GMM

Execute:

`python gmm.py`