# PELT Changepoint Detection for Weather Time Series

## What is PELT?

**PELT (Pruned Exact Linear Time)** is a changepoint detection algorithm that identifies points in time series data where the statistical properties change significantly. It searches for moments when the mean, variance, or other characteristics of the data shift, dividing the time series into distinct segments with homogeneous properties.

For weather data, PELT can detect:
- Climate regime shifts
- Changes in temperature patterns
- Variations in precipitation trends
- Transitions between different weather states

The algorithm works by minimizing a cost function that balances segment fit quality against the number of segments, controlled by a penalty parameter.

## How does it assign state (enso_anomaly) ?
1. it applies PELT on each site's multivariate time series to find the changing point, and find the time segments spliting by these changing points.
2. For each site, it calculates the segment average time series data, and compare the L2 distance between this segment avg data with the overall avg data of this site.
    Large distance → unusual climate period.  Small distance → normal climate period

## What This Script Does

This script processes historical weather data (1901-2019) and uses PELT to automatically segment the time series into periods with statistically similar characteristics. It:

1. **Loads multi-site weather data** from a CSV file containing monthly weather observations
2. **Applies PELT changepoint detection** to each site independently using 6 weather features:
   - Mean temperature
   - Max temperature
   - Min temperature
   - Sea level pressure
   - Wind speed
   - Precipitation
3. **Identifies temporal segments** where weather patterns are statistically consistent
4. **Exports results**


## Usage

### Output Format
csv file "PELT_enso_ensemble_results.csv". The final prediciton produced by voting on 21 local sites prediction results.

## Parameters

### `model` (default: `"rbf"`)
- **`"rbf"`**: Detects changes in both mean and variance (Radial Basis Function kernel)
- **`"l2"`**: Detects changes in mean only (L2 norm)

Choose `"rbf"` for comprehensive changepoint detection or `"l2"` for simpler mean-shift detection.

### `penalty` (default: `10`)
Controls the trade-off between segment fit and number of segments:
- **Higher values** (e.g., 20-50): Fewer, longer segments; only detects major changes
- **Lower values** (e.g., 1-5): More, shorter segments; detects subtle changes
- **Typical range**: 5-20 for weather data

Tune this parameter based on your desired segmentation granularity.


