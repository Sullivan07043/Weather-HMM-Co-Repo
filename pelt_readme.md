# PELT Changepoint Detection for Weather Time Series

## What is PELT?

**PELT (Pruned Exact Linear Time)** is a changepoint detection algorithm that identifies points in time series data where the statistical properties change significantly. It searches for moments when the mean, variance, or other characteristics of the data shift, dividing the time series into distinct segments with homogeneous properties.

For weather data, PELT can detect:
- Climate regime shifts
- Changes in temperature patterns
- Variations in precipitation trends
- Transitions between different weather states

The algorithm works by minimizing a cost function that balances segment fit quality against the number of segments, controlled by a penalty parameter.

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
4. **Exports results** with segment boundaries, dates, and indices for further analysis

## Usage


### Input Data Format

The script expects a CSV file with the following columns:
- `site_id`: Unique identifier for weather station
- `date`: Timestamp (convertible to datetime)
- `mean_temp`: Mean temperature
- `max_temp`: Maximum temperature
- `min_temp`: Minimum temperature
- `sea_level_pressure`: Sea level pressure
- `wind_speed`: Wind speed
- `precipitation`: Precipitation amount

### Running the Script

**Basic usage:**
```python
python script_name.py
```

This will:
1. Load `weather_1901_2019_monthly_continuous.csv`
2. Run PELT with default parameters (`model="rbf"`, `penalty=10`)
3. Save results to `pelt_segments_enso24.csv`

**Custom usage in your own code:**
```python
from script_name import load_data, run_pelt_all_sites

# Load data
site_dict = load_data("your_weather_data.csv")

# Run PELT with custom parameters
segments = run_pelt_all_sites(
    site_dict, 
    model="rbf",  # 'rbf' for mean+variance, 'l2' for mean only
    penalty=15    # Higher = fewer segments, Lower = more segments
)

# Save results
segments.to_csv("my_segments.csv", index=False)
```

### Output Format

The output CSV (`pelt_segments_enso24.csv`) contains:
- `site_id`: Weather station identifier
- `segment_id`: Sequential segment number for each site
- `start_idx`: Starting index in the time series
- `end_idx`: Ending index (inclusive)
- `start_date`: First date of segment
- `end_date`: Last date of segment
- `length`: Number of time points in segment

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

## Helper Functions

### `gen_site_csv()`
Generates a `sites_to_process.csv` file with 24 predefined weather station IDs (USAF-WBAN format).

### `load_data(path)`
Loads weather data and returns a dictionary:
```python
{
    "site_id": {
        "X": numpy.ndarray,  # (T, 6) feature matrix
        "dates": list        # List of timestamps
    }
}
```

### `run_pelt_one_site(X, model, penalty)`
Runs PELT on a single site's feature matrix, returns breakpoint indices.

### `breakpoints_to_segments(dates, breakpoints)`
Converts breakpoint indices to segment metadata with dates.

## Example Workflow

```python
# 1. Load your data
site_dict = load_data("weather_data.csv")

# 2. Run segmentation
segments = run_pelt_all_sites(site_dict, model="rbf", penalty=12)

# 3. Analyze results
print(f"Total segments detected: {len(segments)}")
print(f"Average segment length: {segments['length'].mean():.1f} months")

# 4. Filter long-term stable periods
stable_periods = segments[segments['length'] > 120]  # >10 years
print(f"Stable periods (>10 years): {len(stable_periods)}")
```

