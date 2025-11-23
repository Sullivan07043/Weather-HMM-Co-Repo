# ENSO ONI Historical Data (1950-2000)

## Data Source

This dataset is derived from the official **Oceanic Niño Index (ONI)** maintained by NOAA Climate Prediction Center.

**Source**: [https://ggweather.com/enso/oni.htm](https://ggweather.com/enso/oni.htm)

## File: `enso_oni_data_1950_2000.csv`

### Description

A comprehensive record of ENSO (El Niño-Southern Oscillation) events from 1950 to 2000, classifying each year as El Niño, La Niña, or Normal conditions based on official ONI records.

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| `year` | Integer | Calendar year (1950-2000) |
| `enso_type` | String | Event classification: `El_Nino`, `La_Nina`, or `Normal` |
| `is_el_nino` | Binary | 1 if El Niño year, 0 otherwise |
| `is_la_nina` | Binary | 1 if La Niña year, 0 otherwise |
| `enso_anomaly` | Binary | 1 if any ENSO anomaly (El Niño or La Niña), 0 if Normal |

### Data Statistics

**Time Period**: 1950-2000 (51 years)

**Event Distribution**:
- **El Niño**: 18 years (35.3%)
- **La Niña**: 17 years (33.3%)
- **Normal**: 16 years (31.4%)
- **Total Anomalies**: 35 years (68.6%)

### El Niño Years (18 total)

```
1951, 1952, 1953, 1957, 1958, 1963, 1965, 1968, 1969, 1972, 1976, 1977, 1979, 1982, 1986, 1987, 1991, 1994, 1997
```

**Notable Events**:
- **1982-83**: Very Strong El Niño (strongest of 20th century until 1997-98)
- **1997-98**: Very Strong El Niño (strongest on record for the period)
- **1957-58**: Strong El Niño
- **1987-88**: Strong El Niño
- **1991-92**: Strong El Niño

### La Niña Years (17 total)

```
1954, 1955, 1956, 1964, 1970, 1971, 1974, 1975, 1983, 1984, 1985, 1988, 1995, 1998, 1999, 2000
```

**Notable Events**:
- **1973-74**: Strong La Niña
- **1975-76**: Strong La Niña
- **1988-89**: Strong La Niña
- **1998-99**: Strong La Niña
- **1999-00**: Strong La Niña

### Normal Years (16 total)

```
1950, 1959, 1960, 1961, 1962, 1966, 1967, 1973, 1978, 1980, 1981, 1989, 1990, 1992, 1993, 1996
```

## Usage Examples

### Python (pandas)

```python
import pandas as pd

# Load ENSO data
df = pd.read_csv('enso_oni_data_1950_1990.csv')

# Get El Niño years
el_nino_years = df[df['is_el_nino'] == 1]['year'].tolist()

# Get all anomaly years
anomaly_years = df[df['enso_anomaly'] == 1]['year'].tolist()

# Filter by type
la_nina_data = df[df['enso_type'] == 'La_Nina']
```

### R

```r
# Load ENSO data
df <- read.csv('enso_oni_data_1950_1990.csv')

# Get El Niño years
el_nino_years <- df[df$is_el_nino == 1, 'year']

# Get all anomaly years
anomaly_years <- df[df$enso_anomaly == 1, 'year']
```

## ONI Definition

The **Oceanic Niño Index (ONI)** is calculated as the 3-month running mean of sea surface temperature (SST) anomalies in the Niño 3.4 region (5°N-5°S, 120°-170°W).

**Classification Criteria**:
- **El Niño**: ONI ≥ +0.5°C for at least 5 consecutive overlapping 3-month periods
- **La Niña**: ONI ≤ -0.5°C for at least 5 consecutive overlapping 3-month periods
- **Normal**: ONI between -0.5°C and +0.5°C

## Applications

This dataset is used for:

1. **Climate model validation**: Comparing HMM hidden states with known ENSO events
2. **Statistical analysis**: Studying ENSO frequency and persistence patterns
3. **Correlation studies**: Analyzing ENSO impacts on regional weather
4. **Machine learning**: Training and evaluating ENSO detection algorithms

## Data Quality

- **Accuracy**: Data derived from official NOAA ONI records
- **Completeness**: 100% coverage for 1950-1990 period
- **Consistency**: All years classified into one of three mutually exclusive categories

## Citation

If you use this data, please cite:

```
NOAA Climate Prediction Center. Oceanic Niño Index (ONI). 
Retrieved from https://ggweather.com/enso/oni.htm
```

## Related Files

- `Categorical_HMM.py` - HMM implementation using this ENSO data
- `evaluate_enso_f1.py` - Evaluation script using this dataset
- `enso_precision_evaluation.csv` - Performance metrics against this ground truth

## Version History

- **v2.0** (2025-11-23): Updated with official ONI data covering 1950-2000, corrected classification based on NOAA records
- **v1.0** (2025-11-22): Initial release covering 1950-1990

## License

This data is derived from NOAA public domain sources. See NOAA data usage policies.

## Contact

For questions or corrections, please open an issue on the GitHub repository:
[Weather-HMM-Co-Repo](https://github.com/Sullivan07043/Weather-HMM-Co-Repo/tree/HMM/Categorical_HMM)

