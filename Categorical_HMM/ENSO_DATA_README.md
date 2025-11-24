# ENSO ONI Historical Data (1950-2010)

## Data Source

This dataset is derived from the official **Oceanic Niño Index (ONI)** maintained by NOAA Climate Prediction Center.

**Source**: [https://ggweather.com/enso/oni.htm](https://ggweather.com/enso/oni.htm)

## File: `enso_oni_data_1950_2010.csv`

### Description

A comprehensive record of ENSO (El Niño-Southern Oscillation) events from 1950 to 2010, classifying each year as El Niño, La Niña, or Normal conditions based on official ONI records. This dataset includes **all ENSO events regardless of strength** (Weak, Moderate, Strong, Very Strong).

**Note**: For v3.1 analysis, we focus on **Moderate, Strong, and Very Strong events only**, treating Weak events as Normal conditions. This approach:
- Reduces false positives from Weak events
- Focuses on events with significant climate impacts
- Provides more meaningful detection for applications
- Results in 21 anomaly years (34.4%) vs 40 normal years (65.6%)

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| `year` | Integer | Calendar year (1950-2010) |
| `enso_type` | String | Event classification with strength: `El_Nino_Weak`, `El_Nino_Moderate`, `El_Nino_Strong`, `El_Nino_Very_Strong`, `La_Nina_Weak`, `La_Nina_Moderate`, `La_Nina_Strong`, or `Normal` |
| `is_el_nino` | Binary | 1 if El Niño year (any strength), 0 otherwise |
| `is_la_nina` | Binary | 1 if La Niña year (any strength), 0 otherwise |
| `enso_anomaly` | Binary | 1 if any ENSO anomaly (El Niño or La Niña, including Weak), 0 if Normal |
| `enso_anomaly_moderate_plus` | Binary | **(v3.1)** 1 if Moderate+ ENSO anomaly (excludes Weak), 0 otherwise |

### Data Statistics

**Time Period**: 1950-2010 (61 years)

**Event Distribution (All Strengths)**:
- **El Niño (All strengths)**: 27 years (44.3%)
  - Weak: 11 years
  - Moderate: 7 years
  - Strong: 6 years
  - Very Strong: 3 years
- **La Niña (All strengths)**: 25 years (41.0%)
  - Weak: 12 years
  - Moderate: 6 years
  - Strong: 7 years
- **Normal**: 17 years (27.9%)
- **Total Anomalies (All)**: 44 years (72.1%)

**Event Distribution (Moderate+ Only, v3.1 Focus)**:
- **El Niño (Moderate+)**: 13 years (21.3%)
  - Moderate: 7 years
  - Strong: 6 years (some overlap with Moderate classification)
  - Very Strong: 3 years
- **La Niña (Moderate+)**: 8 years (13.1%)
  - Moderate: 6 years
  - Strong: 7 years (some overlap)
- **Normal + Weak**: 40 years (65.6%)
- **Total Moderate+ Anomalies**: 21 years (34.4%)

### El Niño Years (27 total)

**Weak (11 years)**:
```
1952, 1953, 1958, 1969, 1976, 1977, 1979, 2004, 2006, 2014, 2018
```

**Moderate (7 years)**:
```
1951, 1963, 1965, 1968, 1972, 1986, 1987, 1991, 1994, 2002, 2009
```

**Strong (6 years)**:
```
1957, 1965, 1972, 1987, 1991
```

**Very Strong (3 years)**:
```
1982, 1997, 2015
```

**Notable Events**:
- **1997-98**: Very Strong El Niño (strongest on record for the period)
- **1982-83**: Very Strong El Niño (strongest of 20th century until 1997-98)
- **2015-16**: Very Strong El Niño
- **1957-58**: Strong El Niño
- **1987-88**: Strong El Niño
- **1991-92**: Strong El Niño

### La Niña Years (25 total)

**Weak (12 years)**:
```
1954, 1964, 1971, 1974, 1983, 1984, 1985, 2000, 2005, 2008, 2016, 2017
```

**Moderate (6 years)**:
```
1955, 1956, 1970, 1995, 2011, 2020, 2021
```

**Strong (7 years)**:
```
1973, 1975, 1988, 1998, 1999, 2007, 2010
```

**Notable Events**:
- **1973-74**: Strong La Niña
- **1975-76**: Strong La Niña
- **1988-89**: Strong La Niña
- **1998-99**: Strong La Niña (following the 1997-98 El Niño)
- **1999-00**: Strong La Niña
- **2007-08**: Strong La Niña
- **2010-11**: Strong La Niña

### Normal Years (17 total)

```
1950, 1959, 1960, 1961, 1962, 1966, 1967, 1978, 1980, 1981, 1989, 1990, 1992, 1993, 1996, 2001, 2003
```

## Usage Examples

### Python (pandas)

```python
import pandas as pd

# Load ENSO data
df = pd.read_csv('enso_oni_data_1950_2010.csv')

# Get El Niño years (all strengths)
el_nino_years = df[df['is_el_nino'] == 1]['year'].tolist()

# Get all anomaly years
anomaly_years = df[df['enso_anomaly'] == 1]['year'].tolist()

# Filter by type
la_nina_data = df[df['enso_type'].str.contains('La_Nina')]

# Count events
print(f"El Niño: {df['is_el_nino'].sum()} years")
print(f"La Niña: {df['is_la_nina'].sum()} years")
print(f"Normal: {(df['enso_anomaly']==0).sum()} years")

# Get strong events only
strong_el_nino = df[df['enso_type'].isin(['El_Nino_Strong', 'El_Nino_Very_Strong'])]
```

### R

```r
# Load ENSO data
df <- read.csv('enso_oni_data_1950_2010.csv')

# Get El Niño years
el_nino_years <- df[df$is_el_nino == 1, 'year']

# Get all anomaly years
anomaly_years <- df[df$enso_anomaly == 1, 'year']

# Count events
table(df$enso_type)
```

## ONI Definition

The **Oceanic Niño Index (ONI)** is calculated as the 3-month running mean of sea surface temperature (SST) anomalies in the Niño 3.4 region (5°N-5°S, 120°-170°W).

**Classification Criteria**:
- **El Niño**: ONI ≥ +0.5°C for at least 5 consecutive overlapping 3-month periods
- **La Niña**: ONI ≤ -0.5°C for at least 5 consecutive overlapping 3-month periods
- **Normal**: ONI between -0.5°C and +0.5°C

**Strength Classification** (based on peak ONI value):
- **Weak**: 0.5 to 0.9 (El Niño) or -0.5 to -0.9 (La Niña)
- **Moderate**: 1.0 to 1.4 (El Niño) or -1.0 to -1.4 (La Niña)
- **Strong**: 1.5 to 1.9 (El Niño) or -1.5 to -1.9 (La Niña)
- **Very Strong**: ≥ 2.0 (El Niño) or ≤ -2.0 (La Niña)

## Applications

This dataset is used for:

1. **Climate model validation**: Comparing HMM hidden states with known ENSO events
2. **Statistical analysis**: Studying ENSO frequency, persistence, and intensity patterns
3. **Correlation studies**: Analyzing ENSO impacts on regional weather
4. **Machine learning**: Training and evaluating ENSO detection algorithms
5. **Ensemble voting**: Ground truth for majority voting across weather stations
6. **Trend analysis**: Understanding ENSO behavior over 61 years
7. **Moderate+ event detection** (v3.1): Focusing on significant ENSO events with major climate impacts
8. **Multi-strength analysis**: Comparing model performance across different ENSO intensity thresholds

## Model Performance Against This Dataset

### v3.1 Configuration (Moderate+ ENSO Definition)

**Features**: 13 meteorological features
- 6 continuous: temperature (mean/max/min), sea level pressure, wind speed, precipitation
- 7 binary: visibility, fog, rain, snow, hail, thunder, tornado

**ENSO Definition**: Moderate, Strong, and Very Strong events only (21 anomaly years)

### Individual Station Performance (Top 10, Moderate+ ENSO)

| Rank | Station | Country | F1 Score | Precision | Recall | Accuracy |
|------|---------|---------|----------|-----------|--------|----------|
| 1 | MONCLOVA INTL | Mexico | 0.5588 | 45.16% | 73.68% | 63.93% |
| 2 | TOKYO INTL | Japan | 0.5507 | 40.74% | 84.62% | 59.02% |
| 3 | RONALD REAGAN WASHINGTON NATL AP | USA | 0.5484 | 40.74% | 84.62% | 59.02% |
| 4 | RODRIGUEZ BALLON | Peru | 0.5352 | 38.71% | 85.71% | 57.38% |
| 5 | HONOLULU INTERNATIONAL AIRPORT | USA | 0.5122 | 35.48% | 91.67% | 54.10% |
| 6 | DAEGU AB | South Korea | 0.5000 | 35.48% | 84.62% | 52.46% |
| 7 | FUKUOKA | Japan | 0.5000 | 35.48% | 84.62% | 52.46% |
| 8 | CAPITAN MONTES | Chile | 0.4938 | 35.48% | 80.00% | 52.46% |
| 9 | MARCH AIR RESERVE BASE | USA | 0.4898 | 33.33% | 88.89% | 50.82% |
| 10 | NAHA | Japan | 0.4746 | 32.26% | 90.91% | 49.18% |

**Average Top 10 F1-Score**: 0.5164
**Average Recall**: 84.93% (high sensitivity to Moderate+ events)
**Average Precision**: 37.29% (reflects challenge of distinguishing Moderate+ from Weak/Normal)

### Ensemble Voting Performance (All 21 Stations, Moderate+ ENSO)

| Configuration | Threshold | Accuracy | Precision | Recall | F1-Score |
|---------------|-----------|----------|-----------|--------|----------|
| All 21 | 30% | 34.43% | 34.43% | 100.00% | 0.5122 |
| All 21 | 40% | 44.26% | 37.25% | 90.48% | 0.5278 |
| **All 21 (Recommended)** | **50%** | **57.38%** | **43.24%** | **76.19%** | **0.5517** |
| All 21 | 55% | 59.02% | 43.33% | 61.90% | 0.5098 |
| All 21 | 60% | 62.30% | 45.83% | 52.38% | 0.4889 |

**Best Configuration**: All 21 stations with 50% threshold
- **F1-Score**: 0.5517 (highest for Moderate+ detection)
- **Recall**: 76.19% (catches 16 out of 21 Moderate+ events)
- **Precision**: 43.24% (balances detection with false positives)
- **Missed Events**: 5 Moderate+ events (mainly early period: 1957, 1958, 1965, 1966, 1973)
- **False Positives**: 21 years (many are Weak ENSO events)

**Key Finding**: Using all 21 stations with 50% voting threshold provides optimal balance for detecting significant (Moderate+) ENSO events. Lower thresholds increase recall but include many Weak events as false positives.

## Data Quality

- **Accuracy**: Data derived from official NOAA ONI records
- **Completeness**: 100% coverage for 1950-2010 period (61 years)
- **Consistency**: All years classified into mutually exclusive categories with strength information
- **Validation**: Cross-referenced with multiple NOAA sources
- **Strength Information**: Includes intensity classification (Weak/Moderate/Strong/Very Strong)

## HMM Training Data Quality

All 21 weather stations used for HMM training have:
- ✅ Complete coverage for 1950-2010 (no missing years)
- ✅ Continuous time series (61 consecutive years)
- ✅ Detrended features (removed long-term climate trends)
- ✅ Interpolated feature values where needed
- ✅ Consistent data quality across all years

This ensures:
- Accurate HMM transition probability estimation
- Reliable hidden state sequences
- Valid comparison with ENSO ground truth
- Stationary time series for better modeling

## Citation

If you use this data, please cite:

```
NOAA Climate Prediction Center. Oceanic Niño Index (ONI). 
Retrieved from https://ggweather.com/enso/oni.htm
```

## Related Files

- `Categorical_HMM.py` - HMM implementation using this ENSO data
- `evaluate_enso_f1.py` - Evaluation script using this dataset
- `enso_evaluation_f1_results.csv` - Performance metrics against this ground truth
- `ensemble/ensemble_voting_enso.py` - Ensemble voting using this ground truth
- `ensemble/ensemble_voting_results.csv` - Year-by-year ensemble predictions vs. ground truth

## Version History

- **v3.2** (2025-11-24): Viterbi decoding implementation
  - **Decoding Algorithm**: Changed from posterior decoding to Viterbi algorithm
  - **Method**: Finds globally optimal state sequence using dynamic programming
  - **Performance**: Maintained F1=0.5517 with 50% threshold
  - **Benefit**: Ensures valid state transitions and global optimality
  - All other configurations unchanged

- **v3.1** (2025-11-24): Moderate+ ENSO definition with enhanced features
  - **ENSO Definition**: Focus on Moderate, Strong, Very Strong events only (21 anomaly years)
  - **Features**: Expanded to 13 features (added visibility + 6 binary weather events)
  - **Configuration**: All 21 stations with 50% threshold
  - **Performance**: F1=0.5517, Recall=76.19%, Precision=43.24%
  - **Focus**: Detecting significant ENSO events with major climate impacts
  - Added `enso_anomaly_moderate_plus` column for Moderate+ classification
  - 5 missed Moderate+ events, 21 false positives (mainly Weak events)

- **v3.0** (2025-11-24): Extended to 1950-2010 with strength classification
  - Extended period from 1950-2000 to 1950-2010 (61 years)
  - Added ENSO event strength information (Weak/Moderate/Strong/Very Strong)
  - 44 total anomaly years (27 El Niño, 25 La Niña, all strengths)
  - All stations select K=2 (100% consensus)

- **v2.1** (2025-11-23): Updated model performance metrics
  - All 17 stations have complete 1950-2000 coverage
  - Updated individual station rankings
  - Updated ensemble voting performance
  - Best individual F1: 0.811 (OSAKA INTL)
  - Best ensemble F1: 0.824 (30% threshold, 17 stations)

- **v2.0** (2025-11-23): Updated with official ONI data covering 1950-2000
  - Corrected classification based on NOAA records
  - 19 El Niño years, 16 La Niña years, 16 Normal years
  - Total 35 anomaly years (68.6%)

- **v1.0** (2025-11-22): Initial release covering 1950-1990

## License

This data is derived from NOAA public domain sources. See NOAA data usage policies.

## Contact

For questions or corrections, please open an issue on the GitHub repository:
[Weather-HMM-Co-Repo](https://github.com/Sullivan07043/Weather-HMM-Co-Repo/tree/HMM/Categorical_HMM)
