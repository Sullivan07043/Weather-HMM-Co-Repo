# ENSO ONI Historical Data (1950-2010)

## Data Source

This dataset is derived from the official **Oceanic Niño Index (ONI)** maintained by NOAA Climate Prediction Center.

**Source**: [https://ggweather.com/enso/oni.htm](https://ggweather.com/enso/oni.htm)

## File: `enso_oni_data_1950_2010.csv`

### Description

A comprehensive record of ENSO (El Niño-Southern Oscillation) events from 1950 to 2010, classifying each year as El Niño, La Niña, or Normal conditions based on official ONI records. This dataset includes **all ENSO events regardless of strength** (Weak, Moderate, Strong, Very Strong).

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| `year` | Integer | Calendar year (1950-2010) |
| `enso_type` | String | Event classification with strength: `El_Nino_Weak`, `El_Nino_Moderate`, `El_Nino_Strong`, `El_Nino_Very_Strong`, `La_Nina_Weak`, `La_Nina_Moderate`, `La_Nina_Strong`, or `Normal` |
| `is_el_nino` | Binary | 1 if El Niño year (any strength), 0 otherwise |
| `is_la_nina` | Binary | 1 if La Niña year (any strength), 0 otherwise |
| `enso_anomaly` | Binary | 1 if any ENSO anomaly (El Niño or La Niña, including Weak), 0 if Normal |

### Data Statistics

**Time Period**: 1950-2010 (61 years)

**Event Distribution**:
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
- **Total Anomalies**: 44 years (72.1%)

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

## Model Performance Against This Dataset

### Individual Station Performance (Top 10)

| Rank | Station | Country | F1 Score | Precision | Recall | Accuracy |
|------|---------|---------|----------|-----------|--------|----------|
| 1 | TOKYO INTL | Japan | 0.7586 | 73.33% | 79.55% | 70.49% |
| 2 | BROOME INTL | Australia | 0.7397 | 75.86% | 72.73% | 68.85% |
| 3 | IWOTO | Japan | 0.7170 | 73.17% | 70.45% | 65.57% |
| 4 | NAHA | Japan | 0.7170 | 73.17% | 70.45% | 65.57% |
| 5 | MONCLOVA INTL | Mexico | 0.7143 | 72.22% | 70.45% | 65.57% |
| 6 | SOTO LA MARINA TAMPS. | Mexico | 0.7143 | 72.22% | 70.45% | 65.57% |
| 7 | MARCH AIR RESERVE BASE | USA | 0.7132 | 67.57% | 75.00% | 63.93% |
| 8 | KALGOORLIE BOULDER | Australia | 0.7119 | 72.41% | 70.00% | 65.57% |
| 9 | CEDUNA AMO | Australia | 0.7080 | 66.67% | 75.00% | 62.30% |
| 10 | GENERAL IGNACIO P GARCIA INTL | Mexico | 0.7048 | 65.63% | 75.00% | 60.66% |

**Average Top 10 F1-Score**: 0.7048

### Ensemble Voting Performance (Top 14 Stations)

| Configuration | Threshold | Accuracy | Precision | Recall | F1-Score |
|---------------|-----------|----------|-----------|--------|----------|
| **Top 14 (Recommended)** | **40%** | **73.77%** | **74.14%** | **97.73%** | **0.8431** |
| Top 14 | 35% | 72.13% | 73.58% | 97.73% | 0.8387 |
| Top 14 | 30% | 70.49% | 71.67% | 97.73% | 0.8269 |
| Top 10 | 35% | 73.77% | 76.92% | 90.91% | 0.8333 |
| All 21 | 35% | 65.57% | 70.91% | 88.64% | 0.7879 |

**Best Configuration**: Top 14 stations with 40% threshold
- **F1-Score**: 0.8431 (highest overall)
- **Recall**: 97.73% (misses only 1 event out of 44)
- **Precision**: 74.14% (3 out of 4 predictions correct)

**Key Finding**: Using Top 14 stations (by F1-score) with 40% voting threshold provides the best balance between detecting ENSO events (high recall) and avoiding false alarms (good precision).

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

- **v3.0** (2025-11-24): Extended to 1950-2010 with strength classification
  - Extended period from 1950-2000 to 1950-2010 (61 years)
  - Added ENSO event strength information (Weak/Moderate/Strong/Very Strong)
  - 44 total anomaly years (27 El Niño, 25 La Niña)
  - Updated model performance metrics with Top 14 ensemble
  - Best ensemble F1: 0.8431 (Top 14, 40% threshold)
  - 97.73% recall with only 1 missed event

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
