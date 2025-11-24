# Ensemble Voting for ENSO Detection

This directory contains the ensemble voting system that aggregates predictions from multiple weather stations to create a consensus ENSO (El Niño-Southern Oscillation) forecast.

## Overview

The ensemble method uses **majority voting** across selected high-performing station-level HMM predictions to determine whether a given year represents an ENSO anomaly (El Niño or La Niña) or normal conditions.

## Methodology

### Voting Mechanism

For each year from 1950 to 2010:

1. Collect hidden state predictions from selected weather stations
2. Calculate the **anomaly ratio**: fraction of stations predicting state=1 (anomaly)
3. Apply threshold-based classification:
   - If `anomaly_ratio > threshold` → Predict ENSO Anomaly
   - Otherwise → Predict Normal conditions

### Station Selection

**Current Configuration**: All 21 stations
- Total available stations: 21 (all with complete 1950-2010 coverage)
- **All 21 stations used for comprehensive coverage**
- Global distribution across ENSO-sensitive regions

**Why All 21?**
- Maximum geographic coverage for Moderate+ ENSO detection
- Captures diverse regional ENSO impacts
- Better representation of global ENSO signal
- Suitable for detecting significant (Moderate+) ENSO events

### Data Quality

**Complete Coverage**: All selected stations have predictions for every year (1950-2010)
- **Total records**: 1,281 (21 stations × 61 years)
- **No missing data**: `Total_Stations` = 21 for all years
- **Continuous time series**: Uninterrupted yearly sequences
- **Detrended data**: Climate trends removed, preserving ENSO variability
- **Enhanced features**: 13 meteorological features (6 continuous + 7 binary)

### Threshold Selection

We evaluated six different voting thresholds:

| Threshold | Description | Use Case |
|-----------|-------------|----------|
| **30%** | Most sensitive | Maximum ENSO detection |
| **40%** | High sensitivity | Early warning systems |
| **50%** | **Balanced (recommended default)** | Moderate+ ENSO detection |
| **55%** | Conservative | High-confidence scenarios |
| **60%** | Most strict | Very high-confidence requirements |

## Performance Results (Moderate+ ENSO Definition)

### Best Performance: 50% Threshold (Recommended Default)

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 57.38% | Overall correctness |
| **Precision** | 43.24% | Accuracy when predicting anomaly |
| **Recall** | 76.19% | **Detection rate** - catches 16 out of 21 Moderate+ ENSO events |
| **F1-Score** | 0.5517 | **Highest** - optimal balance for Moderate+ detection |

**Confusion Matrix (50% threshold, All 21):**

```
                Predicted: Normal    Predicted: Anomaly
Actual: Normal          19                  21
Actual: Anomaly          5                  16
```

**Key Findings:**
- ✅ **Highest F1-score (0.5517)** at 50% threshold for Moderate+ events
- ✅ **76.19% recall** - Catches most significant ENSO events (16/21)
- ✅ **43.24% precision** - Balances detection with false positive control
- ⚠️ **5 missed events** - Mainly early period Moderate events (1957, 1958, 1965, 1966, 1973)
- ⚠️ **21 false positives** - Includes Weak ENSO events and borderline years

### Performance Comparison Across Thresholds (All 21 Stations, Moderate+ ENSO)

| Threshold | Accuracy | Precision | Recall | F1-Score | Missed Events |
|-----------|----------|-----------|--------|----------|---------------|
| 30% | 34.43% | 34.43% | 100.00% | 0.5122 | 0 |
| 40% | 44.26% | 37.25% | 90.48% | 0.5278 | 2 |
| **50%** ⭐ | **57.38%** | **43.24%** | **76.19%** | **0.5517** | **5** |
| 55% | 59.02% | 43.33% | 61.90% | 0.5098 | 8 |
| 60% | 62.30% | 45.83% | 52.38% | 0.4889 | 10 |

**Key Insights**:
- **50% threshold** provides the best F1-score (0.5517) for Moderate+ ENSO detection
- Lower thresholds (30-40%) maximize recall but include many Weak events (false positives)
- Higher thresholds (55-60%) improve precision but miss too many Moderate events
- 50% represents optimal balance between detecting significant events and controlling false alarms

## Files in This Directory

### Scripts

1. **`ensemble_voting_enso.py`** (18KB)
   - Main analysis script
   - Performs majority voting across All 21 stations
   - Evaluates performance at multiple thresholds (30%, 40%, 45%, 50%, 55%, 60%)
   - **Default threshold**: 50%
   - Focuses on Moderate+ ENSO events
   - Generates CSV results and console reports
   - Validates data completeness

2. **`plot_ensemble_voting.py`** (7.5KB)
   - Visualization script
   - Creates comprehensive analysis plots with color-coded predictions
   - Generates detailed comparison table
   - **Uses 50% threshold as default**

### Data Files

3. **`ensemble_voting_results.csv`** (3.8KB)
   - Complete year-by-year results (1950-2010, 61 years)
   - **All years have `Total_Stations` = 21** (All stations configuration)
   - **ENSO Definition**: Moderate, Strong, Very Strong only
   - Columns:
     - `Year`: Calendar year
     - `ENSO_Type`: Ground truth (El_Nino_Moderate/Strong/Very_Strong, La_Nina_Moderate/Strong, Normal)
     - `Ground_Truth`: Binary anomaly indicator (0=Normal, 1=Moderate+ Anomaly)
     - `Total_Stations`: Number of stations with predictions (always 21)
     - `Anomaly_Votes`: Count of stations predicting anomaly
     - `Anomaly_Ratio`: Fraction of stations predicting anomaly
     - `Ensemble_30pct`, `Ensemble_40pct`, `Ensemble_45pct`, `Ensemble_50pct`, `Ensemble_55pct`, `Ensemble_60pct`: Predictions at each threshold
     - `Match_30pct`, `Match_40pct`, `Match_45pct`, `Match_50pct`, `Match_55pct`, `Match_60pct`: Binary match indicators

### Visualizations

4. **`ensemble_voting_enso_analysis.png`** (550KB)
   - Three-panel comprehensive analysis figure for Moderate+ ENSO
   - **Panel 1**: Time series comparison with color-coded predictions (Green=Correct, Red=Incorrect)
   - **Panel 2**: Station voting ratio over time with threshold lines (30%-60%)
     - Orange dashed line (50%) indicates current default
   - **Panel 3**: Performance metrics bar chart (50% threshold)
   - All legends positioned on the right side for clarity

5. **`ensemble_voting_detailed_comparison.png`** (690KB)
   - Year-by-year detailed comparison table (1950-2010)
   - Shows ENSO type, ground truth, voting percentage, prediction (50% threshold), and match status
   - Color-coded: ✓ (green) for correct, ✗ (red) for misclassified
   - Highlights 5 missed Moderate+ events and 21 false positives (mainly Weak events)

## Usage

### Running the Analysis

```bash
# Navigate to ensemble directory
cd ensemble

# Run ensemble voting analysis (All 21, 50% default threshold, Moderate+ ENSO)
python ensemble_voting_enso.py

# Generate visualizations
python plot_ensemble_voting.py
```

### Output

The scripts will generate:
- Console report with performance metrics for all thresholds
- `ensemble_voting_results.csv` with detailed year-by-year results
- Two PNG visualization files

### Example Output

```
================================================================================
Ensemble Performance Evaluation (All 21 Stations, Moderate+ ENSO)
================================================================================

Threshold: 30% (anomaly votes > 30%)
  Accuracy:  0.3443
  Precision: 0.3443
  Recall:    1.0000
  F1-Score:  0.5122

Threshold: 50% (anomaly votes > 50%) [DEFAULT]
  Accuracy:  0.5738
  Precision: 0.4324
  Recall:    0.7619
  F1-Score:  0.5517
  
[... additional thresholds ...]

================================================================================
Best Threshold: 50% (F1-Score: 0.5517)
================================================================================

Misclassified Years (50% threshold):
  1957: True=El_Nino_Moderate, Predicted=Normal, Vote=47.6%
  1958: True=El_Nino_Moderate, Predicted=Normal, Vote=42.9%
  [... 5 missed events, 21 false positives ...]
```

## Key Insights (Moderate+ ENSO Definition)

### 1. Optimal Threshold for Moderate+ Detection

✅ **All 21 stations with 50% threshold** provides best balance:
- Highest F1-score (0.5517) for Moderate+ ENSO detection
- Good recall (76.19%) - catches 16 out of 21 significant events
- Acceptable precision (43.24%) - balances detection with false positives
- Suitable for identifying impactful ENSO events

### 2. ENSO Strength Classification Impact

**Moderate+ vs All Strengths**:
- **Moderate+ only**: 21 anomaly years (34.4% of period)
- **All strengths**: 44 anomaly years (72.1% of period)
- **Key finding**: Focusing on Moderate+ events reduces false positives from Weak events
- **Trade-off**: Lower overall metrics but more meaningful event detection

### 3. Threshold Sensitivity for Moderate+ Events

The ensemble system shows different behavior for Moderate+ events:

- **Lower thresholds (30-40%)**: High recall but many false positives (Weak events)
- **Optimal threshold (50%)**: Best F1-score, balanced performance
- **Higher thresholds (55-60%)**: Better precision but miss too many Moderate events

**50% threshold selection rationale**:
- Highest F1-score (0.5517) for Moderate+ detection
- Good recall (76.19%) - catches most significant events
- Acceptable precision (43.24%) - reduces Weak event false positives
- 5 missed events vs. 21 false positives (reasonable for significant event detection)

### 4. Station Consensus Patterns

- **Average anomaly voting ratio**: 57.0%
- **Analysis Period**: 1950-2010 (61 years)
- **Total Moderate+ Anomalies**: 21 years (34.4%)
- **All 21 stations participate**: 21 stations × 61 years = 1,281 predictions

The voting patterns suggest:
- Moderate+ ENSO signals are captured globally
- Regional variation in ENSO impact is reflected in voting patterns
- 50% threshold effectively identifies strong consensus for significant events

### 5. Temporal Patterns

Analysis of the voting ratio time series reveals:
- Clear peaks during major ENSO events (e.g., 1982-83, 1997-98 El Niño, 1988-89, 1998-2000 La Niña)
- Moderate signals during Moderate events (often 40-60% voting ratio)
- Baseline voting ratio varies with natural climate variability
- Missed events (1957, 1958, 1965, 1966, 1973) are mainly early period Moderate events with lower voting ratios

### 6. Enhanced Feature Set Impact

Compared to previous 6-feature configuration:
- **Previous (6 features)**: Temperature, pressure, wind, precipitation only
- **Current (13 features)**: Added visibility + 6 binary weather events (fog, rain, snow, hail, thunder, tornado)
- **Improvement**: Better capture of extreme weather patterns associated with ENSO
- **Result**: More robust hidden state detection across diverse meteorological conditions

## Recommended Thresholds by Application (Moderate+ ENSO)

### For Maximum Sensitivity
**Use 30% threshold**
- Priority: Don't miss any Moderate+ ENSO event
- 100% recall ensures all significant events are flagged
- Acceptable: Many false alarms (includes Weak events)
- F1-Score: 0.5122

### For High Sensitivity Early Warning
**Use 40% threshold**
- Priority: High detection rate with some precision
- 90.48% recall catches most Moderate+ events
- Lower precision (37.25%) includes many Weak events
- F1-Score: 0.5278

### For Scientific Research and Climate Analysis (Recommended)
**Use 50% threshold (Default)**
- Priority: Optimal balance for Moderate+ event detection
- **Highest F1-score (0.5517)**
- 76.19% recall catches most significant ENSO events
- 43.24% precision reduces Weak event false positives
- **Best overall performance for Moderate+ detection**

### For Conservative Detection
**Use 55% threshold**
- Priority: Higher confidence in Moderate+ predictions
- 61.90% recall - misses some Moderate events
- Better precision (43.33%)
- F1-Score: 0.5098

### For High-Confidence Applications
**Use 60% threshold**
- Priority: Very high confidence requirements
- 52.38% recall - misses many Moderate events
- Best precision (45.83%)
- F1-Score: 0.4889
- Only flags strong consensus signals

## Misclassified Events Analysis (Moderate+ ENSO, 50% threshold)

### Missed Events (FN=5)
- **1957**: El Niño Moderate, Vote=47.6% (just below threshold)
- **1958**: El Niño Moderate, Vote=42.9% (below threshold)
- **1965**: El Niño Moderate, Vote=33.3% (well below threshold)
- **1966**: El Niño Moderate, Vote=33.3% (well below threshold)
- **1973**: El Niño Strong, Vote=47.6% (just below threshold)

**Analysis**:
- 3 events just below 50% threshold (1957, 1958, 1973)
- 2 events with weak consensus (1965, 1966)
- Mainly early period events (1957-1973)
- Suggests early period data quality or regional signal issues

### False Positives (FP=21)
**Weak ENSO events** (classified as Normal in Moderate+ definition):
- Multiple Weak El Niño and Weak La Niña years
- These years show ENSO-like patterns but below Moderate threshold
- Model correctly detects ENSO signal, but strength is below Moderate

**Borderline years**:
- Years with ENSO-like climate patterns
- Regional climate variations captured by global station network
- Transition periods between ENSO phases

**Possible explanations**:
- Model is sensitive to Weak ENSO events (not classified as anomalies in Moderate+ definition)
- Regional climate variations captured by Pacific Rim stations
- Some years may have had ENSO-like impacts without official Moderate classification
- Binary classification (anomaly vs normal) doesn't capture ENSO strength gradation

## Limitations

1. **ENSO Strength Granularity**
   - Current system uses Moderate+ threshold (binary: Moderate+ vs Normal/Weak)
   - Does not differentiate between El Niño and La Niña
   - Does not distinguish Moderate from Strong/Very Strong events
   - Many false positives are actually Weak ENSO events
   - Extension to multi-class classification (Weak/Moderate/Strong, El Niño/La Niña) is recommended

2. **Station Coverage**
   - Analysis based on All 21 globally distributed stations
   - Limited representation in tropical Pacific (ENSO origin region)
   - More stations from ENSO-sensitive regions could improve performance
   - Current stations primarily in mid-latitudes and subtropics

3. **Temporal Resolution**
   - Yearly data averages may smooth out intra-annual ENSO variations
   - ENSO events typically peak in boreal winter months (DJF)
   - Monthly or seasonal analysis could provide better temporal resolution
   - Current approach may miss short-duration events

4. **Equal Weighting**
   - Each station has equal voting weight
   - Stations closer to ENSO-affected regions might be more informative
   - Weighted voting based on geographic relevance or individual F1-scores could improve results
   - Current approach treats all stations equally regardless of ENSO sensitivity

5. **Threshold Selection**
   - Fixed 50% threshold may not be optimal for all event strengths
   - Adaptive thresholding could improve performance
   - Different thresholds for different ENSO strengths could be beneficial
   - Current threshold optimized for Moderate+ events, may not suit all applications

## Future Improvements

1. **Multi-Class Strength Classification**: Classify ENSO events by strength (Weak/Moderate/Strong/Very Strong) instead of binary
   - Would reduce false positives from Weak events
   - Better align predictions with ENSO impact severity
   - Provide more actionable information for applications

2. **3-Phase Classification**: Distinguish El Niño, La Niña, and Normal separately
   - Different impacts require different responses
   - Current binary approach loses phase information
   - Important for agricultural and water resource planning

3. **Weighted Voting**: Assign higher weights to stations in ENSO-sensitive regions
   - Pacific coast stations (USA, Mexico, Peru, Chile) may be more informative
   - Weight by individual station F1-scores
   - Geographic relevance-based weighting

4. **Monthly or Seasonal Analysis**: Use monthly data for higher temporal resolution
   - ENSO events peak in boreal winter (DJF)
   - Capture intra-annual variability
   - Better align with ONI 3-month running mean definition

5. **Soft Voting**: Use HMM posterior probabilities instead of hard binary votes
   - Incorporate prediction confidence
   - More nuanced ensemble decision-making
   - Better uncertainty quantification

6. **Machine Learning Meta-Classifier**: Train a meta-classifier on station predictions
   - Learn optimal combination of station predictions
   - Capture non-linear relationships
   - Potentially improve beyond simple voting

7. **Feature Engineering**: Incorporate additional climate indices
   - Southern Oscillation Index (SOI)
   - Multivariate ENSO Index (MEI)
   - Pacific Decadal Oscillation (PDO)
   - Better capture large-scale climate patterns

8. **Expand Station Network**: Include more stations from tropical Pacific
   - Closer to ENSO origin region
   - Stronger ENSO signals
   - Better geographic coverage

9. **Adaptive Thresholding**: Dynamic threshold selection
   - Different thresholds for different time periods
   - Strength-dependent thresholds
   - Seasonal threshold adjustment

10. **Confidence Intervals**: Provide uncertainty estimates
    - Bootstrap ensemble predictions
    - Quantify prediction confidence
    - Risk-based decision support

## References

1. **ENSO Ground Truth Data**: NOAA Climate Prediction Center Oceanic Niño Index (ONI)
   - Source: https://ggweather.com/enso/oni.htm
   - See: `../enso_oni_data_1950_2010.csv`

2. **Individual Station Predictions**: Factorized Categorical HMM with trend removal
   - See: `../enso_factorized_categorical_hmm_states.csv`
   - Model: `../Categorical_HMM.py`

3. **Data Preprocessing**: Detrending and time series completion
   - Detrending: `../data/dataloader2.py` (removes climate trends)
   - Time series completion: `../data/fill_missing_years_detrended.py`
   - Ensures continuous yearly sequences for all stations

## Contact

For questions or suggestions about the ensemble voting system, please open an issue on the GitHub repository.

## Version History

- **v3.2** (2025-11-24): Viterbi decoding implementation
  - **Decoding Algorithm**: Changed from posterior decoding to Viterbi algorithm
  - **Method**: Finds globally optimal state sequence for each station
  - **Performance**: Maintained F1=0.5517 with 50% threshold
  - **Benefit**: Ensures valid state transitions and global optimality
  - All other configurations unchanged

- **v3.1** (2025-11-24): Moderate+ ENSO definition with enhanced features
  - **ENSO Definition**: Changed to Moderate, Strong, Very Strong only (21 anomaly years)
  - **Features**: Expanded from 6 to 13 features (added visibility + 6 binary weather events)
  - **Configuration**: All 21 stations with 50% threshold
  - **Best F1-score**: 0.5517 (50% threshold, All 21 stations)
  - **Performance**: 76.19% recall, 43.24% precision for Moderate+ events
  - **Focus**: Detecting significant ENSO events with global consensus
  - 5 missed Moderate+ events, 21 false positives (mainly Weak events)

- **v3.0** (2025-11-24): Extended period with all ENSO strengths
  - Extended period to 1950-2010 (61 years)
  - All ENSO strengths included (44 anomaly years)
  - 21 stations with complete coverage
  - Adaptive trend removal applied

- **v2.1** (2025-11-23): Data completeness update
  - All 17 stations have complete 1950-2000 coverage
  - `Total_Stations` = 17 for all years (no missing data)
  - Updated performance metrics with complete data
  - Best F1-score: 0.824 (30% threshold) with perfect recall
  - Recommended threshold: 40% with F1=0.819

- **v2.0** (2025-11-23): Major update with corrected ENSO data
  - Updated ENSO ground truth data from official ONI records (1950-2000)
  - Changed threshold range from 20-50% to 30-60%
  - Improved visualization: color-coded predictions
  - Moved all legends to right side for better clarity

- **v1.1** (2025-11-22): Added 25% threshold
  - Added 25% threshold evaluation for better balance
  - Updated visualizations to include all 5 thresholds

- **v1.0** (2025-11-22): Initial release
  - Implemented majority voting ensemble
  - Evaluated 4 different thresholds (20%, 30%, 40%, 50%)

## License

This work is part of the CSE250A course project. See main repository for license details.
