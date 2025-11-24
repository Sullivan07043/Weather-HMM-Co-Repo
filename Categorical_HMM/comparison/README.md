# Model Comparison for ENSO Detection

## Overview

This directory contains a comprehensive comparison of four different models for ENSO (El Niño-Southern Oscillation) anomaly detection using global weather station data from 1950-2000.

## Models Compared

### 1. **HMM (Hidden Markov Model)**
- **Type**: Factorized Categorical HMM with temporal dependencies
- **Features**: 13 weather features (temperature, pressure, wind, precipitation, weather events)
- **Stations**: 21 global stations
- **Algorithm**: Forward-backward algorithm + Viterbi decoding
- **Key Strength**: Captures temporal dependencies and state transitions

### 2. **GMM (Gaussian Mixture Model)**
- **Type**: Probabilistic clustering model
- **Features**: Continuous weather features
- **Stations**: 14 global stations
- **Algorithm**: Expectation-Maximization (EM)
- **Key Strength**: Models multivariate Gaussian distributions

### 3. **PELT (Pruned Exact Linear Time)**
- **Type**: Change point detection algorithm
- **Features**: Weather time series
- **Stations**: 21 global stations
- **Algorithm**: Dynamic programming for optimal segmentation
- **Key Strength**: Detects abrupt changes in time series

### 4. **Independent Classifier**
- **Type**: Independent Mixture Model (no temporal dependencies)
- **Features**: Same 13 features as HMM
- **Stations**: 21 global stations
- **Algorithm**: EM algorithm without transition matrix
- **Key Strength**: Baseline for understanding temporal dependency contribution

---

## Performance Summary (50% Ensemble Threshold)

| Model | F1-Score | Accuracy | Precision | Recall | TP | FP | TN | FN |
|-------|----------|----------|-----------|--------|----|----|----|----|
| **HMM** | **0.6383** | **0.6667** | **0.5556** | 0.7500 | 15 | 12 | 19 | 5 |
| **Independent** | 0.5231 | 0.3922 | 0.3778 | **0.8500** | 17 | 28 | 3 | 3 |
| **GMM** | 0.3404 | 0.3922 | 0.2963 | 0.4000 | 8 | 19 | 12 | 12 |
| **PELT** | 0.0000 | 0.6078 | 0.0000 | 0.0000 | 0 | 0 | 31 | 20 |

### Key Findings

1. **HMM is the Best Overall Model**
   - Highest F1-Score (0.6383) and Accuracy (0.6667)
   - Best balance between Precision and Recall
   - Successfully detects 75% of true ENSO anomalies with 55.6% precision

2. **Temporal Dependencies Matter**
   - HMM (with temporal) outperforms Independent Classifier (without temporal)
   - F1 improvement: 0.6383 vs 0.5231 (+22.0%)
   - Demonstrates the value of modeling state transitions

3. **Independent Classifier Has High Recall**
   - Highest Recall (0.8500) but lowest Precision (0.3778)
   - Tends to over-predict anomalies (45/51 years predicted as anomalies)
   - Useful when false negatives are more costly than false positives

4. **GMM Shows Moderate Performance**
   - F1-Score of 0.3404 with balanced errors
   - May benefit from feature engineering or more components

5. **PELT Failed to Detect Anomalies**
   - 0% Recall - detected no anomalies at 50% threshold
   - High Accuracy (0.6078) only because most years are normal (31/51)
   - **Root Cause**: PELT's anomaly voting ratios are all below 40% (max 38%)
   - PELT is designed for change point detection, not binary classification
   - The gradual nature of ENSO transitions may not trigger PELT's change point criteria
   - May require different parameterization, lower threshold, or alternative change point algorithm

---

## Files

### Data Files
- `ensemble_voting_results.csv` - HMM ensemble predictions (1950-2000)
- `gmm_ensemble_voting_results.csv` - GMM ensemble predictions (1950-2010, filtered to 1950-2000)
- `PELT_enso_ensemble_results.csv` - PELT ensemble predictions (1950-2000)
- `independent_ensemble_voting_results.csv` - Independent Classifier predictions (1950-2000)

### Scripts
- `extract_independent_ensemble.py` - Generate Independent Classifier predictions
- `visualize_model_comparison_v2.py` - Create comprehensive comparison visualizations

### Outputs
- `figure1_performance_metrics.png` - Performance metrics and confusion matrix comparison
- `figure2_threshold_analysis.png` - F1-Score vs voting threshold curve
- `figure3_temporal_analysis.png` - Year-by-year anomaly detection timeline (1950-1975 and 1976-2000)
- `figure4_performance_summary.png` - Comprehensive performance summary table
- `README.md` - This documentation file

---

## Visualizations

### Figure 1: Performance Metrics Comparison
- **Left Panel**: Bar chart comparing F1-Score, Accuracy, Precision, and Recall across all models
- **Right Panel**: Confusion matrix breakdown (TN, FP, FN, TP) for each model
- Clear value labels on all bars for easy reading

### Figure 2: F1-Score vs Voting Threshold
- Single focused plot showing how F1-Score changes with voting threshold (30%-60%)
- Red dashed line marks the 50% threshold
- All models (HMM, GMM, PELT, Independent) shown with distinct colors
- Demonstrates model sensitivity to threshold selection

### Figure 3: Year-by-Year Temporal Analysis
- **Upper Panel**: Detection results for 1950-1975
- **Lower Panel**: Detection results for 1976-2000
- **Red background**: True ENSO anomaly years (ground truth)
- **Colored dots**: Model predictions
  - **HMM & Independent**: 50% threshold (standard)
  - **GMM & PELT**: 30% threshold (to show more predictions)
- **Shared legend**: Placed at bottom to avoid obscuring data
- Clear visualization of when each model detects anomalies
- Different thresholds allow fair comparison of conservative models (GMM, PELT)

### Figure 4: Performance Summary Table
- Comprehensive table with all key metrics (F1, Accuracy, Precision, Recall, TP, FP, TN, FN)
- Color-coded by model for easy identification
- Light green highlighting for best performance in each metric
- Clean layout with no overlapping elements

---

## Methodology

### Ensemble Voting
All models use ensemble voting across multiple weather stations:
- Each station's model predicts binary state (0=Normal, 1=Anomaly) for each year
- Votes are aggregated across all stations
- Anomaly ratio = (# stations predicting anomaly) / (total # stations)
- Final prediction based on threshold (e.g., 50% = majority vote)

### Evaluation Period
- **Training**: 1950-2000 (51 years)
- **Ground Truth**: ENSO ONI index with Moderate+ intensity threshold
- **True Anomalies**: 20 years (El Niño and La Niña events)
- **Normal Years**: 31 years

### Metrics
- **F1-Score**: Harmonic mean of Precision and Recall (primary metric)
- **Accuracy**: Overall correct predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)

---

## Conclusions

### Model Selection Recommendations

**For Best Overall Performance**: Use **HMM**
- Highest F1-Score and Accuracy
- Best balance of Precision and Recall
- Captures temporal patterns in weather data

**For High Recall (Don't Miss Anomalies)**: Use **Independent Classifier**
- Detects 85% of true anomalies
- Acceptable when false alarms are tolerable
- Simpler than HMM (no temporal modeling)

**For High Precision (Avoid False Alarms)**: Use **HMM**
- 55.6% Precision is highest among models that detect anomalies
- More reliable when false positives are costly

### Scientific Insights

1. **Temporal Dependencies are Valuable**
   - HMM's superior performance validates the importance of modeling state transitions
   - Weather patterns evolve over time, not independently

2. **Feature Quality Matters**
   - HMM and Independent Classifier (both using 13 engineered features) outperform GMM
   - Categorical binning may be more effective than raw continuous features

3. **Ensemble Voting is Robust**
   - Aggregating predictions across 21 global stations improves reliability
   - Geographic diversity helps capture ENSO's global signature

4. **Change Point Detection Limitations**
   - PELT's failure suggests ENSO anomalies may not manifest as abrupt changes
   - Gradual transitions may be better captured by probabilistic models

---

## Future Work

1. **Hybrid Models**
   - Combine HMM's temporal modeling with GMM's continuous distributions
   - Ensemble multiple model types for improved robustness

2. **Feature Engineering**
   - Add spatial features (pressure gradients, wind divergence)
   - Include lagged features to capture delayed responses

3. **Temporal Resolution**
   - Test with monthly data instead of yearly aggregates
   - May reveal finer-grained temporal patterns

4. **Adaptive Thresholds**
   - Optimize voting threshold per model
   - Consider confidence-weighted voting

5. **PELT Tuning**
   - Experiment with different penalty parameters
   - Try alternative change point algorithms

---

## Usage

### Generate Independent Classifier Results
```bash
cd /Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/comparison
python3 extract_independent_ensemble.py
```

### Create Comparison Visualizations
```bash
python3 visualize_model_comparison_v2.py
```

This will generate 4 separate high-quality figures:
- `figure1_performance_metrics.png` - Metrics comparison
- `figure2_threshold_analysis.png` - Threshold sensitivity
- `figure3_temporal_analysis.png` - Timeline visualization
- `figure4_performance_summary.png` - Summary table

All figures have clean layouts with legends positioned to avoid obscuring data.

---

## References

1. HMM Implementation: `../Categorical_HMM.py`
2. ENSO Ground Truth: `../enso_oni_data_1950_2010.csv`
3. Weather Data: `../data/processed/weather_1901_2019_yearly_detrend_adaptive_bins10.csv`
4. Ablation Studies: `../ablation/`

---

**Generated**: 2024-11-24  
**Analysis Period**: 1950-2000 (51 years) - All models use the same time range  
**Models Compared**: 4 (HMM, GMM, PELT, Independent Classifier)  
**Stations**: 21 global weather stations  
**Visualizations**: 4 high-quality figures with clear layouts  
**Color Scheme**: High contrast - Blue (HMM), Purple (GMM), Gold (PELT), Red (Independent)

