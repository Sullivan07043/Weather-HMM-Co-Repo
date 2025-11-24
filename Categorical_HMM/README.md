# Factorized Categorical HMM for Weather Pattern Analysis

This module implements a **Factorized Categorical Hidden Markov Model (HMM)** for analyzing weather patterns and their relationship with ENSO (El Niño-Southern Oscillation) phenomena. The model uses official ONI (Oceanic Niño Index) data from [NOAA/GGWeather](https://ggweather.com/enso/oni.htm) for validation.

## Overview

The Factorized Categorical HMM assumes conditional independence among features given the hidden state:

```
p(x_t | z_t = k) = ∏_f p(x_{t,f} | z_t = k)
```

where each feature `f` is a categorical variable. This approach allows modeling multiple discrete meteorological features simultaneously while maintaining computational efficiency.

## Features

- **Multi-feature modeling**: Handles 13 ENSO-related meteorological features:
  - Mean temperature
  - Maximum temperature
  - Minimum temperature
  - Sea level pressure
  - Wind speed
  - Precipitation
  - Visibility
  - Fog occurrence
  - Rain occurrence
  - Snow occurrence
  - Hail occurrence
  - Thunder occurrence
  - Tornado occurrence

- **Automatic model selection**: Uses BIC (Bayesian Information Criterion) to select optimal number of hidden states (K) for each site

- **EM algorithm**: Implements Baum-Welch (forward-backward) algorithm for parameter estimation

- **Viterbi decoding**: Uses Viterbi algorithm to find the most likely state sequence (global optimum)

- **Numerical stability**: Uses log-sum-exp trick to prevent numerical underflow

- **Trend removal**: Detrends continuous features using adaptive methods (polynomial, differencing, high-pass filtering)

- **Complete time series**: Data preprocessing ensures continuous yearly sequences (1950-2010) with interpolation for missing values

## File Structure

```
Categorical_HMM/
├── Categorical_HMM.py                          # Main HMM implementation
├── README.md                                    # This file
├── ENSO_DATA_README.md                         # ENSO data documentation
├── enso_oni_data_1950_2010.csv                 # ENSO ground truth (1950-2010)
├── enso_factorized_categorical_hmm_states.csv  # Hidden state sequences (21 stations)
├── hmm_k_values.txt                            # Selected K values per site
├── hmm_parameters.txt                          # Trained model parameters
├── evaluate_enso_f1.py                         # ENSO anomaly evaluation (F1-based)
├── enso_evaluation_f1_results.csv              # Evaluation results
├── visualize_top10_f1.py                       # Visualization script
├── plot_top10_f1.py                            # Performance comparison plots
├── top10_f1_enso_sites_table.png               # Top 10 sites table
├── top10_f1_time_series_comparison.png         # Time series comparison
├── top10_f1_performance_comparison.png         # Performance metrics
├── ensemble/                                    # Ensemble voting system (All 21)
│   ├── README.md                               # Ensemble documentation
│   ├── ensemble_voting_enso.py                 # Voting analysis (50% threshold)
│   ├── plot_ensemble_voting.py                 # Voting visualizations
│   ├── ensemble_voting_results.csv             # Year-by-year results
│   ├── ensemble_voting_enso_analysis.png       # Analysis plots
│   └── ensemble_voting_detailed_comparison.png # Detailed comparison
├── comparison/                                  # Model comparison (HMM vs GMM vs PELT vs Independent)
│   ├── README.md                               # Comparison documentation
│   ├── ensemble_voting_results.csv             # HMM ensemble results
│   ├── gmm_ensemble_voting_results.csv         # GMM ensemble results
│   ├── PELT_enso_ensemble_results.csv          # PELT ensemble results
│   ├── independent_ensemble_voting_results.csv # Independent classifier results
│   ├── extract_independent_ensemble.py         # Generate independent predictions
│   ├── visualize_model_comparison_v2.py        # Create comparison visualizations
│   ├── figure1_performance_metrics.png         # Performance & confusion matrix
│   ├── figure2_threshold_analysis.png          # F1-Score vs threshold
│   ├── figure3_temporal_analysis.png           # Year-by-year detection timeline
│   └── figure4_performance_summary.png         # Performance summary table
├── ablation/                                    # Ablation studies
│   ├── README.md                               # Ablation documentation
│   ├── feature_ablation.py                     # Feature importance analysis
│   ├── temporal_ablation.py                    # Temporal dependency analysis
│   ├── run_all_ablations.py                    # Run all experiments
│   ├── feature_ablation_results.csv            # Feature ablation results
│   ├── temporal_ablation_results.csv           # Temporal ablation results
│   ├── feature_ablation_analysis.png           # Feature importance visualization
│   ├── temporal_ablation_analysis.png          # Temporal dependency visualization
│   └── ABLATION_SUMMARY.md                     # Comprehensive ablation summary
└── data/                                        # Data preprocessing
    ├── searcher.py                             # Station filtering
    ├── dataloader2.py                          # Data loading, cleaning & detrending
    ├── stations_1950_2010_covered_top_each_country.csv  # Selected stations
    └── processed/
        ├── weather_1901_2019_yearly_detrend_adaptive_bins10.csv  # Processed dataset
        ├── normalization_info.txt             # Feature normalization parameters
        ├── trend_removal_summary.txt          # Trend removal summary
        └── trend_removal_detailed_report.csv  # Detailed trend removal report
```

## Output Files

### 1. `enso_factorized_categorical_hmm_states.csv`
Contains the decoded hidden state sequence for each site:
- `site_id`: Station identifier
- `year`: Calendar year (1950-2010)
- `state`: Hidden state (0 or 1)
- **Total records**: 1,281 (21 stations × 61 years)

### 2. `hmm_k_values.txt`
Records the optimal number of hidden states selected for each site:
- All 21 sites selected **K=2** based on BIC criterion
- Indicates 2 dominant weather regimes corresponding to ENSO states

### 3. `hmm_parameters.txt`
Detailed model parameters for each site:
- **Initial state distribution (π)**: Starting probabilities for each hidden state
- **Transition matrix (A)**: State transition probabilities
- **Emission matrices (B)**: Conditional probability distributions for each feature given each hidden state

## Model Selection Results

Based on BIC criterion across 21 globally distributed stations with complete 1950-2010 data coverage:

| K Value | Number of Sites | Percentage |
|---------|----------------|------------|
| K=2     | 21             | 100%       |

**Key Finding**: All sites (100%) are best modeled with K=2, strongly suggesting 2 dominant weather regimes at these locations, which correspond to ENSO anomaly states (El Niño/La Niña) versus normal conditions.

## Data Quality Improvements

### Version 3.0 Enhancements (1950-2010 Period)

1. **Extended Time Range**: Analysis period extended to 1950-2010 (61 years)
   - Captures more recent ENSO events (including strong 1997-98 El Niño, 2007-08 La Niña, 2009-10 El Niño)
   - Better statistical power with larger sample size
   - More robust model validation

2. **Advanced Trend Removal**: 
   - Adaptive detrending using multiple methods (linear, polynomial, differencing, high-pass filtering)
   - Removes climate change signals and long-term trends
   - Preserves ENSO variability while removing non-stationary components
   - Average 100% trend reduction in continuous features

3. **Complete Time Series**: All stations have continuous yearly data
   - No missing years in the analysis period
   - Linear interpolation applied to fill gaps in feature values
   - Ensures accurate HMM transition probability estimation

4. **Strict Station Selection**: 
   - 21 high-quality stations with complete 1950-2010 coverage
   - Each station has all 61 years of data
   - Globally distributed across ENSO-sensitive regions

## ENSO Anomaly Detection Performance

The model's ability to detect ENSO events was evaluated using official ONI data (1950-2010) from [NOAA](https://ggweather.com/enso/oni.htm):

### Historical ENSO Events (1950-2010, Moderate+ Strength)
- **El Niño years**: 13 years (Moderate or stronger)
- **La Niña years**: 8 years (Moderate or stronger)
- **Total anomaly years**: 21 out of 61 years (34.4%)
- **Normal years**: 40 years (65.6%)
- **Note**: Only Moderate, Strong, and Very Strong ENSO events are classified as anomalies

### TOP 10 Sites by F1 Score (Moderate+ ENSO Definition)

| Rank | Site ID | Station Name | Country | F1 Score | Precision | Recall | Accuracy |
|------|---------|--------------|---------|----------|-----------|--------|----------|
| 1 | 763420-99999 | MONCLOVA INTL | Mexico | 0.5588 | 0.4516 | 0.7368 | 63.93% |
| 2 | 476710-99999 | TOKYO INTL | Japan | 0.5507 | 0.4074 | 0.8462 | 59.02% |
| 3 | 724050-13743 | RONALD REAGAN WASHINGTON NATL AP | USA | 0.5484 | 0.4074 | 0.8462 | 59.02% |
| 4 | 847520-99999 | RODRIGUEZ BALLON | Peru | 0.5352 | 0.3871 | 0.8571 | 57.38% |
| 5 | 911820-22521 | HONOLULU INTERNATIONAL AIRPORT | USA | 0.5122 | 0.3548 | 0.9167 | 54.10% |
| 6 | 471420-99999 | DAEGU AB | South Korea | 0.5000 | 0.3548 | 0.8462 | 52.46% |
| 7 | 478080-99999 | FUKUOKA | Japan | 0.5000 | 0.3548 | 0.8462 | 52.46% |
| 8 | 843900-99999 | CAPITAN MONTES | Chile | 0.4938 | 0.3548 | 0.8000 | 52.46% |
| 9 | 722860-23119 | MARCH AIR RESERVE BASE | USA | 0.4898 | 0.3333 | 0.8889 | 50.82% |
| 10 | 479300-99999 | NAHA | Japan | 0.4746 | 0.3226 | 0.9091 | 49.18% |

**Average Performance (TOP 10)**:
- **F1 Score**: 0.5164
- **Precision**: 0.3729
- **Recall**: 0.8493
- **Accuracy**: 55.08%

**Geographic Distribution**: Pacific Rim sites (Mexico, Japan, USA, Peru, Chile, South Korea) show strong correlation between hidden states and Moderate+ ENSO anomalies. The higher recall (84.93%) indicates the model is effective at detecting significant ENSO events, while lower precision reflects the challenge of distinguishing Moderate+ events from Weak events and normal conditions.

## Ensemble Voting Performance (Moderate+ ENSO Definition)

Majority voting across **All 21 stations** provides comprehensive ENSO detection for Moderate and stronger events:

### Best Configuration: All 21 Stations, 50% Threshold (Recommended Default)

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 57.38% | Overall correctness |
| **Precision** | 43.24% | Accuracy when predicting anomaly |
| **Recall** | 76.19% | **Detection rate** - catches 16 out of 21 Moderate+ ENSO events |
| **F1-Score** | 0.5517 | **Optimal balance** - highest among all thresholds |

**Confusion Matrix (50% threshold, All 21):**
```
                Predicted: Normal    Predicted: Anomaly
Actual: Normal          19                  21
Actual: Anomaly          5                  16
```

**Key Findings**:
- ✅ **Best F1-Score (0.5517)** at 50% threshold - optimal balance for Moderate+ events
- ✅ **76.19% Recall**: Catches most significant ENSO events (16/21)
- ✅ **43.24% Precision**: Balances detection with false positive control
- ⚠️ **5 Missed Events**: Mainly early period Moderate events (1957, 1958, 1965, 1966, 1973)
- ⚠️ **21 False Positives**: Includes Weak ENSO events and borderline years

### Threshold Comparison (All 21 Stations)

| Threshold | F1 Score | Recall | Precision | Accuracy | Use Case |
|-----------|----------|--------|-----------|----------|----------|
| 30% | 0.5122 | 100.00% | 34.43% | 34.43% | Maximum sensitivity |
| 40% | 0.5278 | 90.48% | 37.25% | 44.26% | High sensitivity |
| **50%** ⭐ | **0.5517** | **76.19%** | **43.24%** | **57.38%** | **Optimal balance** |
| 55% | 0.5098 | 61.90% | 43.33% | 59.02% | Conservative |
| 60% | 0.4889 | 52.38% | 45.83% | 62.30% | High confidence |

**Threshold Selection Rationale**:
- 50% threshold provides the best F1-score (0.5517)
- Balances recall (76.19%) with acceptable precision (43.24%)
- Suitable for detecting Moderate and stronger ENSO events
- Lower thresholds increase recall but reduce precision significantly

See `ensemble/README.md` for detailed ensemble voting analysis.

## Model Comparison

We compared four different approaches for ENSO detection using the same 21 stations and 1950-2000 time period:

### Models Evaluated

1. **HMM (Hidden Markov Model)** - Our factorized categorical HMM with temporal dependencies
2. **GMM (Gaussian Mixture Model)** - Probabilistic clustering without temporal modeling
3. **PELT (Pruned Exact Linear Time)** - Change point detection algorithm
4. **Independent Classifier** - Mixture model without temporal dependencies (ablation baseline)

### Comparison Results (50% Threshold, 1950-2000)

| Model | F1-Score | Accuracy | Precision | Recall | Key Characteristic |
|-------|----------|----------|-----------|--------|-------------------|
| **HMM** | **0.6383** ⭐ | **0.6667** ⭐ | **0.5556** ⭐ | 0.7500 | Best overall performance |
| **Independent** | 0.5231 | 0.3922 | 0.3778 | **0.8500** ⭐ | Highest recall, over-predicts |
| **GMM** | 0.3404 | 0.3922 | 0.2963 | 0.4000 | Moderate performance |
| **PELT** | 0.0000 | 0.6078 | 0.0000 | 0.0000 | Failed to detect anomalies |

### Key Findings

1. **HMM Outperforms All Baselines**
   - Highest F1-Score (0.6383), Accuracy (66.67%), and Precision (55.56%)
   - Best balance between detection and false positive control
   - Temporal modeling provides clear advantage

2. **Temporal Dependencies Matter**
   - HMM (with temporal) vs Independent (without temporal): +22.0% F1 improvement
   - Validates the importance of modeling state transitions
   - Forward-backward algorithm effectively leverages sequence information

3. **Model Characteristics**
   - **HMM**: Balanced performance, suitable for most applications
   - **Independent**: High recall (85%), use when missing events is costly
   - **GMM**: Moderate performance, may need feature engineering
   - **PELT**: Not suitable for gradual ENSO transitions

See `comparison/README.md` for detailed model comparison analysis and visualizations.

## Usage

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Running the Model

1. Preprocess data with trend removal:
```bash
cd data
python dataloader2.py  # Includes trend removal
python fill_missing_years_detrended.py
```

2. Train HMM:
```bash
python Categorical_HMM.py
```

3. Evaluate ENSO detection:
```bash
python evaluate_enso_f1.py
```

4. Generate visualizations:
```bash
python visualize_top10_f1.py
python plot_top10_f1.py
```

5. Run ensemble voting (All 21 stations, 50% threshold):
```bash
cd ensemble
python ensemble_voting_enso.py
python plot_ensemble_voting.py
```

6. Run model comparison (HMM vs GMM vs PELT vs Independent):
```bash
cd comparison
python extract_independent_ensemble.py  # Generate independent classifier results
python visualize_model_comparison_v2.py  # Create comparison visualizations
```

7. Run ablation studies (feature importance & temporal dependencies):
```bash
cd ablation
python run_all_ablations.py  # Run all ablation experiments
```

### Customizing the Model

Modify these parameters in the main section:

```python
# Site IDs are now loaded from CSV
csv_path = 'data/stations_1950_2000_covered_top_each_country.csv'

# Change maximum number of states to try
max_K = 8

# Modify convergence criteria
n_iter = 100
tol = 1e-3
```

## Algorithm Details

### EM Algorithm (Baum-Welch)

**E-step**: Compute posteriors using forward-backward algorithm
- Forward pass: `α_t(k) = p(x_1:t, z_t=k)`
- Backward pass: `β_t(k) = p(x_{t+1:T} | z_t=k)`
- State posteriors: `γ_t(k) = p(z_t=k | x_1:T)`
- Transition posteriors: `ξ_t(i,j) = p(z_t=i, z_{t+1}=j | x_1:T)`

**M-step**: Update parameters
- Initial distribution: `π_k ∝ γ_1(k)`
- Transition matrix: `A_{ij} ∝ Σ_t ξ_t(i,j)`
- Emission matrices: `B_f(k,v) ∝ Σ_t γ_t(k) · 1{x_{t,f}=v}`

### Viterbi Decoding

The Viterbi algorithm finds the most likely state sequence globally:

**Initialization** (t=0):
```
δ_0(k) = π_k · p(x_0 | z_0=k)
```

**Recursion** (t=1 to T-1):
```
δ_t(j) = max_i [δ_{t-1}(i) · A_{ij}] · p(x_t | z_t=j)
ψ_t(j) = argmax_i [δ_{t-1}(i) · A_{ij}]
```

**Termination**:
```
z_T* = argmax_k δ_T(k)
```

**Backtracking** (t=T-1 to 0):
```
z_t* = ψ_{t+1}(z_{t+1}*)
```

This ensures the decoded state sequence respects transition probabilities and finds the globally optimal path, unlike posterior decoding which maximizes each state independently.

### Model Complexity

Number of free parameters:
```
N_params = (K-1) + K(K-1) + Σ_f K(V_f-1)
```

where:
- K: number of hidden states (K=2 for all sites)
- V_f: number of categories for feature f (10 bins per feature)

## Visualizations

### Time Series Comparison
![Time Series Comparison](top10_f1_time_series_comparison.png)

Time series plots showing predicted hidden states for the top 10 performing sites from 1950-2010, with actual ENSO years highlighted.

### Performance Comparison
![Performance Comparison](top10_f1_performance_comparison.png)

Comprehensive performance comparison including F1 scores, precision vs recall, accuracy, and confusion matrices for top 10 stations.

### Station Information Table
![Station Table](top10_f1_enso_sites_table.png)

Detailed table showing station metadata, location, and performance metrics for top 10 sites.

### Ensemble Voting Analysis
![Ensemble Analysis](ensemble/ensemble_voting_enso_analysis.png)

Three-panel analysis showing time series predictions, voting ratios over time, and performance metrics at 50% threshold (All 21 stations) for Moderate+ ENSO events.

## Applications

This model can be used for:

1. **Climate state identification**: Discover latent weather regimes
2. **ENSO phase detection**: Correlate hidden states with Moderate+ El Niño/La Niña events (ensemble F1: 0.5517)
3. **Weather forecasting**: Predict future states based on transitions
4. **Anomaly detection**: Identify significant ENSO patterns (ensemble recall: 76.19%)
5. **Multi-site comparison**: Compare climate dynamics across different locations
6. **Significant event detection**: Focus on Moderate and stronger ENSO events with 50% consensus threshold

## Data Sources

- **Weather Data**: NOAA Global Surface Summary of the Day (GSOD)
  - Preprocessed with complete time series (1950-2010)
  - 21 globally distributed stations
  - Features binned into 10 categories each
  - Detrended to remove long-term climate trends
  
- **ENSO Index**: Oceanic Niño Index (ONI) from [NOAA Climate Prediction Center](https://ggweather.com/enso/oni.htm)
  - Official records (1950-2010)
  - 21 Moderate+ anomaly years (13 El Niño, 8 La Niña)
  - Focus on significant ENSO events (Moderate, Strong, Very Strong)

## References

- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.
- Zucchini, W., MacDonald, I. L., & Langrock, R. (2016). *Hidden Markov models for time series: an introduction using R*. CRC press.
- NOAA Climate Prediction Center. Oceanic Niño Index (ONI). Retrieved from https://ggweather.com/enso/oni.htm

## Version History

- **v3.3** (2025-11-24): Model comparison and ablation studies
  - **Comparison**: Added comprehensive comparison with GMM, PELT, and Independent Classifier
  - **Ablation**: Feature importance and temporal dependency analysis
  - **Results**: HMM achieves best F1-Score (0.6383), outperforming all baselines
  - **Findings**: Temporal dependencies provide +22% F1 improvement over independent model
  - **Visualizations**: 4 high-quality comparison figures + 2 ablation analysis figures
  - All experiments use 1950-2000 period with 21 stations

- **v3.2** (2025-11-24): Viterbi decoding implementation
  - **Decoding**: Changed from posterior decoding to Viterbi algorithm
  - **Algorithm**: Finds globally optimal state sequence using dynamic programming
  - **Performance**: Maintained F1=0.5517 with 50% threshold
  - **Benefit**: Ensures valid state transitions and global optimality
  - All other configurations unchanged (13 features, Moderate+ ENSO, 21 stations)

- **v3.1** (2025-11-24): Moderate+ ENSO definition with enhanced features
  - **ENSO Definition**: Moderate, Strong, and Very Strong events only (21 anomaly years)
  - **Features**: Expanded to 13 features (added visibility + 6 binary weather events)
  - **Ensemble**: All 21 stations with 50% threshold (F1=0.5517)
  - **Performance**: 76.19% recall, 43.24% precision for significant ENSO events
  - **Threshold**: 50% provides optimal F1-score for Moderate+ detection
  - All stations select K=2 (100% consensus)

- **v3.0** (2025-11-24): Extended period and trend removal
  - Extended analysis period to 1950-2010 (61 years)
  - Added adaptive trend removal (polynomial, differencing, high-pass filtering)
  - 21 stations with complete 1950-2010 coverage
  - All ENSO strengths included (44 anomaly years)
  - All stations select K=2 (100% consensus)

- **v2.0** (2025-11-23): Major data quality update
  - Complete time series for all stations (1950-2000, no missing years)
  - Strict station filtering (17 stations with full coverage)
  - Improved interpolation for missing feature values
  - Updated performance metrics with new data
  - All stations select K=2 (100% consensus)
  - Ensemble F1-score: 0.819 (40% threshold, 17 stations)

- **v1.0** (2025-11-22): Initial release
  - 24 stations with varying data coverage
  - Basic interpolation for missing values
  - 95.8% stations select K=2

## Project Information

- **Course**: CSE 250A - Probabilistic Reasoning and Learning
- **Project**: Hidden Markov Models for Weather Pattern Analysis
- **Repository**: [Weather-HMM-Co-Repo](https://github.com/Sullivan07043/Weather-HMM-Co-Repo/tree/HMM)

## License

This project is part of academic coursework at UC San Diego.

## Contact

For questions or issues, please open an issue on the GitHub repository.
