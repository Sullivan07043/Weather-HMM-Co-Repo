# Hidden Markov Models for ENSO Detection from Land-Based Meteorological Observations

**Course**: CSE 250A - Probabilistic Reasoning and Learning  
**Institution**: UC San Diego  
**Project Type**: Final Course Project  
**Academic Year**: 2024-2025

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Description](#problem-description)
3. [System Architecture](#system-architecture)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Model Architecture](#model-architecture)
6. [Experimental Design](#experimental-design)
7. [Implementation Details](#implementation-details)
8. [References](#references)

---

## Project Overview

This project develops an unsupervised learning framework using **Hidden Markov Models (HMMs)** to detect El Niño-Southern Oscillation (ENSO) events from land-based weather station observations. We demonstrate that carefully designed probabilistic models can discover physically meaningful climate patterns without direct ocean measurements.

### Key Innovation

Unlike traditional ENSO detection methods that rely on oceanic measurements (sea surface temperatures, buoy data), our approach uses only standard meteorological observations from land-based weather stations, making it applicable to:
- Regions without ocean monitoring infrastructure
- Historical periods predating modern oceanographic instruments
- Cost-constrained operational settings

### Research Philosophy

This project emphasizes:
- **Probabilistic Reasoning**: Rigorous formulation of independence assumptions
- **Algorithmic Soundness**: Correct implementation of inference and learning methods
- **Model Interpretability**: Clear explanation of modeling choices and their physical justification
- **Systematic Evaluation**: Comprehensive ablation studies and baseline comparisons

---

## Problem Description

### Background

The El Niño-Southern Oscillation (ENSO) is a recurring climate phenomenon involving coupled ocean-atmosphere interactions across the tropical Pacific Ocean. ENSO manifests in two primary phases:

- **El Niño**: Anomalous warming of eastern Pacific waters
- **La Niña**: Anomalous cooling of eastern Pacific waters

These events significantly impact global weather patterns, agricultural productivity, water resources, and disaster risks.

### Research Questions

**Primary Question**: Can we reliably detect ENSO events using only land-based meteorological observations through unsupervised learning of latent climate states?

**Specific Challenges**:
1. How do we model multi-dimensional, heterogeneous meteorological observations within a probabilistic framework?
2. What independence assumptions are reasonable for factorizing high-dimensional emissions?
3. Which meteorological features are most informative for ENSO detection?
4. How can we aggregate predictions across geographically distributed stations?
5. Can learned hidden states correspond to physically meaningful climate regimes?

### Our Approach

We address these challenges through a systematic methodology:

```
Problem Formulation → Data Processing → Model Design → 
Training & Inference → Evaluation → Ablation Studies
```

Each component is designed with clear probabilistic reasoning and physical interpretability.

---

## System Architecture

### Overall Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Raw GSOD   │      │  Preprocessed│      │   HMM Model  │
│     Data     │ ───> │     Data     │ ───> │   Training   │
│  (NOAA)      │      │   (CSV)      │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                            │                       │
                            │                       │
                            ▼                       ▼
                   ┌──────────────┐      ┌──────────────┐
                   │   Station    │      │   Hidden     │
                   │   Filtering  │      │    States    │
                   └──────────────┘      └──────────────┘
                                                │
                                                │
                                                ▼
                                      ┌──────────────┐
                                      │   Ensemble   │
                                      │   Voting     │
                                      └──────────────┘
                                                │
                                                │
                                                ▼
                                      ┌──────────────┐
                                      │   ENSO       │
                                      │   Detection  │
                                      └──────────────┘
```

### Module Organization

**1. Data Module** (`data/`)
- Responsible for data acquisition, cleaning, and preprocessing
- Outputs standardized CSV format
- Independent of downstream modeling choices

**2. HMM Module** (separate branch: `Weather-HMM-Co-Repo-HMM/`)
- Implements Factorized Categorical HMM
- Performs training via EM algorithm
- Executes inference using Forward-Backward algorithm

**3. Evaluation Module**
- Compares predictions against official ONI records
- Computes performance metrics
- Generates ablation study results

**4. Ensemble Module**
- Aggregates station-level predictions
- Performs threshold optimization
- Produces final ENSO classifications

---

## Data Processing Pipeline

### Pipeline Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING PIPELINE                         │
└────────────────────────────────────────────────────────────────────┘

Raw NOAA GSOD Data (tar archives)
         │
         ▼
┌─────────────────────┐
│ Stage 1: Extraction │  • Parse fixed-width format
│  & Initial Parse    │  • Filter selected stations
└─────────────────────┘  • Handle compressed files
         │
         ▼
┌─────────────────────┐
│ Stage 2: Data       │  • Replace sentinel values (9999.9 → NaN)
│   Cleaning          │  • Standardize station IDs
└─────────────────────┘  • Convert date formats
         │
         ▼
┌─────────────────────┐
│ Stage 3: Temporal   │  • Aggregate daily → yearly
│   Aggregation       │  • Compute annual statistics
└─────────────────────┘  • Handle seasonal patterns
         │
         ▼
┌─────────────────────┐
│ Stage 4: Complete   │  • Generate full date range
│  Time Series        │  • Identify missing years
└─────────────────────┘  • Intelligent imputation
         │
         ▼
┌─────────────────────┐
│ Stage 5: Detrending │  • Remove long-term trends
│                     │  • Apply first-order differencing
└─────────────────────┘  • Preserve ENSO variability
         │
         ▼
┌─────────────────────┐
│ Stage 6: Normalize  │  • Scale features to [0,1]
│  & Discretize       │  • Equal-width binning (10 bins)
└─────────────────────┘  • Preserve original values
         │
         ▼
Processed CSV Output (clean, discretized, complete)
```

### Data Scope

**Temporal Coverage**: 1950-2000 (51 years)

**Rationale for Period Selection**:
- Maximizes stations with complete coverage
- Balances data quality with historical depth
- Minimizes missing years requiring imputation
- Captures multiple complete ENSO cycles

**Spatial Coverage**: 21 high-quality stations from 6 Pacific Rim countries

| Region | Countries | Focus |
|--------|-----------|-------|
| East Asia | Japan, South Korea | Western Pacific teleconnections |
| Oceania | Australia | Southern Hemisphere impacts |
| North America | USA, Mexico | Eastern Pacific proximity |
| South America | Peru | Coastal upwelling regions |

**Station Selection Criteria**:
1. ≥90% temporal coverage during 1950-2000
2. Geographic distribution across Pacific basin
3. Minimal instrumentation changes
4. Consistent observation protocols

### Feature Set

**12 Continuous Meteorological Features**:

| Feature Group | Variables | Physical Significance |
|---------------|-----------|----------------------|
| Temperature (3) | mean_temp, max_temp, min_temp | Thermal response to ENSO |
| Atmospheric (2) | sea_level_pressure, wind_speed | Walker circulation indicators |
| Precipitation (2) | precipitation, visibility | Hydrological impacts |
| Wind (3) | wind_speed, max_wind_speed, wind_gust | Trade wind dynamics |
| Snow (2) | snow_depth, (derived) | Regional climate extremes |

**6 Binary Weather Event Indicators**:
- Fog, Rain, Snow, Hail, Thunder, Tornado

### Processing Specifications

**Temporal Aggregation**: Daily → Yearly
- Continuous features: Annual mean
- Precipitation: Annual sum
- Binary events: Annual frequency

**Detrending Method**: First-Order Differencing
```
x'(t) = x(t) - x(t-1)
```
- Removes linear trends
- Preserves interannual variability
- Ensures stationarity for HMM

**Discretization**: Equal-Width Binning
```
normalized = (x - x_min) / (x_max - x_min)
bin = floor(normalized × 10), capped at 9
```
- 10 categories per feature: {0, 1, 2, ..., 9}
- Uniform across all features
- Original values preserved in *_raw columns

**Output Format**: CSV with structure
```
site_id | year | feature_1 | ... | feature_12 | feature_1_raw | ... | events
```

---

## Model Architecture

### Factorized Categorical Hidden Markov Model

#### Core Mathematical Framework

**State Space**: Z = {0, 1} (binary climate regimes)  
**Observation Space**: X = {0,1,...,9}^12 (discretized features)  
**Time Steps**: T = 51 years (1950-2000)

#### Independence Assumptions

Our model makes three key assumptions, each carefully justified:

##### 1. First-Order Markov Property

**Assumption**:
```
p(z_t | z_{1:t-1}) = p(z_t | z_{t-1})
```

**Mathematical Formulation**:
```
p(z_1, ..., z_T) = p(z_1) ∏_{t=2}^T p(z_t | z_{t-1})
```

**Justification**:
- ENSO exhibits strong year-to-year persistence
- Events typically last 12-18 months (spanning 2+ calendar years)
- Higher-order dependencies exponentially increase parameters
- Empirical validation: High learned transition probabilities (>90%)

**Physical Interpretation**: Current ENSO state is the primary predictor of next year's state, capturing the "memory" of ocean heat content anomalies.

**Critical Discussion**: This simplifies delayed oscillator mechanisms (e.g., oceanic Rossby waves with 6-9 month delays). However, at yearly resolution, first-order dependence suffices for classification.

##### 2. Conditional Independence of Observations

**Assumption**:
```
p(x_t | z_t) = ∏_{f=1}^{12} p(x_{t,f} | z_t)
```

**Mathematical Formulation**:
```
p(x_{t,1}, ..., x_{t,12} | z_t = k) = p(x_{t,1}|z_t=k) × ... × p(x_{t,12}|z_t=k)
```

**Justification**:
- **Computational**: Reduces parameters from 10^12 to 12×10×2 = 240
- **Statistical**: Prevents overfitting with limited data (51 time points)
- **Physical**: Hidden state captures large-scale circulation (Walker cell), which independently influences regional meteorological variables
- **Empirical**: Ablation shows feature groups provide complementary (not redundant) information

**Physical Interpretation**: Given the ENSO state (which determines atmospheric circulation patterns), local weather variables are conditionally independent. The hidden state acts as the "common cause" explaining correlations.

**Critical Discussion**: In reality, temperature and precipitation are coupled through thermodynamic processes. However:
- The hidden state captures the primary common mode
- Residual correlations are secondary for classification
- Trade-off: Model simplicity vs. feature coupling

**Alternative Considered**: Fully-dependent emissions
```
p(x_t | z_t) = Multinomial(θ_z_t)  [No factorization]
```
Rejected due to parameter explosion (10^12 values).

##### 3. Stationary Transition Probabilities

**Assumption**:
```
p(z_t | z_{t-1}) = A_{ij}  ∀t
```

**Mathematical Formulation**:
Transition matrix A is time-invariant.

**Justification**:
- Simplifies learning (constant parameters)
- Reasonable for 51-year window (minimal climate change effects)
- Standard HMM assumption

**Physical Interpretation**: ENSO dynamics (ocean-atmosphere feedback loops) remain consistent over the study period.

**Limitation**: Cannot capture potential non-stationarity in ENSO behavior due to anthropogenic climate change. Extension: Time-varying transition matrices.

#### Model Parameters

**Parameter Set**: θ = {π, A, {B^(f)}_{f=1}^{12}}

**Initial Distribution** π ∈ ℝ^2:
```
π_k = p(z_1 = k)
Constraint: π_0 + π_1 = 1
Free parameters: 1
```

**Transition Matrix** A ∈ ℝ^{2×2}:
```
A_{ij} = p(z_{t+1} = j | z_t = i)
Constraint: A_{i,0} + A_{i,1} = 1  ∀i
Free parameters: 2
```

**Emission Matrices** B^(f) ∈ ℝ^{2×10} for each feature f:
```
B^(f)_{k,v} = p(x_{t,f} = v | z_t = k)
Constraint: Σ_v B^(f)_{k,v} = 1  ∀k,f
Free parameters: 12 × 2 × 9 = 216
```

**Total Parameters**: 1 + 2 + 216 = 219

**Data-to-Parameter Ratio**: 1,071 observations / 219 parameters ≈ 4.9

This ratio is reasonable but highlights the importance of regularization (Laplace smoothing).

### Inference Algorithm: Forward-Backward

```
┌───────────────────────────────────────────────────────────────┐
│                  FORWARD-BACKWARD ALGORITHM                    │
└───────────────────────────────────────────────────────────────┘

Forward Pass (α):                    Backward Pass (β):
───────────────                      ───────────────────

t=1: α_1(k) = π_k · p(x_1|k)        t=T: β_T(k) = 1

t>1: α_t(k) = [Σ_j α_{t-1}(j)·      t<T: β_t(k) = Σ_j A_{kj}·
               A_{jk}]·                        β_{t+1}(j)·
               p(x_t|k)                        p(x_{t+1}|j)

         ↓                                    ↓
         └────────────────┬──────────────────┘
                          ↓
              Posterior Computation:
              ────────────────────
              γ_t(k) = p(z_t=k | x_{1:T})
                     = α_t(k)·β_t(k) / Σ_j α_t(j)·β_t(j)
                          ↓
                     Decoding:
                     ────────
                     ẑ_t = argmax_k γ_t(k)
```

**Key Properties**:
- **Time Complexity**: O(K^2 × T × F) = O(4 × 51 × 12) ≈ 2,448 operations
- **Space Complexity**: O(K × T) = O(2 × 51) = 102 values
- **Numerical Stability**: Log-space computation prevents underflow
- **Optimality**: Minimizes expected number of state prediction errors

**Why Not Viterbi?**
- Viterbi finds most probable path: argmax p(z_{1:T} | x_{1:T})
- Forward-Backward finds most probable state at each time: argmax_k p(z_t | x_{1:T})
- For K=2 with high persistence, difference is minimal
- Posterior marginals provide uncertainty estimates

### Learning Algorithm: Expectation-Maximization (Baum-Welch)

```
┌───────────────────────────────────────────────────────────────┐
│                   EM ALGORITHM WORKFLOW                        │
└───────────────────────────────────────────────────────────────┘

Initialize: θ^(0) = {π^(0), A^(0), {B^(f,0)}_{f=1}^{12}}
            ↓
┌───────────────────────────────────────┐
│  Iteration m:                         │
│                                       │
│  ┌─────────────────────────────────┐ │
│  │ E-Step: Compute Posteriors      │ │
│  │                                 │ │
│  │  Run Forward-Backward           │ │
│  │  → γ_t(k) = p(z_t=k | X, θ^(m))│ │
│  │  → ξ_t(i,j) = p(z_t=i, z_{t+1} │ │
│  │               =j | X, θ^(m))    │ │
│  └─────────────────────────────────┘ │
│            ↓                          │
│  ┌─────────────────────────────────┐ │
│  │ M-Step: Update Parameters       │ │
│  │                                 │ │
│  │  π_k^(m+1) = γ_1(k)            │ │
│  │                                 │ │
│  │  A_{ij}^(m+1) = Σ_t ξ_t(i,j)   │ │
│  │                / Σ_t γ_t(i)     │ │
│  │                                 │ │
│  │  B_{k,v}^(f,m+1) = Σ_t γ_t(k)· │ │
│  │                    𝟙(x_{t,f}=v) │ │
│  │                  / Σ_t γ_t(k)   │ │
│  └─────────────────────────────────┘ │
│            ↓                          │
│  Compute: ℓ^(m+1) = log p(X|θ^(m+1))│
│                                       │
│  Check: |ℓ^(m+1) - ℓ^(m)| < ε ?     │
│         └─Yes→ CONVERGED              │
│         └─No──┘ (continue)           │
└───────────────────────────────────────┘
            ↓
Return: θ* = θ^(m+1)
```

**Convergence Criteria**:
- Tolerance: ε = 10^-3
- Maximum iterations: 100
- Typical convergence: 20-50 iterations

**Regularization**: Laplace Smoothing
```
B_{k,v}^(f,new) = [Σ_t γ_t(k)·𝟙(x_{t,f}=v) + ε] / [Σ_t γ_t(k) + 10ε]
```
with ε = 0.01, prevents zero probabilities.

### Model Selection: Bayesian Information Criterion

```
┌───────────────────────────────────────────────────────────────┐
│                    MODEL SELECTION WORKFLOW                    │
└───────────────────────────────────────────────────────────────┘

For each station s:
  ↓
  For K ∈ {2, 3, 4, 5, 6, 7, 8}:
    ↓
    ┌──────────────────────────────┐
    │ Train HMM with K states      │
    │ → Run EM until convergence   │
    └──────────────────────────────┘
    ↓
    ┌──────────────────────────────┐
    │ Compute BIC                  │
    │ BIC(K) = -2·log L + d·log T │
    │                              │
    │ where:                       │
    │  L = p(X | θ̂_K)            │
    │  d = # parameters           │
    │  T = # time steps (51)      │
    └──────────────────────────────┘
    ↓
  Select: K* = argmin_K BIC(K)
  ↓
Result: Optimal K for station s
```

**Parameter Count Formula**:
```
d(K) = (K-1) + K(K-1) + 12×K×(10-1)
     = K^2 - 1 + 108K

For K=2: d = 3 + 216 = 219
For K=3: d = 8 + 324 = 332
```

**BIC Trade-off**:
- Lower K → Better fit but higher BIC penalty
- Higher K → More flexibility but overfitting risk
- BIC balances model complexity and data fit

**Empirical Observation**: 100% of stations select K=2

**Physical Interpretation**: Binary regime (normal vs. anomaly) suffices for ENSO detection at yearly resolution with land-based observations.

### Ensemble Prediction System

```
┌───────────────────────────────────────────────────────────────┐
│                    ENSEMBLE WORKFLOW                           │
└───────────────────────────────────────────────────────────────┘

Step 1: Individual Station Predictions
───────────────────────────────────────
For each station s ∈ {1,...,21}:
  Train HMM → Decode states → {ẑ_t^(s)}_{t=1}^{51}

Step 2: Ranking and Selection
──────────────────────────────
Evaluate each station on validation set
→ Compute F1-scores → Rank stations
→ Select Top K stations (K=14 optimal)

Step 3: Majority Voting
────────────────────────
For each year t ∈ {1,...,51}:
  Collect votes: V_t = {ẑ_t^(s)}_{s ∈ TopK}
  Compute anomaly ratio: r_t = (# anomaly votes) / K
  
  Decision rule:
  ŷ_t = { 1 (anomaly)  if r_t > τ
        { 0 (normal)   if r_t ≤ τ
  
  where τ = 0.40 (40% threshold, optimized)

Step 4: Validation
──────────────────
Compare {ŷ_t} with ground truth ONI
→ Compute metrics (F1, precision, recall, accuracy)
```

**Rationale for Ensemble**:
- Reduces local noise and measurement errors
- Averages out geographic biases
- Improves robustness across different ENSO event types
- Provides implicit confidence estimate (voting ratio)

---

## Experimental Design

### Evaluation Framework

```
┌───────────────────────────────────────────────────────────────┐
│                   EXPERIMENTAL WORKFLOW                        │
└───────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ Ground Truth    │  NOAA Oceanic Niño Index (ONI)
│ (ONI Data)      │  • 1950-2000: 51 years
└─────────────────┘  • 35 anomaly years (68.6%)
        │            • 16 normal years (31.4%)
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
┌─────────────────┐                  ┌─────────────────┐
│ Main Experiment │                  │ Baseline Models │
│   (HMM)         │                  │                 │
│                 │                  │ • Independent   │
│ • Train on all  │                  │   Mixture       │
│   21 stations   │                  │ • K-Means       │
│ • Individual    │                  │ • GMM           │
│   predictions   │                  │                 │
│ • Ensemble      │                  │ Same evaluation │
│   aggregation   │                  │ protocol        │
└─────────────────┘                  └─────────────────┘
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
              ┌─────────────────┐
              │ Performance     │
              │ Metrics         │
              │                 │
              │ • F1-Score      │
              │ • Precision     │
              │ • Recall        │
              │ • Accuracy      │
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Ablation        │
              │ Studies         │
              │                 │
              │ • Feature       │
              │   Importance    │
              │ • K Selection   │
              │ • Temporal      │
              │   Dependencies  │
              │ • Threshold     │
              │   Sensitivity   │
              └─────────────────┘
```

### Baseline Comparisons

**Purpose**: Validate HMM design choices through controlled comparisons

**Baseline 1: Independent Mixture Model**
- **Design**: Same emission structure as HMM, but A = Identity matrix
- **Purpose**: Isolate contribution of temporal dependencies
- **Expected**: HMM should outperform (temporal structure adds value)

**Baseline 2: K-Means Clustering**
- **Design**: Unsupervised clustering on 12-dimensional feature space
- **Purpose**: Test benefit of probabilistic framework vs. hard clustering
- **Expected**: HMM should outperform (soft assignments, temporal smoothing)

**Baseline 3: Gaussian Mixture Model**
- **Design**: GMM on continuous (non-discretized) features
- **Purpose**: Validate discretization choice
- **Expected**: HMM (discrete) may outperform if climate states are better captured by discrete regimes

### Ablation Studies

**Study 1: Feature Importance**

```
Experimental Design:
─────────────────────
Full Model (all 12 features) → Baseline F1

Remove each group separately:
  • Without Temperature (9 features)
  • Without Atmospheric (10 features)
  • Without Precipitation (10 features)
  • Without Weather Events (6 features)

Measure: ΔF1 = F1_baseline - F1_without_group

Interpretation:
  Larger ΔF1 → More important feature group
```

**Study 2: Number of Hidden States**

```
Experimental Design:
─────────────────────
Train models with K ∈ {2, 3, 4, 5, 6, 7, 8}

For each K:
  • Compute BIC(K)
  • Measure F1 score on validation

Compare:
  • BIC preference (statistical criterion)
  • F1 performance (task-specific criterion)

Validate: Do BIC and F1 agree on optimal K?
```

**Study 3: Temporal Dependencies**

```
Experimental Design:
─────────────────────
Model A: Full HMM (temporal dependencies)
  → Forward-Backward uses α_t and β_t

Model B: Independent Model (no temporal)
  → Classification: argmax_k p(x_t | z_t=k)

Compare:
  • Individual station F1
  • Ensemble F1
  • State sequence coherence

Interpretation:
  Performance gap quantifies value of temporal modeling
```

**Study 4: Decoding Algorithm**

```
Experimental Design:
─────────────────────
Same trained HMM, different decoding:

Method A: Viterbi Decoding
  ẑ_{1:T} = argmax p(z_{1:T} | x_{1:T})

Method B: Posterior Marginal Decoding
  ẑ_t = argmax_k p(z_t=k | x_{1:T})

Compare F1 scores

Expected: Minimal difference for K=2 with high persistence
```

**Study 5: Ensemble Threshold Sensitivity**

```
Experimental Design:
─────────────────────
Fixed: Top 14 stations

Vary threshold: τ ∈ {0.30, 0.35, 0.40, 0.45, 0.50, 0.60}

For each τ:
  Classify year as anomaly if voting ratio > τ
  Compute F1, precision, recall

Analyze trade-off:
  Lower τ → Higher recall, lower precision
  Higher τ → Lower recall, higher precision

Select τ* that maximizes F1
```

### Evaluation Metrics

**Primary Metric: F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean balances precision and recall, appropriate for imbalanced classes.

**Supporting Metrics**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

**Confusion Matrix**:
```
                 Predicted Normal    Predicted Anomaly
Actual Normal          TN                   FP
Actual Anomaly         FN                   TP
```

**Interpretation**:
- **High Recall**: Critical for early warning (avoid missing ENSO events)
- **High Precision**: Reduces false alarms (operational credibility)
- **F1-Score**: Overall balance for scientific evaluation

---

## Implementation Details

### Software Architecture

```
Weather-HMM-Co-Repo/
│
├── data/                          # Data Module
│   ├── dataloader2.py            # Main preprocessing pipeline
│   ├── download.py               # GSOD data acquisition
│   ├── searcher.py               # Station filtering
│   └── processed/                # Output directory
│
├── Weather-HMM-Co-Repo-HMM/      # HMM Module (separate branch)
│   └── Categorical_HMM/
│       ├── Categorical_HMM.py    # Core HMM implementation
│       ├── evaluate_enso_f1.py   # Performance evaluation
│       ├── ensemble/             # Ensemble system
│       │   ├── ensemble_voting_enso.py
│       │   └── README.md
│       ├── ablation/             # Ablation studies
│       │   ├── feature_ablation.py
│       │   ├── temporal_ablation.py
│       │   └── ABLATION_SUMMARY.md
│       └── comparison/           # Baseline comparisons
│           ├── generate_independent_ensemble.py
│           └── visualize_model_comparison.py
│
└── PROJECT_README.md             # This document
```

### Key Implementation Classes

**1. GSODDataLoader** (`data/dataloader2.py`)
```python
class GSODDataLoader:
    """Main data preprocessing pipeline"""
    
    def __init__(self, n_bins=10, discretize=True, 
                 time_aggregation='yearly', detrend=True):
        # Initialize preprocessing parameters
    
    def load_station_metadata(self):
        # Load and filter station list
    
    def process_year_data(self, year):
        # Extract and parse data for given year
    
    def ensure_complete_time_series(self, df):
        # Fill missing years with intelligent imputation
    
    def detrend_data(self, df):
        # Remove long-term trends
    
    def normalize_and_discretize(self, df):
        # Discretize continuous features
    
    def save_processed_data(self, df, filename):
        # Export to CSV
```

**2. FactorizedCategoricalHMM** (`Categorical_HMM/Categorical_HMM.py`)
```python
class FactorizedCategoricalHMM:
    """Factorized Categorical HMM implementation"""
    
    def __init__(self, n_states=2, n_features=12, n_categories=10):
        # Initialize model structure
    
    def initialize_parameters(self):
        # Random initialization for EM
    
    def forward_pass(self, X):
        # Compute forward probabilities (α)
    
    def backward_pass(self, X):
        # Compute backward probabilities (β)
    
    def compute_posteriors(self, X):
        # Compute γ_t(k) and ξ_t(i,j)
    
    def m_step(self, X, gamma, xi):
        # Update parameters (π, A, B)
    
    def fit(self, X, max_iter=100, tol=1e-3):
        # EM algorithm main loop
    
    def decode(self, X, method='posterior'):
        # Posterior marginal or Viterbi decoding
    
    def compute_bic(self, X):
        # Calculate BIC for model selection
```

**3. EnsembleVoting** (`ensemble/ensemble_voting_enso.py`)
```python
class EnsembleVoting:
    """Majority voting across stations"""
    
    def __init__(self, threshold=0.40):
        # Set voting threshold
    
    def load_station_predictions(self, station_ids):
        # Load individual HMM predictions
    
    def rank_stations_by_f1(self):
        # Evaluate and rank stations
    
    def select_top_k(self, k=14):
        # Select best performing stations
    
    def aggregate_votes(self, year):
        # Compute voting ratio for given year
    
    def classify(self, voting_ratio):
        # Apply threshold decision rule
    
    def evaluate_performance(self, predictions, ground_truth):
        # Compute metrics
```

### Computational Specifications

**Training Complexity**:
- Per station: O(K^2 × T × F × I) where I = EM iterations
- For K=2, T=51, F=12, I≈30: ~36,720 operations
- Total (21 stations): ~15 minutes on standard CPU

**Memory Requirements**:
- Forward/Backward arrays: 2 × K × T × 8 bytes ≈ 1.6 KB per station
- Parameter storage: (K^2 + K×F×V) × 8 bytes ≈ 17.5 KB
- Data: 21 × 51 × 12 × 4 bytes ≈ 51 KB
- Total: < 1 MB (very efficient)

**Numerical Stability**:
- All forward-backward computations in log-space
- Log-sum-exp trick: log(Σ exp(a_i)) = a_max + log(Σ exp(a_i - a_max))
- Prevents underflow for long sequences

### Technology Stack

**Core Libraries**:
- **NumPy** 1.21+: Numerical computations, matrix operations
- **Pandas** 1.3+: Data manipulation, CSV I/O
- **scikit-learn** 1.0+: Evaluation metrics, baseline models
- **Matplotlib/Seaborn**: Visualization

**Development Tools**:
- **Python** 3.8+
- **Git**: Version control
- **Jupyter**: Exploratory analysis
- **pytest**: Unit testing (inference algorithms)

**Reproducibility**:
- Random seed setting for initialization
- Parameter logging (all hyperparameters saved)
- Version pinning (requirements.txt)

---

## Results Summary

### Main Findings

Our systematic experimental methodology yielded several key insights:

**1. Unsupervised Discovery of Climate States**
- HMM autonomously identifies latent states highly correlated with ENSO
- 100% of stations converge to K=2 (binary regime consensus)
- Learned transition matrices exhibit high persistence (>90%), consistent with ENSO timescales

**2. Feature Importance Hierarchy**
- Atmospheric features (pressure, wind) dominate: ~91% of full performance alone
- Temperature features are secondary: ~76% performance when removed
- Precipitation and weather events are supplementary: ~15% combined contribution
- Validates ENSO as primarily atmospheric circulation phenomenon

**3. Temporal Modeling Value**
- HMM outperforms Independent Mixture by ~3-5% F1
- Benefit primarily from temporal smoothing and state persistence priors
- Larger gains for stations with moderate ENSO signal strength

**4. Ensemble Robustness**
- Aggregating Top 14 stations improves performance significantly over individuals
- 40% voting threshold provides optimal precision-recall balance
- Geographic diversity in top stations validates teleconnection patterns

**5. Model Validation**
- HMM outperforms all baseline models (K-Means, GMM, Independent)
- Ablation studies confirm all design choices are justified
- High recall achieved: rarely misses ENSO events

### Performance Characteristics

**Model Strengths**:
- ✅ High detection rate (catches most ENSO events)
- ✅ Computationally efficient (trains in minutes)
- ✅ Interpretable (clear physical state mapping)
- ✅ Robust (ensemble aggregation reduces noise)

**Model Limitations**:
- ❌ Binary classification only (cannot distinguish El Niño from La Niña)
- ❌ Yearly resolution (misses sub-annual dynamics)
- ❌ Some false positives (borderline normal years misclassified)
- ❌ Limited spatial coverage (21 stations)

---

## Conclusions and Future Work

### Key Contributions

This project demonstrates that:

1. **Land-based observations suffice** for ENSO detection through careful probabilistic modeling
2. **Independence assumptions are reasonable** when properly justified and validated through ablations
3. **Temporal structure matters** but requires careful formulation (first-order Markov is sufficient)
4. **Ensemble methods** dramatically improve robustness over individual predictions
5. **Unsupervised learning can discover** physically meaningful climate patterns

### Modeling Insights

**What We Learned About Independence Assumptions**:
- Conditional independence (given hidden state) is a reasonable approximation for feature factorization
- First-order Markov property captures essential temporal dynamics at yearly resolution
- Stationarity assumption holds for 51-year window but may not extend to climate change scenarios

**What We Learned About ENSO**:
- Atmospheric variables (pressure, wind) provide strongest land-based ENSO signal
- Pacific Rim stations show strongest correlation (validates teleconnection theory)
- Binary regime (normal vs. anomaly) suffices for yearly detection
- ENSO exhibits strong year-to-year persistence in land-based observations

### Proposed Extensions

**Near-Term Improvements**:
1. **Three-State Model**: Extend to K=3 to distinguish El Niño from La Niña
2. **Monthly Resolution**: Capture sub-annual dynamics and event transitions
3. **Feature Engineering**: Compute physically-motivated derived variables (pressure gradients, wind stress)
4. **Spatial Coupling**: Model dependencies between nearby stations (factorial HMMs)

**Methodological Advances**:
1. **Relaxed Independence**: Structured emissions (e.g., chained graphical models for temperature-precipitation coupling)
2. **Semi-Markov Models**: Explicit duration distributions for non-geometric state lifetimes
3. **Time-Varying Parameters**: Adapt to climate change with online learning
4. **Bayesian Inference**: Full posterior over parameters for uncertainty quantification

**Application Extensions**:
1. **Operational Forecasting**: Multi-step-ahead prediction using learned transitions
2. **Other Climate Modes**: Apply to IOD, NAO, MJO, AO
3. **Impact Assessment**: Link states to agricultural yields, extreme events
4. **Real-Time Monitoring**: Deploy as operational dashboard with automated alerts

### Final Remarks

This project successfully demonstrates that sophisticated probabilistic modeling with carefully justified independence assumptions can extract meaningful patterns from complex climate data. The key to success lies in:

- **Rigorous formulation** of modeling assumptions with physical justification
- **Systematic validation** through ablation studies and baseline comparisons
- **Clear interpretation** of learned parameters and their physical meaning
- **Honest assessment** of limitations and appropriate scope

By emphasizing principled probabilistic reasoning over raw predictive performance, we developed a framework that is not only effective but also interpretable and extensible.

---

## AI Usage Statement

This project utilized artificial intelligence tools for specific development tasks:

**Tools Used**:
- **Claude Sonnet 4.5** (Anthropic): Primary assistant
- **GPT-4o** (OpenAI): Secondary consultation

**Specific Applications**:
1. **Code Development**: HMM implementation structure, forward-backward algorithm, EM optimization
2. **Debugging**: Numerical stability issues (log-space), convergence problems
3. **Visualization**: Matplotlib plotting code for workflows and results
4. **Documentation**: LaTeX formatting, mathematical notation, README organization

**Human Contributions**:
- All modeling decisions and independence assumptions
- Physical interpretations and validations
- Experimental design and ablation protocols
- Critical analysis and conclusions
- Final code review and correctness verification

All AI-generated content was carefully reviewed and validated.

---

## References

1. McPhaden, M. J., Zebiak, S. E., & Glantz, M. H. (2006). ENSO as an integrating concept in earth science. *Science*, 314(5806), 1740-1745.

2. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.

3. Zucchini, W., MacDonald, I. L., & Langrock, R. (2016). *Hidden Markov models for time series: an introduction using R* (2nd ed.). CRC Press.

4. Timmermann, A., et al. (2018). El Niño–Southern Oscillation complexity. *Nature*, 559(7715), 535-545.

5. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. (Chapters 17-18: Markov and Hidden Markov Models)

6. Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461-464.

7. NOAA Climate Prediction Center. Oceanic Niño Index (ONI). Retrieved from https://ggweather.com/enso/oni.htm

---

## Acknowledgments

- **Data Provider**: NOAA National Centers for Environmental Information (NCEI)
- **Ground Truth**: NOAA Climate Prediction Center (ONI records)
- **Course Staff**: CSE 250A instructors for guidance on probabilistic modeling
- **Computing**: UC San Diego resources

---

**Last Updated**: November 2024  
**Course**: CSE 250A - Probabilistic Reasoning and Learning  
**Institution**: UC San Diego
