# Ablation Studies for ENSO Detection HMM

This directory contains systematic ablation experiments to understand the key components contributing to ENSO detection performance.

## 🎯 Experiment Overview

### 1. **Feature Ablation Study** (`feature_ablation.py`)

**Research Question**: Which feature groups are most important for ENSO detection?

**Methodology**:
- **Baseline**: Train HMM with all 13 features
- **Ablation**: Remove each feature group and measure performance drop
- **Isolation**: Train with only one feature group at a time

**Feature Groups**:
1. **Temperature** (3 features): mean_temp, max_temp, min_temp
2. **Atmospheric** (2 features): sea_level_pressure, wind_speed
3. **Precipitation** (2 features): precipitation, visibility
4. **Weather Events** (6 features): fog, rain, snow, hail, thunder, tornado

**Key Insights**:
- Identifies which features contribute most to ENSO detection
- Reveals redundancy between feature groups
- Guides feature selection for model simplification

---

### 2. **Temporal Dependency Ablation** (`temporal_ablation.py`)

**Research Question**: Do temporal dependencies (Markov transitions) improve ENSO detection?

**Models Compared**:
1. **Full HMM**: Uses forward-backward algorithm, considers temporal sequence
2. **Independent Mixture Model**: Treats each time point independently, no sequence information

**Analysis**:
- Measures the contribution of the Markov assumption
- Examines learned transition matrices across stations
- Quantifies state persistence vs transition probabilities

**Key Insights**:
- Determines if ENSO events exhibit temporal autocorrelation
- Reveals whether year-to-year dependencies are meaningful
- Validates the HMM architecture choice

---

## 📊 Running the Experiments

### Prerequisites
```bash
cd /path/to/Categorical_HMM/ablation
```

Ensure the parent directory contains:
- `Categorical_HMM.py` (HMM implementation)
- `data/` directory with processed weather data
- `enso_oni_data_1950_2010.csv` (ground truth)

### Run Individual Experiments

```bash
# Feature ablation (estimated time: 30-45 minutes)
python3 feature_ablation.py

# Temporal ablation (estimated time: 15-20 minutes)
python3 temporal_ablation.py
```

### Run All Experiments
```bash
# Run all ablation studies sequentially
python3 feature_ablation.py && python3 temporal_ablation.py
```

---

## 📈 Output Files

Each experiment generates:

### Feature Ablation
- `feature_ablation_results.csv` - Performance metrics for each configuration
- `feature_ablation_analysis.png` - Comprehensive visualization (4 subplots)

### Temporal Ablation
- `temporal_ablation_results.csv` - HMM vs Independent classifier comparison
- `temporal_ablation_analysis.png` - Temporal dependency analysis (6 subplots)

---

## 🔬 Experimental Design Principles

### 1. **Fair Comparison**
- All experiments use K=2 hidden states (fixed for consistency)
- Same random seed (0) for reproducibility
- Same training parameters (100 iterations, tol=1e-3)
- Same evaluation period (1950-2000)

### 2. **Comprehensive Evaluation**
- **Station-level metrics**: Individual station F1-scores
- **Ensemble metrics**: Majority voting performance (50% threshold)
- **Multiple metrics**: F1, Accuracy, Precision, Recall

### 3. **Statistical Rigor**
- 21 stations provide multiple independent evaluations
- 51 years (1950-2000) for robust temporal assessment
- Confusion matrices for detailed error analysis

---

## 📊 Interpretation Guide

### Performance Metrics

| Metric | Interpretation |
|--------|---------------|
| **F1-Score** | Harmonic mean of precision and recall (primary metric) |
| **Accuracy** | Overall correctness (can be misleading with class imbalance) |
| **Precision** | How many predicted anomalies are correct |
| **Recall** | How many true anomalies are detected |

### Ablation Results

**Feature Ablation**:
- Large F1 drop when removing a group → That group is important
- High F1 with only one group → That group is sufficient
- Similar F1 with/without a group → That group is redundant

**Temporal Ablation**:
- HMM > Independent → Temporal dependencies are valuable (forward-backward helps)
- HMM ≈ Independent → Year-to-year transitions are weak
- High persistence (diagonal) → States are stable, HMM leverages this pattern

---

## 🎓 Scientific Value

These ablation studies provide:

1. **Model Understanding**: Which components are essential vs optional
2. **Feature Engineering**: Guide for data collection and preprocessing
3. **Architecture Validation**: Justify the HMM design choices
4. **Performance Bounds**: Upper and lower limits of model capability
5. **Future Directions**: Identify areas for improvement

---

## 📝 Citation

If you use these ablation studies, please cite:

```
Factorized Categorical HMM for ENSO Detection
Ablation Studies: Feature Importance and Temporal Dependencies
Analysis Period: 1950-2000 (51 years)
Stations: 21 global weather stations
```

---

## 🔄 Version History

- **v1.0** (2024-11-24): Initial ablation study framework
  - Feature ablation with 4 feature groups
  - Temporal dependency analysis (HMM vs Independent)

---

## 📧 Contact

For questions about these ablation experiments, please refer to the main project README.

---

## ⚠️ Notes

- **Computational Cost**: Feature ablation is most expensive (trains 9 model configurations)
- **Memory Usage**: Each experiment loads full dataset into memory (~1GB)
- **Reproducibility**: Fixed random seed ensures consistent results across runs
- **Extensibility**: Easy to add new feature groups or model architectures

---

## 🚀 Future Ablation Studies

Potential extensions:
1. **Data Ablation**: Impact of training data size (temporal subsampling)
2. **Spatial Ablation**: Geographic region importance
3. **Preprocessing Ablation**: Detrending vs raw data
4. **Discretization Ablation**: Number of bins (5, 10, 15, 20)
5. **State Number Ablation**: K=2 vs K=3 vs K=4 (systematic comparison)

---

**Last Updated**: 2024-11-23

