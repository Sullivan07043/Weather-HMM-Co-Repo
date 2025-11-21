# Factorized Categorical HMM for Weather Pattern Analysis

This module implements a **Factorized Categorical Hidden Markov Model (HMM)** for analyzing weather patterns and their relationship with ENSO (El Niño-Southern Oscillation) phenomena.

## Overview

The Factorized Categorical HMM assumes conditional independence among features given the hidden state:

```
p(x_t | z_t = k) = ∏_f p(x_{t,f} | z_t = k)
```

where each feature `f` is a categorical variable. This approach allows modeling multiple discrete meteorological features simultaneously while maintaining computational efficiency.

## Features

- **Multi-feature modeling**: Handles 6 ENSO-related meteorological features:
  - Mean temperature
  - Maximum temperature
  - Minimum temperature
  - Sea level pressure
  - Wind speed
  - Precipitation

- **Automatic model selection**: Uses BIC (Bayesian Information Criterion) to select optimal number of hidden states (K) for each site

- **EM algorithm**: Implements Baum-Welch (forward-backward) algorithm for parameter estimation

- **Numerical stability**: Uses log-sum-exp trick to prevent numerical underflow

## File Structure

```
Categorical_HMM/
├── Categorical_HMM.py              # Main implementation
├── README.md                        # This file
├── enso_factorized_categorical_hmm_states.csv  # Hidden state sequences
├── hmm_k_values.txt                # Selected K values per site
└── hmm_parameters.txt              # Trained model parameters
```

## Output Files

### 1. `enso_factorized_categorical_hmm_states.csv`
Contains the decoded hidden state sequence for each site:
- `site_id`: Station identifier
- `t`: Time index
- `state`: Hidden state (0 to K-1)

### 2. `hmm_k_values.txt`
Records the optimal number of hidden states selected for each site:
- Summary statistics showing K value distribution across all sites

### 3. `hmm_parameters.txt`
Detailed model parameters for each site:
- **Initial state distribution (π)**: Starting probabilities for each hidden state
- **Transition matrix (A)**: State transition probabilities
- **Emission matrices (B)**: Conditional probability distributions for each feature given each hidden state

## Model Selection Results

Based on BIC criterion across 24 ENSO-sensitive sites:

| K Value | Number of Sites | Percentage |
|---------|----------------|------------|
| K=2     | 6              | 25.0%      |
| K=3     | 7              | 29.2%      |
| K=4     | 5              | 20.8%      |
| K=5     | 4              | 16.7%      |
| K=6     | 2              | 8.3%       |
| K=7     | 1              | 4.2%       |

**Key Finding**: Most sites (54%) are best modeled with K=2 or K=3, suggesting 2-3 dominant weather regimes at these locations. K=3 may correspond to El Niño, La Niña, and neutral phases.

## Usage

### Prerequisites
```bash
pip install numpy pandas
```

### Running the Model

1. Ensure your data is in the correct format:
   - CSV file with columns: `site_id`, `date`, and meteorological features
   - Features should be discretized into categorical bins

2. Run the script:
```bash
python Categorical_HMM.py
```

### Customizing the Model

Modify these parameters in the main section:

```python
# Select different sites
site_ids = ["942030-99999", "943350-99999", ...]

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

### Model Complexity

Number of free parameters:
```
N_params = (K-1) + K(K-1) + Σ_f K(V_f-1)
```

where:
- K: number of hidden states
- V_f: number of categories for feature f

## Applications

This model can be used for:

1. **Climate state identification**: Discover latent weather regimes
2. **ENSO phase detection**: Correlate hidden states with El Niño/La Niña events
3. **Weather forecasting**: Predict future states based on transitions
4. **Anomaly detection**: Identify unusual weather patterns
5. **Multi-site comparison**: Compare climate dynamics across different locations

## References

- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.
- Zucchini, W., MacDonald, I. L., & Langrock, R. (2016). *Hidden Markov models for time series: an introduction using R*. CRC press.

## Project Information

- **Course**: CSE 250A - Probabilistic Reasoning and Learning
- **Project**: Hidden Markov Models for Weather Pattern Analysis
- **Repository**: [Weather-HMM-Co-Repo](https://github.com/Sullivan07043/Weather-HMM-Co-Repo/tree/HMM)

## License

This project is part of academic coursework at UC San Diego.

## Contact

For questions or issues, please open an issue on the GitHub repository.

