"""
Extract Independent Classifier Ensemble Voting Results from Ablation Study
Generate year-by-year results compatible with comparison visualization
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Categorical_HMM import FactorizedCategoricalHMM, load_data


class IndependentMixtureModel:
    """
    Independent Mixture Model - No temporal dependencies
    Each time point is classified independently based on emission probabilities only
    """
    
    def __init__(self, n_states=2, n_categories=None, max_iter=100, tol=1e-4, random_state=42):
        self.K = n_states
        self.n_categories = n_categories if isinstance(n_categories, (list, tuple)) else [n_categories]
        self.F = len(self.n_categories)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Model parameters (no transition matrix!)
        self.pi = None  # Initial state distribution (also serves as marginal state probability)
        self.B = None   # List of emission probability matrices [B_f for f in features]
        
    def _initialize_parameters(self):
        """Initialize model parameters randomly"""
        np.random.seed(self.random_state)
        
        # Initialize state probabilities (uniform + small noise)
        self.pi = np.ones(self.K) / self.K + np.random.randn(self.K) * 0.01
        self.pi = np.abs(self.pi)
        self.pi /= self.pi.sum()
        
        # Initialize emission probabilities for each feature
        self.B = []
        for f in range(self.F):
            C_f = self.n_categories[f]
            B_f = np.random.dirichlet(np.ones(C_f), size=self.K)
            self.B.append(B_f)
    
    def _compute_log_emission(self, X):
        """
        Compute log emission probabilities for all time steps
        X: (T, F) array of observations
        Returns: (T, K) array of log P(x_t | z_t = k)
        """
        T = len(X)
        log_emission = np.zeros((T, self.K))
        
        for t in range(T):
            for k in range(self.K):
                log_prob = 0.0
                for f in range(self.F):
                    x_tf = int(X[t, f])
                    if 0 <= x_tf < self.n_categories[f]:
                        log_prob += np.log(self.B[f][k, x_tf] + 1e-10)
                    else:
                        log_prob += -1e10
                log_emission[t, k] = log_prob
        
        return log_emission
    
    def _e_step(self, X):
        """
        E-step: Compute posterior probabilities independently for each time step
        Returns: gamma (T, K) - posterior probabilities
        """
        T = len(X)
        log_emission = self._compute_log_emission(X)
        
        # For each time step independently: P(z_t=k | x_t) ∝ P(x_t | z_t=k) * P(z_t=k)
        log_gamma = log_emission + np.log(self.pi + 1e-10)
        
        # Normalize (log-sum-exp trick)
        log_gamma_max = np.max(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma - log_gamma_max)
        gamma /= gamma.sum(axis=1, keepdims=True)
        
        return gamma
    
    def _m_step(self, X, gamma):
        """
        M-step: Update parameters based on posteriors
        """
        T = len(X)
        
        # Update state probabilities (average over all time steps)
        self.pi = gamma.mean(axis=0)
        self.pi /= self.pi.sum()
        
        # Update emission probabilities for each feature
        for f in range(self.F):
            for k in range(self.K):
                for c in range(self.n_categories[f]):
                    mask = (X[:, f] == c)
                    numerator = (gamma[:, k] * mask).sum()
                    denominator = gamma[:, k].sum()
                    self.B[f][k, c] = (numerator + 1e-10) / (denominator + 1e-10)
            
            # Normalize
            self.B[f] = self.B[f] / self.B[f].sum(axis=1, keepdims=True)
    
    def _compute_log_likelihood(self, X):
        """Compute log-likelihood of the data"""
        T = len(X)
        log_emission = self._compute_log_emission(X)
        
        # For each time step: P(x_t) = sum_k P(x_t | z_t=k) * P(z_t=k)
        log_likelihood = 0.0
        for t in range(T):
            log_probs = log_emission[t] + np.log(self.pi + 1e-10)
            log_likelihood += np.logaddexp.reduce(log_probs)
        
        return log_likelihood
    
    def fit(self, X):
        """
        Fit the independent mixture model using EM algorithm
        X: (T, F) array of observations
        """
        self._initialize_parameters()
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step
            gamma = self._e_step(X)
            
            # M-step
            self._m_step(X, gamma)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            
            prev_log_likelihood = log_likelihood
        
        return self
    
    def predict(self, X):
        """
        Predict states independently for each time step
        Returns: (T,) array of predicted states
        """
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)


def main():
    """Main execution"""
    
    # Configuration
    csv_path = "/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/data/processed/weather_1901_2019_yearly_detrend_adaptive_bins10.csv"
    enso_path = "/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/enso_oni_data_1950_2010.csv"
    top_sites_path = "/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/data/stations_1950_2010_covered_top_each_country.csv"
    
    # Load ENSO data
    enso_df = pd.read_csv(enso_path)
    enso_dict = dict(zip(enso_df['year'], enso_df['enso_type']))
    enso_labels = dict(zip(enso_df['year'], enso_df['enso_anomaly']))
    
    # Load top sites
    top_sites_df = pd.read_csv(top_sites_path)
    # Construct site_id from USAF-WBAN
    site_ids = (top_sites_df['USAF'].astype(str) + '-' + top_sites_df['WBAN'].astype(str)).tolist()
    
    print(f"Loading data for {len(site_ids)} stations...")
    
    # Load data using the same function as HMM
    EL_NINO_FEATURES = [
        'mean_temp', 'max_temp', 'min_temp',
        'sea_level_pressure', 'wind_speed',
        'precipitation', 'visibility',
        'fog', 'rain', 'snow', 'hail', 'thunder', 'tornado'
    ]
    
    data_dict, lengths_dict, feature_cols, n_categories = load_data(
        csv_path, site_ids=site_ids, feature_cols=EL_NINO_FEATURES
    )
    
    print(f"Loaded {len(data_dict)} stations")
    print(f"Features: {len(feature_cols)}")
    print(f"Categories per feature: {n_categories}")
    
    # Filter to 1950-2000 (51 years)
    # First, determine the actual year range for each station
    csv_df = pd.read_csv(csv_path, parse_dates=["date"])
    csv_df = csv_df[csv_df['site_id'].isin(site_ids)]
    
    # Filter to 1950-2000
    csv_df['year'] = pd.to_datetime(csv_df['date']).dt.year
    csv_df = csv_df[(csv_df['year'] >= 1950) & (csv_df['year'] <= 2000)]
    
    # Rebuild data_dict with filtered data
    years = list(range(1950, 2001))
    filtered_data_dict = {}
    
    for sid in site_ids:
        station_data = csv_df[csv_df['site_id'] == sid].sort_values('date')
        if len(station_data) > 0:
            X = station_data[EL_NINO_FEATURES].to_numpy(dtype=int)
            filtered_data_dict[sid] = X
        else:
            print(f"Warning: No data for station {sid} in 1950-2000")
    
    data_dict = filtered_data_dict
    
    # Train independent models for each station
    print("\n" + "="*80)
    print("Training Independent Mixture Models")
    print("="*80)
    
    predictions = {}
    
    for i, sid in enumerate(site_ids, 1):
        print(f"[{i}/{len(site_ids)}] Training {sid}...", end=" ")
        
        X = data_dict[sid]
        
        # Train independent model
        model = IndependentMixtureModel(
            n_states=2,
            n_categories=n_categories,
            max_iter=100,
            tol=1e-4,
            random_state=42
        )
        
        model.fit(X)
        pred = model.predict(X)
        predictions[sid] = pred
        
        print(f"Done (State 0: {np.sum(pred==0)}, State 1: {np.sum(pred==1)})")
    
    # Ensemble voting
    print("\n" + "="*80)
    print("Performing Ensemble Voting")
    print("="*80)
    
    results = []
    
    for t, year in enumerate(years):
        # Get ground truth
        ground_truth = enso_labels.get(year, 0)
        enso_type = enso_dict.get(year, 'Unknown')
        
        # Count anomaly votes across all stations
        anomaly_votes = sum(predictions[sid][t] for sid in site_ids)
        total_stations = len(site_ids)
        anomaly_ratio = anomaly_votes / total_stations
        
        # Ensemble predictions at different thresholds
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        ensemble_preds = {f"Ensemble_{int(th*100)}pct": int(anomaly_ratio >= th) for th in thresholds}
        matches = {f"Match_{int(th*100)}pct": int(ensemble_preds[f"Ensemble_{int(th*100)}pct"] == ground_truth) for th in thresholds}
        
        result = {
            'Year': year,
            'ENSO_Type': enso_type,
            'Ground_Truth': ground_truth,
            'Total_Stations': total_stations,
            'Anomaly_Votes': anomaly_votes,
            'Anomaly_Ratio': anomaly_ratio,
            **ensemble_preds,
            **matches
        }
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = '/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/comparison/independent_ensemble_voting_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("Performance Summary (50% threshold)")
    print("="*80)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_true = results_df['Ground_Truth'].values
    y_pred = results_df['Ensemble_50pct'].values
    
    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")


if __name__ == "__main__":
    main()

