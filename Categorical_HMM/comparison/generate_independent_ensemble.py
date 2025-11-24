"""
Generate Independent Classifier Ensemble Voting Results
Compares with HMM, GMM, and PELT models
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Categorical_HMM import FactorizedCategoricalHMM


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
                print(f"  Converged at iteration {iteration + 1}")
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


def load_processed_data(csv_path):
    """Load and prepare data for modeling"""
    df = pd.read_csv(csv_path)
    
    # Get unique stations
    stations = df['station_id'].unique()
    
    # Prepare data dictionary
    data_dict = {}
    labels_dict = {}
    
    for station in stations:
        station_data = df[df['station_id'] == station].sort_values('year')
        
        # Extract features (binned)
        feature_cols = [col for col in df.columns if col.endswith('_binned')]
        X = station_data[feature_cols].values
        
        # Extract labels
        y = station_data['enso_anomaly_moderate_plus'].values
        
        data_dict[station] = X
        labels_dict[station] = y
    
    return data_dict, labels_dict, stations


def train_independent_models(data_dict, labels_dict, stations):
    """Train independent mixture models for each station"""
    models = {}
    predictions = {}
    
    print("\n" + "="*80)
    print("Training Independent Mixture Models (No Temporal Dependencies)")
    print("="*80)
    
    for i, station in enumerate(stations, 1):
        print(f"\n[{i}/{len(stations)}] Training station: {station}")
        
        X = data_dict[station]
        y = labels_dict[station]
        
        # Determine number of categories for each feature
        n_categories = [int(X[:, f].max()) + 1 for f in range(X.shape[1])]
        
        # Train independent model
        model = IndependentMixtureModel(
            n_states=2,
            n_categories=n_categories,
            max_iter=100,
            tol=1e-4,
            random_state=42
        )
        
        model.fit(X)
        
        # Predict
        pred = model.predict(X)
        
        models[station] = model
        predictions[station] = pred
        
        print(f"  State distribution: {np.bincount(pred)}")
    
    return models, predictions


def ensemble_voting(predictions, labels_dict, stations, years):
    """Perform ensemble voting across all stations"""
    
    # Get ENSO type information
    enso_df = pd.read_csv('/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/data/enso_oni_data_1950_2010.csv')
    enso_dict = dict(zip(enso_df['year'], enso_df['enso_type']))
    
    results = []
    
    for t, year in enumerate(years):
        # Get ground truth (same for all stations at this year)
        ground_truth = labels_dict[stations[0]][t]
        enso_type = enso_dict.get(year, 'Unknown')
        
        # Count anomaly votes across all stations
        anomaly_votes = sum(predictions[station][t] for station in stations)
        total_stations = len(stations)
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
    
    return pd.DataFrame(results)


def main():
    """Main execution"""
    
    # Load data
    print("Loading processed data...")
    data_path = '/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/data/processed/weather_1901_2019_yearly_detrend_adaptive_bins10.csv'
    data_dict, labels_dict, stations = load_processed_data(data_path)
    
    # Filter to 1950-2000
    years = list(range(1950, 2001))
    for station in stations:
        data_dict[station] = data_dict[station][:51]  # 1950-2000
        labels_dict[station] = labels_dict[station][:51]
    
    print(f"\nLoaded data for {len(stations)} stations")
    print(f"Time period: 1950-2000 ({len(years)} years)")
    print(f"Features: {data_dict[stations[0]].shape[1]}")
    
    # Train models
    models, predictions = train_independent_models(data_dict, labels_dict, stations)
    
    # Ensemble voting
    print("\n" + "="*80)
    print("Performing Ensemble Voting")
    print("="*80)
    
    results_df = ensemble_voting(predictions, labels_dict, stations, years)
    
    # Save results
    output_path = '/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/comparison/independent_ensemble_voting_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Print summary statistics
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

