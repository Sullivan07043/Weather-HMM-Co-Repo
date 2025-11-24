"""
Temporal Dependency Ablation Study (Revised)
=============================================

Compare models with and without temporal dependencies:
1. Full HMM (with learned transition matrix and temporal modeling)
2. Independent Mixture Model (GMM-like, no temporal dependencies)

Key Difference:
- HMM: Uses forward-backward algorithm, considers temporal sequence
- Independent: Treats each time point independently, no sequence information

This study answers:
- Do temporal dependencies improve ENSO detection?
- Is the Markov assumption valuable for year-to-year prediction?
- How much does sequence modeling contribute?
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from Categorical_HMM import FactorizedCategoricalHMM, load_data, EL_NINO_FEATURES

class IndependentMixtureModel:
    """
    Independent Mixture Model - treats each observation independently
    Uses a mixture of categorical distributions (like GMM but for discrete data)
    NO temporal dependencies - each time point classified independently
    """
    
    def __init__(self, n_components=2, n_features=13, n_categories=10, 
                 n_iter=100, tol=1e-3, random_state=0):
        self.n_components = n_components
        self.n_features = n_features
        self.n_categories = n_categories
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize parameters
        self.pi_ = np.ones(n_components) / n_components  # Mixture weights
        # Emission probabilities: [K, D, M]
        self.B_ = np.random.dirichlet(np.ones(n_categories), 
                                      size=(n_components, n_features))
    
    def _compute_log_emission(self, X):
        """Compute log emission probabilities for all observations"""
        T = X.shape[0]
        K = self.n_components
        
        log_B_all = np.zeros((T, K))
        
        for t in range(T):
            for k in range(K):
                log_prob = 0.0
                for d in range(self.n_features):
                    x_td = X[t, d]
                    if 0 <= x_td < self.n_categories:
                        log_prob += np.log(self.B_[k, d, x_td] + 1e-300)
                log_B_all[t, k] = log_prob
        
        return log_B_all
    
    def fit(self, X):
        """
        Fit independent mixture model using EM algorithm
        Key difference from HMM: NO forward-backward, each observation independent
        """
        X = np.asarray(X, dtype=int)
        T = X.shape[0]
        K = self.n_components
        
        log_B_all = self._compute_log_emission(X)
        prev_loglik = -np.inf
        
        for iteration in range(self.n_iter):
            # E-step: Compute responsibilities (independent for each time point)
            # gamma[t, k] = P(z_t = k | x_t) - NO temporal smoothing
            log_gamma = np.zeros((T, K))
            
            for t in range(T):
                log_numerator = np.log(self.pi_ + 1e-300) + log_B_all[t, :]
                log_denominator = self._log_sum_exp(log_numerator)
                log_gamma[t, :] = log_numerator - log_denominator
            
            gamma = np.exp(log_gamma)
            
            # M-step: Update parameters
            # Update mixture weights
            self.pi_ = np.sum(gamma, axis=0) / T
            
            # Update emission probabilities
            for k in range(K):
                for d in range(self.n_features):
                    numerator = np.zeros(self.n_categories)
                    denominator = 0.0
                    
                    for t in range(T):
                        x_td = X[t, d]
                        if 0 <= x_td < self.n_categories:
                            numerator[x_td] += gamma[t, k]
                            denominator += gamma[t, k]
                    
                    if denominator > 0:
                        self.B_[k, d, :] = (numerator + 1e-10) / (denominator + 1e-10 * self.n_categories)
                    else:
                        self.B_[k, d, :] = np.ones(self.n_categories) / self.n_categories
            
            # Recompute log emission
            log_B_all = self._compute_log_emission(X)
            
            # Compute log-likelihood
            loglik = 0.0
            for t in range(T):
                log_prob_t = self._log_sum_exp(np.log(self.pi_ + 1e-300) + log_B_all[t, :])
                loglik += log_prob_t
            
            # Check convergence
            if iteration > 0 and abs(loglik - prev_loglik) < self.tol:
                if iteration % 5 == 0 or iteration < 10:
                    print(f"      iter {iteration:3d}: loglik = {loglik:.2f}")
                print(f"      EM converged at iter {iteration}, loglik = {loglik:.2f}")
                break
            
            if iteration % 5 == 0:
                print(f"      iter {iteration:3d}: loglik = {loglik:.2f}")
            
            prev_loglik = loglik
    
    def predict(self, X):
        """
        Predict states independently for each time point
        NO Viterbi, NO forward-backward - pure independent classification
        """
        X = np.asarray(X, dtype=int)
        T = X.shape[0]
        
        log_B_all = self._compute_log_emission(X)
        states = np.zeros(T, dtype=int)
        
        for t in range(T):
            # Classify each time point independently
            log_prob = np.log(self.pi_ + 1e-300) + log_B_all[t, :]
            states[t] = np.argmax(log_prob)
        
        return states
    
    def _log_sum_exp(self, log_probs):
        """Numerically stable log-sum-exp"""
        max_log = np.max(log_probs)
        return max_log + np.log(np.sum(np.exp(log_probs - max_log)))


def train_and_evaluate(ModelClass, data_dict, n_categories, df_truth, years, model_name):
    """Train model and evaluate performance"""
    
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")
    
    predictions_dict = {}
    station_f1_scores = []
    
    # Train for each station
    for site_id, obs in data_dict.items():
        model = ModelClass(
            n_components=2,
            n_features=obs.shape[1],
            n_categories=n_categories,
            n_iter=100,
            tol=1e-3,
            random_state=0
        )
        model.fit(obs)
        states = model.predict(obs)
        
        # Evaluate station
        y_true = []
        y_pred = []
        
        for year_idx, year in enumerate(years):
            if year_idx < len(states):
                truth_row = df_truth[df_truth['year'] == year]
                if len(truth_row) > 0:
                    y_true.append(truth_row.iloc[0]['enso_anomaly'])
                    y_pred.append(states[year_idx])
        
        if len(y_true) > 0:
            # Determine state mapping
            state_0_anomaly = sum([1 for i in range(len(y_true)) if y_pred[i] == 0 and y_true[i] == 1])
            state_1_anomaly = sum([1 for i in range(len(y_true)) if y_pred[i] == 1 and y_true[i] == 1])
            
            if state_0_anomaly > state_1_anomaly:
                y_pred = [1 - p for p in y_pred]
            
            f1 = f1_score(y_true, y_pred, zero_division=0)
            station_f1_scores.append(f1)
            predictions_dict[site_id] = y_pred
        
        print(f"  ✓ {site_id}: F1={f1:.4f}")
    
    # Ensemble voting
    ensemble_predictions = []
    ground_truth = []
    
    for year_idx, year in enumerate(years):
        votes = 0
        total = 0
        
        for site_id in predictions_dict.keys():
            if year_idx < len(predictions_dict[site_id]):
                votes += predictions_dict[site_id][year_idx]
                total += 1
        
        ensemble_pred = 1 if (votes / total) > 0.5 else 0
        ensemble_predictions.append(ensemble_pred)
        
        truth_row = df_truth[df_truth['year'] == year]
        if len(truth_row) > 0:
            ground_truth.append(truth_row.iloc[0]['enso_anomaly'])
        else:
            ground_truth.append(0)
    
    # Calculate metrics
    ensemble_f1 = f1_score(ground_truth, ensemble_predictions)
    ensemble_acc = accuracy_score(ground_truth, ensemble_predictions)
    ensemble_prec = precision_score(ground_truth, ensemble_predictions, zero_division=0)
    ensemble_rec = recall_score(ground_truth, ensemble_predictions, zero_division=0)
    
    avg_station_f1 = np.mean(station_f1_scores)
    
    print(f"\nResults for {model_name}:")
    print(f"  Average Station F1: {avg_station_f1:.4f}")
    print(f"  Ensemble F1:        {ensemble_f1:.4f}")
    print(f"  Ensemble Accuracy:  {ensemble_acc:.4f}")
    print(f"  Ensemble Precision: {ensemble_prec:.4f}")
    print(f"  Ensemble Recall:    {ensemble_rec:.4f}")
    
    return {
        'model': model_name,
        'avg_station_f1': avg_station_f1,
        'ensemble_f1': ensemble_f1,
        'ensemble_accuracy': ensemble_acc,
        'ensemble_precision': ensemble_prec,
        'ensemble_recall': ensemble_rec,
        'station_f1_scores': station_f1_scores
    }


def analyze_transition_matrices(data_dict, n_categories):
    """Analyze learned transition matrices from HMM"""
    
    print(f"\n{'='*80}")
    print("Analyzing Transition Matrices (HMM only)")
    print(f"{'='*80}")
    
    all_transitions = []
    
    for site_id, obs in data_dict.items():
        model = FactorizedCategoricalHMM(
            n_components=2,
            n_features=obs.shape[1],
            n_categories=n_categories,
            n_iter=100,
            tol=1e-3,
            random_state=0
        )
        model.fit(obs)
        all_transitions.append(model.A_)
    
    avg_transition = np.mean(all_transitions, axis=0)
    std_transition = np.std(all_transitions, axis=0)
    
    print("\nAverage Transition Matrix:")
    print(avg_transition)
    print("\nStandard Deviation:")
    print(std_transition)
    
    # Calculate persistence (diagonal values)
    persistence_0 = avg_transition[0, 0]
    persistence_1 = avg_transition[1, 1]
    
    print(f"\nState Persistence:")
    print(f"  State 0 -> State 0: {persistence_0:.4f} (±{std_transition[0,0]:.4f})")
    print(f"  State 1 -> State 1: {persistence_1:.4f} (±{std_transition[1,1]:.4f})")
    
    print(f"\nTransition Probabilities:")
    print(f"  State 0 -> State 1: {avg_transition[0,1]:.4f} (±{std_transition[0,1]:.4f})")
    print(f"  State 1 -> State 0: {avg_transition[1,0]:.4f} (±{std_transition[1,0]:.4f})")
    
    return avg_transition, std_transition, all_transitions


def main():
    print("="*80)
    print("Temporal Dependency Ablation Study (Revised)")
    print("="*80)
    
    # Load data
    station_csv = "../data/stations_1950_2010_covered_top_each_country.csv"
    df_stations = pd.read_csv(station_csv)
    site_ids = (df_stations['USAF'].astype(str).str.zfill(6) + '-' + 
                df_stations['WBAN'].astype(str).str.zfill(5)).tolist()
    
    csv_path = "../data/processed/weather_1901_2019_yearly_detrend_adaptive_bins10.csv"
    data_dict, _, _, n_categories = load_data(
        csv_path, site_ids=site_ids, feature_cols=EL_NINO_FEATURES
    )
    # Get scalar value for Independent Model (use first value, they should all be 10)
    n_cat_scalar = n_categories[0] if isinstance(n_categories, (list, tuple)) else n_categories
    
    df_truth = pd.read_csv('../enso_oni_data_1950_2010.csv')
    df_truth = df_truth[(df_truth['year'] >= 1950) & (df_truth['year'] <= 2000)]
    years = list(range(1950, 2001))
    
    # Experiment 1: Full HMM (with temporal dependencies)
    result_hmm = train_and_evaluate(
        FactorizedCategoricalHMM, data_dict, n_categories, df_truth, years,
        "Full HMM (with temporal dependencies)"
    )
    
    # Experiment 2: Independent Mixture Model (no temporal dependencies)
    result_independent = train_and_evaluate(
        IndependentMixtureModel, data_dict, n_cat_scalar, df_truth, years,
        "Independent Mixture Model (no temporal dependencies)"
    )
    
    # Analyze transition matrices
    avg_trans, std_trans, all_trans = analyze_transition_matrices(data_dict, n_categories)
    
    # Save results
    results = [result_hmm, result_independent]
    df_results = pd.DataFrame(results)
    df_results.to_csv('temporal_ablation_results.csv', index=False)
    
    print("\n" + "="*80)
    print("Results saved to: temporal_ablation_results.csv")
    print("="*80)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Ensemble F1 comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = [r['model'].split('(')[0].strip() for r in results]
    f1_scores = [r['ensemble_f1'] for r in results]
    colors = ['green', 'red']
    
    bars = ax1.bar(range(len(models)), f1_scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Ensemble F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('Temporal Dependencies Impact on Performance', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: All metrics comparison
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(results))
    width = 0.2
    
    ax2.bar(x - 1.5*width, [r['ensemble_f1'] for r in results], width, 
           label='F1-Score', alpha=0.8)
    ax2.bar(x - 0.5*width, [r['ensemble_accuracy'] for r in results], width, 
           label='Accuracy', alpha=0.8)
    ax2.bar(x + 0.5*width, [r['ensemble_precision'] for r in results], width, 
           label='Precision', alpha=0.8)
    ax2.bar(x + 1.5*width, [r['ensemble_recall'] for r in results], width, 
           label='Recall', alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['HMM', 'Independent'], fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('All Metrics Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Station-level F1 distribution
    ax3 = fig.add_subplot(gs[0, 2])
    
    hmm_f1 = result_hmm['station_f1_scores']
    ind_f1 = result_independent['station_f1_scores']
    
    ax3.hist(hmm_f1, bins=15, alpha=0.6, label='Full HMM', color='green', edgecolor='black')
    ax3.hist(ind_f1, bins=15, alpha=0.6, label='Independent', color='red', edgecolor='black')
    ax3.set_xlabel('Station F1-Score', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Stations', fontsize=12, fontweight='bold')
    ax3.set_title('Station-Level F1 Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: HMM vs Independent scatter
    ax4 = fig.add_subplot(gs[1, 0])
    
    ax4.scatter(ind_f1, hmm_f1, s=100, alpha=0.6, edgecolors='black', color='purple')
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Equal Performance')
    ax4.set_xlabel('Independent Model F1', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Full HMM F1', fontsize=12, fontweight='bold')
    ax4.set_title('HMM vs Independent: Station Comparison', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    better_hmm = np.sum(np.array(hmm_f1) > np.array(ind_f1))
    better_ind = np.sum(np.array(ind_f1) > np.array(hmm_f1))
    
    ax4.text(0.05, 0.95, f'HMM better: {better_hmm}\nIndependent better: {better_ind}',
            transform=ax4.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 5: Average transition matrix heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    
    sns.heatmap(avg_trans, annot=True, fmt='.4f', cmap='YlOrRd', 
                xticklabels=['State 0', 'State 1'],
                yticklabels=['State 0', 'State 1'],
                ax=ax5, cbar_kws={'label': 'Probability'})
    ax5.set_title('Average Transition Matrix\n(Learned from Data)', 
                  fontsize=13, fontweight='bold')
    ax5.set_xlabel('To State', fontsize=11, fontweight='bold')
    ax5.set_ylabel('From State', fontsize=11, fontweight='bold')
    
    # Plot 6: Transition matrix variability
    ax6 = fig.add_subplot(gs[1, 2])
    
    persistence_values = [trans[0, 0] for trans in all_trans] + [trans[1, 1] for trans in all_trans]
    transition_values = [trans[0, 1] for trans in all_trans] + [trans[1, 0] for trans in all_trans]
    
    ax6.boxplot([persistence_values, transition_values], 
                labels=['Persistence\n(stay in state)', 'Transition\n(change state)'])
    ax6.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax6.set_title('Transition Probability Distribution\nAcross All Stations', 
                  fontsize=13, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    plt.savefig('temporal_ablation_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: temporal_ablation_analysis.png")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    improvement = result_hmm['ensemble_f1'] - result_independent['ensemble_f1']
    print(f"\nTemporal Dependencies Impact:")
    print(f"  Full HMM F1:        {result_hmm['ensemble_f1']:.4f}")
    print(f"  Independent F1:     {result_independent['ensemble_f1']:.4f}")
    print(f"  Improvement:        {improvement:.4f} ({improvement/result_independent['ensemble_f1']*100:.2f}%)")
    
    print(f"\nStation-level Analysis:")
    print(f"  Stations where HMM is better: {better_hmm}/{len(hmm_f1)}")
    print(f"  Stations where Independent is better: {better_ind}/{len(ind_f1)}")
    
    print(f"\nTransition Matrix Analysis:")
    print(f"  Average persistence: {(avg_trans[0,0] + avg_trans[1,1])/2:.4f}")
    print(f"  Average transition:  {(avg_trans[0,1] + avg_trans[1,0])/2:.4f}")
    
    if improvement > 0.01:
        print(f"\n✓ Temporal dependencies provide meaningful improvement!")
        print(f"  HMM's forward-backward algorithm leverages sequence information")
        print(f"  Transition matrix captures year-to-year ENSO dynamics")
    elif improvement < -0.01:
        print(f"\n✗ Temporal dependencies hurt performance!")
    else:
        print(f"\n≈ Temporal dependencies have minimal impact.")
    
    print("\n" + "="*80)
    print("Temporal Ablation Study Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

