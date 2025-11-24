"""
Feature Ablation Study for ENSO Detection
==========================================

This script performs systematic feature ablation experiments to understand
the contribution of different feature groups to ENSO detection performance.

Feature Groups:
1. Temperature features (mean_temp, max_temp, min_temp)
2. Atmospheric features (sea_level_pressure, wind_speed)
3. Precipitation features (precipitation, visibility)
4. Weather events (fog, rain, snow, hail, thunder, tornado)

Experiment Design:
- Train HMM with all features (baseline)
- Train HMM with each feature group removed (ablation)
- Train HMM with only one feature group (isolation)
- Compare F1-scores and ensemble performance
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from Categorical_HMM import FactorizedCategoricalHMM, load_data

# Define feature groups
FEATURE_GROUPS = {
    'temperature': ['mean_temp', 'max_temp', 'min_temp'],
    'atmospheric': ['sea_level_pressure', 'wind_speed'],
    'precipitation': ['precipitation', 'visibility'],
    'weather_events': ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']
}

ALL_FEATURES = [
    'mean_temp', 'max_temp', 'min_temp',
    'sea_level_pressure', 'wind_speed',
    'precipitation', 'visibility',
    'fog', 'rain', 'snow', 'hail', 'thunder', 'tornado'
]

def train_and_evaluate_hmm(data_dict, n_categories, df_truth, feature_name="baseline"):
    """Train HMM on all stations and evaluate ensemble performance"""
    
    results = {}
    predictions = {}
    
    # Train HMM for each station
    for site_id, obs in data_dict.items():
        # Train with K=2 (fixed for fair comparison)
        model = FactorizedCategoricalHMM(
            n_components=2,
            n_features=obs.shape[1],
            n_categories=n_categories,
            n_iter=100,
            tol=1e-3,
            random_state=0
        )
        model.fit(obs)
        states = model.predict(obs)
        
        results[site_id] = {
            'model': model,
            'states': states
        }
        
        # Store predictions by year
        predictions[site_id] = states
    
    # Evaluate ensemble voting (50% threshold)
    years = range(1950, 2001)
    ensemble_predictions = []
    ground_truth = []
    
    for year_idx, year in enumerate(years):
        # Count anomaly votes
        votes = 0
        total = 0
        for site_id in data_dict.keys():
            if year_idx < len(predictions[site_id]):
                votes += predictions[site_id][year_idx]
                total += 1
        
        # Ensemble prediction (50% threshold)
        ensemble_pred = 1 if (votes / total) > 0.5 else 0
        ensemble_predictions.append(ensemble_pred)
        
        # Get ground truth
        truth_row = df_truth[df_truth['year'] == year]
        if len(truth_row) > 0:
            ground_truth.append(truth_row.iloc[0]['enso_anomaly'])
        else:
            ground_truth.append(0)
    
    # Calculate metrics
    f1 = f1_score(ground_truth, ensemble_predictions)
    acc = accuracy_score(ground_truth, ensemble_predictions)
    prec = precision_score(ground_truth, ensemble_predictions, zero_division=0)
    rec = recall_score(ground_truth, ensemble_predictions, zero_division=0)
    
    print(f"\n{feature_name}:")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    
    return {
        'name': feature_name,
        'f1': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'n_features': obs.shape[1]
    }

def main():
    print("="*80)
    print("Feature Ablation Study for ENSO Detection")
    print("="*80)
    
    # Load station list
    station_csv = "../data/stations_1950_2010_covered_top_each_country.csv"
    df_stations = pd.read_csv(station_csv)
    site_ids = (df_stations['USAF'].astype(str).str.zfill(6) + '-' + 
                df_stations['WBAN'].astype(str).str.zfill(5)).tolist()
    
    # Load ground truth
    df_truth = pd.read_csv('../enso_oni_data_1950_2010.csv')
    df_truth = df_truth[(df_truth['year'] >= 1950) & (df_truth['year'] <= 2000)]
    
    csv_path = "../data/processed/weather_1901_2019_yearly_detrend_adaptive_bins10.csv"
    
    results_list = []
    
    # Experiment 1: Baseline (all features)
    print("\n" + "="*80)
    print("Experiment 1: Baseline (All Features)")
    print("="*80)
    data_dict, _, _, n_categories = load_data(
        csv_path, site_ids=site_ids, feature_cols=ALL_FEATURES
    )
    result = train_and_evaluate_hmm(data_dict, n_categories, df_truth, "Baseline (All 13 features)")
    results_list.append(result)
    
    # Experiment 2: Remove each feature group (ablation)
    print("\n" + "="*80)
    print("Experiment 2: Remove Each Feature Group (Ablation)")
    print("="*80)
    
    for group_name, group_features in FEATURE_GROUPS.items():
        remaining_features = [f for f in ALL_FEATURES if f not in group_features]
        print(f"\nRemoving {group_name}: {group_features}")
        print(f"Remaining features: {len(remaining_features)}")
        
        data_dict, _, _, n_categories = load_data(
            csv_path, site_ids=site_ids, feature_cols=remaining_features
        )
        result = train_and_evaluate_hmm(
            data_dict, n_categories, df_truth, 
            f"Without {group_name} ({len(remaining_features)} features)"
        )
        results_list.append(result)
    
    # Experiment 3: Use only one feature group (isolation)
    print("\n" + "="*80)
    print("Experiment 3: Use Only One Feature Group (Isolation)")
    print("="*80)
    
    for group_name, group_features in FEATURE_GROUPS.items():
        print(f"\nUsing only {group_name}: {group_features}")
        
        data_dict, _, _, n_categories = load_data(
            csv_path, site_ids=site_ids, feature_cols=group_features
        )
        result = train_and_evaluate_hmm(
            data_dict, n_categories, df_truth,
            f"Only {group_name} ({len(group_features)} features)"
        )
        results_list.append(result)
    
    # Save results
    df_results = pd.DataFrame(results_list)
    df_results.to_csv('feature_ablation_results.csv', index=False)
    print("\n" + "="*80)
    print("Results saved to: feature_ablation_results.csv")
    print("="*80)
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: F1-Score comparison
    ax = axes[0, 0]
    colors = ['green'] + ['red']*4 + ['blue']*4
    bars = ax.barh(range(len(df_results)), df_results['f1'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_results)))
    ax.set_yticklabels(df_results['name'], fontsize=9)
    ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Ablation: F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.axvline(df_results.iloc[0]['f1'], color='green', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2: All metrics comparison
    ax = axes[0, 1]
    x = np.arange(len(df_results))
    width = 0.2
    ax.bar(x - 1.5*width, df_results['f1'], width, label='F1-Score', alpha=0.8)
    ax.bar(x - 0.5*width, df_results['accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x + 0.5*width, df_results['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + 1.5*width, df_results['recall'], width, label='Recall', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(range(len(df_results)), fontsize=10)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Feature count vs F1-Score
    ax = axes[1, 0]
    scatter_colors = ['green'] + ['red']*4 + ['blue']*4
    ax.scatter(df_results['n_features'], df_results['f1'], 
               c=scatter_colors, s=200, alpha=0.7, edgecolors='black')
    for idx, row in df_results.iterrows():
        ax.annotate(f"{idx}", (row['n_features'], row['f1']), 
                   fontsize=9, ha='center', va='center', fontweight='bold')
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Count vs Performance', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Plot 4: Performance drop when removing each group
    ax = axes[1, 1]
    ablation_results = df_results.iloc[1:5]  # Only ablation experiments
    baseline_f1 = df_results.iloc[0]['f1']
    performance_drop = baseline_f1 - ablation_results['f1']
    group_names = [name.split('(')[0].replace('Without ', '').strip() 
                   for name in ablation_results['name']]
    
    bars = ax.bar(range(len(performance_drop)), performance_drop, 
                  color='red', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(performance_drop)))
    ax.set_xticklabels(group_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('F1-Score Drop', fontsize=12, fontweight='bold')
    ax.set_title('Performance Impact of Removing Feature Groups', 
                 fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, performance_drop)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_ablation_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_ablation_analysis.png")
    
    # Summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"\nBaseline F1-Score: {baseline_f1:.4f}")
    print(f"\nMost important feature group (largest drop when removed):")
    max_drop_idx = performance_drop.idxmax()
    print(f"  {group_names[max_drop_idx - 1]}: -{performance_drop.iloc[max_drop_idx - 1]:.4f}")
    
    print(f"\nBest single feature group (isolation):")
    isolation_results = df_results.iloc[5:]
    best_isolation = isolation_results.loc[isolation_results['f1'].idxmax()]
    print(f"  {best_isolation['name']}: {best_isolation['f1']:.4f}")
    
    print("\n" + "="*80)
    print("Feature Ablation Study Complete!")
    print("="*80)

if __name__ == "__main__":
    main()

