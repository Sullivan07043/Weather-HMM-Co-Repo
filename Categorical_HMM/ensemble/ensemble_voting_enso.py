"""
Ensemble Voting for ENSO Detection
Uses majority voting across all station predictions to create a consensus ENSO forecast
Compares with ground truth and visualizes results
"""

import pandas as pd
import numpy as np

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    PLOTTING_AVAILABLE = True
    
    # Set plot style
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 100
except ImportError:
    PLOTTING_AVAILABLE = False
    # Define simple metric functions if sklearn not available
    def accuracy_score(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def precision_score(y_true, y_pred, zero_division=0):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if tp + fp == 0:
            return zero_division
        return tp / (tp + fp)
    
    def recall_score(y_true, y_pred, zero_division=0):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        if tp + fn == 0:
            return zero_division
        return tp / (tp + fn)
    
    def f1_score(y_true, y_pred, zero_division=0):
        prec = precision_score(y_true, y_pred, zero_division)
        rec = recall_score(y_true, y_pred, zero_division)
        if prec + rec == 0:
            return zero_division
        return 2 * prec * rec / (prec + rec)
    
    def confusion_matrix(y_true, y_pred):
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return np.array([[tn, fp], [fn, tp]])

# Read data
print("="*80)
print("Loading data...")
print("="*80)

# Load HMM predictions for all stations
df_states = pd.read_csv('../enso_factorized_categorical_hmm_states.csv')
print(f"Loaded HMM states: {len(df_states)} records from {df_states['site_id'].nunique()} stations")

# Filter to 1950-2000 time window
df_states = df_states[(df_states['year'] >= 1950) & (df_states['year'] <= 2000)]
print(f"Filtered to 1950-2000: {len(df_states)} records")

# Load F1 evaluation results for reference
df_f1 = pd.read_csv('../enso_evaluation_f1_results.csv')
df_f1_sorted = df_f1.sort_values('f1_score', ascending=False)

print(f"\nUsing ALL {df_states['site_id'].nunique()} stations for ensemble voting")
print(f"Station F1-scores (sorted):")
for i, (idx, row) in enumerate(df_f1_sorted.iterrows(), 1):
    print(f"  #{i:2d}. {row['site_id']} - {row['station_name']:30s} F1={row['f1_score']:.4f}")

# Use all stations for ensemble (no filtering)
print(f"\n✓ Using all {df_states['site_id'].nunique()} stations: {len(df_states)} records")

# Load ground truth ENSO data
df_truth = pd.read_csv('../enso_oni_data_1950_2010.csv')
df_truth = df_truth[(df_truth['year'] >= 1950) & (df_truth['year'] <= 2000)]
print(f"Loaded ground truth: {len(df_truth)} years (1950-2000)")

print(f"\nTotal stations in ensemble: {df_states['site_id'].nunique()}")
print(f"Time range: {df_states['year'].min()} to {df_states['year'].max()}")

# ============================================================================
# Majority Voting
# ============================================================================
print("\n" + "="*80)
print("Performing Majority Voting Ensemble...")
print("="*80)

# Group by year and count votes
voting_results = df_states.groupby('year').agg({
    'state': ['sum', 'count', 'mean']
}).reset_index()

voting_results.columns = ['year', 'anomaly_votes', 'total_stations', 'anomaly_ratio']

# Majority voting: try different thresholds
voting_results['ensemble_prediction_30'] = (voting_results['anomaly_ratio'] > 0.3).astype(int)
voting_results['ensemble_prediction_35'] = (voting_results['anomaly_ratio'] > 0.35).astype(int)
voting_results['ensemble_prediction_40'] = (voting_results['anomaly_ratio'] > 0.4).astype(int)
voting_results['ensemble_prediction_45'] = (voting_results['anomaly_ratio'] > 0.45).astype(int)
voting_results['ensemble_prediction_50'] = (voting_results['anomaly_ratio'] > 0.5).astype(int)
voting_results['ensemble_prediction_55'] = (voting_results['anomaly_ratio'] > 0.55).astype(int)
voting_results['ensemble_prediction_60'] = (voting_results['anomaly_ratio'] > 0.6).astype(int)
voting_results['ensemble_prediction'] = voting_results['ensemble_prediction_50']  # Use 50% as default

print(f"\nVoting statistics:")
print(f"  Years analyzed: {len(voting_results)}")
print(f"  Average stations per year: {voting_results['total_stations'].mean():.1f}")
print(f"  Average anomaly ratio: {voting_results['anomaly_ratio'].mean():.3f}")

# Merge with ground truth
df_comparison = voting_results.merge(
    df_truth[['year', 'enso_type', 'enso_anomaly']], 
    on='year',
    how='inner'
)

print(f"  Years with ground truth: {len(df_comparison)}")

# ============================================================================
# Performance Evaluation
# ============================================================================
print("\n" + "="*80)
print("Ensemble Performance Evaluation")
print("="*80)

y_true = df_comparison['enso_anomaly'].values
y_pred_30 = df_comparison['ensemble_prediction_30'].values
y_pred_35 = df_comparison['ensemble_prediction_35'].values
y_pred_40 = df_comparison['ensemble_prediction_40'].values
y_pred_45 = df_comparison['ensemble_prediction_45'].values
y_pred_50 = df_comparison['ensemble_prediction_50'].values
y_pred_55 = df_comparison['ensemble_prediction_55'].values
y_pred_60 = df_comparison['ensemble_prediction_60'].values

thresholds = {
    '30%': y_pred_30,
    '35%': y_pred_35,
    '40%': y_pred_40,
    '45%': y_pred_45,
    '50%': y_pred_50,
    '55%': y_pred_55,
    '60%': y_pred_60
}

results = []
for threshold_name, y_pred in thresholds.items():
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    results.append({
        'Threshold': threshold_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    print(f"\nThreshold: {threshold_name} (anomaly votes > {threshold_name})")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

df_results = pd.DataFrame(results)

# Find best threshold by F1-score
best_idx = df_results['F1-Score'].idxmax()
best_threshold = df_results.loc[best_idx, 'Threshold']
best_f1 = df_results.loc[best_idx, 'F1-Score']

print(f"\n{'='*80}")
print(f"Best Threshold: {best_threshold} (F1-Score: {best_f1:.4f})")
print(f"{'='*80}")

# Use 50% threshold as default for detailed analysis
y_pred = y_pred_50

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print(f"\nConfusion Matrix (50% threshold):")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# ============================================================================
# Visualization
# ============================================================================
if PLOTTING_AVAILABLE:
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)

    # Create a comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # ============================================================================
    # Plot 1: Time Series Comparison
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, :])

    years = df_comparison['year'].values
    y_true_plot = y_true * 0.95  # Offset for visibility
    y_pred_plot = y_pred * 1.05

    ax1.fill_between(years, 0, y_true_plot, where=(y_true==1), 
                     alpha=0.3, color='red', label='Ground Truth: ENSO Anomaly', step='mid')
    ax1.fill_between(years, 0, y_pred_plot, where=(y_pred==1), 
                     alpha=0.3, color='blue', label='Ensemble Prediction: Anomaly', step='mid')

    # Mark ENSO types
    for idx, row in df_comparison.iterrows():
        if row['enso_anomaly'] == 1:
            color = 'darkred' if row['enso_type'] == 'El_Nino' else 'darkblue'
            marker = '^' if row['enso_type'] == 'El_Nino' else 'v'
            ax1.scatter(row['year'], 0.9, color=color, marker=marker, 
                       s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ENSO Anomaly (0=Normal, 1=Anomaly)', fontsize=12, fontweight='bold')
    ax1.set_title('Ensemble ENSO Prediction vs Ground Truth (1950-2000)\nAll Stations (21 sites)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(-0.1, 1.2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', framealpha=0.9)

    # Add custom legend for ENSO types
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='darkred', 
               markersize=10, label='El Niño Year', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='darkblue', 
               markersize=10, label='La Niña Year', markeredgecolor='black', markeredgewidth=0.5),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    # ============================================================================
    # Plot 2: Anomaly Voting Ratio Time Series
    # ============================================================================
    ax2 = fig.add_subplot(gs[1, :])

    ax2.plot(years, df_comparison['anomaly_ratio'], 
             color='purple', linewidth=2, label='Station Anomaly Ratio', alpha=0.7)
    ax2.axhline(y=0.3, color='green', linestyle=':', linewidth=1.5, 
                label='30% Threshold', alpha=0.5)
    ax2.axhline(y=0.4, color='blue', linestyle=':', linewidth=1.5, 
                label='40% Threshold', alpha=0.5)
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, 
                label='50% Threshold (Default)', alpha=0.7)
    ax2.axhline(y=0.6, color='red', linestyle=':', linewidth=1.5, 
                label='60% Threshold', alpha=0.5)

    # Shade ground truth anomaly periods
    for idx, row in df_comparison.iterrows():
        if row['enso_anomaly'] == 1:
            ax2.axvspan(row['year']-0.5, row['year']+0.5, 
                       alpha=0.1, color='red')

    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fraction of Stations Predicting Anomaly', fontsize=12, fontweight='bold')
    ax2.set_title('Station Voting Ratio Over Time', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', framealpha=0.9)

    # ============================================================================
    # Plot 3: Confusion Matrix Heatmap
    # ============================================================================
    ax3 = fig.add_subplot(gs[2, 0])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                xticklabels=['Predicted: Normal', 'Predicted: Anomaly'],
                yticklabels=['Actual: Normal', 'Actual: Anomaly'],
                ax=ax3, annot_kws={'size': 14, 'weight': 'bold'})

    ax3.set_title('Confusion Matrix (35% Threshold)', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=11, fontweight='bold')

    # ============================================================================
    # Plot 4: Performance Metrics Comparison
    # ============================================================================
    ax4 = fig.add_subplot(gs[2, 1])

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.15

    for i, threshold_name in enumerate(['30%', '35%', '40%', '45%', '50%', '60%']):
        values = df_results[df_results['Threshold'] == threshold_name][metrics].values[0]
        offset = (i - 2.5) * width
        bars = ax4.bar(x + offset, values, width, label=f'Threshold {threshold_name}', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax4.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax4.set_title('Performance Metrics by Voting Threshold', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.set_ylim(0, 1.1)
    ax4.legend(loc='lower right', framealpha=0.9)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.suptitle('Ensemble ENSO Detection: Majority Voting Across Top 14 Stations', 
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('ensemble_voting_enso_analysis.png', dpi=300, bbox_inches='tight')
    print("  Saved: ensemble_voting_enso_analysis.png")

    # ============================================================================
    # Additional Visualization: Detailed Comparison Table
    # ============================================================================
    fig2, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    table_data.append(['Year', 'ENSO Type', 'Ground Truth', 'Stations Voting\nAnomaly (%)', 
                      'Ensemble\nPrediction', 'Match'])

    for idx, row in df_comparison.iterrows():
        year = int(row['year'])
        enso_type = row['enso_type'].replace('_', ' ')
        truth = 'Anomaly' if row['enso_anomaly'] == 1 else 'Normal'
        vote_pct = f"{row['anomaly_ratio']*100:.1f}%"
        prediction = 'Anomaly' if row['ensemble_prediction'] == 1 else 'Normal'
        match = '✓' if row['enso_anomaly'] == row['ensemble_prediction'] else '✗'
        
        table_data.append([year, enso_type, truth, vote_pct, prediction, match])

    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.12, 0.18, 0.18, 0.20, 0.18, 0.14])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=10)

    # Style data rows with alternating colors
    for i in range(1, len(table_data)):
        for j in range(6):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')
            
            # Highlight mismatches in red
            if j == 5 and table_data[i][5] == '✗':
                cell.set_text_props(color='red', weight='bold', fontsize=11)
            # Highlight matches in green
            elif j == 5 and table_data[i][5] == '✓':
                cell.set_text_props(color='green', weight='bold', fontsize=11)

    plt.title('Year-by-Year Ensemble Prediction vs Ground Truth (1950-2000)\nAll Stations (21 sites)', 
              fontsize=14, fontweight='bold', pad=20)

    plt.savefig('ensemble_voting_detailed_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: ensemble_voting_detailed_comparison.png")
else:
    print("\n" + "="*80)
    print("Matplotlib not available. Skipping visualizations.")
    print("You can install it with: pip install matplotlib seaborn scikit-learn")
    print("="*80)

# ============================================================================
# Save results to CSV
# ============================================================================
output_df = df_comparison[[
    'year', 'enso_type', 'enso_anomaly', 
    'total_stations', 'anomaly_votes', 'anomaly_ratio',
    'ensemble_prediction_30', 'ensemble_prediction_35', 'ensemble_prediction_40', 
    'ensemble_prediction_45', 'ensemble_prediction_50', 'ensemble_prediction_55', 'ensemble_prediction_60'
]].copy()

output_df.columns = [
    'Year', 'ENSO_Type', 'Ground_Truth', 
    'Total_Stations', 'Anomaly_Votes', 'Anomaly_Ratio',
    'Ensemble_30pct', 'Ensemble_35pct', 'Ensemble_40pct', 'Ensemble_45pct', 'Ensemble_50pct', 'Ensemble_55pct', 'Ensemble_60pct'
]

# Add match indicator
output_df['Match_30pct'] = (output_df['Ground_Truth'] == output_df['Ensemble_30pct']).astype(int)
output_df['Match_35pct'] = (output_df['Ground_Truth'] == output_df['Ensemble_35pct']).astype(int)
output_df['Match_40pct'] = (output_df['Ground_Truth'] == output_df['Ensemble_40pct']).astype(int)
output_df['Match_45pct'] = (output_df['Ground_Truth'] == output_df['Ensemble_45pct']).astype(int)
output_df['Match_50pct'] = (output_df['Ground_Truth'] == output_df['Ensemble_50pct']).astype(int)
output_df['Match_55pct'] = (output_df['Ground_Truth'] == output_df['Ensemble_55pct']).astype(int)
output_df['Match_60pct'] = (output_df['Ground_Truth'] == output_df['Ensemble_60pct']).astype(int)

output_df.to_csv('ensemble_voting_results.csv', index=False)
print("  Saved: ensemble_voting_results.csv")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*80)
print("ENSEMBLE VOTING SUMMARY")
print("="*80)

print(f"\nData Overview:")
print(f"  Analysis Period: 1950-2000 ({len(df_comparison)} years)")
print(f"  Ensemble Stations: All Stations")
print(f"  Total Stations: {df_comparison['total_stations'].iloc[0]}")
print(f"  Ground Truth Anomalies: {y_true.sum()}/{len(y_true)} years ({y_true.sum()/len(y_true)*100:.1f}%)")

print(f"\nBest Performance ({best_threshold} Threshold):")
print(f"  Accuracy:  {df_results.loc[best_idx, 'Accuracy']:.4f}")
print(f"  Precision: {df_results.loc[best_idx, 'Precision']:.4f}")
print(f"  Recall:    {df_results.loc[best_idx, 'Recall']:.4f}")
print(f"  F1-Score:  {df_results.loc[best_idx, 'F1-Score']:.4f}")

print(f"\nMisclassified Years ({best_threshold} threshold):")
# Get predictions for best threshold
# Map threshold like "30%" to column "ensemble_prediction_30"
threshold_num = best_threshold.replace('%', '')
best_threshold_col = f"ensemble_prediction_{threshold_num}"
misclassified = df_comparison[df_comparison['enso_anomaly'] != df_comparison[best_threshold_col]]
if len(misclassified) > 0:
    for idx, row in misclassified.iterrows():
        pred_val = row[best_threshold_col]
        print(f"  {int(row['year'])}: True={row['enso_type']:10s}, "
              f"Predicted={'Anomaly' if pred_val==1 else 'Normal':7s}, "
              f"Vote={row['anomaly_ratio']*100:5.1f}%")
else:
    print("  None! Perfect classification!")

print("\n" + "="*80)
print("Ensemble analysis complete!")
print("="*80)

