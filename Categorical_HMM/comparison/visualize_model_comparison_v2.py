"""
Visualize Model Comparison: HMM vs GMM vs PELT vs Independent Classifier
Create multiple clean figures with better layout
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def load_ensemble_results():
    """Load ensemble voting results from all models (1950-2000)"""
    
    models = {
        'HMM': '/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/comparison/ensemble_voting_results.csv',
        'GMM': '/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/comparison/gmm_ensemble_voting_results.csv',
        'PELT': '/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/comparison/PELT_enso_ensemble_results.csv',
        'Independent': '/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/comparison/independent_ensemble_voting_results.csv'
    }
    
    data = {}
    for name, path in models.items():
        try:
            df = pd.read_csv(path)
            # Filter to 1950-2000
            df = df[(df['Year'] >= 1950) & (df['Year'] <= 2000)]
            data[name] = df
            print(f"✓ Loaded {name}: {len(df)} years (1950-2000)")
        except FileNotFoundError:
            print(f"✗ {name} results not found at {path}")
    
    return data


def compute_metrics(df, threshold='50pct'):
    """Compute performance metrics for a given threshold"""
    
    y_true = df['Ground_Truth'].values
    y_pred = df[f'Ensemble_{threshold}'].values
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['tn'] = cm[0, 0]
    metrics['fp'] = cm[0, 1]
    metrics['fn'] = cm[1, 0]
    metrics['tp'] = cm[1, 1]
    
    return metrics


def plot_figure1_performance_metrics(data, colors):
    """Figure 1: Performance Metrics Comparison"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Bar chart of metrics
    metrics_data = []
    for model_name, df in data.items():
        m = compute_metrics(df, '50pct')
        metrics_data.append({
            'Model': model_name,
            'F1-Score': m['f1'],
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    x = np.arange(len(metrics_df))
    width = 0.2
    
    metrics_list = ['F1-Score', 'Accuracy', 'Precision', 'Recall']
    metric_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (metric, color) in enumerate(zip(metrics_list, metric_colors)):
        bars = ax1.bar(x + i*width, metrics_df[metric], width, label=metric, 
                      alpha=0.85, color=color, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, metrics_df[metric])):
            if val > 0.05:  # Only show label if value is significant
                ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax1.set_title('Performance Metrics Comparison\n(50% Ensemble Threshold)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(metrics_df['Model'], fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='black')
    
    # Subplot 2: Confusion Matrix Comparison
    model_names = list(data.keys())
    cm_comparison = []
    
    for model_name in model_names:
        m = compute_metrics(data[model_name], '50pct')
        cm_comparison.append([m['tn'], m['fp'], m['fn'], m['tp']])
    
    cm_df = pd.DataFrame(cm_comparison, 
                         columns=['TN', 'FP', 'FN', 'TP'],
                         index=model_names)
    
    # Create grouped bar chart for confusion matrix
    x2 = np.arange(len(model_names))
    width2 = 0.2
    
    cm_colors = ['#4CAF50', '#FF5252', '#FFC107', '#2196F3']
    cm_labels = ['TN', 'FP', 'FN', 'TP']
    
    for i, (label, color) in enumerate(zip(cm_labels, cm_colors)):
        bars = ax2.bar(x2 + i*width2, cm_df[label], width2, label=label,
                      alpha=0.85, color=color, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar, val in zip(bars, cm_df[label]):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                        f'{int(val)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax2.set_title('Confusion Matrix Comparison\n(50% Ensemble Threshold)',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x2 + width2 * 1.5)
    ax2.set_xticklabels(model_names, fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='black',
              title='Confusion Matrix', title_fontsize=11)
    
    plt.tight_layout()
    output_path = '/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/comparison/figure1_performance_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: figure1_performance_metrics.png")
    plt.close()


def plot_figure2_threshold_analysis(data, colors):
    """Figure 2: F1-Score vs Threshold (single plot)"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # F1-Score vs Threshold
    thresholds = ['30pct', '35pct', '40pct', '45pct', '50pct', '55pct', '60pct']
    threshold_values = [30, 35, 40, 45, 50, 55, 60]
    
    for model_name, df in data.items():
        f1_scores = []
        for th in thresholds:
            if f'Ensemble_{th}' in df.columns:
                m = compute_metrics(df, th)
                f1_scores.append(m['f1'])
            else:
                f1_scores.append(np.nan)
        
        ax.plot(threshold_values, f1_scores, marker='o', linewidth=3.5, 
                label=model_name, color=colors[model_name], markersize=12,
                markeredgecolor='black', markeredgewidth=2)
    
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2.5, alpha=0.6, label='50% Threshold')
    ax.set_xlabel('Ensemble Voting Threshold (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax.set_title('F1-Score vs Voting Threshold', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    
    plt.tight_layout()
    output_path = '/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/comparison/figure2_threshold_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: figure2_threshold_analysis.png")
    plt.close()


def plot_figure3_temporal_analysis(data, colors):
    """Figure 3: Year-by-Year Detection (split into two periods, shared legend)"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    
    # Get years and ground truth
    first_model = list(data.values())[0]
    years = first_model['Year'].values
    ground_truth = first_model['Ground_Truth'].values
    
    # Split into two periods
    mid_year = 1975
    period1_mask = years <= mid_year
    period2_mask = years > mid_year
    
    # Period 1: 1950-1975
    years_p1 = years[period1_mask]
    gt_p1 = ground_truth[period1_mask]
    
    # Plot ground truth background
    anomaly_years_p1 = years_p1[gt_p1 == 1]
    for year in anomaly_years_p1:
        ax1.axvspan(year-0.4, year+0.4, alpha=0.2, color='red', zorder=0)
    
    # Plot predictions (use 30% threshold for GMM and PELT, 50% for others)
    offset = 0
    handles = []
    labels = []
    for model_name, df in data.items():
        df_p1 = df[df['Year'] <= mid_year]
        
        # Use 30% threshold for GMM and PELT to show more predictions
        if model_name in ['GMM', 'PELT']:
            pred_p1 = df_p1['Ensemble_30pct'].values
        else:
            pred_p1 = df_p1['Ensemble_50pct'].values
            
        years_pred_p1 = df_p1['Year'].values
        
        anomaly_mask = pred_p1 == 1
        scatter = ax1.scatter(years_pred_p1[anomaly_mask], 
                   np.ones(sum(anomaly_mask)) + offset,
                   marker='o', s=150,
                   color=colors[model_name], alpha=0.9, 
                   edgecolors='black', linewidth=2, zorder=5)
        handles.append(scatter)
        labels.append(model_name)
        offset += 0.25
    
    ax1.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Model Predictions', fontsize=13, fontweight='bold')
    ax1.set_title('Year-by-Year ENSO Anomaly Detection: 1950-1975',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_yticks([])
    ax1.set_xlim(years_p1[0]-1, years_p1[-1]+1)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Period 2: 1976-2000
    years_p2 = years[period2_mask]
    gt_p2 = ground_truth[period2_mask]
    
    # Plot ground truth background
    anomaly_years_p2 = years_p2[gt_p2 == 1]
    for year in anomaly_years_p2:
        ax2.axvspan(year-0.4, year+0.4, alpha=0.2, color='red', zorder=0)
    
    # Plot predictions (use 30% threshold for GMM and PELT, 50% for others)
    offset = 0
    for model_name, df in data.items():
        df_p2 = df[df['Year'] > mid_year]
        
        # Use 30% threshold for GMM and PELT to show more predictions
        if model_name in ['GMM', 'PELT']:
            pred_p2 = df_p2['Ensemble_30pct'].values
        else:
            pred_p2 = df_p2['Ensemble_50pct'].values
            
        years_pred_p2 = df_p2['Year'].values
        
        anomaly_mask = pred_p2 == 1
        ax2.scatter(years_pred_p2[anomaly_mask],
                   np.ones(sum(anomaly_mask)) + offset,
                   marker='o', s=150,
                   color=colors[model_name], alpha=0.9,
                   edgecolors='black', linewidth=2, zorder=5)
        offset += 0.25
    
    ax2.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Model Predictions', fontsize=13, fontweight='bold')
    ax2.set_title('Year-by-Year ENSO Anomaly Detection: 1976-2000',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_yticks([])
    ax2.set_xlim(years_p2[0]-1, years_p2[-1]+1)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add shared legend at the bottom
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12, 
              framealpha=0.95, edgecolor='black', fancybox=True, shadow=True,
              bbox_to_anchor=(0.5, 0.02))
    
    # Add note about red background at the very bottom, below legend
    fig.text(0.5, -0.01, 'Red background = True ENSO Anomaly Years | Dots = Model Predictions (HMM & Independent: 50% threshold, GMM & PELT: 30% threshold)',
            ha='center', fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    output_path = '/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/comparison/figure3_temporal_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: figure3_temporal_analysis.png")
    plt.close()


def plot_figure4_model_agreement(data, colors):
    """Figure 4: Performance Summary Table"""
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    model_names = list(data.keys())
    
    # Create summary data
    summary_data = []
    for model_name, df in data.items():
        m = compute_metrics(df, '50pct')
        summary_data.append([
            model_name,
            f"{m['f1']:.4f}",
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['tp']}",
            f"{m['fp']}",
            f"{m['tn']}",
            f"{m['fn']}"
        ])
    
    # Create table
    col_labels = ['Model', 'F1-Score', 'Accuracy', 'Precision', 'Recall', 'TP', 'FP', 'TN', 'FN']
    table = ax.table(cellText=summary_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    # Style header row
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=13)
        cell.set_edgecolor('black')
        cell.set_linewidth(2)
    
    # Style data rows
    for i, model_name in enumerate(model_names, 1):
        # Model name cell
        cell = table[(i, 0)]
        cell.set_facecolor(colors[model_name])
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
        
        # Other cells
        for j in range(1, len(col_labels)):
            cell = table[(i, j)]
            cell.set_edgecolor('black')
            cell.set_linewidth(1)
            cell.set_text_props(fontsize=11)
            
            # Highlight best values in each column
            col_values = [float(summary_data[k][j]) if j < 5 else int(summary_data[k][j]) 
                         for k in range(len(model_names))]
            if j < 5:  # Metrics columns
                if float(summary_data[i-1][j]) == max(col_values) and max(col_values) > 0:
                    cell.set_facecolor('#E8F5E9')  # Light green for best
    
    ax.set_title('Performance Summary (50% Threshold, 1950-2000)', 
                fontsize=16, fontweight='bold', pad=30)
    
    plt.tight_layout()
    output_path = '/Users/shuhaozhang/PycharmProjects/CSE250A/HW/Categorical_HMM/comparison/figure4_performance_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: figure4_performance_summary.png")
    plt.close()


def print_detailed_comparison(data):
    """Print detailed comparison statistics"""
    
    print("\n" + "="*80)
    print("DETAILED MODEL COMPARISON (50% Threshold, 1950-2000)")
    print("="*80)
    
    for model_name, df in data.items():
        print(f"\n{'='*40}")
        print(f"Model: {model_name}")
        print(f"{'='*40}")
        
        m = compute_metrics(df, '50pct')
        
        print(f"\nPerformance Metrics:")
        print(f"  F1-Score:  {m['f1']:.4f}")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN={m['tn']}, FP={m['fp']}")
        print(f"  FN={m['fn']}, TP={m['tp']}")
        
        print(f"\nAnomalies Detected: {m['tp'] + m['fp']} / {len(df)} years")
        print(f"True Anomalies: {m['tp'] + m['fn']} years")


def main():
    """Main execution"""
    
    print("="*80)
    print("Model Comparison Visualization (1950-2000)")
    print("="*80)
    
    # Load data
    data = load_ensemble_results()
    
    if len(data) < 2:
        print("\n❌ Need at least 2 models to compare!")
        return
    
    # Color scheme - high contrast
    colors = {
        'HMM': '#0066CC',        # Bright blue
        'GMM': '#9933FF',        # Bright purple
        'PELT': '#FFD700',       # Bright yellow (gold)
        'Independent': '#FF0000' # Bright red
    }
    
    # Print detailed comparison
    print_detailed_comparison(data)
    
    # Create visualizations
    print("\n" + "="*80)
    print("Generating Visualizations...")
    print("="*80 + "\n")
    
    plot_figure1_performance_metrics(data, colors)
    plot_figure2_threshold_analysis(data, colors)
    plot_figure3_temporal_analysis(data, colors)
    plot_figure4_model_agreement(data, colors)
    
    print("\n" + "="*80)
    print("✅ All visualizations complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - figure1_performance_metrics.png")
    print("  - figure2_threshold_analysis.png")
    print("  - figure3_temporal_analysis.png")
    print("  - figure4_model_agreement.png")


if __name__ == "__main__":
    main()

