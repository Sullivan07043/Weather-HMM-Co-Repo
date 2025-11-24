"""
Generate visualizations for ensemble voting results
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('ensemble_voting_results.csv')

print("Generating visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# ============================================================================
# Plot 1: Time Series Comparison
# ============================================================================
ax = axes[0]

years = df['Year'].values
ground_truth = df['Ground_Truth'].values

# Use 50% threshold as default
prediction_50 = df['Ensemble_50pct'].values
match_50 = df['Match_50pct'].values

# Plot bars with color based on match
x = np.arange(len(years))
width = 0.6

# Create colors based on match: green for correct, red for incorrect
colors = ['green' if m == 1 else 'red' for m in match_50]

# Plot single bars
bars = ax.bar(x, ground_truth, width, alpha=0.7, color=colors, 
              edgecolor='black', linewidth=1)

# Add prediction markers on top
for i, (pred, truth) in enumerate(zip(prediction_50, ground_truth)):
    if pred == 1:
        # Show prediction as a marker
        marker_color = 'darkgreen' if pred == truth else 'darkred'
        ax.plot(i, 0.9, marker='v', markersize=10, color=marker_color, 
                markeredgecolor='black', markeredgewidth=0.5)

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Correct Prediction'),
    Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Incorrect Prediction'),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='darkgreen', 
               markersize=8, markeredgecolor='black', label='Predicted Anomaly (Correct)'),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='darkred', 
               markersize=8, markeredgecolor='black', label='Predicted Anomaly (Wrong)')
]

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('ENSO Anomaly (0=Normal, 1=Anomaly)', fontsize=11, fontweight='bold')
ax.set_title('Ensemble ENSO Prediction vs Ground Truth (1950-2010)\nAll Stations (21 sites) - Green=Match, Red=Mismatch', 
             fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x[::5])  # Show every 5th year
ax.set_xticklabels(years[::5], rotation=45)
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), 
          framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim(-0.1, 1.1)

# ============================================================================
# Plot 2: Anomaly Voting Ratio Time Series
# ============================================================================
ax = axes[1]

ax.plot(years, df['Anomaly_Ratio'], color='purple', linewidth=2, 
        label='Station Anomaly Ratio', alpha=0.8, marker='o', markersize=4)
ax.axhline(y=0.3, color='green', linestyle=':', linewidth=1.5, 
          label='30% Threshold', alpha=0.6)
ax.axhline(y=0.4, color='cyan', linestyle=':', linewidth=1.5, 
          label='40% Threshold', alpha=0.6)
ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, 
          label='50% Threshold (Default)', alpha=0.8)
ax.axhline(y=0.6, color='red', linestyle=':', linewidth=1.5, 
          label='60% Threshold', alpha=0.6)

# Shade ground truth anomaly periods
for idx, row in df.iterrows():
    if row['Ground_Truth'] == 1:
        ax.axvspan(row['Year']-0.5, row['Year']+0.5, alpha=0.1, color='red')

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Fraction of Stations Predicting Anomaly', fontsize=11, fontweight='bold')
ax.set_title('Station Voting Ratio Over Time', fontsize=13, fontweight='bold', pad=10)
ax.set_ylim(0, 1)
ax.set_xlim(years[0]-1, years[-1]+1)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), framealpha=0.9, fontsize=9)

# ============================================================================
# Plot 3: Performance Metrics
# ============================================================================
ax = axes[2]

# Calculate metrics using 50% threshold
prediction_50_metrics = df['Ensemble_50pct'].values

tn = np.sum((df['Ground_Truth'] == 0) & (prediction_50_metrics == 0))
fp = np.sum((df['Ground_Truth'] == 0) & (prediction_50_metrics == 1))
fn = np.sum((df['Ground_Truth'] == 1) & (prediction_50_metrics == 0))
tp = np.sum((df['Ground_Truth'] == 1) & (prediction_50_metrics == 1))

accuracy = (tp + tn) / len(df)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = ax.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

# Add confusion matrix text on the right side
cm_text = f'Confusion Matrix:\nTN={tn}, FP={fp}\nFN={fn}, TP={tp}'
ax.text(1.05, 0.5, cm_text, transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        verticalalignment='center')

ax.set_xlabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Ensemble Performance Metrics (50% Threshold)', 
             fontsize=13, fontweight='bold', pad=10)
ax.set_xlim(0, 1.1)
ax.grid(True, alpha=0.3, axis='x', linestyle='--')

plt.suptitle('Ensemble ENSO Detection: Majority Voting Across All Stations (21 sites)', 
             fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for legends
plt.savefig('ensemble_voting_enso_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ensemble_voting_enso_analysis.png")

# ============================================================================
# Create detailed comparison table visualization
# ============================================================================
fig2, ax = plt.subplots(figsize=(12, 16))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Year', 'ENSO Type', 'Truth', 'Vote %', 'Prediction', 'Match'])

for idx, row in df.iterrows():
    year = int(row['Year'])
    enso_type = row['ENSO_Type'].replace('_', ' ')
    truth = 'Anomaly' if row['Ground_Truth'] == 1 else 'Normal'
    vote_pct = f"{row['Anomaly_Ratio']*100:.1f}%"
    prediction = 'Anomaly' if row['Ensemble_50pct'] == 1 else 'Normal'
    match = '✓' if row['Match_50pct'] == 1 else '✗'
    
    table_data.append([year, enso_type, truth, vote_pct, prediction, match])

# Create table - position it lower in the figure
table = ax.table(cellText=table_data, cellLoc='center', loc='upper center',
                colWidths=[0.15, 0.20, 0.17, 0.15, 0.18, 0.15],
                bbox=[0, 0, 1, 0.92])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', fontsize=10)

# Style data rows
for i in range(1, len(table_data)):
    for j in range(6):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#FFFFFF')
        
        # Highlight mismatches in red
        if j == 5 and table_data[i][5] == '✗':
            cell.set_text_props(color='red', weight='bold', fontsize=10)
        # Highlight matches in green
        elif j == 5 and table_data[i][5] == '✓':
            cell.set_text_props(color='green', weight='bold', fontsize=10)

# Add title with more space above the table
ax.text(0.5, 0.98, 'Year-by-Year Ensemble Prediction vs Ground Truth (1950-2010)\nAll Stations (21 sites)', 
        ha='center', va='top', fontsize=14, fontweight='bold', 
        transform=ax.transAxes)

plt.savefig('ensemble_voting_detailed_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ensemble_voting_detailed_comparison.png")

print("\nVisualization complete!")

