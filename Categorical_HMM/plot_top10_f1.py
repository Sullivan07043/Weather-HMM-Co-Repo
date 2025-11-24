"""
Visualize top 10 F1-score stations' ENSO detection performance
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 10

# Load results
df_results = pd.read_csv('enso_evaluation_f1_results.csv')
df_results = df_results.sort_values('f1_score', ascending=False)

# Get top 10
top10 = df_results.head(10).copy()

# Station info is already in the CSV from evaluate_enso_f1.py

# Create figure with 2 subplots
fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.4)

# ============================================================================
# Plot 1: Performance Metrics Comparison
# ============================================================================
ax1 = fig.add_subplot(gs[0])

x = np.arange(len(top10))
width = 0.2

metrics = ['accuracy', 'precision', 'recall', 'f1_score']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
    offset = width * (i - 1.5)
    bars = ax1.bar(x + offset, top10[metric], width, label=label, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0.05:  # Only show label if bar is visible
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=7, rotation=0)

# Customize plot
ax1.set_xlabel('Station', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Top 10 Stations: ENSO Detection Performance Metrics (1950-2010)', 
             fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels([f"#{i+1}\n{row['site_id']}\n{row['country']}" 
                     for i, (idx, row) in enumerate(top10.iterrows())], 
                    rotation=0, fontsize=9)
ax1.set_ylim(0, 1.15)
ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add horizontal line at 0.7 for reference
ax1.axhline(y=0.7, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='0.7 Reference')

# ============================================================================
# Plot 2: Confusion Matrix Heatmap
# ============================================================================
ax2 = fig.add_subplot(gs[1])

# Prepare data for heatmap
cm_data = []
station_labels = []

for i, (idx, row) in enumerate(top10.iterrows()):
    # Normalize confusion matrix to percentages
    total = row['tn'] + row['fp'] + row['fn'] + row['tp']
    if total > 0:
        cm_norm = np.array([[row['tn'], row['fp']], 
                           [row['fn'], row['tp']]]) / total * 100
    else:
        cm_norm = np.zeros((2, 2))
    
    cm_data.append(cm_norm.flatten())
    station_labels.append(f"#{i+1} {row['site_id']}")

cm_array = np.array(cm_data)

# Create heatmap
im = ax2.imshow(cm_array.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Set ticks
ax2.set_xticks(np.arange(len(station_labels)))
ax2.set_yticks(np.arange(4))
ax2.set_xticklabels(station_labels, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(['TN (%)', 'FP (%)', 'FN (%)', 'TP (%)'], fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, orientation='vertical', pad=0.02)
cbar.set_label('Percentage (%)', rotation=270, labelpad=20, fontsize=10)

# Add text annotations
for i in range(len(station_labels)):
    for j in range(4):
        text = ax2.text(i, j, f'{cm_array[i, j]:.1f}%',
                       ha="center", va="center", color="black", fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, pad=0.3))

ax2.set_title('Confusion Matrix Breakdown (% of total predictions per station)', 
             fontsize=12, fontweight='bold', pad=15)
ax2.set_xlabel('Station Rank', fontsize=12, fontweight='bold')

# Add overall title
plt.suptitle('ENSO Detection: Top 10 Stations by F1-Score\nBased on Official ONI Records (1950-2010)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('top10_f1_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: top10_f1_performance_comparison.png")

# ============================================================================
# Create detailed table visualization
# ============================================================================
fig2, ax = plt.subplots(figsize=(14, 12))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Rank', 'Station ID', 'Location', 'Country', 'Years', 'Acc', 'Prec', 'Rec', 'F1', 'TN', 'FP', 'FN', 'TP'])

for i, (idx, row) in enumerate(top10.iterrows()):
    # Truncate station name if too long
    location = row['station_name'][:25] if len(row['station_name']) > 25 else row['station_name']
    
    table_data.append([
        f"#{i+1}",
        row['site_id'],
        location,
        row['country'],
        int(row['years_evaluated']),
        f"{row['accuracy']:.3f}",
        f"{row['precision']:.3f}",
        f"{row['recall']:.3f}",
        f"{row['f1_score']:.3f}",
        int(row['tn']),
        int(row['fp']),
        int(row['fn']),
        int(row['tp'])
    ])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.06, 0.12, 0.18, 0.08, 0.06, 0.07, 0.07, 0.07, 0.07, 0.05, 0.05, 0.05, 0.05])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header row
for i in range(13):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white', fontsize=10)

# Style data rows - alternate colors
for i in range(1, len(table_data)):
    for j in range(13):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        else:
            cell.set_facecolor('#ffffff')
        
        # Highlight F1-score column (best scores in green)
        if j == 8:  # F1-score column
            f1_val = float(table_data[i][8])
            if f1_val >= 0.8:
                cell.set_facecolor('#2ecc71')
                cell.set_text_props(weight='bold', color='white')
            elif f1_val >= 0.7:
                cell.set_facecolor('#f39c12')
                cell.set_text_props(weight='bold', color='white')

plt.title('Top 10 ENSO Detection Stations: Detailed Performance Breakdown\nOfficial ONI Records (1950-2010)', 
         fontsize=14, fontweight='bold', pad=20)

plt.savefig('top10_f1_enso_sites_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: top10_f1_enso_sites_table.png")

print("\nVisualization complete!")
print(f"Top 1: {top10.iloc[0]['site_id']} (F1={top10.iloc[0]['f1_score']:.4f})")
print(f"Top 10 average F1: {top10['f1_score'].mean():.4f}")

