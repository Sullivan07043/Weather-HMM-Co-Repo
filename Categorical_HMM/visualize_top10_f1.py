"""
Visualize top 10 F1-score stations with detailed table and time series comparison
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load results
df_results = pd.read_csv('enso_evaluation_f1_results.csv')
df_results = df_results.sort_values('f1_score', ascending=False)

# Get top 10
top10 = df_results.head(10).copy()

print("="*80)
print("Generating Visualizations")
print("="*80)

# ============================================================================
# Figure 1: Detailed Performance Table
# ============================================================================
fig1, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Rank', 'Station ID', 'Station Name', 'Country', 'Lat', 'Lon', 
                   'Years', 'Acc', 'Prec', 'Rec', 'F1', 'TN', 'FP', 'FN', 'TP'])

for i, (idx, row) in enumerate(top10.iterrows()):
    # Truncate station name if too long
    location = row['station_name'][:30] if len(str(row['station_name'])) > 30 else str(row['station_name'])
    
    table_data.append([
        f"#{i+1}",
        row['site_id'],
        location,
        row['country'],
        f"{row['lat']:.2f}",
        f"{row['lon']:.2f}",
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
                colWidths=[0.05, 0.10, 0.16, 0.06, 0.06, 0.06, 0.05, 
                          0.06, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.8)

# Style header row
for i in range(15):
    cell = table[(0, i)]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(weight='bold', color='white', fontsize=10)

# Style data rows
for i in range(1, len(table_data)):
    for j in range(15):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        else:
            cell.set_facecolor('#ffffff')
        
        # Highlight F1-score column
        if j == 10:  # F1-score column
            f1_val = float(table_data[i][10])
            if f1_val >= 0.8:
                cell.set_facecolor('#27ae60')
                cell.set_text_props(weight='bold', color='white')
            elif f1_val >= 0.7:
                cell.set_facecolor('#f39c12')
                cell.set_text_props(weight='bold', color='white')
            elif f1_val >= 0.6:
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white')

plt.title('Top 10 ENSO Detection Stations by F1-Score\nBased on Official ONI Records (1950-2000)', 
         fontsize=16, fontweight='bold', pad=20)

plt.savefig('top10_f1_enso_sites_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: top10_f1_enso_sites_table.png")

# ============================================================================
# Figure 2: Time Series Comparison (Top 10 stations vs Ground Truth)
# ============================================================================

# Load HMM predictions and ground truth
df_states = pd.read_csv('enso_factorized_categorical_hmm_states.csv')
df_states = df_states[(df_states['year'] >= 1950) & (df_states['year'] <= 2000)]
df_truth = pd.read_csv('enso_oni_data_1950_2010.csv')
df_truth = df_truth[(df_truth['year'] >= 1950) & (df_truth['year'] <= 2000)]

# Create a large figure with subplots for top 10 stations
fig2 = plt.figure(figsize=(18, 16))
gs = fig2.add_gridspec(5, 2, hspace=0.5, wspace=0.3)

# Store handles and labels for unified legend
legend_handles = None
legend_labels = None

for i, (idx, row) in enumerate(top10.iterrows()):
    site_id = row['site_id']
    station_name = row['station_name'][:25] if len(str(row['station_name'])) > 25 else str(row['station_name'])
    country = row['country']
    f1 = row['f1_score']
    
    # Get subplot position
    row_pos = i // 2
    col_pos = i % 2
    ax = fig2.add_subplot(gs[row_pos, col_pos])
    
    # Get predictions for this station
    df_site = df_states[df_states['site_id'] == site_id].copy()
    df_site = df_site.merge(df_truth[['year', 'enso_anomaly']], on='year', how='inner')
    df_site = df_site.sort_values('year')
    
    years = df_site['year'].values
    ground_truth = df_site['enso_anomaly'].values
    prediction = df_site['state'].values
    
    # Determine if state 0 or 1 represents anomaly
    state_1_anomalies = np.sum((prediction == 1) & (ground_truth == 1))
    state_0_anomalies = np.sum((prediction == 0) & (ground_truth == 1))
    
    if state_0_anomalies > state_1_anomalies:
        # State 0 = Anomaly, flip prediction
        prediction = 1 - prediction
    
    # Plot ground truth as filled area
    ax.fill_between(years, 0, ground_truth, alpha=0.3, color='red', 
                     label='Ground Truth (ENSO)', step='mid')
    
    # Plot prediction as line
    ax.plot(years, prediction, color='blue', linewidth=2, 
            label='HMM Prediction', alpha=0.8, marker='o', markersize=3)
    
    # Mark mismatches
    mismatch_years = years[ground_truth != prediction]
    mismatch_truth = ground_truth[ground_truth != prediction]
    ax.scatter(mismatch_years, mismatch_truth, color='red', s=50, 
              marker='x', linewidths=2, zorder=5, label='Mismatch')
    
    # Capture legend handles and labels from first subplot
    if i == 0:
        legend_handles, legend_labels = ax.get_legend_handles_labels()
    
    # Customize subplot
    ax.set_title(f'#{i+1}. {site_id} - {station_name} ({country})\nF1={f1:.3f}', 
                fontsize=10, fontweight='bold')
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Anomaly', fontsize=9)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlim(1948, 2012)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Normal', 'ENSO'])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# Add unified legend at the top right of the figure
fig2.legend(legend_handles, legend_labels, loc='upper right', 
           bbox_to_anchor=(0.98, 0.98), fontsize=11, framealpha=0.95,
           edgecolor='black', fancybox=True)

plt.suptitle('Top 10 Stations: ENSO Detection Time Series (1950-2000)\nBlue Line = HMM Prediction, Red Fill = Ground Truth, Red X = Mismatch', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('top10_f1_time_series_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: top10_f1_time_series_comparison.png")

print("\n" + "="*80)
print("Visualization Complete!")
print("="*80)
print(f"Top 1 Station: {top10.iloc[0]['site_id']} - {top10.iloc[0]['station_name']}")
print(f"  Country: {top10.iloc[0]['country']}")
print(f"  F1-Score: {top10.iloc[0]['f1_score']:.4f}")
print(f"  Precision: {top10.iloc[0]['precision']:.4f}")
print(f"  Recall: {top10.iloc[0]['recall']:.4f}")
print(f"\nTop 10 Average F1-Score: {top10['f1_score'].mean():.4f}")

