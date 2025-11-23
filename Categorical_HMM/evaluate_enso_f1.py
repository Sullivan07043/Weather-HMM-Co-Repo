import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Historical El Niño and La Niña years (1950-1990) - both considered as anomalies
# Based on ONI (Oceanic Niño Index) from https://ggweather.com/enso/oni.htm
EL_NINO_YEARS = [1951, 1953, 1957, 1958, 1963, 1965, 1969, 1972, 1976, 1977, 1982, 1986, 1987]
LA_NINA_YEARS = [1950, 1954, 1955, 1956, 1964, 1970, 1971, 1973, 1974, 1975, 1983, 1984, 1985, 1988, 1989]
ENSO_YEARS = sorted(EL_NINO_YEARS + LA_NINA_YEARS)

# Exclude site
EXCLUDED_SITES = ['760500-99999']

# Read the results (already calculated)
results_df = pd.read_csv('enso_precision_evaluation.csv')

# Filter out excluded sites
results_df = results_df[~results_df['site_id'].isin(EXCLUDED_SITES)]

# Sort by F1 score
results_df = results_df.sort_values('f1', ascending=False)

print("="*80)
print("ENSO Anomaly Detection Evaluation - TOP 10 Sites by F1 Score")
print("="*80)
print(f"\n{'Rank':<5} {'Site ID':<17} {'Site Name':<30} {'Country':<5}")
print(f"{'':5} {'':17} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<8}")
print("-"*80)

for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
    print(f"{i:<5} {row['site_id']:<17} {row['name'][:28]:<30} {row['country']:<5}")
    print(f"{'':5} {'':17} {row['f1']:<8.3f} {row['precision']:<10.3f} {row['recall']:<8.3f} {row['accuracy']:<8.3f}")

print("\n" + "="*80)
print("DETAILED RESULTS FOR TOP 10 SITES")
print("="*80)

# Read the raw data for visualization
df = pd.read_csv('enso_factorized_categorical_hmm_states.csv')
start_year = 1950
end_year = 1990

for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
    print(f"\n{'='*80}")
    print(f"Rank {i}: {row['name']}, {row['country']} ({row['site_id']})")
    print(f"{'='*80}")
    print(f"Location: Lat {row['lat']:.2f}°, Lon {row['lon']:.2f}°")
    print(f"Data years: {row['n_years']} years")
    print(f"Best mapping: {row['mapping']}")
    
    print(f"\nPerformance Metrics:")
    print(f"  - F1 Score:  {row['f1']:.3f} ⭐ (PRIMARY METRIC)")
    print(f"  - Precision: {row['precision']:.3f}")
    print(f"  - Recall:    {row['recall']:.3f}")
    print(f"  - Accuracy:  {row['accuracy']:.3f}")
    
    print(f"\nENSO Anomaly Detection:")
    print(f"  - Total ENSO years in data: {row['enso_years_in_data']}")
    print(f"  - Correctly detected: {row['enso_correct']}")
    print(f"  - Detection rate: {row['enso_correct']/row['enso_years_in_data']*100:.1f}%")
    
    print(f"\nBreakdown by Type:")
    print(f"  - El Niño: {row['el_nino_correct']}/{row['el_nino_years']} detected ({row['el_nino_correct']/row['el_nino_years']*100:.1f}%)")
    print(f"  - La Niña: {row['la_nina_correct']}/{row['la_nina_years']} detected ({row['la_nina_correct']/row['la_nina_years']*100:.1f}%)")
    
    # Calculate confusion matrix from metrics
    tp = row['enso_correct']
    fn = row['enso_years_in_data'] - row['enso_correct']
    
    if row['precision'] > 0:
        fp = int(tp / row['precision'] - tp)
    else:
        fp = 0
    
    normal_years = row['n_years'] - row['enso_years_in_data']
    tn = normal_years - fp
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Normal  Anomaly")
    print(f"  Actual Normal       {tn:<6}  {fp:<6}")
    print(f"  Actual Anomaly      {fn:<6}  {tp:<6}")
    
    # Calculate additional metrics
    if tn + fp > 0:
        fpr = fp / (tn + fp)
        print(f"\nFalse Positive Rate: {fpr:.1%} ({fp} false alarms / {tn + fp} normal years)")
    
    if fn + tp > 0:
        fnr = fn / (fn + tp)
        print(f"False Negative Rate: {fnr:.1%} ({fn} missed / {fn + tp} anomaly years)")

# Create comprehensive visualization for top 10 sites
print("\n" + "="*80)
print("Generating visualization for TOP 10 sites...")
print("="*80)

fig = plt.figure(figsize=(20, 24))

for plot_idx, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
    # Create subplot for time series
    ax = plt.subplot(10, 1, plot_idx)
    site_id = row['site_id']
    
    # Get data
    df_site = df[df['site_id'] == site_id].copy()
    df_site['year'] = df_site['t'].apply(lambda t: start_year + t)
    df_site = df_site[(df_site['year'] >= start_year) & (df_site['year'] <= end_year)].copy()
    df_site = df_site.sort_values('year')
    
    years_available = df_site['year'].values
    states = df_site['state'].values
    
    # Flip if needed
    if row['mapping'] == "State 0 = Anomaly":
        states = 1 - states
    
    # Plot predicted states
    ax.plot(years_available, states, marker='o', markersize=3, linewidth=1.5, 
            label='Predicted Anomaly', alpha=0.8, color='blue', zorder=3)
    
    # Highlight ENSO periods (background)
    ax.fill_between(years_available, -0.1, 1.1, 
                     where=[y in ENSO_YEARS for y in years_available],
                     alpha=0.25, color='orange', label='Actual ENSO Years', zorder=1)
    
    # Mark El Niño years with red vertical lines
    for year in EL_NINO_YEARS:
        if start_year <= year <= end_year and year in years_available:
            ax.axvline(x=year, color='red', alpha=0.3, linestyle='--', linewidth=0.8, zorder=2)
    
    # Mark La Niña years with blue vertical lines
    for year in LA_NINA_YEARS:
        if start_year <= year <= end_year and year in years_available:
            ax.axvline(x=year, color='darkblue', alpha=0.3, linestyle='--', linewidth=0.8, zorder=2)
    
    # Title with rank and metrics
    title = (f"#{plot_idx}: {row['name']}, {row['country']} ({row['site_id']})\n"
             f"F1: {row['f1']:.3f} | Precision: {row['precision']:.3f} | "
             f"Recall: {row['recall']:.3f} | Accuracy: {row['accuracy']:.3f}")
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    
    ax.set_ylabel('State\n(1=Anomaly)', fontsize=9)
    ax.set_ylim(-0.15, 1.15)
    ax.set_xlim(start_year-1, end_year+1)
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Add legend only for first plot
    if plot_idx == 1:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Add detection breakdown text
    text_str = (f"El Nino: {row['el_nino_correct']}/{row['el_nino_years']} "
                f"({row['el_nino_correct']/row['el_nino_years']*100:.0f}%), "
                f"La Nina: {row['la_nina_correct']}/{row['la_nina_years']} "
                f"({row['la_nina_correct']/row['la_nina_years']*100:.0f}%)")
    ax.text(0.02, 0.95, text_str, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Only show x-axis label on bottom plot
    if plot_idx == 10:
        ax.set_xlabel('Year', fontsize=9)
    else:
        ax.set_xticklabels([])

plt.tight_layout()
plt.savefig('top10_f1_enso_sites.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'top10_f1_enso_sites.png'")

# Create a summary comparison chart
print("\nGenerating performance comparison charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

top10 = results_df.head(10)

# Plot 1: F1 Score comparison
ax = axes[0, 0]
bars = ax.barh(range(10), top10['f1'].values, color='steelblue', alpha=0.8)
ax.set_yticks(range(10))
ax.set_yticklabels([f"#{i+1}: {row['name'][:20]}" for i, (_, row) in enumerate(top10.iterrows())], fontsize=9)
ax.set_xlabel('F1 Score', fontsize=11, fontweight='bold')
ax.set_title('TOP 10 Sites - F1 Score Comparison', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
            ha='left', va='center', fontsize=8, fontweight='bold')

# Plot 2: Precision vs Recall
ax = axes[0, 1]
scatter = ax.scatter(top10['recall'].values, top10['precision'].values, 
                     s=top10['f1'].values*500, alpha=0.6, c=range(10), cmap='viridis')
for i, (_, row) in enumerate(top10.iterrows()):
    ax.annotate(f"#{i+1}", (row['recall'], row['precision']), 
                fontsize=9, ha='center', va='center', fontweight='bold')
ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
ax.set_title('TOP 10 Sites - Precision vs Recall\n(Circle size = F1 Score)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.plot([0, 1], [0, 1], 'r--', alpha=0.3, linewidth=1)

# Plot 3: Accuracy comparison
ax = axes[1, 0]
bars = ax.barh(range(10), top10['accuracy'].values, color='forestgreen', alpha=0.8)
ax.set_yticks(range(10))
ax.set_yticklabels([f"#{i+1}: {row['name'][:20]}" for i, (_, row) in enumerate(top10.iterrows())], fontsize=9)
ax.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
ax.set_title('TOP 10 Sites - Accuracy Comparison', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
            ha='left', va='center', fontsize=8, fontweight='bold')

# Plot 4: Detection rates breakdown
ax = axes[1, 1]
x = np.arange(10)
width = 0.35
el_nino_rates = (top10['el_nino_correct'] / top10['el_nino_years']).values
la_nina_rates = (top10['la_nina_correct'] / top10['la_nina_years']).values

bars1 = ax.bar(x - width/2, el_nino_rates, width, label='El Nino Detection Rate', 
               color='crimson', alpha=0.8)
bars2 = ax.bar(x + width/2, la_nina_rates, width, label='La Nina Detection Rate', 
               color='navy', alpha=0.8)

ax.set_ylabel('Detection Rate', fontsize=11, fontweight='bold')
ax.set_xlabel('Site Rank', fontsize=11, fontweight='bold')
ax.set_title('TOP 10 Sites - El Nino vs La Nina Detection Rates', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"#{i+1}" for i in range(10)])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('top10_f1_performance_comparison.png', dpi=300, bbox_inches='tight')
print("Performance comparison chart saved as 'top10_f1_performance_comparison.png'")

print("\n" + "="*80)
print("SUMMARY STATISTICS (TOP 10 Sites)")
print("="*80)
print(f"Average F1 Score:  {top10['f1'].mean():.3f}")
print(f"Average Precision: {top10['precision'].mean():.3f}")
print(f"Average Recall:    {top10['recall'].mean():.3f}")
print(f"Average Accuracy:  {top10['accuracy'].mean():.3f}")
print(f"\nHighest F1 Score:  {top10['f1'].max():.3f}")
print(f"Lowest F1 Score:   {top10['f1'].min():.3f}")

# Geographic distribution
print("\n" + "="*80)
print("GEOGRAPHIC DISTRIBUTION (TOP 10 Sites)")
print("="*80)
country_counts = top10['country'].value_counts()
for country, count in country_counts.items():
    print(f"  {country}: {count} sites")

print("\n" + "="*80)
print("Analysis complete. Results saved.")
print("="*80)
