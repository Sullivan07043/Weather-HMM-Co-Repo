import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Historical El Niño and La Niña years (1950-1990) - both considered as anomalies
# Based on ONI (Oceanic Niño Index) from https://ggweather.com/enso/oni.htm
EL_NINO_YEARS = [1951, 1953, 1957, 1958, 1963, 1965, 1969, 1972, 1976, 1977, 1982, 1986, 1987]
LA_NINA_YEARS = [1950, 1954, 1955, 1956, 1964, 1970, 1971, 1973, 1974, 1975, 1983, 1984, 1985, 1988, 1989]
ENSO_YEARS = sorted(EL_NINO_YEARS + LA_NINA_YEARS)  # All anomaly years

print("="*80)
print("ENSO Anomaly Detection Evaluation for K=2 Sites (1950-1990)")
print("Both El Niño and La Niña are considered as anomalies (State = 1)")
print("="*80)
print(f"\nEl Niño years ({len(EL_NINO_YEARS)}): {EL_NINO_YEARS}")
print(f"\nLa Niña years ({len(LA_NINA_YEARS)}): {LA_NINA_YEARS}")
print(f"\nTotal ENSO anomaly years: {len(ENSO_YEARS)}")
print(f"Normal years: {41 - len(ENSO_YEARS)}")

# Read the results
df = pd.read_csv('enso_factorized_categorical_hmm_states.csv')

# Read K values
k_values = pd.read_csv('hmm_k_values.txt', sep='\t')
k2_sites = k_values[k_values['K'] == 2]['site_id'].tolist()

# Read station metadata
stations = pd.read_csv('data/stations_1960_2000_covered_top_each_country.csv')
station_dict = {}
for _, row in stations.iterrows():
    site_id = f"{row['USAF']}-{row['WBAN']}"
    station_dict[site_id] = {
        'name': row['Name'],
        'country': row['Country'],
        'lat': row['LAT'],
        'lon': row['LON']
    }

# Time range
start_year = 1950
end_year = 1990

# Create ground truth labels: 1 for ENSO anomaly (El Niño or La Niña), 0 for normal
years = list(range(start_year, end_year + 1))
ground_truth = [1 if year in ENSO_YEARS else 0 for year in years]

results = []

print("\n" + "="*80)
print("Evaluating each K=2 site...")
print("="*80)

for site_id in k2_sites:
    # Filter data for this site
    df_site = df[df['site_id'] == site_id].copy()
    
    # Add year column
    df_site['year'] = df_site['t'].apply(lambda t: start_year + t)
    
    # Filter to date range and sort by year
    df_site = df_site[(df_site['year'] >= start_year) & (df_site['year'] <= end_year)].copy()
    df_site = df_site.sort_values('year')
    
    if len(df_site) == 0:
        continue
    
    # Get predictions (state)
    predictions = df_site['state'].values
    
    # Align with ground truth
    years_available = df_site['year'].values
    ground_truth_aligned = [1 if year in ENSO_YEARS else 0 for year in years_available]
    
    # Try both mappings: state 1 = anomaly or state 0 = anomaly
    metrics_state1 = {}
    metrics_state0 = {}
    
    # Case 1: State 1 represents anomaly
    pred_state1 = predictions
    metrics_state1['accuracy'] = accuracy_score(ground_truth_aligned, pred_state1)
    metrics_state1['precision'] = precision_score(ground_truth_aligned, pred_state1, zero_division=0)
    metrics_state1['recall'] = recall_score(ground_truth_aligned, pred_state1, zero_division=0)
    metrics_state1['f1'] = f1_score(ground_truth_aligned, pred_state1, zero_division=0)
    cm1 = confusion_matrix(ground_truth_aligned, pred_state1)
    
    # Case 2: State 0 represents anomaly (flip predictions)
    pred_state0 = 1 - predictions
    metrics_state0['accuracy'] = accuracy_score(ground_truth_aligned, pred_state0)
    metrics_state0['precision'] = precision_score(ground_truth_aligned, pred_state0, zero_division=0)
    metrics_state0['recall'] = recall_score(ground_truth_aligned, pred_state0, zero_division=0)
    metrics_state0['f1'] = f1_score(ground_truth_aligned, pred_state0, zero_division=0)
    cm0 = confusion_matrix(ground_truth_aligned, pred_state0)
    
    # Choose the better mapping based on precision
    if metrics_state1['precision'] >= metrics_state0['precision']:
        best_metrics = metrics_state1
        best_mapping = "State 1 = Anomaly"
        best_pred = pred_state1
        best_cm = cm1
    else:
        best_metrics = metrics_state0
        best_mapping = "State 0 = Anomaly"
        best_pred = pred_state0
        best_cm = cm0
    
    # Get station info
    station_info = station_dict.get(site_id, {})
    name = station_info.get('name', site_id)
    country = station_info.get('country', '')
    lat = station_info.get('lat', 'N/A')
    lon = station_info.get('lon', 'N/A')
    
    # Count correct predictions
    enso_years_available = [y for y in years_available if y in ENSO_YEARS]
    enso_correct = sum([best_pred[i] == 1 for i, y in enumerate(years_available) if y in ENSO_YEARS])
    
    # Count El Niño and La Niña separately
    el_nino_years_available = [y for y in years_available if y in EL_NINO_YEARS]
    la_nina_years_available = [y for y in years_available if y in LA_NINA_YEARS]
    el_nino_correct = sum([best_pred[i] == 1 for i, y in enumerate(years_available) if y in EL_NINO_YEARS])
    la_nina_correct = sum([best_pred[i] == 1 for i, y in enumerate(years_available) if y in LA_NINA_YEARS])
    
    results.append({
        'site_id': site_id,
        'name': name,
        'country': country,
        'lat': lat,
        'lon': lon,
        'n_years': len(df_site),
        'mapping': best_mapping,
        'accuracy': best_metrics['accuracy'],
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1'],
        'enso_years_in_data': len(enso_years_available),
        'enso_correct': enso_correct,
        'el_nino_years': len(el_nino_years_available),
        'el_nino_correct': el_nino_correct,
        'la_nina_years': len(la_nina_years_available),
        'la_nina_correct': la_nina_correct,
        'confusion_matrix': best_cm
    })

# Convert to DataFrame and sort by precision (descending)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('precision', ascending=False)

# Save results
results_df.to_csv('enso_precision_evaluation.csv', index=False)

print("\n" + "="*80)
print("TOP 10 BEST PERFORMING SITES (by Precision)")
print("="*80)
print(f"\n{'Rank':<5} {'Site ID':<17} {'Name':<30} {'Country':<5} {'Prec':<7} {'Recall':<7} {'F1':<7} {'Acc':<7}")
print("-"*80)

for idx, row in results_df.head(10).iterrows():
    print(f"{results_df.index.get_loc(idx)+1:<5} {row['site_id']:<17} {row['name'][:28]:<30} "
          f"{row['country']:<5} {row['precision']:<7.3f} {row['recall']:<7.3f} "
          f"{row['f1']:<7.3f} {row['accuracy']:<7.3f}")

print("\n" + "="*80)
print("DETAILED RESULTS FOR TOP 5 SITES (by Precision)")
print("="*80)

for idx, row in results_df.head(5).iterrows():
    print(f"\n{'='*80}")
    print(f"Rank {results_df.index.get_loc(idx)+1}: {row['name']}, {row['country']} ({row['site_id']})")
    print(f"{'='*80}")
    print(f"Location: Lat {row['lat']:.2f}, Lon {row['lon']:.2f}")
    print(f"Years of data: {row['n_years']}")
    print(f"Best mapping: {row['mapping']}")
    print(f"\nPerformance Metrics:")
    print(f"  - Precision: {row['precision']:.3f} ⭐ (PRIMARY METRIC)")
    print(f"  - Recall:    {row['recall']:.3f}")
    print(f"  - F1 Score:  {row['f1']:.3f}")
    print(f"  - Accuracy:  {row['accuracy']:.3f}")
    print(f"\nENSO Anomaly Detection:")
    print(f"  - Total ENSO years in data: {row['enso_years_in_data']}")
    print(f"  - Correctly detected: {row['enso_correct']}")
    print(f"  - Detection rate: {row['enso_correct']/row['enso_years_in_data']*100:.1f}%")
    print(f"\nBreakdown by type:")
    print(f"  - El Niño: {row['el_nino_correct']}/{row['el_nino_years']} detected ({row['el_nino_correct']/row['el_nino_years']*100:.1f}%)")
    print(f"  - La Niña: {row['la_nina_correct']}/{row['la_nina_years']} detected ({row['la_nina_correct']/row['la_nina_years']*100:.1f}%)")
    print(f"\nConfusion Matrix:")
    cm = row['confusion_matrix']
    print(f"                    Predicted")
    print(f"                    Normal  Anomaly")
    print(f"  Actual Normal       {cm[0,0]:<6}  {cm[0,1]:<6}")
    print(f"  Actual Anomaly      {cm[1,0]:<6}  {cm[1,1]:<6}")
    
    # Calculate false positive rate
    if cm[0,0] + cm[0,1] > 0:
        fpr = cm[0,1] / (cm[0,0] + cm[0,1])
        print(f"\nFalse Positive Rate: {fpr:.3f} ({cm[0,1]} false alarms out of {cm[0,0] + cm[0,1]} normal years)")

print("\n" + "="*80)
print("BOTTOM 5 SITES (Lowest Precision)")
print("="*80)

for idx, row in results_df.tail(5).iterrows():
    print(f"\n{row['name']}, {row['country']} ({row['site_id']})")
    print(f"  Precision: {row['precision']:.3f}, Recall: {row['recall']:.3f}, "
          f"F1: {row['f1']:.3f}, Accuracy: {row['accuracy']:.3f}")

# Create visualization for top 5 sites
print("\n" + "="*80)
print("Creating visualization for top 5 sites (by precision)...")
print("="*80)

fig, axes = plt.subplots(5, 1, figsize=(18, 15))

for plot_idx, (idx, row) in enumerate(results_df.head(5).iterrows()):
    ax = axes[plot_idx]
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
    ax.plot(years_available, states, marker='o', markersize=4, linewidth=1.5, 
            label='Predicted Anomaly', alpha=0.7, color='blue')
    
    # Plot actual ENSO years with different colors
    for year in EL_NINO_YEARS:
        if start_year <= year <= end_year and year in years_available:
            ax.axvline(x=year, color='red', alpha=0.4, linestyle='--', linewidth=1)
    
    for year in LA_NINA_YEARS:
        if start_year <= year <= end_year and year in years_available:
            ax.axvline(x=year, color='blue', alpha=0.4, linestyle='--', linewidth=1)
    
    # Highlight ENSO periods
    ax.fill_between(years_available, 0, 1, where=[y in ENSO_YEARS for y in years_available],
                     alpha=0.2, color='orange', label='Actual ENSO Years (El Niño + La Niña)')
    
    ax.set_title(f"Rank {plot_idx+1}: {row['name']}, {row['country']} - "
                 f"Precision: {row['precision']:.3f}, Recall: {row['recall']:.3f}, F1: {row['f1']:.3f}", 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('State (1=Anomaly)', fontsize=10)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlim(start_year-1, end_year+1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    # Add text with breakdown
    text_str = f"El Niño: {row['el_nino_correct']}/{row['el_nino_years']}, La Niña: {row['la_nina_correct']}/{row['la_nina_years']}"
    ax.text(0.02, 0.02, text_str, transform=ax.transAxes, 
            fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('top5_enso_precision.png', dpi=300, bbox_inches='tight')
print("Saved visualization as 'top5_enso_precision.png'")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nTotal sites evaluated: {len(results_df)}")
print(f"Average Precision: {results_df['precision'].mean():.3f}")
print(f"Average Recall: {results_df['recall'].mean():.3f}")
print(f"Average F1 score: {results_df['f1'].mean():.3f}")
print(f"Average Accuracy: {results_df['accuracy'].mean():.3f}")
print(f"\nBest Precision: {results_df['precision'].max():.3f}")
print(f"Worst Precision: {results_df['precision'].min():.3f}")

# Additional analysis: which sites detect El Niño vs La Niña better
print("\n" + "="*80)
print("ANALYSIS: El Niño vs La Niña Detection")
print("="*80)
results_df['el_nino_rate'] = results_df['el_nino_correct'] / results_df['el_nino_years']
results_df['la_nina_rate'] = results_df['la_nina_correct'] / results_df['la_nina_years']
results_df['detection_bias'] = results_df['el_nino_rate'] - results_df['la_nina_rate']

print(f"\nAverage El Niño detection rate: {results_df['el_nino_rate'].mean():.3f}")
print(f"Average La Niña detection rate: {results_df['la_nina_rate'].mean():.3f}")

print("\nSites with strong El Niño bias (detect El Niño better):")
el_nino_biased = results_df.nlargest(3, 'detection_bias')[['site_id', 'name', 'country', 'el_nino_rate', 'la_nina_rate', 'detection_bias']]
for _, row in el_nino_biased.iterrows():
    print(f"  {row['name']}, {row['country']}: El Niño {row['el_nino_rate']:.2%}, La Niña {row['la_nina_rate']:.2%} (bias: {row['detection_bias']:+.3f})")

print("\nSites with strong La Niña bias (detect La Niña better):")
la_nina_biased = results_df.nsmallest(3, 'detection_bias')[['site_id', 'name', 'country', 'el_nino_rate', 'la_nina_rate', 'detection_bias']]
for _, row in la_nina_biased.iterrows():
    print(f"  {row['name']}, {row['country']}: El Niño {row['el_nino_rate']:.2%}, La Niña {row['la_nina_rate']:.2%} (bias: {row['detection_bias']:+.3f})")

print("\n" + "="*80)
print("Results saved to 'enso_precision_evaluation.csv'")
print("="*80)

