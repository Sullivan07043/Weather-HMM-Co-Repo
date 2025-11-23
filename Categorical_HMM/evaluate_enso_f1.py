"""
Evaluate individual station ENSO detection performance using F1-score
Based on corrected official ONI data (1950-2000)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load HMM predictions
df_states = pd.read_csv('enso_factorized_categorical_hmm_states.csv')

# Filter to 1950-2000 period
df_states = df_states[(df_states['year'] >= 1950) & (df_states['year'] <= 2000)]

# Load ground truth ENSO data (corrected official ONI records)
df_truth = pd.read_csv('enso_oni_data_1950_2000.csv')

# Load station metadata from official ISD history
df_stations = pd.read_csv('/Users/shuhaozhang/PycharmProjects/CSE250A/HW/kaggle_data/datasets/noaa/noaa-global-surface-summary-of-the-day/versions/2/isd-history.csv')
df_stations['USAF-WBAN'] = df_stations['USAF'].astype(str) + '-' + df_stations['WBAN'].astype(str).str.zfill(5)

print("="*80)
print("ENSO Detection Performance Evaluation (Individual Stations)")
print("="*80)
print(f"\nGround Truth: Official ONI records (1950-2000)")
print(f"Source: https://ggweather.com/enso/oni.htm")
print(f"\nTotal years: {len(df_truth)}")
print(f"ENSO Anomalies: {df_truth['enso_anomaly'].sum()} years ({df_truth['enso_anomaly'].sum()/len(df_truth)*100:.1f}%)")
print(f"  - El Niño: {df_truth['is_el_nino'].sum()} years")
print(f"  - La Niña: {df_truth['is_la_nina'].sum()} years")
print(f"  - Normal: {(df_truth['enso_anomaly']==0).sum()} years")

# Evaluate each station
results = []

for site_id in df_states['site_id'].unique():
    # Get predictions for this station
    df_site = df_states[df_states['site_id'] == site_id].copy()
    
    # Merge with ground truth by year
    df_eval = df_site.merge(df_truth[['year', 'enso_type', 'enso_anomaly']], on='year', how='inner')
    
    if len(df_eval) < 30:  # Skip if too few years
        continue
    
    y_true = df_eval['enso_anomaly'].values
    y_pred = df_eval['state'].values
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Get station info
    station_info = df_stations[df_stations['USAF-WBAN'] == site_id]
    if len(station_info) > 0:
        station_name = station_info.iloc[0]['STATION NAME']
        country = station_info.iloc[0]['CTRY']
        lat = station_info.iloc[0]['LAT']
        lon = station_info.iloc[0]['LON']
    else:
        station_name = "Unknown"
        country = "Unknown"
        lat = 0
        lon = 0
    
    # Determine which state represents anomaly
    state_0_anomalies = np.sum((y_pred == 0) & (y_true == 1))
    state_1_anomalies = np.sum((y_pred == 1) & (y_true == 1))
    
    if state_1_anomalies >= state_0_anomalies:
        anomaly_state = "State 1 = Anomaly"
    else:
        anomaly_state = "State 0 = Anomaly"
    
    results.append({
        'site_id': site_id,
        'station_name': station_name,
        'country': country,
        'lat': lat,
        'lon': lon,
        'years_evaluated': len(df_eval),
        'anomaly_interpretation': anomaly_state,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'tn': cm[0,0],
        'fp': cm[0,1],
        'fn': cm[1,0],
        'tp': cm[1,1],
        'confusion_matrix': str(cm.tolist())
    })

# Create results dataframe and sort by F1-score
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('f1_score', ascending=False)

# Display top 10
print("\n" + "="*80)
print("Top 10 Stations by F1-Score")
print("="*80)

for i, row in df_results.head(10).iterrows():
    print(f"\n#{df_results.index.get_loc(i)+1}. {row['site_id']} - {row['station_name']} ({row['country']})")
    print(f"   Location: {row['lat']:.3f}°, {row['lon']:.3f}°")
    print(f"   Years Evaluated: {row['years_evaluated']}")
    print(f"   Interpretation: {row['anomaly_interpretation']}")
    print(f"   Accuracy:  {row['accuracy']:.4f}")
    print(f"   Precision: {row['precision']:.4f}")
    print(f"   Recall:    {row['recall']:.4f}")
    print(f"   F1-Score:  {row['f1_score']:.4f}")
    print(f"   Confusion Matrix: TN={row['tn']}, FP={row['fp']}, FN={row['fn']}, TP={row['tp']}")

# Save results
df_results.to_csv('enso_evaluation_f1_results.csv', index=False)
print(f"\n{'='*80}")
print(f"Results saved to: enso_evaluation_f1_results.csv")
print(f"Total stations evaluated: {len(df_results)}")
print(f"{'='*80}")

