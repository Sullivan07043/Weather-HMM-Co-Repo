import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Read the results
df = pd.read_csv('enso_factorized_categorical_hmm_states.csv')

# Read station metadata
stations = pd.read_csv('data/stations_1960_2000_covered_top_each_country.csv')
station_dict = {}
for _, row in stations.iterrows():
    site_id = f"{row['USAF']}-{row['WBAN']}"
    station_dict[site_id] = {
        'name': row['Name'],
        'country': row['Country']
    }

# Select two K=2 sites: one from Pacific East Coast, one from West Coast
# K=2 West Coast sites: 471080-99999 (Korea), 474250-99999 (Japan), 477590-99999 (Japan), 
#                       942030-99999 (Australia), 943350-99999 (Australia), 944760-99999 (Australia)
# K=2 East Coast sites: 843900-99999 (Peru), 844520-99999 (Peru), 846910-99999 (Peru), 847520-99999 (Peru)

east_site = '843900-99999'   # CAPITAN MONTES, Peru (K=2, East Coast)
west_site = '943350-99999'   # CLONCURRY AIRPORT, Australia (K=2, West Coast)

# Filter data
df_east = df[df['site_id'] == east_site].copy()
df_west = df[df['site_id'] == west_site].copy()

# Create time axis: 1950 to 1990 (yearly data)
start_year = 1950
end_year = 1990
years = list(range(start_year, end_year + 1))

# Add year column to dataframes
df_east['year'] = [years[t] if t < len(years) else None for t in df_east['t']]
df_west['year'] = [years[t] if t < len(years) else None for t in df_west['t']]

# Filter to year range
df_east = df_east[df_east['year'].notna()].copy()
df_west = df_west[df_west['year'].notna()].copy()

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

# Plot East Coast site (Peru)
ax1.plot(df_east['year'], df_east['state'], linewidth=2, marker='o', 
         markersize=4, color='#2E86AB', alpha=0.8)
east_name = station_dict.get(east_site, {}).get('name', east_site)
east_country = station_dict.get(east_site, {}).get('country', '')
ax1.set_title(f'Hidden States Time Series (1950-1990) - Pacific East Coast\n{east_name}, {east_country} ({east_site})', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Hidden State', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.5, df_east['state'].max() + 0.5)
ax1.set_xlim(start_year, end_year)

# Add horizontal lines for each state
for state in range(int(df_east['state'].max()) + 1):
    ax1.axhline(y=state, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)

# Plot West Coast site (Australia)
ax2.plot(df_west['year'], df_west['state'], linewidth=2, marker='o', 
         markersize=4, color='#A23B72', alpha=0.8)
west_name = station_dict.get(west_site, {}).get('name', west_site)
west_country = station_dict.get(west_site, {}).get('country', '')
ax2.set_title(f'Hidden States Time Series (1950-1990) - Pacific West Coast\n{west_name}, {west_country} ({west_site})', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Hidden State', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.5, df_west['state'].max() + 0.5)
ax2.set_xlim(start_year, end_year)

# Add horizontal lines for each state
for state in range(int(df_west['state'].max()) + 1):
    ax2.axhline(y=state, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)

plt.tight_layout()
plt.savefig('hidden_states_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'hidden_states_comparison.png'")

# Print statistics
print("\n" + "="*70)
east_name = station_dict.get(east_site, {}).get('name', east_site)
east_country = station_dict.get(east_site, {}).get('country', '')
print(f"Pacific East Coast Site ({east_site}):")
print(f"  - Name: {east_name}, {east_country}")
print(f"  - Time range: {int(df_east['year'].min())} to {int(df_east['year'].max())}")
print(f"  - Total time points: {len(df_east)} years")
print(f"  - Number of states (K): {df_east['state'].nunique()}")
print(f"  - State distribution:")
for state in sorted(df_east['state'].unique()):
    count = (df_east['state'] == state).sum()
    pct = count / len(df_east) * 100
    print(f"    State {state}: {count} times ({pct:.1f}%)")

print("\n" + "="*70)
west_name = station_dict.get(west_site, {}).get('name', west_site)
west_country = station_dict.get(west_site, {}).get('country', '')
print(f"Pacific West Coast Site ({west_site}):")
print(f"  - Name: {west_name}, {west_country}")
print(f"  - Time range: {int(df_west['year'].min())} to {int(df_west['year'].max())}")
print(f"  - Total time points: {len(df_west)} years")
print(f"  - Number of states (K): {df_west['state'].nunique()}")
print(f"  - State distribution:")
for state in sorted(df_west['state'].unique()):
    count = (df_west['state'] == state).sum()
    pct = count / len(df_west) * 100
    print(f"    State {state}: {count} times ({pct:.1f}%)")
print("="*70)

plt.show()
