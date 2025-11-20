Weather-HMM-Co-Repo

Collaborative repository for the CSE 250A Final Project: Hidden Markov Models for Weather Pattern Analysis

⸻

Repository Structure and Interface Design

This repository follows a modular, team-collaborative structure. Each module exposes a clear interface, allowing the Data, HMM, Baseline, and Evaluation groups to develop independently while maintaining full compatibility.

⸻

1. Data Module (data team)

Responsibility:
Process raw weather data from the Kaggle Global Weather Repository and produce a single cleaned CSV file for downstream modules.

Output format:
The processed CSV should contain:
	•	site_id — weather station / region ID
	•	date — chronological date
	•	processed meteorological features (e.g., temperature, humidity, precipitation, condition encoding, etc.)
	•	any other optional metadata fields

Rows must be grouped by site and sorted by date within each site.

No Python object output — only the CSV file is produced by this module.

⸻

2. HMM Module (HMM team)

The HMM team will implement:
	•	data loading
	•	model training (Baum–Welch)
	•	inference (Forward–Backward, Viterbi)
	•	generation of result files

load_data() specification (HMM team version)

Given the CSV from the data module, load_data() should:
	1.	Read the CSV
	2.	Convert it into a two-level dictionary:
  data[site_id][t] = feature_vector
  	•	site_id: string or integer
	•	t: integer time index within that site
	•	feature_vector: processed numeric feature array

This structure ensures each site is treated as an independent observation sequence for the HMM.

No dependency on the data preprocessing code—the HMM team independently loads and structures the CSV.

Output from HMM module

Write model outputs in a unified results.csv format.

⸻

3. Baseline Module (baseline team)

The Baseline team (e.g., GMM, k-means, naive temporal baseline) follows the same interface as the HMM team:
	•	They write their own load_data() reading the same CSV
	•	They output their predictions, cluster assignments, or state sequences into a unified results.csv format

This ensures fair comparison between HMM and alternative models.

⸻

4. Evaluation Module (evaluation team)

The evaluation module:
	•	only reads results.csv
	•	does not depend on internal representations of HMM/Baseline
	•	computes metrics such as:
	•	likelihood-based scores
	•	temporal consistency
	•	cross-site synchronization
	•	ENSO-related correlations
	•	visualization of hidden-state sequences

The evaluation module remains completely independent, making the pipeline modular and reproducible.
