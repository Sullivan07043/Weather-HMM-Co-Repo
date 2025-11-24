# Weather-HMM-Co-Repo

Collaborative repository for the CSE 250A Final Project: Hidden Markov Models for Weather Pattern Analysis.

---

## Repository Structure and Module Interfaces

This repository follows a modular design. Each team works independently but adheres to a unified interface.

---

## 1. Data Module (data team)

**Responsibility:**  
Process the raw Kaggle Global Weather Repository data and output a single cleaned CSV file.

**Output CSV Requirements:**  
Each row must contain:
- `site_id`
- `date` (chronologically ordered within each site)
- processed meteorological features (temperature, humidity, precipitation, condition-encoded values, etc.)
- optional metadata fields

**Important:**  
The data team outputs **only the CSV file**. No Python objects, no pickles, no NumPy arrays.

Rows must be grouped by site and sorted by date.

---

## 2. HMM Module (HMM team)

**Responsibility:**  
Train HMMs on the processed CSV and produce model outputs.

### load_data() specification (HMM team version)
Given the CSV from the data module, `load_data()` must:
1. Read the CSV  
2. Convert it into a two-level dictionary:
   `data[site_id][t] = feature_vector`
- `site_id`: region/station identifier  
- `t`: integer index representing time order within that site  
- `feature_vector`: numerical feature array derived from the CSV (continuous + encoded categorical fields)

Each site is treated as a separate observation sequence.

### Output  
The HMM module writes its results to a unified `results.csv` file.

---

## 3. Baseline Module (baseline team)

**Responsibility:**  
Implement alternative (non-HMM) unsupervised baselines, such as:
- Gaussian Mixture Models  
- k-means  
- simple temporal baselines  

The baseline team:
- implements its own `load_data()` reading the same CSV  
- outputs predictions or state sequences to the same unified `results.csv` format

This ensures fair comparison with the HMM model.

---

## 4. Evaluation Module (evaluation team)

**Responsibility:**  
Evaluation depends **only on** `results.csv`, independent of model internals.

Tasks may include:
- computing temporal metrics  
- assessing sequence consistency  
- comparing across sites  
- analyzing correlations with ENSO phases  
- visualizing hidden state sequences  

The evaluation module does not access or assume anything about how HMM or baseline models were implemented.

---
