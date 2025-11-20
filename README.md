# Weather-HMM-Co-Repo
This is a collaborative repository of CSE 250A Final project: HMM for Weather

## Interface Design:

- **Data**: Responsible for `load_data()`, outputs:
  - `X`: all time steps stacked into one array, shape = `(n_samples, n_features)`
  - `lengths`: list of sequence lengths, e.g. `[T1, T2, ...]` (used to **split X along time into different sequences**, **not** to split features; Each portion is a observation sequence from a stie/station)
  - `meta`: a DataFrame storing `seq_id / t / date / site_id / ...`

- **HMM & Baseline**:
  - Read data from `load_data()`
  - Write results in a unified `results.csv` format

- **Evaluation**: Only depends on `results.csv`, independent of model internals