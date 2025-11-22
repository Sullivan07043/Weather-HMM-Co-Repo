import numpy as np
import pandas as pd

# ========================
# 1. ENSO-related Features
# ========================
EL_NINO_FEATURES = [
    "mean_temp",
    "max_temp",
    "min_temp",
    "sea_level_pressure",
    "wind_speed",
    "precipitation",
]


# ========================
# 2. Utility Function: log-sum-exp
# ========================
def logsumexp(a, axis=None):
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)
    res = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        res = np.squeeze(res, axis=axis)
    return res


# ========================
# 3. Factorized Categorical HMM
# ========================
class FactorizedCategoricalHMM:
    """
    Factorized Categorical HMM:
      p(x_t | z_t = k) = prod_f p(x_{t,f} | z_t = k)
    where each feature f is a categorical variable in [0, V_f-1]
    """

    def __init__(
        self,
        n_components,
        n_features,
        n_categories,
        n_iter=100,
        tol=1e-3,
        random_state=0,
        smoothing=1e-2,
    ):
        """
        Parameters:
        -----------
        n_components : int
            Number of hidden states (K)
        n_features : int
            Number of features (F)
        n_categories : list
            List of length F, where each element is the number of categories V_f for feature f
        n_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence tolerance for log-likelihood
        random_state : int
            Random seed
        smoothing : float
            Laplace smoothing parameter
        """
        self.n_components = n_components
        self.n_features = n_features
        self.n_categories = list(n_categories)
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)
        self.smoothing = smoothing

        # Model parameters
        self.pi_ = None                     # Initial state distribution (K,)
        self.A_ = None                      # Transition matrix (K,K)
        self.B_ = None                      # Emission matrices: list of F arrays, each (K, V_f)

    # -----------------------
    # Initialize parameters
    # -----------------------
    def _init_params(self, X):
        K = self.n_components
        F = self.n_features

        # Initial state distribution
        pi = self.random_state.rand(K) + self.smoothing
        self.pi_ = pi / pi.sum()

        # Transition matrix
        A = self.random_state.rand(K, K) + self.smoothing
        A /= A.sum(axis=1, keepdims=True)
        self.A_ = A

        # Emission matrices: one matrix (K, V_f) per feature
        B = []
        for f in range(F):
            V_f = self.n_categories[f]
            mat = self.random_state.rand(K, V_f) + self.smoothing
            mat /= mat.sum(axis=1, keepdims=True)
            B.append(mat)
        self.B_ = B

    # -----------------------
    # Compute log emission probabilities
    # -----------------------
    def _compute_log_emission(self, X):
        """
        Compute log emission probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (T, F)
            Observation sequence with integer values
            
        Returns:
        --------
        log_B : array, shape (T, K)
            Log emission probabilities for each time step and state
        """
        T, F = X.shape
        K = self.n_components

        log_B = np.zeros((T, K), dtype=float)
        for f in range(F):
            obs_f = X[:, f]   # (T,)
            B_f = self.B_[f]  # (K, V_f)
            # Extract p(x_{t,f} | z_t=k)
            prob_f = B_f[:, obs_f]   # shape (K, T)
            # Accumulate log probabilities
            log_B += np.log(prob_f.T + 1e-300)  # -> (T,K)

        return log_B  # (T,K)

    # -----------------------
    # E-step: forward-backward algorithm
    # -----------------------
    def _forward_backward(self, X):
        """
        Forward-backward algorithm for computing posteriors.
        
        Parameters:
        -----------
        X : array-like, shape (T, F)
            Observation sequence
            
        Returns:
        --------
        loglik : float
            Log-likelihood of the sequence
        gamma : array, shape (T, K)
            Posterior probabilities of states
        xi : array, shape (T-1, K, K)
            Posterior probabilities of state transitions
        """
        T = X.shape[0]
        K = self.n_components

        log_pi = np.log(self.pi_ + 1e-300)
        log_A = np.log(self.A_ + 1e-300)
        log_B = self._compute_log_emission(X)   # (T,K)

        # Forward pass
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            log_alpha[t] = log_B[t] + logsumexp(
                log_alpha[t - 1][:, None] + log_A, axis=0
            )

        loglik = logsumexp(log_alpha[-1], axis=0)

        # Backward pass
        log_beta = np.zeros((T, K))
        for t in reversed(range(T - 1)):
            tmp = log_A + log_B[t + 1] + log_beta[t + 1]
            log_beta[t] = logsumexp(tmp, axis=1)

        # Compute gamma (state posteriors)
        log_gamma = log_alpha + log_beta - loglik
        gamma = np.exp(log_gamma)

        # Compute xi (transition posteriors)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            tmp = (
                log_alpha[t][:, None]
                + log_A
                + log_B[t + 1][None, :]
                + log_beta[t + 1][None, :]
                - loglik
            )
            xi[t] = np.exp(tmp)

        return loglik, gamma, xi

    # -----------------------
    # M-step
    # -----------------------
    def _m_step(self, X, gamma, xi):
        T, F = X.shape
        K = self.n_components

        # Update initial state distribution
        self.pi_ = gamma[0] + self.smoothing
        self.pi_ /= self.pi_.sum()

        # Update transition matrix
        sum_xi = xi.sum(axis=0) + self.smoothing
        self.A_ = sum_xi / sum_xi.sum(axis=1, keepdims=True)

        # Update emission matrices
        for f in range(F):
            V_f = self.n_categories[f]
            B_f = np.zeros((K, V_f)) + self.smoothing

            obs_f = X[:, f]  # (T,)
            for v in range(V_f):
                mask = (obs_f == v)
                if not np.any(mask):
                    continue
                # For all time steps t where x_{t,f} = v, accumulate gamma_t(k)
                B_f[:, v] += gamma[mask].sum(axis=0)

            # Normalize
            B_f /= B_f.sum(axis=1, keepdims=True)
            self.B_[f] = B_f

    # -----------------------
    # Training (EM algorithm)
    # -----------------------
    def fit(self, X):
        X = np.asarray(X, dtype=int)
        T = X.shape[0]

        self._init_params(X)

        prev_loglik = -np.inf
        for it in range(self.n_iter):
            loglik, gamma, xi = self._forward_backward(X)
            self._m_step(X, gamma, xi)

            if it % 5 == 0:
                print(f"      iter {it:3d}: loglik = {loglik:.2f}")

            if np.abs(loglik - prev_loglik) < self.tol:
                print(f"      EM converged at iter {it}, loglik = {loglik:.2f}")
                break
            prev_loglik = loglik

        return self

    # -----------------------
    # Compute log-likelihood
    # -----------------------
    def score(self, X):
        X = np.asarray(X, dtype=int)
        loglik, _, _ = self._forward_backward(X)
        return loglik

    # -----------------------
    # Decode hidden states using posterior argmax
    # -----------------------
    def predict(self, X):
        X = np.asarray(X, dtype=int)
        _, gamma, _ = self._forward_backward(X)
        return np.argmax(gamma, axis=1)


# ========================
# 4. Data Loading: Interpolation + ENSO Features
# ========================
def load_data(csv_path, site_ids=None, feature_cols=None):
    """
    Load and preprocess weather data from CSV.
    
    Parameters:
    -----------
    csv_path : str
        Path to the weather CSV file
    site_ids : list or None
        List of site IDs to include. If None, use all sites.
    feature_cols : list or None
        List of feature column names to use. If None, use EL_NINO_FEATURES.
        
    Returns:
    --------
    data_dict : dict
        Dictionary mapping site_id to observation matrix (T, F)
    lengths_dict : dict
        Dictionary mapping site_id to sequence length T
    feature_cols : list
        List of feature names used
    n_categories : list
        List of number of categories for each feature
    """
    if feature_cols is None:
        feature_cols = EL_NINO_FEATURES

    df = pd.read_csv(csv_path, parse_dates=["date"])

    if site_ids is not None:
        df = df[df["site_id"].isin(site_ids)]

    df = df.sort_values(["site_id", "date"]).reset_index(drop=True)

    # ---- Interpolation ----
    # 1. Check for required features
    for feat in feature_cols:
        if feat not in df.columns:
            raise ValueError(f"Missing feature column in CSV: {feat}")

    # 2. Interpolate missing values for each site and feature
    def interp_one_feature(series, col):
        # series is already a single feature Series
        s_interp = series.interpolate(limit_direction="both")
        if s_interp.isna().any():
            # If there are still NaN values, fill with global median
            global_median = df[col].median()
            s_interp = s_interp.fillna(global_median)
        return s_interp

    for feat in feature_cols:
        df[feat] = (
            df.groupby("site_id")[feat]
              .apply(lambda s: interp_one_feature(s, feat))
              .reset_index(level=0, drop=True)
        )
        # After interpolation, round to discrete categories
        df[feat] = df[feat].round().astype(int)

    # 3. Check for remaining NaN values
    if df[feature_cols].isna().any().any():
        raise ValueError("Features still contain NaN after interpolation. Please check data or interpolation logic.")

    # 4. Count number of categories for each feature (assuming 0-indexed)
    n_categories = []
    for feat in feature_cols:
        max_v = int(df[feat].max())
        min_v = int(df[feat].min())
        if min_v < 0:
            raise ValueError(f"Feature {feat} has negative values. Currently assumes non-negative integer categories.")
        n_categories.append(max_v + 1)

    data_dict = {}
    lengths_dict = {}

    for sid, sub in df.groupby("site_id"):
        X = sub[feature_cols].to_numpy(dtype=int)  # (T,F)
        data_dict[sid] = X
        lengths_dict[sid] = len(sub)

    return data_dict, lengths_dict, feature_cols, n_categories


# ========================
# 5. AIC/BIC Computation (Factorized)
# ========================
def compute_aic_bic_factorized(model, logL, n_obs, n_categories):
    """
    Compute AIC and BIC for a factorized categorical HMM.
    
    Parameters:
    -----------
    model : FactorizedCategoricalHMM
        Trained HMM model
    logL : float
        Log-likelihood of the sequence
    n_obs : int
        Number of observations (T)
    n_categories : list
        List of number of categories for each feature
        
    Returns:
    --------
    AIC : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    """
    K = model.n_components
    # π: K-1 free parameters
    # A: K*(K-1) free parameters
    # B_f: K*(V_f-1) free parameters for each feature f
    num_params = (K - 1) + K * (K - 1) + sum(K * (V_f - 1) for V_f in n_categories)

    AIC = 2 * num_params - 2 * logL
    BIC = num_params * np.log(n_obs) - 2 * logL
    return AIC, BIC


# ========================
# 6. Train HMM for a Single Site (Factorized)
# ========================
def train_hmm_for_one_site_factorized(
    obs_mat,
    n_categories,
    max_K=8,
    n_iter=100,
    tol=1e-3,
    random_state=0,
):
    """
    Train HMM for a single site with model selection via BIC.
    
    Parameters:
    -----------
    obs_mat : array-like, shape (T, F)
        Observation matrix with integer values
    n_categories : list
        List of number of categories for each feature
    max_K : int
        Maximum number of hidden states to try
    n_iter : int
        Maximum number of EM iterations
    tol : float
        Convergence tolerance
    random_state : int
        Random seed
        
    Returns:
    --------
    best_model : FactorizedCategoricalHMM
        Best model selected by BIC
    best_K : int
        Optimal number of hidden states
    """
    obs_mat = np.asarray(obs_mat, dtype=int)
    T, F = obs_mat.shape

    best_model = None
    best_K = None
    best_BIC = np.inf

    for K in range(2, max_K + 1):
        print(f"  >>> Try K={K}")
        model = FactorizedCategoricalHMM(
            n_components=K,
            n_features=F,
            n_categories=n_categories,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
        )
        model.fit(obs_mat)
        logL = model.score(obs_mat)
        AIC, BIC = compute_aic_bic_factorized(model, logL, T, n_categories)

        print(f"Site Factorized HMM: K={K}, LogL={logL:.2f}, AIC={AIC:.2f}, BIC={BIC:.2f}")

        if BIC < best_BIC:
            best_BIC = BIC
            best_K = K
            best_model = model

    print(f"  >>> Selected K={best_K} with BIC={best_BIC:.2f}")
    return best_model, best_K


# ========================
# 7. Train Multiple Sites & Output Results
# ========================
def train_all_sites_factorized(data_dict, n_categories, max_K=8):
    """
    Train HMMs for all sites.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary mapping site_id to observation matrix
    n_categories : list
        List of number of categories for each feature
    max_K : int
        Maximum number of hidden states to try
        
    Returns:
    --------
    results : dict
        Dictionary containing model, K, logL, and hidden_states for each site
    """
    results = {}
    for sid, obs in data_dict.items():
        print(f"\n===== Training site {sid} =====")
        model, K = train_hmm_for_one_site_factorized(
            obs,
            n_categories=n_categories,
            max_K=max_K,
        )
        hidden_states = model.predict(obs)
        logL = model.score(obs)
        results[sid] = {
            "model": model,
            "K": K,
            "logL": logL,
            "hidden_states": hidden_states,
        }
    return results


def output_results(results, out_path):
    """
    Save hidden state sequences to CSV.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each site
    out_path : str
        Output CSV file path
    """
    rows = []
    for sid, info in results.items():
        states = info["hidden_states"]
        for t, s in enumerate(states):
            rows.append([sid, t, s])

    df = pd.DataFrame(rows, columns=["site_id", "t", "state"])
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


def save_k_values(results, out_path):
    """
    Save selected K values for each site.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each site
    out_path : str
        Output text file path
    """
    with open(out_path, 'w') as f:
        f.write("site_id\tK\tlogL\n")
        for sid, info in results.items():
            f.write(f"{sid}\t{info['K']}\t{info['logL']:.4f}\n")
    print(f"Saved K values to {out_path}")


def save_parameters(results, out_path, feature_cols):
    """
    Save model parameters for each site.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each site
    out_path : str
        Output text file path
    feature_cols : list
        List of feature names
    """
    with open(out_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Factorized Categorical HMM Parameters\n")
        f.write("=" * 80 + "\n\n")
        
        for sid, info in results.items():
            model = info['model']
            K = info['K']
            logL = info['logL']
            
            f.write(f"\n{'='*80}\n")
            f.write(f"Site ID: {sid}\n")
            f.write(f"Number of States (K): {K}\n")
            f.write(f"Log-Likelihood: {logL:.4f}\n")
            f.write(f"{'='*80}\n\n")
            
            # Initial state distribution
            f.write("Initial State Distribution (π):\n")
            f.write("-" * 40 + "\n")
            for k in range(K):
                f.write(f"  State {k}: {model.pi_[k]:.6f}\n")
            f.write("\n")
            
            # Transition matrix
            f.write("Transition Matrix (A):\n")
            f.write("-" * 40 + "\n")
            f.write("     " + "".join([f"State{k:>2}" for k in range(K)]) + "\n")
            for i in range(K):
                f.write(f"State{i:>2} ")
                for j in range(K):
                    f.write(f"{model.A_[i, j]:>7.4f} ")
                f.write("\n")
            f.write("\n")
            
            # Emission probability matrices (one per feature)
            f.write("Emission Probability Matrices (B):\n")
            f.write("-" * 40 + "\n")
            for feat_idx, feat_name in enumerate(feature_cols):
                B_f = model.B_[feat_idx]
                V_f = B_f.shape[1]
                f.write(f"\nFeature: {feat_name} ({V_f} categories)\n")
                f.write("     " + "".join([f"Cat{v:>2}" for v in range(min(V_f, 10))]))
                if V_f > 10:
                    f.write(" ...")
                f.write("\n")
                for k in range(K):
                    f.write(f"State{k:>2} ")
                    for v in range(min(V_f, 10)):
                        f.write(f"{B_f[k, v]:>6.3f} ")
                    if V_f > 10:
                        f.write("...")
                    f.write("\n")
            f.write("\n")
    
    print(f"Saved parameters to {out_path}")


# ========================
# 8. Main: ENSO Sites + Monthly CSV
# ========================
if __name__ == "__main__":
    # Selected ENSO-sensitive sites (USAF-WBAN format)
    site_ids = [
        "942030-99999",
        "943350-99999",
        "943740-99999",
        "944760-99999",
        "474250-99999",
        "477590-99999",
        "477710-99999",
        "478030-99999",
        "471100-99999",
        "471080-99999",
        "471420-99999",
        "471510-99999",
        "760500-99999",
        "760610-99999",
        "761130-99999",
        "761220-99999",
        "843900-99999",
        "844520-99999",
        "846910-99999",
        "847520-99999",
        "726810-24131",
        "726815-24106",
        "722860-23119",
        "722265-13821",
    ]

    # Path to preprocessed monthly data with discretized features
    csv_path = "data/processed/weather_1901_2019_yearly_bins10.csv"

    data_dict, lengths_dict, feature_cols, n_categories = load_data(
        csv_path,
        site_ids=site_ids,
        feature_cols=EL_NINO_FEATURES,
    )

    print("Features used:", feature_cols)
    print("Number of categories per feature:", n_categories)

    results = train_all_sites_factorized(
        data_dict,
        n_categories=n_categories,
        max_K=8,
    )
    output_results(results, "enso_factorized_categorical_hmm_states.csv")
    save_k_values(results, "hmm_k_values.txt")
    save_parameters(results, "hmm_parameters.txt", feature_cols)
