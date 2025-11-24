import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

'''
Used for dataloader2.py, the site to process parameters
'''
def gen_site_csv(site = None):
        if site==None:
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

                rows = []
                for sid in site_ids:
                    usaf, wban = sid.split("-")
                    rows.append({"USAF": usaf, "WBAN": wban})

                df = pd.DataFrame(rows)
                df.to_csv("sites_to_process.csv", index=False)

        else:
                print("check gen_site_csv() for supporting passing parameter")
                exit(1)

'''
path:  path to 23 sites data "weather_1901_2019_xxxx_xxxxx.csv"
return val: dict[site_id] = {"X":(T, d), "dates" = [list of timestamps}}. 
        dict[site_id]["X"] -> X(T, d), T is the time series size, d is num of featrues
'''
def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = [
        "mean_temp", "max_temp", "min_temp",
        "sea_level_pressure", "wind_speed", "precipitation"
    ]

    site_dict = {}

    for site_id, g in df.groupby("site_id"):
        g = g.sort_values("date")

        X = g[feature_cols].to_numpy(dtype=float)
        dates = g["date"].to_list()

        site_dict[site_id] = {
            "X": X,          # shape (T, d)
            "dates": dates,  # list of timestamps
        }

    return site_dict

def run_pelt_one_site(X, model="rbf", penalty=10):
    """
    X: shape (T, d)
    model: 'rbf' (captures mean+variance changes) or 'l2' (mean changes)
    penalty: larger => fewer segments; smaller => more
    """
    algo = rpt.Pelt(model=model).fit(X)
    breakpoints = algo.predict(pen=penalty)   # e.g. [50, 120, 200, T]
    return breakpoints


def breakpoints_to_segments(dates, breakpoints):
        """
        dates: list of timestamps (length T)777
        breakpoints: list of end indices from ruptures

        Returns list of dicts:
          [{'start_date': ..., 'end_date': ..., 'start_idx':..., 'end_idx':...}, ...]
        """
        segments = []
        prev = 0
        T = len(dates)

        for bp in breakpoints:
                end = min(bp, T)
                segments.append({
                        "start_idx": prev,
                        "end_idx": end - 1,  # inclusive
                        "start_date": dates[prev],
                        "end_date": dates[end - 1],
                })
                prev = end

        return segments

def run_pelt_all_sites(site_dict, model="rbf", penalty=0.1):
    all_rows = []

    for site_id, data in site_dict.items():
        X = data["X"]       # (T, d)
        dates = data["dates"]

        if len(X) < 2:
            # cannot segment
            continue

        # run PELT on this site
        # breakpoints
        bps = run_pelt_one_site(X, model=model, penalty=penalty)

        # turn breakpoints into time segments
        segments = breakpoints_to_segments(dates, bps)

        for seg_id, seg in enumerate(segments):
            all_rows.append({
                "site_id": site_id,
                "segment_id": seg_id,
                "start_idx": seg["start_idx"],
                "end_idx": seg["end_idx"],
                "start_date": seg["start_date"],
                "end_date": seg["end_date"],
                "length": seg["end_idx"] - seg["start_idx"] + 1,
            })

    return pd.DataFrame(all_rows)


def assign_states(site_dict, segments_df, quantile=0.75):
    """
    Assign anomaly states to segments.

    state = 0 -> normal
    state = 1 -> anomaly (large deviation from site mean climate)

    quantile controls how strict the anomaly threshold is.
    """

    feature_cols = [
        "mean_temp", "max_temp", "min_temp",
        "sea_level_pressure", "wind_speed", "precipitation"
    ]

    # We must re-load the full data to compute features per segment
    df_states = []
    distances = []

    for idx, row in segments_df.iterrows():
        sid = row["site_id"]
        start = row["start_idx"]
        end = row["end_idx"]

        # X for this site
        X = site_dict[sid]["X"]

        seg_X = X[start:end + 1]
        seg_mean = seg_X.mean(axis=0)

        global_mean = X.mean(axis=0)

        # L2 norm distance between segment mean and long-term mean
        dist = np.linalg.norm(seg_mean - global_mean)

        distances.append(dist)

        df_states.append({
            "site_id": sid,
            "segment_id": row["segment_id"],
            "start_idx": start,
            "end_idx": end,
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "length": row["length"],
            "dist": dist,
        })

    df_states = pd.DataFrame(df_states)

    # Compute threshold
    threshold = df_states["dist"].quantile(quantile)

    # Assign anomaly state
    df_states["state"] = (df_states["dist"] > threshold).astype(int)

    return df_states


def expand_segments_to_years(segments_df):
    """
    Input: segments_df with columns:
       - site_id
       - segment_id
       - start_date (Timestamp)
       - end_date   (Timestamp)
       - state (optional)

    Output:
       DataFrame where each row is a single year with the segment state.
    """

    rows = []

    for idx, row in segments_df.iterrows():
        sid = row["site_id"]
        seg_id = row["segment_id"]
        state = row.get("state", None)

        start_year = row["start_date"].year
        end_year = row["end_date"].year

        for year in range(start_year, end_year + 1):
            rows.append({
                "site_id": sid,
                "segment_id": seg_id,
                "year": year,
                "state": state,
            })

    return pd.DataFrame(rows)


def load_enso_ground_truth(path):
    """
    Load ENSO ONI ground truth.
    Returns a DataFrame with only:
       - year
       - enso_anomaly (0 = normal, 1 = ENSO event)
    """
    df = pd.read_csv(path)

    # Keep only relevant columns
    df = df[["year", "enso_type", "enso_anomaly"]]

    df["year"] = df["year"].astype(int)
    df["enso_anomaly"] = df["enso_anomaly"].astype(int)

    return df

def plot_site_predictions(site_id, df_site, gt_df, ax=None, title_prefix=""):
    """
    df_site: predicted states for one site (year, state)
    gt_df:   ground truth ENSO anomaly (year, enso_anomaly)
    """

    # -------- Restrict to ground-truth year range --------
    yr_min = gt_df["year"].min()
    yr_max = gt_df["year"].max()

    df_site = df_site[(df_site["year"] >= yr_min) & (df_site["year"] <= yr_max)]
    gt_df = gt_df[(gt_df["year"] >= yr_min) & (gt_df["year"] <= yr_max)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))

    # -------- Merge for metrics --------
    merged = df_site.merge(gt_df, on="year", how="inner")
    y_true = merged["enso_anomaly"].astype(int).values
    y_pred = merged["state"].astype(int).values

    # -------- Compute metrics --------
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-9)

    # -------- Plot predicted anomaly states --------
    years = df_site["year"].values
    preds = df_site["state"].values
    ax.plot(years, preds, "-o", color="blue", label="Predicted Anomaly")

    # -------- Shade ENSO anomaly years --------
    first_shade = True
    for _, row in gt_df[gt_df["enso_anomaly"] == 1].iterrows():
        y = row["year"]
        ax.axvspan(
            y - 0.5,
            y + 0.5,
            color="orange",
            alpha=0.3,
            label="Actual ENSO Years" if first_shade else ""
        )
        first_shade = False

    # -------- Title & Metrics --------
    title = f"{title_prefix} {site_id}"
    metrics = (
        f"F1: {f1:.3f} | Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | Accuracy: {accuracy:.3f}"
    )

    # Title above
    ax.set_title(title, fontsize=12, pad=22)

    # Metrics above plot area (avoid overlap)
    ax.text(0.0, 1.12, metrics,
            transform=ax.transAxes,
            fontsize=10, va='bottom')

    # -------- Formatting --------
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_ylabel("State\n(1 = Anomaly)")
    ax.set_xlabel("Year")

    ax.legend(loc="upper left")
    return ax


# ============================================================
#  PLOT MANY SITES (stacked vertically)
# ============================================================

def plot_all_sites(expanded_states, gt_df, max_sites=21):
    site_list = expanded_states["site_id"].unique()[:max_sites]

    fig, axes = plt.subplots(
        len(site_list), 1,
        figsize=(18, 3.3 * len(site_list)),
        sharex=True
    )

    if len(site_list) == 1:
        axes = [axes]

    for i, site_id in enumerate(site_list):
        df_site = expanded_states[expanded_states["site_id"] == site_id]
        plot_site_predictions(
            site_id,
            df_site,
            gt_df,
            ax=axes[i],
            title_prefix=f"#{i+1}:"
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.55)   # avoid overlap
    fig.savefig("pelt_vs_gt.png", dpi=150, facecolor="white")
    plt.show()

def compute_enso_ensemble(expanded_states, gt_df, total_sites=21):
    """
    Compute year-level ensemble ENSO predictions.
    """

    # 1. Aggregate per year
    year_votes = (
        expanded_states.groupby("year")["state"]
        .agg(Anomaly_Votes="sum")
        .reset_index()
    )

    year_votes["Total_Stations"] = total_sites
    year_votes["Anomaly_Ratio"] = year_votes["Anomaly_Votes"] / total_sites

    # 2. Merge with ground truth
    merged = year_votes.merge(gt_df, on="year", how="left")

    merged.rename(columns={
        "year": "Year",
        "enso_type": "ENSO_Type",
        "enso_anomaly": "Ground_Truth"
    }, inplace=True)

    # 3. Ensemble thresholds
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    for th in thresholds:
        pct = int(th * 100)

        pred_col = f"Ensemble_{pct}pct"
        match_col = f"Match_{pct}pct"

        merged[pred_col] = (merged["Anomaly_Ratio"] >= th).astype(int)
        merged[match_col] = (merged[pred_col] == merged["Ground_Truth"]).astype(int)

    # 4. Reorder columns
    ordered_cols = [
        "Year",
        "ENSO_Type",
        "Ground_Truth",
        "Total_Stations",
        "Anomaly_Votes",
        "Anomaly_Ratio",
        "Ensemble_30pct",
        "Ensemble_35pct",
        "Ensemble_40pct",
        "Ensemble_45pct",
        "Ensemble_50pct",
        "Ensemble_55pct",
        "Ensemble_60pct",
        "Match_30pct",
        "Match_35pct",
        "Match_40pct",
        "Match_45pct",
        "Match_50pct",
        "Match_55pct",
        "Match_60pct"
    ]

    merged = merged[ordered_cols]
    return merged




if __name__ == "__main__":
        site_dict =  load_data("./weather_1901_2019_yearly_continuous_filled.csv")
        segments = run_pelt_all_sites(site_dict)
        segments.to_csv("pelt_segments_enso24_yearly.csv", index=False)

        segments_with_states = assign_states(site_dict, segments, quantile=0.75)
        # segments_with_states.to_csv("pelt_segments_with_states.csv", index = False)

        expanded_states = expand_segments_to_years(segments_with_states)
        expanded_states.to_csv("pelt_states_expanded.csv", index = False)

        ground_truth = load_enso_ground_truth("enso_oni_data_1950_2010.csv")

        # only do from 1950 to 2000
        expanded_states = expanded_states[
            (expanded_states["year"] >= 1950) &
            (expanded_states["year"] <= 2000)
            ]

        ground_truth = ground_truth[
            (ground_truth["year"] >= 1950) &
            (ground_truth["year"] <= 2000)
            ]

        # plot_all_sites(expanded_states, ground_truth, max_sites=21)

        ensemble_df = compute_enso_ensemble(expanded_states, ground_truth)
        ensemble_df.to_csv("PELT_enso_ensemble_results.csv", index=False)