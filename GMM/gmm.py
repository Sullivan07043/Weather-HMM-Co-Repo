import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

INPUT_CSV = "weather_1901_2019_yearly_detrend_adaptive_continuous.csv"
OUTPUT_CSV = "gmm_results.csv"

def run_gmm_per_site(
    input_csv: str = INPUT_CSV,
    n_components: int = 2,
    output_csv: str = OUTPUT_CSV,
):
    df = pd.read_csv(input_csv)

    # 使用的特征
    feature_cols = [
        "mean_temp",
        "max_temp",
        "min_temp",
        "sea_level_pressure",
        "wind_speed",
        "precipitation",
    ]

    # 防御性检查特征列是否都存在
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing}")

    all_results = []

    for site_id, g in df.groupby("site_id"):
        g = g.sort_values("date")
        n_samples_site = len(g)

        # 1）样本太少，没法拟合 GMM
        if n_samples_site < 2:
            print(
                f"[WARN] site_id={site_id} has only {n_samples_site} sample(s), "
                f"skip GMM and set state = -1"
            )

            g_res = g.copy()

            # ---- 新增：生成 year 列 ----
            g_res["year"] = pd.to_datetime(g_res["date"]).dt.year

            # ---- 新增：重命名 pred_state → state ----
            g_res["state"] = -1
            g_res["model_name"] = f"gmm_{n_components}_insufficient_samples"

            # 只保留所需列
            g_res = g_res[["site_id", "year"] + feature_cols + ["state", "model_name"]]
            all_results.append(g_res)
            continue

        # 2）样本数 < n_components 时，调小组件数
        n_components_site = min(n_components, n_samples_site)

        X = g[feature_cols].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        gmm = GaussianMixture(
            n_components=n_components_site,
            covariance_type="full",
            random_state=0,
        )

        labels = gmm.fit_predict(X_scaled)

        g_res = g.copy()

        # decode year
        g_res["year"] = pd.to_datetime(g_res["date"]).dt.year


        g_res["state"] = labels
        g_res["model_name"] = f"gmm_{n_components_site}"

        # 只保留所需列
        g_res = g_res[["site_id", "year"] + feature_cols + ["state", "model_name"]]
        all_results.append(g_res)

    # 合并全部
    out_df = pd.concat(all_results, ignore_index=True)
    out_df = out_df.sort_values(["site_id", "year"])

    # 保存
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / output_csv
    out_df.to_csv(out_path, index=False)

    print("Saved GMM results to:", out_path)


if __name__ == "__main__":
    run_gmm_per_site()