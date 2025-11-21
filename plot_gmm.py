import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_gmm_states_all_sites(
    input_csv: str = "outputs/gmm_per_site_results.csv",
    output_dir: str = "plots/gmm_states_per_site",
):
    # 1. 读数据
    df = pd.read_csv(input_csv)

    # 2. 处理日期
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a 'date' column.")
    if "site_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'site_id' column.")
    if "pred_state" not in df.columns:
        raise ValueError("Input CSV must contain a 'pred_state' column.")

    df["date"] = pd.to_datetime(df["date"])

    # 3. 输出目录
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. 按站点画图
    site_ids = df["site_id"].unique()
    print(f"Found {len(site_ids)} sites. Generating one figure per site...")

    for site in site_ids:
        g = df[df["site_id"] == site].copy()
        g = g.sort_values("date")

        if g.empty:
            continue

        plt.figure(figsize=(10, 4))
        plt.plot(g["date"], g["pred_state"], marker="o", linestyle="-")
        plt.yticks(sorted(g["pred_state"].unique()))
        plt.xlabel("Date")
        plt.ylabel("GMM state")
        plt.title(f"GMM states over time, site {site}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 文件名里把特殊字符去掉/替换
        safe_site = str(site).replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"gmm_states_{safe_site}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved figure for site {site} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    plot_gmm_states_all_sites(
        input_csv="outputs/gmm_per_site_results.csv",
        output_dir="plots/gmm_states_per_site",
    )