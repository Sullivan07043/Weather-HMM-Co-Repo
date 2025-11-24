import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
INPUT_CSV = "outputs/gmm_results.csv"
OUTPUT_DIR = "outputs/plots/gmm_states_per_site"

def plot_gmm_states_all_sites(
    input_csv: str = INPUT_CSV,
    output_dir: str = OUTPUT_DIR,
):
    # 1. 读数据
    df = pd.read_csv(input_csv)

    # 2. 处理日期
    if "year" not in df.columns:
        raise ValueError("Input CSV must contain a 'date' column.")
    if "site_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'site_id' column.")
    if "state" not in df.columns:
        raise ValueError("Input CSV must contain a 'pred_state' column.")

    df["year"] = pd.to_datetime(df["year"])

    # 3. 输出目录
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. 按站点画图
    site_ids = df["site_id"].unique()
    print(f"Found {len(site_ids)} sites. Generating one figure per site...")

    for site in site_ids:
        g = df[df["site_id"] == site].copy()
        g = g.sort_values("year")

        if g.empty:
            continue

        plt.figure(figsize=(10, 4))
        plt.plot(g["year"], g["state"], marker="o", linestyle="-")
        plt.yticks(sorted(g["state"].unique()))
        plt.xlabel("year")
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
        input_csv=INPUT_CSV,
        output_dir=OUTPUT_DIR,
    )