import pandas as pd

# ================================
# 可调参数
# ================================
TARGET_COUNTRIES = {"JA", "KS", "AS", "US", "MX", "PE"}  # 日本、韩国、澳大利亚、美国、墨西哥、秘鲁
MIN_TOTAL_YEARS = 50          # 站点总长度至少多少年
WINDOW_START = 1950           # 要求覆盖的时间窗口起点
WINDOW_END = 2000             # 要求覆盖的时间窗口终点
CSV_PATH = "isd-history.csv"  # 你的 CSV 路径
TOP_K = 4                    # 每个国家最多取多少个站点

# 必须存在的列名（完全按你给的表头来）
REQUIRED_COLUMNS = {
    "USAF", "WBAN", "STATION NAME", "CTRY", "STATE",
    "ICAO", "LAT", "LON", "ELEV(M)", "BEGIN", "END"
}

# ================================
# 1. 读 CSV + 检查列
# ================================
df = pd.read_csv(CSV_PATH)

missing = REQUIRED_COLUMNS - set(df.columns)
if missing:
    raise ValueError(f"CSV 缺失字段: {missing}")

# ================================
# 2. 按国家过滤
# ================================
df = df[df["CTRY"].isin(TARGET_COUNTRIES)].copy()

# 如果一个目标国家都没有，就直接报一下
if df.empty:
    print("在 CSV 中没有匹配到任何目标国家的站点。检查一下 CTRY 列是否是 JA/KS/AS/US/MX/PE 这些代码。")
else:
    # ================================
    # 3. 计算年份 & 覆盖 1960–2000
    # ================================
    # BEGIN / END 格式是 YYYYMMDD，这里取前 4 位
    df["BeginYear"] = pd.to_numeric(df["BEGIN"].astype(str).str[:4], errors="coerce")
    df["EndYear"]   = pd.to_numeric(df["END"].astype(str).str[:4], errors="coerce")

    # 去掉年份解析失败的行
    df = df.dropna(subset=["BeginYear", "EndYear"]).copy()
    df["BeginYear"] = df["BeginYear"].astype(int)
    df["EndYear"]   = df["EndYear"].astype(int)

    # 站点总时长
    df["Years"] = df["EndYear"] - df["BeginYear"] + 1

    # 1）总时长过滤
    df = df[df["Years"] >= MIN_TOTAL_YEARS].copy()

    # 2）要求 1960–2000 这一整段被包含在站点记录里：
    #    记录开始年份 <= 1960 且 结束年份 >= 2000
    df = df[(df["BeginYear"] <= WINDOW_START) & (df["EndYear"] >= WINDOW_END)].copy()

    if df.empty:
        print("没有任何站点同时满足：目标国家 + 总长度 >= MIN_TOTAL_YEARS + 完整覆盖 1960–2000。")
    else:
        # ================================
        # 4. 每个国家选 Top K（按 Years 从大到小）
        # ================================
        df_sorted = df.sort_values(["CTRY", "Years"], ascending=[True, False])

        df_top = (
            df_sorted
            .groupby("CTRY")
            .head(TOP_K)
            .reset_index(drop=True)
        )

        # 整理一个更简洁的输出列名（方便你后面用）
        out = df_top.rename(columns={
            "STATION NAME": "Name",
            "CTRY": "Country",
            "STATE": "State",
            "ELEV(M)": "Elev(m)"
        })[[
            "USAF", "WBAN", "Name", "Country", "State",
            "ICAO", "LAT", "LON", "Elev(m)", "BEGIN", "END", "Years"
        ]]

        print(out)

        out.to_csv("stations_1960_2000_covered_top_each_country.csv", index=False)
        print("\nSaved to stations_1960_2000_covered_top_each_country.csv")