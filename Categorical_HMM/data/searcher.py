import pandas as pd
import zipfile
from pathlib import Path
from tqdm import tqdm

# ================================
# 可调参数
# ================================
TARGET_COUNTRIES = {"JA", "KS", "AS", "US", "MX", "PE"}  # 日本、韩国、澳大利亚、美国、墨西哥、秘鲁
MIN_TOTAL_YEARS = 50          # 站点总长度至少多少年
WINDOW_START = 1950           # 要求覆盖的时间窗口起点
WINDOW_END = 2010             # 要求覆盖的时间窗口终点
GSOD_ZIP_PATH = "/Users/shuhaozhang/PycharmProjects/CSE250A/HW/kaggle_data/datasets/noaa/noaa-global-surface-summary-of-the-day/versions/2/gsod_all_years.zip"
ISD_HISTORY_CSV = "/Users/shuhaozhang/PycharmProjects/CSE250A/HW/kaggle_data/datasets/noaa/noaa-global-surface-summary-of-the-day/versions/2/isd-history.csv"
TOP_K = 4                    # 每个国家最多取多少个站点

# 必须存在的列名
REQUIRED_COLUMNS = {
    "USAF", "WBAN", "STATION NAME", "CTRY", "STATE",
    "ICAO", "LAT", "LON", "ELEV(M)", "BEGIN", "END"
}

print("=" * 80)
print("从GSOD原始数据筛选站点")
print("=" * 80)

# ================================
# 1. 读取ISD元数据以获取站点地理信息和国家
# ================================
print("\n【步骤1】读取ISD元数据...")
df_isd = pd.read_csv(ISD_HISTORY_CSV)

# 清理列名
missing = REQUIRED_COLUMNS - set(df_isd.columns)
if missing:
    raise ValueError(f"ISD历史文件缺失字段: {missing}")

# 创建site_id
df_isd['USAF'] = df_isd['USAF'].astype(str).str.zfill(6)
df_isd['WBAN'] = df_isd['WBAN'].astype(str).str.zfill(5)
df_isd['site_id'] = df_isd['USAF'] + '-' + df_isd['WBAN']

print(f"✓ ISD元数据: {len(df_isd)} 个站点记录")

# 先按目标国家过滤，减少后续处理量
df_isd_filtered = df_isd[df_isd['CTRY'].isin(TARGET_COUNTRIES)].copy()
print(f"✓ 目标国家筛选后: {len(df_isd_filtered)} 个站点")
print(f"  国家分布:")
for country, count in df_isd_filtered['CTRY'].value_counts().items():
    print(f"    {country}: {count} 个站点")

target_site_ids = set(df_isd_filtered['site_id'].unique())
print(f"\n目标站点数: {len(target_site_ids)}")

# ================================
# 2. 从ZIP文件读取GSOD数据并统计站点年份范围
# ================================
print("\n【步骤2】从GSOD ZIP文件读取数据并统计站点年份范围...")
print(f"ZIP文件: {GSOD_ZIP_PATH}")

# 检查文件是否存在
if not Path(GSOD_ZIP_PATH).exists():
    raise FileNotFoundError(f"找不到文件: {GSOD_ZIP_PATH}")

station_years = {}  # {site_id: set of years}

import tarfile
import io

with zipfile.ZipFile(GSOD_ZIP_PATH, 'r') as zf:
    # 获取所有tar文件
    tar_files = [f for f in zf.namelist() if f.endswith('.tar')]
    print(f"✓ ZIP中共有 {len(tar_files)} 个tar文件")
    
    print("\n正在处理目标国家的站点数据...")
    processed_count = 0
    
    for tar_file in tqdm(tar_files, desc="处理tar文件"):
        try:
            # 读取tar文件
            with zf.open(tar_file) as tar_bytes:
                # 将字节流转换为tar文件对象
                tar_data = io.BytesIO(tar_bytes.read())
                with tarfile.open(fileobj=tar_data, mode='r') as tf:
                    # 遍历tar中的所有.op.gz文件
                    for member in tf.getmembers():
                        if not member.name.endswith('.op.gz'):
                            continue
                        
                        # 从文件名提取site_id和年份 (格式: ./USAF-WBAN-YEAR.op.gz)
                        basename = Path(member.name).stem  # 去掉.op.gz，得到USAF-WBAN-YEAR.op
                        basename = basename.replace('.op', '')  # 去掉.op，得到USAF-WBAN-YEAR
                        parts = basename.split('-')
                        
                        if len(parts) < 3:
                            continue
                        
                        # 提取site_id (USAF-WBAN)
                        site_id = f"{parts[0]}-{parts[1]}"
                        
                        # 只处理目标站点
                        if site_id not in target_site_ids:
                            continue
                        
                        # 提取年份
                        try:
                            year = int(parts[2])
                        except ValueError:
                            continue
                        
                        # 记录该站点的年份（累积，因为跨多个tar文件）
                        if site_id in station_years:
                            station_years[site_id].add(year)
                        else:
                            station_years[site_id] = {year}
                        processed_count += 1
                
        except Exception as e:
            # 跳过有问题的文件
            continue
    
    print(f"\n✓ 成功处理 {processed_count} 个CSV文件")
    print(f"✓ 找到 {len(station_years)} 个唯一站点")

# ================================
# 3. 统计各站点的年份范围
# ================================
print("\n【步骤3】统计各站点的年份范围...")
station_stats = []

for site_id, years in station_years.items():
    if len(years) == 0:
        continue
    
    start_year = min(years)
    end_year = max(years)
    total_years = len(years)
    
    # 检查是否覆盖1950-2000窗口（起点在1950之前或等于，终点在2000之后或等于）
    covers_window = (start_year <= WINDOW_START) and (end_year >= WINDOW_END)
    
    station_stats.append({
        'site_id': site_id,
        'actual_start': start_year,
        'actual_end': end_year,
        'total_years': total_years,
        'covers_1950_2000': covers_window
    })

df_stats = pd.DataFrame(station_stats)
print(f"✓ 统计完成: {len(df_stats)} 个站点有数据")

# ================================
# 4. 应用筛选条件
# ================================
print("\n【步骤4】应用筛选条件...")
print(f"  条件1: 总年数 >= {MIN_TOTAL_YEARS}")
print(f"  条件2: 起始年份 <= {WINDOW_START}，结束年份 >= {WINDOW_END}")

df_filtered = df_stats[
    (df_stats['total_years'] >= MIN_TOTAL_YEARS) &
    (df_stats['covers_1950_2000'] == True)
].copy()

print(f"✓ 筛选后剩余: {len(df_filtered)} 个站点")

if df_filtered.empty:
    print("\n⚠️  没有站点同时满足所有条件！")
    print("\n统计信息:")
    print(f"  总年数 >= {MIN_TOTAL_YEARS}: {(df_stats['total_years'] >= MIN_TOTAL_YEARS).sum()} 个站点")
    print(f"  覆盖 {WINDOW_START}-{WINDOW_END}: {df_stats['covers_1950_2000'].sum()} 个站点")
    exit(1)

# 显示一些被筛掉的站点信息
df_rejected = df_stats[~df_stats.index.isin(df_filtered.index)]
if len(df_rejected) > 0:
    print(f"\n被筛掉的站点数: {len(df_rejected)}")
    print("主要原因统计:")
    insufficient_years = df_rejected[df_rejected['total_years'] < MIN_TOTAL_YEARS]
    print(f"  - 总年数不足 {MIN_TOTAL_YEARS}: {len(insufficient_years)} 个")
    
    not_covering = df_rejected[
        (df_rejected['total_years'] >= MIN_TOTAL_YEARS) & 
        (~df_rejected['covers_1950_2000'])
    ]
    if len(not_covering) > 0:
        print(f"  - 未覆盖 {WINDOW_START}-{WINDOW_END}: {len(not_covering)} 个")
        print(f"    示例（显示前5个）:")
        for idx, row in not_covering.head(5).iterrows():
            print(f"      {row['site_id']}: {row['actual_start']}-{row['actual_end']} ({row['total_years']}年)")

# ================================
# 5. 合并ISD元数据
# ================================
print("\n【步骤5】合并ISD元数据...")
df_merged = df_filtered.merge(
    df_isd[['site_id', 'USAF', 'WBAN', 'STATION NAME', 'CTRY', 'STATE', 
            'ICAO', 'LAT', 'LON', 'ELEV(M)', 'BEGIN', 'END']],
    on='site_id',
    how='left'
)

# 检查缺失元数据的站点
missing_meta = df_merged['CTRY'].isna().sum()
if missing_meta > 0:
    print(f"⚠️  警告: {missing_meta} 个站点在ISD历史文件中找不到元数据")
    print(f"  这些站点将被排除")
    df_merged = df_merged.dropna(subset=['CTRY'])

print(f"✓ 合并完成: {len(df_merged)} 个站点有完整信息")

# 再次确认国家过滤（应该已经是目标国家了）
df_merged = df_merged[df_merged['CTRY'].isin(TARGET_COUNTRIES)].copy()
print(f"✓ 确认目标国家: {len(df_merged)} 个站点")

if df_merged.empty:
    print(f"\n⚠️  没有目标国家的站点满足条件！")
    exit(1)

print(f"  国家分布:")
for country, count in df_merged['CTRY'].value_counts().items():
    print(f"    {country}: {count} 个站点")

# ================================
# 6. 每个国家选 Top K（按实际年数从大到小）
# ================================
print(f"\n【步骤6】每个国家选Top {TOP_K}...")
df_sorted = df_merged.sort_values(['CTRY', 'total_years'], ascending=[True, False])

df_top = (
    df_sorted
    .groupby('CTRY')
    .head(TOP_K)
    .reset_index(drop=True)
)

print(f"✓ 最终选择: {len(df_top)} 个站点")
print(f"  国家分布:")
for country, count in df_top['CTRY'].value_counts().items():
    print(f"    {country}: {count} 个站点")

# ================================
# 7. 整理输出格式
# ================================
print("\n【步骤7】整理输出...")
out = df_top.rename(columns={
    'STATION NAME': 'Name',
    'CTRY': 'Country',
    'STATE': 'State',
    'ELEV(M)': 'Elev(m)'
})

# 添加实际年份信息列
out['Years'] = out['total_years']
out['ActualBegin'] = out['actual_start'].astype(int)
out['ActualEnd'] = out['actual_end'].astype(int)

# 选择输出列
out = out[[
    'USAF', 'WBAN', 'Name', 'Country', 'State',
    'ICAO', 'LAT', 'LON', 'Elev(m)', 
    'BEGIN', 'END', 'Years',
    'ActualBegin', 'ActualEnd'
]]

# 按国家和年数排序
out = out.sort_values(['Country', 'Years'], ascending=[True, False])

print("\n" + "=" * 80)
print("筛选结果预览:")
print("=" * 80)
print(out[['USAF', 'WBAN', 'Name', 'Country', 'ActualBegin', 'ActualEnd', 'Years']].to_string())

# ================================
# 8. 保存结果
# ================================
output_file = "stations_1950_2010_covered_top_each_country.csv"
out.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print(f"✅ 已保存到: {output_file}")
print("=" * 80)
print(f"  总站点数: {len(out)}")
print(f"  国家数: {out['Country'].nunique()}")
print(f"  筛选条件: 起始年份 <= {WINDOW_START}，结束年份 >= {WINDOW_END}")
print(f"  数据质量: 不要求每年都有数据，只要时间跨度覆盖即可")
print("=" * 80)
