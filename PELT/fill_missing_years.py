"""
填充缺失年份并插值
为每个站点创建完整的年份序列（1950-2000），然后对缺失年份进行插值
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# ================================
# 配置参数
# ================================
INPUT_CSV = "weather_1901_2019_yearly_continuous.csv"
OUTPUT_CSV = "weather_1901_2019_yearly_continuous_filled.csv"
YEAR_START = 1950
YEAR_END = 2000

# 需要插值的连续特征（离散化后的特征）
CONTINUOUS_FEATURES = [
    'mean_temp', 'dew_point', 'max_temp', 'min_temp',
    'sea_level_pressure', 'station_pressure',
    'visibility', 'wind_speed', 'max_wind_speed', 'wind_gust',
    'precipitation', 'snow_depth'
]

# 二进制特征（天气事件）
BINARY_FEATURES = ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']

print("=" * 80)
print("填充缺失年份并插值")
print("=" * 80)

# ================================
# 1. 加载数据
# ================================
print(f"\n【步骤1】加载数据: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

print(f"✓ 加载完成: {len(df)} 行数据")
print(f"  站点数: {df['site_id'].nunique()}")
print(f"  年份范围: {df['year'].min()} - {df['year'].max()}")

# ================================
# 2. 统计缺失情况
# ================================
print(f"\n【步骤2】统计 {YEAR_START}-{YEAR_END} 年间的缺失情况...")

df_window = df[(df['year'] >= YEAR_START) & (df['year'] <= YEAR_END)]
expected_years = set(range(YEAR_START, YEAR_END + 1))

missing_stats = []
for site_id in df['site_id'].unique():
    site_data = df_window[df_window['site_id'] == site_id]
    years_present = set(site_data['year'].unique())
    missing_years = expected_years - years_present
    
    if missing_years:
        missing_stats.append({
            'site_id': site_id,
            'years_present': len(years_present),
            'years_missing': len(missing_years),
            'missing_years': sorted(missing_years)
        })

if missing_stats:
    print(f"\n发现 {len(missing_stats)} 个站点在 {YEAR_START}-{YEAR_END} 年间有缺失:")
    for stat in sorted(missing_stats, key=lambda x: x['years_missing'], reverse=True)[:10]:
        print(f"  {stat['site_id']}: 缺失 {stat['years_missing']} 年")
        if stat['years_missing'] <= 5:
            print(f"    缺失年份: {stat['missing_years']}")
else:
    print(f"✓ 所有站点在 {YEAR_START}-{YEAR_END} 年间数据完整")

# ================================
# 3. 为每个站点创建完整的年份骨架
# ================================
print(f"\n【步骤3】为每个站点创建完整的年份骨架 ({YEAR_START}-{YEAR_END})...")

filled_data = []

for site_id in tqdm(df['site_id'].unique(), desc="处理站点"):
    # 获取该站点的所有数据
    site_data = df[df['site_id'] == site_id].copy()
    
    # 创建完整的年份序列
    complete_years = pd.DataFrame({
        'year': range(YEAR_START, YEAR_END + 1),
        'site_id': site_id
    })
    
    # 合并：保留原有数据，添加缺失年份（值为NaN）
    site_filled = complete_years.merge(
        site_data,
        on=['site_id', 'year'],
        how='left'
    )
    
    # 按年份排序
    site_filled = site_filled.sort_values('year').reset_index(drop=True)
    
    # ================================
    # 4. 插值填充缺失年份的特征值
    # ================================
    
    # 4.1 处理date列（为缺失年份创建日期）
    site_filled['date'] = pd.to_datetime(site_filled['year'].astype(str) + '-01-01')
    
    # 4.2 插值连续特征（离散化后的特征）
    for feature in CONTINUOUS_FEATURES:
        if feature not in site_filled.columns:
            continue
        
        # 线性插值
        site_filled[feature] = site_filled[feature].interpolate(
            method='linear',
            limit_direction='both'
        )
        
        # 前向填充（处理开头的NaN）
        site_filled[feature] = site_filled[feature].fillna(method='ffill')
        
        # 后向填充（处理结尾的NaN）
        site_filled[feature] = site_filled[feature].fillna(method='bfill')
        
        # 如果仍有NaN，用该站点的均值填充
        if site_filled[feature].isna().any():
            mean_val = site_filled[feature].mean()
            if pd.notna(mean_val):
                site_filled[feature] = site_filled[feature].fillna(mean_val)
            else:
                # 如果均值也是NaN，用0填充
                site_filled[feature] = site_filled[feature].fillna(0)
        
        # 四舍五入到整数（因为是离散化后的特征）
        site_filled[feature] = site_filled[feature].round().astype(int)
    
    # 4.3 处理原始值列（*_raw）
    raw_features = [col for col in site_filled.columns if col.endswith('_raw')]
    for feature in raw_features:
        # 线性插值
        site_filled[feature] = site_filled[feature].interpolate(
            method='linear',
            limit_direction='both'
        )
        site_filled[feature] = site_filled[feature].fillna(method='ffill')
        site_filled[feature] = site_filled[feature].fillna(method='bfill')
        
        # 用均值填充剩余NaN
        if site_filled[feature].isna().any():
            mean_val = site_filled[feature].mean()
            if pd.notna(mean_val):
                site_filled[feature] = site_filled[feature].fillna(mean_val)
            else:
                site_filled[feature] = site_filled[feature].fillna(0)
    
    # 4.4 处理二进制特征（天气事件）
    for feature in BINARY_FEATURES:
        if feature not in site_filled.columns:
            continue
        
        # 二进制特征：缺失值填充为0（表示事件未发生）
        site_filled[feature] = site_filled[feature].fillna(0).astype(int)
    
    filled_data.append(site_filled)

# ================================
# 5. 合并所有站点数据
# ================================
print("\n【步骤4】合并所有站点数据...")
df_filled = pd.concat(filled_data, ignore_index=True)
df_filled = df_filled.sort_values(['site_id', 'year']).reset_index(drop=True)

print(f"✓ 合并完成: {len(df_filled)} 行数据")

# ================================
# 6. 验证填充结果
# ================================
print(f"\n【步骤5】验证填充结果...")

df_filled_window = df_filled[(df_filled['year'] >= YEAR_START) & (df_filled['year'] <= YEAR_END)]

print(f"\n{YEAR_START}-{YEAR_END} 年间的数据完整性:")
for site_id in df_filled['site_id'].unique():
    site_data = df_filled_window[df_filled_window['site_id'] == site_id]
    years_count = len(site_data)
    expected_count = YEAR_END - YEAR_START + 1
    
    if years_count != expected_count:
        print(f"  ⚠️  {site_id}: {years_count}/{expected_count} 年")
    else:
        print(f"  ✓ {site_id}: {years_count}/{expected_count} 年 - 完整")

# 检查是否有NaN
print(f"\n检查是否有剩余NaN:")
for col in df_filled.columns:
    na_count = df_filled[col].isna().sum()
    if na_count > 0:
        print(f"  ⚠️  {col}: {na_count} 个NaN")

if df_filled.isna().sum().sum() == 0:
    print("  ✓ 无NaN，数据完整")

# ================================
# 7. 保存结果
# ================================
print(f"\n【步骤6】保存结果到: {OUTPUT_CSV}")

# 删除临时的year列（因为已经有date列）
if 'year' in df_filled.columns:
    df_filled = df_filled.drop(columns=['year'])

df_filled.to_csv(OUTPUT_CSV, index=False)

print("=" * 80)
print("✅ 填充完成！")
print("=" * 80)
print(f"  输入文件: {INPUT_CSV}")
print(f"  输出文件: {OUTPUT_CSV}")
print(f"  原始行数: {len(df)}")
print(f"  填充后行数: {len(df_filled)}")
print(f"  新增行数: {len(df_filled) - len(df)}")
print(f"  站点数: {df_filled['site_id'].nunique()}")
print(f"  {YEAR_START}-{YEAR_END} 年间每个站点都有 {YEAR_END - YEAR_START + 1} 年完整数据")
print("=" * 80)

