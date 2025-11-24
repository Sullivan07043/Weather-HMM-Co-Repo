# 快速开始指南

## 🚀 5分钟上手

### 步骤 1: 下载数据

```bash
cd Weather-HMM-Co-Repo/data
python download.py
```

### 步骤 2: 处理数据（推荐方式）

```bash
python dataloader.py
```

**选择设置**：
1. 离散化？→ 输入 `1`（是）
2. 组数？→ 输入 `5`（推荐）
3. 年份？→ 输入 `3`（1973-2019，推荐）

等待处理完成（约30-60分钟），输出文件：
- `processed/weather_data_1973_2019_full.csv`
- `processed/normalization_info.txt`

## 📊 输出数据格式

### 离散化后（默认）

```csv
site_id,date,mean_temp,mean_temp_raw,dew_point,dew_point_raw,...
010080-99999,2015-01-01,2,26.0,2,22.4,...
```

**说明**：
- `mean_temp`: 离散值（0-4）← HMM 使用这个
- `mean_temp_raw`: 原始值（26.0°F）← Baseline 可以使用这个

### 主要特征

**12个离散化特征**（值范围0-4）：
- 温度：mean_temp, dew_point, max_temp, min_temp
- 气压：sea_level_pressure, station_pressure
- 能见度：visibility
- 风速：wind_speed, max_wind_speed, wind_gust
- 降水：precipitation, snow_depth

**6个二元特征**（值0/1）：
- fog, rain, snow, hail, thunder, tornado

**12个原始值列**（*_raw）：
- 保存原始连续值，便于需要时使用

## 💻 在代码中使用

### HMM 模型（使用离散值）

```python
import pandas as pd

# 读取离散化数据
df = pd.read_csv('processed/weather_data_1973_2019_full.csv')

# 选择离散特征
features = ['mean_temp', 'dew_point', 'precipitation', 
            'fog', 'rain', 'snow']

# 转换为 HMM 格式
data = {}
for site_id, site_data in df.groupby('site_id'):
    # 每行是一个时间步的观测向量
    observations = site_data[features].values
    data[site_id] = {t: observations[t] for t in range(len(observations))}

# data[site_id][t] 是一个向量，包含该站点在时间t的所有特征
# 所有特征都是离散值（0-4或0-1）
```

### Baseline 模型（使用连续值）

```python
import pandas as pd

# 方法1：重新处理为连续值
from dataloader import GSODDataLoader
loader = GSODDataLoader(discretize=False)  # 禁用离散化
df = loader.process_year_data(2015)
cleaned_df = loader.clean_and_transform(df)

# 方法2：从离散化数据中提取原始值
df = pd.read_csv('processed/weather_data_1973_2019_full.csv')
continuous_features = ['mean_temp_raw', 'dew_point_raw', 
                       'wind_speed_raw', 'precipitation_raw']
# 使用 *_raw 列获取连续值
```

## 🎯 不同场景的配置

### 场景 1：HMM 模型（离散观测）⭐

```python
# 推荐配置
loader = GSODDataLoader(n_bins=5, discretize=True)

# 处理数据
df = loader.process_year_data(2015)
cleaned_df = loader.clean_and_transform(df)
loader.save_processed_data(cleaned_df, 'hmm_data.csv')

# 使用离散特征
features = ['mean_temp', 'dew_point', 'wind_speed', 
            'fog', 'rain', 'snow']  # 所有值都是0-4或0-1
```

### 场景 2：GMM/k-means（连续值）

```python
# 保持连续值
loader = GSODDataLoader(discretize=False)

df = loader.process_year_data(2015)
cleaned_df = loader.clean_and_transform(df)
loader.save_processed_data(cleaned_df, 'continuous_data.csv')

# 使用连续特征
features = ['mean_temp', 'dew_point', 'wind_speed']  # 连续浮点数
```

### 场景 3：快速测试

```python
# 快速测试（少量数据）
loader = GSODDataLoader(n_bins=3, discretize=True)
df = loader.process_year_data(2015, max_stations=20)
cleaned_df = loader.clean_and_transform(df)
loader.save_processed_data(cleaned_df, 'test.csv')
```

## 📁 输出文件

处理完成后会生成：

```
processed/
├── weather_data_1973_2019_full.csv  # 主数据文件
└── normalization_info.txt            # 归一化参数记录
```

### normalization_info.txt 内容示例

```
连续特征归一化和离散化信息
=================================================================

离散化组数: 5
离散化方法: 等宽分箱 (equal-width binning)
组标签: 0, 1, 2, ..., 4

mean_temp:
  原始范围: [-73.50, 92.50]
  唯一值数: 962
  离散化后: 5 组 (0-4)
...
```

## ⚙️ 高级配置

### 自定义组数

```python
# 使用7组（更细粒度）
loader = GSODDataLoader(n_bins=7, discretize=True)

# 或使用3组（粗粒度，快速训练）
loader = GSODDataLoader(n_bins=3, discretize=True)
```

### 处理特定年份

```python
loader = GSODDataLoader(n_bins=5, discretize=True)

# 处理单年
df = loader.process_year_data(2015)

# 处理多年
years = [2015, 2016, 2017, 2018, 2019]
df = loader.process_multiple_years(years)

# 处理所有年份（1901-2019）
years = list(range(1901, 2020))
df = loader.process_multiple_years(years)
```

## 📊 数据验证

### 快速验证脚本

```python
import pandas as pd

df = pd.read_csv('processed/weather_data_1973_2019_full.csv')

print(f"数据行数: {len(df):,}")
print(f"站点数: {df['site_id'].nunique()}")
print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")

# 验证离散值范围
print("\n离散特征值范围:")
for feat in ['mean_temp', 'wind_speed', 'precipitation']:
    values = sorted(df[feat].dropna().unique())
    print(f"  {feat}: {values}")  # 应为 [0, 1, 2, 3, 4]
```

## 🔍 常见问题

### Q: 如何选择组数？

**A:** 
- 小数据（<10k样本）：3-5组
- 中等数据（10k-100k）：5-7组
- 大数据（>100k）：7-10组
- **推荐：5组（平衡精度和速度）**

### Q: 所有特征必须使用相同组数吗？

**A:** 
- 是的，所有连续特征使用相同的组数
- 这简化了模型，便于特征间比较
- 对于HMM，这是标准做法

### Q: 如何获取原始连续值？

**A:**
```python
df = pd.read_csv('processed/weather_data.csv')

# 离散值
discrete_temp = df['mean_temp']  # 0-4

# 原始值
original_temp = df['mean_temp_raw']  # 连续浮点数
```

### Q: 处理需要多长时间？

**A:**
- 单年（部分站点）：<1分钟
- 单年（所有站点）：2-5分钟
- 1973-2019：30-60分钟
- 1901-2019：1-3小时

## 📚 更多文档

- `README.md` - 完整使用文档
- `DISCRETIZATION_GUIDE.md` - 离散化详细指南
- `CHANGES.md` - 更新日志

## 💡 提示

1. **推荐从少量数据开始测试**
2. **HMM 建议使用离散化数据**
3. **Baseline 可以根据模型选择连续或离散**
4. **离散化组数影响模型复杂度，从5开始调整**

---

**版本**: 2.0  
**最后更新**: 2024-11-20

