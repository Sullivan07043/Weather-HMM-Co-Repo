# 数据处理模块

本模块负责处理 NOAA Global Surface Summary of Day (GSOD) 数据集，输出符合项目要求的清洗后 CSV 文件。

## 数据来源

- **数据集**: NOAA Global Surface Summary of Day
- **下载来源**: Kaggle
- **数据路径**: `../../kaggle_data/datasets/noaa/noaa-global-surface-summary-of-the-day/`

## ✨ 新功能：连续特征归一化和离散化

### 归一化和离散化处理

所有连续特征（温度、气压、风速、降水等）都会：
1. **归一化**：将原始值映射到 [0, 1] 区间
2. **离散化**：使用等宽分箱分为 N 组（默认5组）

**公式**：
```
归一化: normalized = (value - min) / (max - min)
离散化: bin = floor(normalized * N_bins)，取值 0 到 N_bins-1
```

**关键特点**：
- ✅ **所有连续特征使用相同的组数**（N_bins）
- ✅ 保留原始值作为 `*_raw` 列
- ✅ 离散化后的值为整数：0, 1, 2, ..., N_bins-1

## 使用方法

### 1. 下载数据

首先运行下载脚本：

```bash
python download.py
```

### 2. 处理数据

运行数据加载器（**交互式**）：

```bash
python dataloader.py
```

**处理选项**：

#### 步骤 1：选择是否离散化
```
1. 是 - 归一化到[0,1]并离散化为N组（推荐用于HMM）
2. 否 - 保持原始连续值
```

#### 步骤 2：选择组数（如果启用离散化）
```
离散化组数 (3-10) [默认: 5]
```
- 推荐值：5组（适合大多数HMM应用）
- 范围：3-10组

#### 步骤 3：选择年份范围
```
1. 快速测试（2015年，前50个站点）
2. 单年完整（2015年，所有站点）
3. 近期数据（1973-2019年，推荐：数据更完整）⭐
4. 全部数据（1901-2019年，119年完整历史数据）
```

### 3. 在 Python 中使用

#### 示例 1：启用离散化（推荐用于 HMM）

```python
from dataloader import GSODDataLoader

# 创建加载器，启用离散化，使用5组
loader = GSODDataLoader(n_bins=5, discretize=True)

# 处理数据
df = loader.process_year_data(2015, max_stations=50)
cleaned_df = loader.clean_and_transform(df)

# 保存
loader.save_processed_data(cleaned_df, 'weather_data_discretized.csv')

# 查看离散化结果
print("mean_temp 离散化后的唯一值:", sorted(cleaned_df['mean_temp'].dropna().unique()))
# 输出: [0, 1, 2, 3, 4]

print("原始值范围:", cleaned_df['mean_temp_raw'].min(), "-", cleaned_df['mean_temp_raw'].max())
# 输出: -20.5 - 85.3
```

#### 示例 2：保持连续值（用于 Baseline）

```python
from dataloader import GSODDataLoader

# 创建加载器，不离散化
loader = GSODDataLoader(discretize=False)

# 处理数据
df = loader.process_year_data(2015)
cleaned_df = loader.clean_and_transform(df)

# 保存
loader.save_processed_data(cleaned_df, 'weather_data_continuous.csv')

# 所有特征保持为原始连续值
```

#### 示例 3：处理完整数据（1973-2019）

```python
from dataloader import GSODDataLoader

# 离散化为8组（更细粒度）
loader = GSODDataLoader(n_bins=8, discretize=True)

# 处理近期数据
years = list(range(1973, 2020))
cleaned_df = loader.process_multiple_years(years)

# 保存
loader.save_processed_data(cleaned_df, 'weather_1973_2019_8bins.csv')
```

## 输出格式

### 启用离散化时的输出

输出的 CSV 文件包含以下列：

#### 1. 标识列（2列）
- `site_id`: 站点ID
- `date`: 观测日期

#### 2. 离散化特征（12列） - **值范围: 0 到 N_bins-1**

| 特征名 | 离散值范围 | 含义 |
|--------|-----------|------|
| `mean_temp` | 0-4 (默认5组) | 平均温度的离散级别 |
| `dew_point` | 0-4 | 露点的离散级别 |
| `max_temp` | 0-4 | 最高温度的离散级别 |
| `min_temp` | 0-4 | 最低温度的离散级别 |
| `sea_level_pressure` | 0-4 | 海平面气压的离散级别 |
| `station_pressure` | 0-4 | 站点气压的离散级别 |
| `visibility` | 0-4 | 能见度的离散级别 |
| `wind_speed` | 0-4 | 风速的离散级别 |
| `max_wind_speed` | 0-4 | 最大风速的离散级别 |
| `wind_gust` | 0-4 | 阵风的离散级别 |
| `precipitation` | 0-4 | 降水量的离散级别 |
| `snow_depth` | 0-4 | 雪深的离散级别 |

#### 3. 原始值列（12列） - **连续浮点数**

每个离散化特征都有对应的 `*_raw` 列保存原始值：
- `mean_temp_raw`: 原始平均温度（华氏度）
- `dew_point_raw`: 原始露点（华氏度）
- ... 等等

#### 4. 天气标记（6列） - **二元 0/1**
- `fog`, `rain`, `snow`, `hail`, `thunder`, `tornado`

### 数据示例

#### 离散化后的数据：
```csv
site_id,date,mean_temp,mean_temp_raw,dew_point,dew_point_raw,...,fog,rain,snow
010080-99999,2015-01-01,2,26.0,2,22.4,...,0,0,1
010080-99999,2015-01-02,2,26.9,2,21.7,...,0,0,1
010080-99999,2015-01-03,3,30.5,2,22.9,...,0,0,1
```

**解读**：
- `mean_temp=2`：温度级别为2（在5个级别中的第3级）
- `mean_temp_raw=26.0`：原始温度为26.0°F

## 归一化信息

处理完成后会生成 `normalization_info.txt` 文件，记录每个特征的归一化参数：

```
连续特征归一化和离散化信息
================================================================================

离散化组数: 5
离散化方法: 等宽分箱 (equal-width binning)
组标签: 0, 1, 2, ..., 4

mean_temp:
  原始范围: [-73.50, 92.50]
  唯一值数: 962
  离散化后: 5 组 (0-4)

wind_speed:
  原始范围: [0.00, 50.40]
  唯一值数: 315
  离散化后: 5 组 (0-4)
...
```

## 数据特点

### 离散化后的数据特性

1. **统一的组数**：所有连续特征都使用相同的组数（N_bins）
2. **整数编码**：离散值为整数 0, 1, 2, ..., N_bins-1
3. **保留原始值**：可通过 `*_raw` 列访问原始连续值
4. **等宽分箱**：每个组的区间宽度相等

### 适用场景

| 场景 | 推荐设置 | 说明 |
|------|----------|------|
| **HMM 模型** | `discretize=True, n_bins=5` | 离散观测值，适合HMM |
| **GMM/k-means** | `discretize=False` | 连续值，适合这些模型 |
| **高精度 HMM** | `discretize=True, n_bins=8-10` | 更细粒度的离散化 |
| **快速原型** | `discretize=True, n_bins=3` | 粗粒度，快速训练 |

## 数据质量

### 缺失率统计（基于测试数据）

- 温度特征: < 1% 缺失率
- 气压特征: 35-60% 缺失率（部分站点不记录海平面气压）
- 风特征: 5-10% 缺失率
- 降水特征: ~18% 缺失率
- 雪深: ~91% 缺失率（大部分地区无雪）
- 天气标记: 0% 缺失率（已编码为 0/1）

### 站点覆盖

- 全球 9000+ 个气象站
- 数据时间范围: 1929 年至今
- 建议使用 1973 年后的数据（更完整）

## 与其他模块的接口

### HMM 模块

使用离散化后的数据：

```python
import pandas as pd

df = pd.read_csv('weather_data_discretized.csv')

# 离散特征（用于 HMM 观测）
discrete_features = ['mean_temp', 'dew_point', 'precipitation', 
                     'wind_speed', 'fog', 'rain', 'snow']

# 转换为 HMM 格式
data = {}
for site_id, site_data in df.groupby('site_id'):
    observations = site_data[discrete_features].values  # 每行是一个观测向量
    data[site_id] = {t: observations[t] for t in range(len(observations))}
```

### Baseline 模块

可选择使用连续值或离散值：

```python
import pandas as pd

# 选项1：使用连续值
df = pd.read_csv('weather_data_continuous.csv')

# 选项2：使用离散值
df = pd.read_csv('weather_data_discretized.csv')

# 选项3：混合使用
df = pd.read_csv('weather_data_discretized.csv')
features = ['mean_temp', 'wind_speed']  # 离散特征
raw_features = ['mean_temp_raw', 'wind_speed_raw']  # 连续特征
```

## 技术细节

### 等宽分箱算法

```python
# 对每个特征 f:
min_f = f.min()
max_f = f.max()

# 归一化到 [0, 1]
normalized = (f - min_f) / (max_f - min_f)

# 分箱
bin_width = 1.0 / n_bins
bin_index = floor(normalized / bin_width)
bin_index = min(bin_index, n_bins - 1)  # 确保最大值也在范围内
```

### 为什么所有特征使用相同组数？

1. **简化模型**：HMM 中所有观测维度使用相同的离散级别数
2. **便于比较**：不同特征的离散值具有相同的语义（0=最低，N-1=最高）
3. **减少参数**：统一的组数减少需要调整的超参数
4. **提高泛化**：避免某些特征过度离散化

## 注意事项

1. **缺失值**：离散化前会跳过 NaN，离散化后 NaN 保持为 NaN
2. **边界情况**：最大值会被分配到最后一组（N_bins-1）
3. **单值特征**：如果某特征的所有值相同，跳过离散化
4. **内存使用**：离散化会创建额外的 `*_raw` 列，增加约2倍内存

## 文件说明

- `download.py`: 数据下载脚本
- `dataloader.py`: 数据加载和处理主模块
- `test_discretization.py`: 离散化功能测试
- `processed/`: 处理后的 CSV 文件存放目录
- `processed/normalization_info.txt`: 归一化参数记录

## 依赖包

```
pandas
numpy
tqdm
```

安装：
```bash
pip install pandas numpy tqdm
```

---

**更新日期**: 2024-11-20  
**版本**: v2.0 - 添加归一化和离散化功能
