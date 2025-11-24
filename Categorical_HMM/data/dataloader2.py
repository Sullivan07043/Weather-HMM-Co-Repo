"""
数据加载和处理模块
负责处理 NOAA Global Surface Summary of Day 数据集
输出清洗后的 CSV 文件供 HMM 和 Baseline 模块使用
"""

import os
import gzip
import tarfile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')


class GSODDataLoader:
    """NOAA GSOD 数据加载和处理器"""

    def __init__(
        self,
        data_root=None,
        output_dir=None,
        n_bins=5,
        discretize=True,
        station_list_csv=None,   # 新增：只处理这些站点
        time_aggregation='daily',   # 新增：时间聚合方式 ('daily', 'monthly', 'quarterly', 'yearly')
        detrend=False,   # 新增：是否去趋势
        detrend_method='difference',   # 新增：去趋势方法
        complete_time_series=True,   # 新增：是否确保完整时间序列
    ):
        """
        初始化数据加载器

        Args:
            data_root: 数据根目录路径
            output_dir: 输出目录路径
            n_bins: 离散化的组数（默认5组，所有连续特征使用相同组数）
            discretize: 是否对连续特征进行归一化和离散化（默认True）
            station_list_csv: 站点列表 CSV 路径（包含 USAF、WBAN 等），
                              若为 None，则默认使用 isd-history.csv 全部站点
            time_aggregation: 时间聚合方式
                - 'daily': 保持每日数据（默认）
                - 'monthly': 聚合为月平均
                - 'quarterly': 聚合为季度平均
                - 'yearly': 聚合为年平均
            detrend: 是否对连续特征进行去趋势处理（默认False）
            detrend_method: 去趋势方法
                - 'adaptive': 自适应方法（推荐）- 自动测试多种方法并选择效果最好的
                  包括：线性回归、二次多项式、一阶差分、高通滤波（Butterworth）
                - 'difference': 一阶差分
                - 'linear': 线性去趋势
                - 'moving_average': 移动平均去趋势
                - 'seasonal': 季节性差分（适合有明显季节性的数据）
            complete_time_series: 是否确保每个站点都有完整的时间序列（默认True）
                - True: 为每个站点创建完整日期范围，缺失日期用插值填充
                - False: 只填充已有记录中的缺失值，不补全缺失的日期
        """
        if data_root is None:
            # 自动检测数据路径
            current_file = Path(__file__).resolve()
            proj_root = current_file.parent.parent.parent
            data_root = proj_root / "kaggle_data" / "datasets" / "noaa" / \
                        "noaa-global-surface-summary-of-the-day" / "versions" / "2"

        self.data_root = Path(data_root)
        self.gsod_dir = self.data_root / "gsod_all_years"

        # 站点元数据 CSV：可以是完整 isd-history.csv，也可以是你筛好的那份
        if station_list_csv is not None:
            self.station_info_path = Path(station_list_csv)
        else:
            # 使用 NOAA 提供的完整站点历史记录
            self.station_info_path = self.data_root / "isd-history.csv"

        if output_dir is None:
            output_dir = Path(__file__).parent / "processed"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 离散化参数
        self.n_bins = n_bins
        self.discretize = discretize
        
        # 时间聚合参数
        if time_aggregation not in ['daily', 'monthly', 'quarterly', 'yearly']:
            raise ValueError("time_aggregation 必须是 'daily', 'monthly', 'quarterly' 或 'yearly'")
        self.time_aggregation = time_aggregation
        
        # 去趋势参数
        self.detrend = detrend
        if detrend_method not in ['difference', 'linear', 'moving_average', 'seasonal', 'adaptive']:
            raise ValueError("detrend_method 必须是 'difference', 'linear', 'moving_average', 'seasonal' 或 'adaptive'")
        self.detrend_method = detrend_method
        
        # 完整时间序列参数
        self.complete_time_series = complete_time_series

        # 数据格式定义（基于 GSOD 文档）
        self.missing_values = {
            'TEMP': 9999.9,
            'DEWP': 9999.9,
            'SLP': 9999.9,
            'STP': 9999.9,
            'VISIB': 999.9,
            'WDSP': 999.9,
            'MXSPD': 999.9,
            'GUST': 999.9,
            'MAX': 9999.9,
            'MIN': 9999.9,
            'PRCP': 99.99,
            'SNDP': 999.9
        }

        # 连续特征列表（需要归一化和离散化的特征）
        self.continuous_features = [
            'mean_temp', 'dew_point', 'max_temp', 'min_temp',
            'sea_level_pressure', 'station_pressure',
            'visibility', 'wind_speed', 'max_wind_speed', 'wind_gust',
            'precipitation', 'snow_depth'
        ]
        
        # 二元特征列表（也需要归一化和离散化）
        self.binary_features = ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']

        # 这里先占一个属性，load_station_metadata 里会真正填充
        self.target_site_ids = None

    def load_station_metadata(self):
        """
        加载站点元数据（可以是完整 isd-history.csv，也可以是你筛好的那份）

        返回:
            df_meta: 包含至少 USAF, WBAN, site_id 的 DataFrame
        """
        print("加载站点元数据...")
        df_raw = pd.read_csv(self.station_info_path)

        # 统一列名格式：去空格，去引号，大写
        col_map = {c: c.strip().replace('"', '') for c in df_raw.columns}
        df_raw = df_raw.rename(columns=col_map)
        upper = {c.upper(): c for c in df_raw.columns}

        def get_col(*names):
            for n in names:
                if n in upper:
                    return upper[n]
            return None

        usaf_col = get_col("USAF")
        wban_col = get_col("WBAN")

        if usaf_col is None or wban_col is None:
            raise ValueError("站点列表 CSV 中缺少 USAF 或 WBAN 列")

        name_col = get_col("STATION NAME", "NAME")
        ctry_col = get_col("CTRY", "COUNTRY")
        state_col = get_col("STATE", "ST")
        icao_col = get_col("ICAO")
        lat_col = get_col("LAT")
        lon_col = get_col("LON")
        elev_col = get_col("ELEV(M)", "ELEV", "ELEV(M)")
        begin_col = get_col("BEGIN")
        end_col = get_col("END")
        years_col = get_col("YEARS")

        df_meta = pd.DataFrame()
        df_meta["USAF"] = df_raw[usaf_col].astype(str).str.strip()
        df_meta["WBAN"] = df_raw[wban_col].astype(str).str.strip()

        if name_col:
            df_meta["Name"] = df_raw[name_col].astype(str).str.strip()
        if ctry_col:
            df_meta["Country"] = df_raw[ctry_col].astype(str).str.strip()
        if state_col:
            df_meta["State"] = df_raw[state_col].astype(str).str.strip()
        if icao_col:
            df_meta["ICAO"] = df_raw[icao_col].astype(str).str.strip()
        if lat_col:
            df_meta["LAT"] = df_raw[lat_col]
        if lon_col:
            df_meta["LON"] = df_raw[lon_col]
        if elev_col:
            df_meta["Elev(m)"] = df_raw[elev_col]
        if begin_col:
            df_meta["Begin"] = df_raw[begin_col].astype(str).str.strip()
        if end_col:
            df_meta["End"] = df_raw[end_col].astype(str).str.strip()
        if years_col:
            df_meta["Years"] = df_raw[years_col]

        # 创建 site_id (USAF-WBAN)
        df_meta["site_id"] = df_meta["USAF"].str.zfill(6) + "-" + df_meta["WBAN"].str.zfill(5)

        # 保存目标站点集合，用于后续过滤 tar 内文件
        self.target_site_ids = set(df_meta["site_id"].unique())
        print(f"   站点总数: {len(df_meta)}，目标 site_id 数: {len(self.target_site_ids)}")

        return df_meta

    def parse_gsod_file(self, filepath):
        """
        解析单个 GSOD 文件（使用固定宽度格式）

        Args:
            filepath: 文件路径（可以是 .gz 或 .op 文件）

        Returns:
            DataFrame: 解析后的数据
        """
        try:
            colspecs = [
                (0, 6),     # STN
                (7, 12),    # WBAN
                (14, 22),   # YEARMODA
                (24, 30),   # TEMP
                (31, 33),   # TEMP_COUNT
                (35, 41),   # DEWP
                (42, 44),   # DEWP_COUNT
                (46, 52),   # SLP
                (53, 55),   # SLP_COUNT
                (57, 63),   # STP
                (64, 66),   # STP_COUNT
                (68, 73),   # VISIB
                (74, 76),   # VISIB_COUNT
                (78, 83),   # WDSP
                (84, 86),   # WDSP_COUNT
                (88, 93),   # MXSPD
                (95, 100),  # GUST
                (102, 108), # MAX
                (108, 109), # MAX_FLAG
                (110, 116), # MIN
                (116, 117), # MIN_FLAG
                (118, 123), # PRCP
                (123, 124), # PRCP_FLAG
                (125, 130), # SNDP
                (132, 138)  # FRSHTT
            ]

            names = [
                'STN---', 'WBAN', 'YEARMODA', 'TEMP', 'TEMP_COUNT', 'DEWP', 'DEWP_COUNT',
                'SLP', 'SLP_COUNT', 'STP', 'STP_COUNT', 'VISIB', 'VISIB_COUNT',
                'WDSP', 'WDSP_COUNT', 'MXSPD', 'GUST', 'MAX', 'MAX_FLAG',
                'MIN', 'MIN_FLAG', 'PRCP', 'PRCP_FLAG', 'SNDP', 'FRSHTT'
            ]

            if str(filepath).endswith('.gz'):
                with gzip.open(filepath, 'rt') as f:
                    df = pd.read_fwf(f, colspecs=colspecs, names=names, skiprows=1)
            else:
                df = pd.read_fwf(filepath, colspecs=colspecs, names=names, skiprows=1)

            return df
        except Exception as e:
            print(f"   解析文件失败 {filepath}: {e}")
            return None

    def _filter_members_by_site_id(self, members):
        """
        根据 target_site_ids，用文件名里的 USAF-WBAN 过滤 tar 成员
        """
        if not self.target_site_ids:
            return members

        filtered = []
        for m in members:
            base = os.path.basename(m.name)
            if not (base.endswith(".op") or base.endswith(".op.gz")):
                continue

            # 常见文件名格式: "010010-99999-2010.op.gz"
            parts = base.split('-')
            if len(parts) < 2:
                continue

            stn = parts[0]
            wban_part = parts[1]  # 例如 "99999-2010.op.gz"
            wban = wban_part.split('.')[0].split('_')[0]
            # 去掉可能带的年份部分
            wban = wban.split('-')[0]

            site_id = f"{stn.zfill(6)}-{wban.zfill(5)}"
            if site_id in self.target_site_ids:
                filtered.append(m)

        return filtered

    def process_year_data(self, year, max_stations=None):
        """
        处理指定年份的数据

        Args:
            year: 年份（如 2015）
            max_stations: 最大处理站点数（用于测试，None表示不额外限制）

        Returns:
            DataFrame: 处理后的数据
        """
        print(f"\n处理 {year} 年数据...")

        tar_path = self.gsod_dir / f"gsod_{year}.tar"
        if not tar_path.exists():
            print(f"   文件不存在: {tar_path}")
            return None

        all_data = []

        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()

            # 使用站点列表过滤 tar 成员
            members = self._filter_members_by_site_id(members)
            if not members:
                print("   该年份中没有匹配目标站点的文件")
                return None

            if max_stations is not None:
                members = members[:max_stations]

            print(f"   需要处理的站点文件数: {len(members)}")

            for member in tqdm(members, desc=f"   解析 {year}"):
                if not (member.name.endswith('.op') or member.name.endswith('.op.gz')):
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                try:
                    if member.name.endswith('.gz'):
                        content = gzip.decompress(f.read()).decode('utf-8')
                    else:
                        content = f.read().decode('utf-8')

                    from io import StringIO
                    colspecs = [
                        (0, 6), (7, 12), (14, 22),
                        (24, 30), (31, 33),
                        (35, 41), (42, 44),
                        (46, 52), (53, 55),
                        (57, 63), (64, 66),
                        (68, 73), (74, 76),
                        (78, 83), (84, 86),
                        (88, 93), (95, 100),
                        (102, 108), (108, 109),
                        (110, 116), (116, 117),
                        (118, 123), (123, 124),
                        (125, 130), (132, 138)
                    ]
                    names = [
                        'STN---', 'WBAN', 'YEARMODA', 'TEMP', 'TEMP_COUNT', 'DEWP', 'DEWP_COUNT',
                        'SLP', 'SLP_COUNT', 'STP', 'STP_COUNT', 'VISIB', 'VISIB_COUNT',
                        'WDSP', 'WDSP_COUNT', 'MXSPD', 'GUST', 'MAX', 'MAX_FLAG',
                        'MIN', 'MIN_FLAG', 'PRCP', 'PRCP_FLAG', 'SNDP', 'FRSHTT'
                    ]
                    df = pd.read_fwf(StringIO(content), colspecs=colspecs, names=names, skiprows=1)
                    if len(df) > 0:
                        all_data.append(df)
                except Exception:
                    continue

        if not all_data:
            print("   没有成功解析任何数据")
            return None

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"   成功处理 {len(combined_df)} 行数据，来自 {len(all_data)} 个站点文件")

        return combined_df

    def clean_and_transform(self, df):
        """
        清洗和转换数据
        """
        print("\n清洗和转换数据...")

        df['STN---'] = df['STN---'].astype(str).str.zfill(6)
        df['WBAN'] = df['WBAN'].astype(str).str.zfill(5)
        df['site_id'] = df['STN---'] + '-' + df['WBAN']

        df['YEARMODA'] = df['YEARMODA'].astype(str).str.zfill(8)
        df['date'] = pd.to_datetime(df['YEARMODA'], format='%Y%m%d', errors='coerce')

        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            print(f"   删除 {invalid_dates} 行无效日期数据")
            df = df.dropna(subset=['date'])

        for col, missing_val in self.missing_values.items():
            if col in df.columns:
                df[col] = df[col].replace(missing_val, np.nan)

        if 'FRSHTT' in df.columns:
            frshtt_str = df['FRSHTT'].astype(str).str.zfill(6)
            df['fog'] = frshtt_str.str[0].astype(int)
            df['rain'] = frshtt_str.str[1].astype(int)
            df['snow'] = frshtt_str.str[2].astype(int)
            df['hail'] = frshtt_str.str[3].astype(int)
            df['thunder'] = frshtt_str.str[4].astype(int)
            df['tornado'] = frshtt_str.str[5].astype(int)

        feature_cols = {
            'site_id': 'site_id',
            'date': 'date',
            'TEMP': 'mean_temp',
            'DEWP': 'dew_point',
            'SLP': 'sea_level_pressure',
            'STP': 'station_pressure',
            'VISIB': 'visibility',
            'WDSP': 'wind_speed',
            'MXSPD': 'max_wind_speed',
            'GUST': 'wind_gust',
            'MAX': 'max_temp',
            'MIN': 'min_temp',
            'PRCP': 'precipitation',
            'SNDP': 'snow_depth',
            'fog': 'fog',
            'rain': 'rain',
            'snow': 'snow',
            'hail': 'hail',
            'thunder': 'thunder',
            'tornado': 'tornado'
        }

        available_cols = {k: v for k, v in feature_cols.items() if k in df.columns}
        df_cleaned = df[list(available_cols.keys())].rename(columns=available_cols)

        df_cleaned = df_cleaned.sort_values(['site_id', 'date']).reset_index(drop=True)

        print(f"   清洗完成，保留 {len(df_cleaned)} 行，{len(df_cleaned.columns)} 列特征")

        # ========================================================================
        # 数据处理流程（按正确顺序）
        # ========================================================================
        
        # 步骤1: 时间聚合（如果需要）
        # 说明：先聚合可以减少数据量，提高后续处理效率
        if self.time_aggregation == 'monthly':
            print("\n【步骤1/5】时间聚合: 聚合为月平均...")
            df_cleaned = self.aggregate_to_monthly(df_cleaned)
        elif self.time_aggregation == 'quarterly':
            print("\n【步骤1/5】时间聚合: 聚合为季度平均...")
            df_cleaned = self.aggregate_to_quarterly(df_cleaned)
        elif self.time_aggregation == 'yearly':
            print("\n【步骤1/5】时间聚合: 聚合为年平均...")
            df_cleaned = self.aggregate_to_yearly(df_cleaned)
        else:
            print("\n【步骤1/5】时间聚合: 保持每日数据，跳过聚合")
        
        # 步骤2: 确保完整时间序列（填补缺失的日期记录）
        # 说明：在聚合后进行，确保每个站点在聚合后的时间粒度上都有完整序列
        if self.complete_time_series:
            print("\n【步骤2/5】完整时间序列: 填补缺失的时间点...")
            df_cleaned = self.ensure_complete_time_series(df_cleaned)
        else:
            print("\n【步骤2/5】完整时间序列: 跳过（仅处理已有记录）")
        
        # 步骤3: 插值填充缺失值
        # 说明：在完整时间序列创建后，对所有缺失值进行智能插值
        print("\n【步骤3/5】插值填充: 填充缺失值...")
        df_cleaned = self.fill_missing_values(df_cleaned)
        
        # 步骤4: 去趋势处理（如果启用）
        # 说明：在插值后、离散化前进行，保证去趋势在连续值上操作
        if self.detrend:
            print(f"\n【步骤4/5】去趋势: 使用 {self.detrend_method} 方法...")
            df_cleaned = self.detrend_data(df_cleaned)
        else:
            print("\n【步骤4/5】去趋势: 跳过（保留原始趋势）")
        
        # 步骤5: 归一化和离散化（如果启用）
        # 说明：最后一步，将连续值转换为离散类别
        if self.discretize:
            print(f"\n【步骤5/5】离散化: 归一化并离散化为 {self.n_bins} 组...")
            df_cleaned = self.normalize_and_discretize(df_cleaned)
        else:
            print("\n【步骤5/5】离散化: 跳过（保持连续值）")

        print("\n" + "="*80)
        print("✅ 数据处理流程完成")
        print("="*80)

        return df_cleaned

    def ensure_complete_time_series(self, df):
        """
        确保每个站点都有完整的时间序列记录
        
        问题：有些站点可能连续几年都没有记录（整行缺失）
        解决：为每个站点创建完整的日期范围，缺失的日期用NaN填充
        
        注意：此方法应在时间聚合之后调用，会根据时间聚合方式自动调整频率
        
        Args:
            df: 输入数据框
            
        Returns:
            DataFrame: 包含完整时间序列的数据框
        """
        print("   确保每个站点都有完整的时间序列...")
        
        # 获取全局日期范围
        min_date = df['date'].min()
        max_date = df['date'].max()
        print(f"      数据日期范围: {min_date} 到 {max_date}")
        
        # 根据时间聚合方式确定频率
        freq_map = {
            'daily': 'D',       # 每日
            'monthly': 'MS',    # 每月第一天
            'quarterly': 'QS',  # 每季度第一天
            'yearly': 'YS'      # 每年第一天
        }
        freq = freq_map.get(self.time_aggregation, 'D')
        
        # 创建完整的日期序列
        full_date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
        
        time_unit_map = {
            'daily': '天',
            'monthly': '月',
            'quarterly': '季度',
            'yearly': '年'
        }
        time_unit = time_unit_map.get(self.time_aggregation, '天')
        print(f"      完整时间序列长度: {len(full_date_range)} {time_unit}")
        
        # 获取所有站点列表
        all_sites = df['site_id'].unique()
        print(f"      站点数量: {len(all_sites)}")
        
        # 统计原始数据行数
        original_rows = len(df)
        
        # 为每个站点创建完整的时间序列
        complete_data_list = []
        
        for site_id in tqdm(all_sites, desc="      创建完整时间序列", leave=False):
            # 获取该站点的实际数据
            site_data = df[df['site_id'] == site_id].copy()
            
            # 创建该站点的完整日期框架
            site_complete = pd.DataFrame({
                'site_id': site_id,
                'date': full_date_range
            })
            
            # 将实际数据合并到完整框架中
            # 使用左连接，保留所有日期，缺失的特征值会是NaN
            site_complete = site_complete.merge(
                site_data,
                on=['site_id', 'date'],
                how='left'
            )
            
            complete_data_list.append(site_complete)
        
        # 合并所有站点的数据
        df_complete = pd.concat(complete_data_list, ignore_index=True)
        
        # 按站点和日期排序
        df_complete = df_complete.sort_values(['site_id', 'date']).reset_index(drop=True)
        
        # 统计结果
        new_rows = len(df_complete)
        added_rows = new_rows - original_rows
        
        print(f"      ✅ 完整时间序列创建完成")
        print(f"         原始数据行数: {original_rows:,}")
        print(f"         补全后行数: {new_rows:,}")
        print(f"         新增行数: {added_rows:,} ({added_rows/original_rows*100:.1f}%)")
        print(f"         平均每站点: {new_rows/len(all_sites):.1f} 条记录")
        
        return df_complete

    def fill_missing_values(self, df):
        """
        填充缺失值（按站点进行时间序列插值）
        
        策略：
        1. 对每个站点的连续特征，使用线性插值
        2. 如果开头/结尾仍有NaN，使用前向/后向填充
        3. 如果整列都是NaN，使用全局均值
        4. 二进制特征用0填充（表示事件未发生）
        
        注意：此方法应在 ensure_complete_time_series 之后调用
        """
        print("   填充缺失值...")
        
        df_filled = df.copy()
        
        # 统计原始缺失情况
        original_na_counts = {}
        for feature in self.continuous_features:
            if feature in df_filled.columns:
                original_na_counts[feature] = df_filled[feature].isna().sum()
        
        # 按站点分组处理连续特征
        print("      对连续特征进行时间序列插值...")
        
        # 首先计算每个特征的全局统计量（用于长期缺失的站点）
        global_stats = {}
        for feature in self.continuous_features:
            if feature in df_filled.columns:
                global_stats[feature] = {
                    'mean': df_filled[feature].mean(),
                    'median': df_filled[feature].median(),
                    'std': df_filled[feature].std()
                }
        
        for site_id in tqdm(df_filled['site_id'].unique(), desc="      处理站点", leave=False):
            site_mask = df_filled['site_id'] == site_id
            
            for feature in self.continuous_features:
                if feature not in df_filled.columns:
                    continue
                
                # 获取该站点该特征的数据
                site_feature_data = df_filled.loc[site_mask, feature].copy()
                
                if site_feature_data.isna().all():
                    # 如果该站点该特征全是NaN，用全局均值填充
                    if pd.notna(global_stats[feature]['mean']):
                        df_filled.loc[site_mask, feature] = global_stats[feature]['mean']
                    continue
                
                # 检查缺失比例
                missing_ratio = site_feature_data.isna().sum() / len(site_feature_data)
                
                if missing_ratio > 0.8:
                    # 如果缺失超过80%，使用全局中位数作为基准，再加上站点偏移
                    valid_data = site_feature_data.dropna()
                    if len(valid_data) > 0:
                        site_offset = valid_data.mean() - global_stats[feature]['mean']
                        interpolated = site_feature_data.fillna(
                            global_stats[feature]['mean'] + site_offset
                        )
                    else:
                        interpolated = site_feature_data.fillna(global_stats[feature]['mean'])
                else:
                    # 缺失比例较小，使用多种插值方法
                    
                    # 1. 首先使用三次样条插值（对于平滑的数据效果更好）
                    try:
                        interpolated = site_feature_data.interpolate(
                            method='cubic',
                            limit_direction='both',
                            limit=365  # 最多插值365天
                        )
                    except:
                        # 如果三次样条失败，降级为线性插值
                        interpolated = site_feature_data.interpolate(
                            method='linear',
                            limit_direction='both',
                            limit=365
                        )
                    
                    # 2. 对于仍然缺失的值（超过365天的间隔），使用时间加权的全局均值
                    if interpolated.isna().any():
                        # 使用相同日期的全局平均值（考虑季节性）
                        for idx in interpolated[interpolated.isna()].index:
                            date = df_filled.loc[idx, 'date']
                            # 获取相同月份和日期的所有站点数据
                            same_period_mask = (
                                (df_filled['date'].dt.month == date.month) &
                                (df_filled['date'].dt.day == date.day)
                            )
                            same_period_mean = df_filled.loc[same_period_mask, feature].mean()
                            
                            if pd.notna(same_period_mean):
                                interpolated.loc[idx] = same_period_mean
                            else:
                                # 如果同期数据也缺失，使用全局均值
                                interpolated.loc[idx] = global_stats[feature]['mean']
                    
                    # 3. 前向填充（处理开头的NaN）
                    interpolated = interpolated.ffill()
                    
                    # 4. 后向填充（处理结尾的NaN）
                    interpolated = interpolated.bfill()
                
                # 更新数据
                df_filled.loc[site_mask, feature] = interpolated
        
        # 4. 如果仍有NaN（整个站点都缺失或某些特殊情况），使用全局均值
        for feature in self.continuous_features:
            if feature in df_filled.columns:
                remaining_na = df_filled[feature].isna().sum()
                if remaining_na > 0:
                    global_mean = df_filled[feature].mean()
                    if pd.notna(global_mean):
                        df_filled[feature] = df_filled[feature].fillna(global_mean)
                        print(f"      ⚠️  {feature}: 用全局均值 {global_mean:.2f} 填充 {remaining_na} 个剩余NaN")
                    else:
                        # 如果全局均值都是NaN（整列都缺失），填充为0
                        df_filled[feature] = df_filled[feature].fillna(0)
                        print(f"      ⚠️  {feature}: 全部缺失，用0填充")
        
        # 5. 二进制特征（天气事件）：NaN填充为0（表示未发生）
        binary_features = ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']
        for feature in binary_features:
            if feature in df_filled.columns:
                na_count = df_filled[feature].isna().sum()
                if na_count > 0:
                    df_filled[feature] = df_filled[feature].fillna(0)
        
        # 汇总填充结果
        print(f"\n      缺失值填充结果：")
        for feature in self.continuous_features:
            if feature in df_filled.columns and feature in original_na_counts:
                original_na = original_na_counts[feature]
                remaining_na = df_filled[feature].isna().sum()
                if original_na > 0:
                    print(f"         {feature:25s}: {original_na:6d} → {remaining_na:6d} NaN")
        
        print(f"      ✅ 缺失值填充完成")
        
        return df_filled

    def remove_trends_adaptive(self, site_data, feature_name):
        """
        自适应去趋势：尝试多种方法，选择效果最好的
        
        Args:
            site_data: 单个站点的某个特征的时间序列数据（Series）
            feature_name: 特征名称
            
        Returns:
            tuple: (去趋势后的数据, 趋势信息字典)
        """
        original_series = site_data.dropna()
        if len(original_series) < 2:
            return site_data, None
        
        # 原始趋势检测
        x_orig = np.arange(len(original_series))
        try:
            slope_orig, intercept_orig, r_value_orig, p_value_orig, std_err_orig = linregress(x_orig, original_series)
        except:
            return site_data, None
        
        # 如果没有显著趋势，直接返回
        if p_value_orig >= 0.05:
            return site_data, {
                'method': 'None (no significant trend)',
                'original_slope': slope_orig,
                'original_p_value': p_value_orig,
                'slope_reduction': 0.0,
                'final_slope': slope_orig,
                'final_p_value': p_value_orig
            }
        
        best_detrended_series = original_series
        best_slope_reduction = 0
        best_method = "None"
        best_final_slope = slope_orig
        best_final_p_value = p_value_orig
        
        # 1. 线性回归去趋势
        try:
            detrended_linear = original_series - (slope_orig * x_orig + intercept_orig)
            slope_linear, _, _, p_linear, _ = linregress(x_orig, detrended_linear)
            slope_reduction_linear = 1 - abs(slope_linear / slope_orig) if slope_orig != 0 else 1
            if slope_reduction_linear > best_slope_reduction:
                best_slope_reduction = slope_reduction_linear
                best_detrended_series = detrended_linear
                best_method = "Linear Regression"
                best_final_slope = slope_linear
                best_final_p_value = p_linear
        except:
            pass
        
        # 2. 多项式去趋势 (二次)
        if len(original_series) >= 3:
            try:
                poly_coeffs = np.polyfit(x_orig, original_series, 2)
                poly_trend = np.polyval(poly_coeffs, x_orig)
                detrended_poly = original_series - poly_trend
                slope_poly, _, _, p_poly, _ = linregress(x_orig, detrended_poly)
                slope_reduction_poly = 1 - abs(slope_poly / slope_orig) if slope_orig != 0 else 1
                if slope_reduction_poly > best_slope_reduction:
                    best_slope_reduction = slope_reduction_poly
                    best_detrended_series = detrended_poly
                    best_method = "Polynomial (2nd order)"
                    best_final_slope = slope_poly
                    best_final_p_value = p_poly
            except:
                pass
        
        # 3. 差分去趋势
        if len(original_series) >= 2:
            try:
                detrended_diff = original_series.diff().dropna()
                if not detrended_diff.empty and len(detrended_diff) > 1:
                    x_diff = np.arange(len(detrended_diff))
                    slope_diff, _, _, p_diff, _ = linregress(x_diff, detrended_diff)
                    slope_reduction_diff = 1 - abs(slope_diff / slope_orig) if slope_orig != 0 else 1
                    if slope_reduction_diff > best_slope_reduction:
                        best_slope_reduction = slope_reduction_diff
                        best_detrended_series = detrended_diff
                        best_method = "Differencing"
                        best_final_slope = slope_diff
                        best_final_p_value = p_diff
            except:
                pass
        
        # 4. 高通滤波去趋势 (Butterworth)
        if len(original_series) > 24:  # 至少2年的数据（假设年度数据）
            try:
                fs = 1.0  # 采样频率（每年一个点）
                cutoff_freq = 1/10.0  # 移除10年以上的趋势
                nyquist = 0.5 * fs
                normal_cutoff = cutoff_freq / nyquist
                
                if 0 < normal_cutoff < 1:
                    b, a = butter(2, normal_cutoff, btype='high', analog=False)
                    detrended_filter = pd.Series(filtfilt(b, a, original_series.values), index=original_series.index)
                    slope_filter, _, _, p_filter, _ = linregress(x_orig, detrended_filter)
                    slope_reduction_filter = 1 - abs(slope_filter / slope_orig) if slope_orig != 0 else 1
                    if slope_reduction_filter > best_slope_reduction:
                        best_slope_reduction = slope_reduction_filter
                        best_detrended_series = detrended_filter
                        best_method = "High-pass Filter (Butterworth)"
                        best_final_slope = slope_filter
                        best_final_p_value = p_filter
            except:
                pass
        
        # 将去趋势后的数据重新对齐到原始索引
        result = site_data.copy()
        result.loc[original_series.index] = best_detrended_series.values
        
        trend_info = {
            'method': best_method,
            'original_slope': slope_orig,
            'original_p_value': p_value_orig,
            'slope_reduction': best_slope_reduction,
            'final_slope': best_final_slope,
            'final_p_value': best_final_p_value
        }
        
        return result, trend_info

    def detrend_data(self, df):
        """
        对连续特征进行去趋势处理（按站点分组）
        
        Args:
            df: 输入数据框
            
        Returns:
            DataFrame: 去趋势后的数据
        """
        print(f"\n对连续特征进行去趋势处理（方法: {self.detrend_method}）...")
        
        df_detrended = df.copy()
        detrend_info = {}
        all_trend_stats = []  # 用于收集所有趋势统计信息
        
        # 确保数据按站点和日期排序
        df_detrended = df_detrended.sort_values(['site_id', 'date']).reset_index(drop=True)
        
        for feature in self.continuous_features:
            if feature not in df_detrended.columns:
                continue
            
            # 保存原始值
            df_detrended[f'{feature}_before_detrend'] = df_detrended[feature].copy()
            
            removed_count = 0
            
            # 按站点分组处理
            for site_id in tqdm(df_detrended['site_id'].unique(), desc=f"   去趋势 {feature}", leave=False):
                site_mask = df_detrended['site_id'] == site_id
                site_data = df_detrended.loc[site_mask, feature].copy()
                
                # 跳过全是 NaN 的数据
                if site_data.isna().all():
                    continue
                
                # 应用不同的去趋势方法
                if self.detrend_method == 'adaptive':
                    # 使用自适应方法
                    detrended, trend_stat = self.remove_trends_adaptive(site_data, feature)
                    if trend_stat is not None:
                        trend_stat['site_id'] = site_id
                        trend_stat['feature'] = feature
                        all_trend_stats.append(trend_stat)
                
                elif self.detrend_method == 'difference':
                    # 一阶差分: X(t) - X(t-1)
                    detrended = site_data.diff()
                    # 第一个值会是 NaN，我们保留原值或删除
                    # 这里选择删除第一个观测（因为没有前一个值可以计算差分）
                    removed_count += 1
                    
                elif self.detrend_method == 'linear':
                    # 线性去趋势：拟合线性趋势并减去
                    valid_mask = site_data.notna()
                    if valid_mask.sum() > 1:
                        x = np.arange(len(site_data))
                        valid_x = x[valid_mask]
                        valid_y = site_data[valid_mask].values
                        
                        # 拟合线性回归
                        coeffs = np.polyfit(valid_x, valid_y, 1)
                        trend = np.polyval(coeffs, x)
                        
                        # 减去趋势
                        detrended = site_data - trend
                    else:
                        detrended = site_data
                
                elif self.detrend_method == 'moving_average':
                    # 移动平均去趋势：减去移动平均值
                    # 窗口大小根据时间聚合方式调整
                    window_size_map = {
                        'daily': 30,      # 30天
                        'monthly': 12,    # 12个月
                        'quarterly': 4,   # 4个季度
                        'yearly': 5       # 5年
                    }
                    window = window_size_map.get(self.time_aggregation, 30)
                    
                    # 计算移动平均（中心对齐）
                    ma = site_data.rolling(window=window, center=True, min_periods=1).mean()
                    detrended = site_data - ma
                
                elif self.detrend_method == 'seasonal':
                    # 季节性差分
                    # 周期根据时间聚合方式调整
                    period_map = {
                        'daily': 365,     # 年周期
                        'monthly': 12,    # 年周期
                        'quarterly': 4,   # 年周期
                        'yearly': 1       # 不适用
                    }
                    period = period_map.get(self.time_aggregation, 12)
                    
                    if period > 1:
                        detrended = site_data.diff(periods=period)
                        removed_count += period
                    else:
                        # 年度数据不适合季节性差分，使用一阶差分
                        detrended = site_data.diff()
                        removed_count += 1
                
                # 更新数据
                df_detrended.loc[site_mask, feature] = detrended
            
            # 统计信息
            detrend_info[feature] = {
                'method': self.detrend_method,
                'removed_per_site': removed_count / df_detrended['site_id'].nunique() if df_detrended['site_id'].nunique() > 0 else 0
            }
            
            print(f"      {feature:25s}: 完成")
        
        # 如果使用差分方法，删除产生的 NaN 行
        if self.detrend_method in ['difference', 'seasonal']:
            original_len = len(df_detrended)
            
            # 只删除因去趋势产生的 NaN（所有连续特征都是 NaN 的行）
            continuous_cols_in_df = [f for f in self.continuous_features if f in df_detrended.columns]
            if continuous_cols_in_df:
                df_detrended = df_detrended.dropna(subset=continuous_cols_in_df, how='all')
                removed_rows = original_len - len(df_detrended)
                print(f"\n   因差分产生的 NaN 行已删除: {removed_rows:,} 行")
        
        # 生成详细的趋势统计报告（仅对自适应方法）
        if self.detrend_method == 'adaptive' and all_trend_stats:
            self._generate_trend_report(all_trend_stats)
        
        print(f"   ✅ 去趋势处理完成")
        print(f"   保留数据行数: {len(df_detrended):,}")
        
        return df_detrended
    
    def _generate_trend_report(self, trend_stats):
        """
        生成详细的趋势移除统计报告
        
        Args:
            trend_stats: 趋势统计信息列表
        """
        print("\n" + "="*80)
        print("📊 自适应去趋势效果验证和统计")
        print("="*80)
        
        df_stats = pd.DataFrame(trend_stats)
        
        # 1. 方法使用统计
        print("\n【1】去趋势方法使用统计:")
        method_counts = df_stats['method'].value_counts()
        for method, count in method_counts.items():
            percentage = count / len(df_stats) * 100
            print(f"   {method:35s}: {count:4d} 次 ({percentage:5.1f}%)")
        
        # 2. 按特征统计
        print("\n【2】各特征去趋势效果:")
        print(f"   {'特征名称':<30s} {'处理次数':>8s} {'平均趋势减少':>12s} {'显著趋势数':>10s}")
        print("   " + "-"*70)
        
        for feature in df_stats['feature'].unique():
            feature_data = df_stats[df_stats['feature'] == feature]
            
            # 只统计有显著趋势的（p < 0.05）
            significant = feature_data[feature_data['original_p_value'] < 0.05]
            
            if len(significant) > 0:
                avg_reduction = significant['slope_reduction'].mean() * 100
                print(f"   {feature:<30s} {len(feature_data):>8d} {avg_reduction:>11.1f}% {len(significant):>10d}")
            else:
                print(f"   {feature:<30s} {len(feature_data):>8d} {'N/A':>12s} {len(significant):>10d}")
        
        # 3. 整体效果统计
        print("\n【3】整体去趋势效果:")
        significant_trends = df_stats[df_stats['original_p_value'] < 0.05]
        
        if len(significant_trends) > 0:
            print(f"   总处理次数: {len(df_stats)}")
            print(f"   显著趋势数 (p<0.05): {len(significant_trends)} ({len(significant_trends)/len(df_stats)*100:.1f}%)")
            print(f"   平均原始斜率: {significant_trends['original_slope'].abs().mean():.6f}")
            print(f"   平均最终斜率: {significant_trends['final_slope'].abs().mean():.6f}")
            print(f"   平均趋势减少率: {significant_trends['slope_reduction'].mean()*100:.1f}%")
            
            # 效果分级
            excellent = (significant_trends['slope_reduction'] >= 0.9).sum()
            good = ((significant_trends['slope_reduction'] >= 0.7) & (significant_trends['slope_reduction'] < 0.9)).sum()
            moderate = ((significant_trends['slope_reduction'] >= 0.5) & (significant_trends['slope_reduction'] < 0.7)).sum()
            poor = (significant_trends['slope_reduction'] < 0.5).sum()
            
            print(f"\n   效果分级:")
            print(f"      优秀 (≥90%减少): {excellent} ({excellent/len(significant_trends)*100:.1f}%)")
            print(f"      良好 (70-90%):   {good} ({good/len(significant_trends)*100:.1f}%)")
            print(f"      中等 (50-70%):   {moderate} ({moderate/len(significant_trends)*100:.1f}%)")
            print(f"      较差 (<50%):     {poor} ({poor/len(significant_trends)*100:.1f}%)")
        else:
            print(f"   未检测到显著趋势 (所有 p-value ≥ 0.05)")
        
        # 4. 保存详细报告到文件
        report_path = self.output_dir / "trend_removal_detailed_report.csv"
        df_stats.to_csv(report_path, index=False)
        print(f"\n   ✅ 详细趋势统计已保存到: {report_path}")
        
        # 5. 保存文本摘要
        summary_path = self.output_dir / "trend_removal_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("自适应去趋势效果验证和统计报告\n")
            f.write("="*80 + "\n\n")
            f.write(f"去趋势方法: 自适应选择（线性/多项式/差分/高通滤波）\n")
            f.write(f"显著性水平: p < 0.05\n")
            f.write(f"处理日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("="*80 + "\n")
            f.write("1. 方法使用统计\n")
            f.write("="*80 + "\n")
            for method, count in method_counts.items():
                percentage = count / len(df_stats) * 100
                f.write(f"{method:35s}: {count:4d} 次 ({percentage:5.1f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("2. 各特征去趋势效果\n")
            f.write("="*80 + "\n")
            for feature in df_stats['feature'].unique():
                feature_data = df_stats[df_stats['feature'] == feature]
                significant = feature_data[feature_data['original_p_value'] < 0.05]
                
                f.write(f"\n{feature}:\n")
                f.write(f"  总处理次数: {len(feature_data)}\n")
                f.write(f"  显著趋势数: {len(significant)}\n")
                
                if len(significant) > 0:
                    f.write(f"  平均趋势减少: {significant['slope_reduction'].mean()*100:.1f}%\n")
                    f.write(f"  平均原始斜率: {significant['original_slope'].abs().mean():.6f}\n")
                    f.write(f"  平均最终斜率: {significant['final_slope'].abs().mean():.6f}\n")
                    
                    # 列出使用的方法
                    methods_used = significant['method'].value_counts()
                    f.write(f"  使用的方法:\n")
                    for method, count in methods_used.items():
                        f.write(f"    - {method}: {count} 次\n")
            
            if len(significant_trends) > 0:
                f.write("\n" + "="*80 + "\n")
                f.write("3. 整体效果统计\n")
                f.write("="*80 + "\n")
                f.write(f"总处理次数: {len(df_stats)}\n")
                f.write(f"显著趋势数 (p<0.05): {len(significant_trends)} ({len(significant_trends)/len(df_stats)*100:.1f}%)\n")
                f.write(f"平均趋势减少率: {significant_trends['slope_reduction'].mean()*100:.1f}%\n\n")
                f.write(f"效果分级:\n")
                f.write(f"  优秀 (≥90%): {excellent} ({excellent/len(significant_trends)*100:.1f}%)\n")
                f.write(f"  良好 (70-90%): {good} ({good/len(significant_trends)*100:.1f}%)\n")
                f.write(f"  中等 (50-70%): {moderate} ({moderate/len(significant_trends)*100:.1f}%)\n")
                f.write(f"  较差 (<50%): {poor} ({poor/len(significant_trends)*100:.1f}%)\n")
        
        print(f"   ✅ 趋势摘要已保存到: {summary_path}")
        print("="*80)

    def normalize_and_discretize(self, df):
        """
        对连续特征和二元特征进行归一化和离散化
        """
        print("\n归一化和离散化特征...")
        print(f"   离散化组数: {self.n_bins}")

        df_processed = df.copy()
        normalization_info = {}

        # 处理连续特征
        print("\n   处理连续特征:")
        for feature in self.continuous_features:
            if feature not in df_processed.columns:
                continue

            valid_mask = df_processed[feature].notna()
            valid_data = df_processed.loc[valid_mask, feature]

            if len(valid_data) == 0:
                print(f"      {feature}: 全部缺失，跳过")
                continue

            min_val = valid_data.min()
            max_val = valid_data.max()

            if max_val > min_val:
                normalized = (valid_data - min_val) / (max_val - min_val)
                discretized = pd.cut(
                    normalized,
                    bins=self.n_bins,
                    labels=range(self.n_bins),
                    include_lowest=True
                )
                discretized = discretized.astype(float)

                df_processed[f'{feature}_raw'] = df_processed[feature]
                df_processed.loc[valid_mask, feature] = discretized

                normalization_info[feature] = {
                    'type': 'continuous',
                    'min': min_val,
                    'max': max_val,
                    'bins': self.n_bins,
                    'unique_values': len(valid_data.unique())
                }

                print(f"      {feature:25s}: [{min_val:.2f}, {max_val:.2f}] -> [0, {self.n_bins-1}]")
            else:
                print(f"      {feature}: 值范围为 0，跳过离散化")

        # 处理二元特征
        print("\n   处理二元特征:")
        for feature in self.binary_features:
            if feature not in df_processed.columns:
                continue

            valid_mask = df_processed[feature].notna()
            valid_data = df_processed.loc[valid_mask, feature]

            if len(valid_data) == 0:
                print(f"      {feature}: 全部缺失，跳过")
                continue

            min_val = valid_data.min()
            max_val = valid_data.max()

            # 归一化到 [0, 1]
            if max_val > min_val:
                normalized = (valid_data - min_val) / (max_val - min_val)
            else:
                # 如果所有值相同，保持为0
                normalized = pd.Series(0, index=valid_data.index)

            # 离散化为 n_bins 组
            discretized = pd.cut(
                normalized,
                bins=self.n_bins,
                labels=range(self.n_bins),
                include_lowest=True
            )
            discretized = discretized.astype(float)

            df_processed[f'{feature}_raw'] = df_processed[feature]
            df_processed.loc[valid_mask, feature] = discretized

            normalization_info[feature] = {
                'type': 'binary',
                'min': min_val,
                'max': max_val,
                'bins': self.n_bins,
                'unique_values': len(valid_data.unique())
            }

            print(f"      {feature:25s}: [{min_val:.2f}, {max_val:.2f}] -> [0, {self.n_bins-1}]")

        info_path = self.output_dir / "normalization_info.txt"
        with open(info_path, 'w') as f:
            f.write("特征归一化和离散化信息\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"离散化组数: {self.n_bins}\n")
            f.write("离散化方法: 等宽分箱\n")
            f.write(f"组标签: 0, 1, ..., {self.n_bins-1}\n\n")

            f.write("连续特征:\n")
            f.write("-" * 40 + "\n")
            for feature, info in normalization_info.items():
                if info['type'] == 'continuous':
                    f.write(f"\n{feature}:\n")
                    f.write(f"  原始范围: [{info['min']:.2f}, {info['max']:.2f}]\n")
                    f.write(f"  唯一值数: {info['unique_values']}\n")
                    f.write(f"  离散化后: {info['bins']} 组 (0-{info['bins']-1})\n")

            f.write("\n二元特征:\n")
            f.write("-" * 40 + "\n")
            for feature, info in normalization_info.items():
                if info['type'] == 'binary':
                    f.write(f"\n{feature}:\n")
                    f.write(f"  原始范围: [{info['min']:.2f}, {info['max']:.2f}]\n")
                    f.write(f"  唯一值数: {info['unique_values']}\n")
                    f.write(f"  离散化后: {info['bins']} 组 (0-{info['bins']-1})\n")

        print(f"\n   归一化信息已保存到: {info_path}")
        continuous_count = sum(1 for info in normalization_info.values() if info['type'] == 'continuous')
        binary_count = sum(1 for info in normalization_info.values() if info['type'] == 'binary')
        print(f"   离散化完成: {continuous_count} 个连续特征, {binary_count} 个二元特征")

        return df_processed

    def aggregate_to_quarterly(self, df):
        """
        将每日数据聚合为季度平均数据
        
        季度划分：
        - Q1: 1-3月（冬春）
        - Q2: 4-6月（春夏）
        - Q3: 7-9月（夏秋）
        - Q4: 10-12月（秋冬）
        
        Args:
            df: 包含每日观测的 DataFrame
            
        Returns:
            DataFrame: 季度平均数据
        """
        print("   聚合为季度平均数据...")
        
        # 提取年份和季度信息
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter  # Pandas 自动计算季度 (1-4)
        
        # 定义聚合规则
        agg_dict = {}
        
        # 连续特征：计算平均值
        for feature in self.continuous_features:
            if feature in df.columns:
                agg_dict[feature] = 'mean'
                # 如果有原始值列，也计算平均
                if f'{feature}_raw' in df.columns:
                    agg_dict[f'{feature}_raw'] = 'mean'
        
        # 二进制特征（天气事件）：计算发生天数占比
        binary_features = ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']
        for feature in binary_features:
            if feature in df.columns:
                agg_dict[feature] = 'mean'
        
        # 日期：取该季度的第一天作为代表
        agg_dict['date'] = 'first'
        
        # 按站点、年、季度分组聚合
        print(f"      按站点和季度分组聚合...")
        df_quarterly = df.groupby(['site_id', 'year', 'quarter'], as_index=False).agg(agg_dict)
        
        # 将日期设置为每季度第一个月的1号
        # Q1: 1月1日, Q2: 4月1日, Q3: 7月1日, Q4: 10月1日
        quarter_to_month = {1: '01', 2: '04', 3: '07', 4: '10'}
        df_quarterly['date'] = pd.to_datetime(
            df_quarterly['year'].astype(str) + '-' + 
            df_quarterly['quarter'].map(quarter_to_month) + '-01'
        )
        
        # 删除临时的 year 和 quarter 列
        df_quarterly = df_quarterly.drop(columns=['year', 'quarter'])
        
        # 重新排序
        df_quarterly = df_quarterly.sort_values(['site_id', 'date']).reset_index(drop=True)
        
        print(f"      聚合完成:")
        print(f"         原始数据: {len(df):,} 行")
        print(f"         季度数据: {len(df_quarterly):,} 行")
        print(f"         平均每站点: {len(df_quarterly) / df_quarterly['site_id'].nunique():.1f} 个季度")
        
        return df_quarterly

    def aggregate_to_monthly(self, df):
        """
        将每日数据聚合为月平均数据
        
        Args:
            df: 包含每日观测的 DataFrame
            
        Returns:
            DataFrame: 月平均数据
        """
        print("   聚合为月平均数据...")
        
        # 提取年-月信息
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # 定义聚合规则
        agg_dict = {}
        
        # 连续特征：计算平均值
        for feature in self.continuous_features:
            if feature in df.columns:
                agg_dict[feature] = 'mean'
                # 如果有原始值列，也计算平均
                if f'{feature}_raw' in df.columns:
                    agg_dict[f'{feature}_raw'] = 'mean'
        
        # 二进制特征（天气事件）：计算发生天数占比或总次数
        binary_features = ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']
        for feature in binary_features:
            if feature in df.columns:
                # 计算该月该事件发生的天数占比（0-1之间）
                agg_dict[feature] = 'mean'
        
        # 日期：取该月的第一天作为代表
        agg_dict['date'] = 'first'
        
        # 按站点、年、月分组聚合
        print(f"      按站点和月份分组聚合...")
        df_monthly = df.groupby(['site_id', 'year', 'month'], as_index=False).agg(agg_dict)
        
        # 将日期设置为每月1号
        df_monthly['date'] = pd.to_datetime(
            df_monthly['year'].astype(str) + '-' + 
            df_monthly['month'].astype(str).str.zfill(2) + '-01'
        )
        
        # 删除临时的 year 和 month 列
        df_monthly = df_monthly.drop(columns=['year', 'month'])
        
        # 重新排序
        df_monthly = df_monthly.sort_values(['site_id', 'date']).reset_index(drop=True)
        
        print(f"      聚合完成:")
        print(f"         原始数据: {len(df):,} 行")
        print(f"         月度数据: {len(df_monthly):,} 行")
        print(f"         平均每站点: {len(df_monthly) / df_monthly['site_id'].nunique():.1f} 个月")
        
        return df_monthly

    def aggregate_to_yearly(self, df):
        """将每日数据聚合为年平均数据"""
        print("   聚合为年平均数据...")

        df['year'] = df['date'].dt.year

        agg_dict = {}
        for feature in self.continuous_features:
            if feature in df.columns:
                agg_dict[feature] = 'mean'
                if f'{feature}_raw' in df.columns:
                    agg_dict[f'{feature}_raw'] = 'mean'

        binary_features = ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']
        for feature in binary_features:
            if feature in df.columns:
                agg_dict[feature] = 'mean'

        agg_dict['date'] = 'first'

        print("      按站点和年份分组聚合...")
        df_yearly = df.groupby(['site_id', 'year'], as_index=False).agg(agg_dict)

        df_yearly['date'] = pd.to_datetime(df_yearly['year'].astype(str) + '-01-01')

        df_yearly = df_yearly.drop(columns=['year'])
        df_yearly = df_yearly.sort_values(['site_id', 'date']).reset_index(drop=True)

        print(f"      聚合完成:")
        print(f"         原始数据: {len(df):,} 行")
        print(f"         年度数据: {len(df_yearly):,} 行")
        print(f"         平均每站点: {len(df_yearly) / df_yearly['site_id'].nunique():.1f} 年")

        return df_yearly

    def save_processed_data(self, df, filename='processed_weather_data.csv'):
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        
        # 时间粒度显示
        time_granularity_map = {
            'daily': '每日观测',
            'monthly': '月平均',
            'quarterly': '季度平均',
            'yearly': '年平均'
        }
        
        print(f"\n✅ 数据已保存到: {output_path}")
        print(f"   行数: {len(df):,}")
        print(f"   列数: {len(df.columns)}")
        print(f"   站点数: {df['site_id'].nunique()}")
        print(f"   日期范围: {df['date'].min()} 到 {df['date'].max()}")
        print(f"   时间粒度: {time_granularity_map[self.time_aggregation]}")
        print(f"   完整时间序列: {'是' if self.complete_time_series else '否'}")
        
        if self.detrend:
            detrend_method_map = {
                'adaptive': '自适应方法（自动选择最佳）',
                'difference': '一阶差分',
                'linear': '线性去趋势',
                'moving_average': '移动平均去趋势',
                'seasonal': '季节性差分'
            }
            print(f"   去趋势方法: {detrend_method_map[self.detrend_method]}")
            before_detrend_cols = [c for c in df.columns if c.endswith('_before_detrend')]
            print(f"   去趋势前的原始值列（*_before_detrend）: {len(before_detrend_cols)} 个")

        if self.discretize:
            raw_cols = [c for c in df.columns if c.endswith('_raw')]
            print(f"   离散化特征数: {len(self.continuous_features)}")
            print(f"   离散化组数: {self.n_bins}")
            print(f"   原始值列（*_raw）: {len(raw_cols)} 个")

    def process_multiple_years(self, years, max_stations_per_year=None):
        all_years_data = []

        for year in years:
            df = self.process_year_data(year, max_stations=max_stations_per_year)
            if df is not None:
                all_years_data.append(df)

        if not all_years_data:
            print("没有成功处理任何年份的数据")
            return None

        print(f"\n合并 {len(all_years_data)} 个年份的数据...")
        combined_df = pd.concat(all_years_data, ignore_index=True)
        cleaned_df = self.clean_and_transform(combined_df)
        return cleaned_df

    def generate_summary_statistics(self, df):
        print("\n数据摘要统计:")
        print("=" * 60)

        print(f"\n基本信息:")
        print(f"  总行数: {len(df):,}")
        print(f"  总站点数: {df['site_id'].nunique():,}")
        print(f"  日期范围: {df['date'].min()} 到 {df['date'].max()}")

        obs_per_site = df.groupby('site_id').size()
        print(f"\n每个站点的观测天数:")
        print(f"  平均: {obs_per_site.mean():.1f} 天")
        print(f"  中位数: {obs_per_site.median():.1f} 天")
        print(f"  最小: {obs_per_site.min()} 天")
        print(f"  最大: {obs_per_site.max()} 天")

        print(f"\n特征缺失率:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_rate = df[col].isna().sum() / len(df) * 100
            print(f"  {col}: {missing_rate:.2f}%")

        print("=" * 60)


def main():
    print("=" * 80)
    print("🌤️  NOAA GSOD 数据处理工具")
    print("=" * 80)

    # 1. 时间粒度选择
    print("\n【1/4】选择时间粒度：")
    print("1. 每日观测（Daily）- 保持原始每日数据")
    print("2. 月平均（Monthly）- 将每日数据聚合为月平均（数据量约减少 1/30）")
    print("3. 季度平均（Quarterly）- 将每日数据聚合为季度平均（数据量约减少 1/90）")
    print("4. 年平均（Yearly）- 将每日数据聚合为年平均（数据量约减少 1/365）")
    
    time_choice = input("\n请选择 (1/2/3/4) [默认: 1]: ").strip() or "1"
    
    time_aggregation_map = {
        "1": "daily",
        "2": "monthly",
        "3": "quarterly",
        "4": "yearly"
    }
    time_aggregation = time_aggregation_map.get(time_choice, "daily")
    
    time_display = {
        "daily": "每日观测",
        "monthly": "月平均聚合",
        "quarterly": "季度平均聚合",
        "yearly": "年平均聚合"
    }
    print(f"✓ 已选择: {time_display[time_aggregation]}")

    # 2. 去趋势选择
    print("\n【2/5】是否对连续特征进行去趋势（Detrend）处理？")
    print("说明：去趋势可以移除数据中的长期趋势，使数据更加平稳（stationary）")
    print("1. 否 - 保留原始趋势（默认）")
    print("2. 是 - 移除趋势")
    
    detrend_choice = input("\n请选择 (1/2) [默认: 1]: ").strip() or "1"
    detrend = (detrend_choice == "2")
    
    detrend_method = 'adaptive'  # 默认方法改为自适应
    if detrend:
        print("\n   选择去趋势方法：")
        print("   1. 自适应方法（Adaptive）- 🌟推荐：自动选择最佳方法（线性/多项式/差分/高通滤波）")
        print("   2. 一阶差分（First Difference）- 计算相邻值差异")
        print("   3. 线性去趋势（Linear Detrend）- 移除线性趋势")
        print("   4. 移动平均去趋势（Moving Average）- 减去移动平均值")
        print("   5. 季节性差分（Seasonal Difference）- 适合有明显季节性的数据")
        
        method_choice = input("\n   请选择去趋势方法 (1/2/3/4/5) [默认: 1-自适应]: ").strip() or "1"
        method_map = {
            "1": "adaptive",
            "2": "difference",
            "3": "linear",
            "4": "moving_average",
            "5": "seasonal"
        }
        detrend_method = method_map.get(method_choice, "adaptive")
        
        method_display = {
            "adaptive": "自适应方法（自动选择最佳）",
            "difference": "一阶差分",
            "linear": "线性去趋势",
            "moving_average": "移动平均去趋势",
            "seasonal": "季节性差分"
        }
        print(f"   ✓ 将使用 {method_display[detrend_method]} 方法")
    else:
        print("   ✓ 保留原始趋势")

    # 3. 缺失值处理策略
    print("\n【3/5】缺失值处理策略：")
    print("说明：部分站点可能连续几年都没有数据记录")
    print("1. 完整填充（推荐）- 为每个站点创建完整时间序列，使用智能插值填补所有缺失")
    print("2. 仅填充已有记录 - 只对已有记录中的缺失值进行填充，不补全缺失的日期")
    
    complete_choice = input("\n请选择 (1/2) [默认: 1]: ").strip() or "1"
    complete_time_series = (complete_choice == "1")
    
    if complete_time_series:
        print("   ✓ 将创建完整时间序列并智能填充所有缺失值")
    else:
        print("   ✓ 仅填充已有记录中的缺失值")

    # 4. 离散化选择
    print("\n【4/5】是否对连续特征进行归一化和离散化？")
    print("1. 是 - 归一化到 [0,1] 并离散化为 N 组（推荐用于 HMM）")
    print("2. 否 - 保持原始连续值")

    discretize_choice = input("\n请选择 (1/2) [默认: 1]: ").strip() or "1"
    discretize = (discretize_choice == "1")

    n_bins = 5
    if discretize:
        n_bins_input = input(f"   离散化组数 (3-10) [默认: 5]: ").strip()
        if n_bins_input.isdigit():
            n_bins = int(n_bins_input)
            n_bins = max(3, min(10, n_bins))
        print(f"   ✓ 将使用 {n_bins} 组进行离散化")
    else:
        print("   ✓ 保持连续值")

    # 5. 站点选择
    print("\n【5/5】选择要处理的站点：")
    station_csv_input = input(
        "   站点列表 CSV 路径（包含 USAF, WBAN；留空使用默认站点列表）: "
    ).strip() or None
    
    if station_csv_input:
        print(f"   ✓ 将使用自定义站点列表: {station_csv_input}")
    else:
        print("   ✓ 使用默认站点列表")

    loader = GSODDataLoader(
        n_bins=n_bins,
        discretize=discretize,
        station_list_csv=station_csv_input,
        time_aggregation=time_aggregation,
        detrend=detrend,
        detrend_method=detrend_method,
        complete_time_series=complete_time_series
    )

    # 加载站点元数据（这里会生成 target_site_ids）
    print("\n" + "=" * 80)
    loader.load_station_metadata()

    print("\n检测可用年份...")
    available_years = []
    for tar_file in sorted(loader.gsod_dir.glob("gsod_*.tar")):
        year = int(tar_file.stem.split('_')[1])
        available_years.append(year)

    print(f"   找到 {len(available_years)} 个年份: {available_years[0]} - {available_years[-1]}")

    # 6. 年份范围选择
    print("\n【6/6】选择处理哪些年份的数据：")
    print("1. 快速测试（2015年，前50个目标站点文件）")
    print("2. 单年完整（2015年，所有目标站点文件）")
    print("3. 近期数据（1973-2019 年，所有目标站点）")
    print("4. 全部数据（1901-2019 年，所有目标站点）")

    choice = input(f"\n请选择 (1/2/3/4) [默认: 3]: ").strip() or "3"
    
    print("\n" + "=" * 80)
    print("🚀 开始处理数据...")
    print("=" * 80)

    # 根据选项生成文件名后缀
    time_suffix = time_aggregation  # 'daily', 'monthly', 'quarterly', 'yearly'
    detrend_suffix = f"detrend_{detrend_method}" if detrend else "raw"
    discrete_suffix = f"bins{n_bins}" if discretize else "continuous"
    
    if choice == "1":
        print("\n📝 快速测试模式...")
        df = loader.process_year_data(2015, max_stations=50)
        if df is not None:
            cleaned_df = loader.clean_and_transform(df)
            filename = f'weather_2015_sample_{time_suffix}_{detrend_suffix}_{discrete_suffix}.csv'
            loader.save_processed_data(cleaned_df, filename)
            loader.generate_summary_statistics(cleaned_df)

    elif choice == "2":
        print("\n📝 处理 2015 年完整数据（仅目标站点）...")
        df = loader.process_year_data(2015)
        if df is not None:
            cleaned_df = loader.clean_and_transform(df)
            filename = f'weather_2015_full_{time_suffix}_{detrend_suffix}_{discrete_suffix}.csv'
            loader.save_processed_data(cleaned_df, filename)
            loader.generate_summary_statistics(cleaned_df)

    elif choice == "3":
        print("\n📝 处理近期数据（1973-2019，目标站点）...")
        years = [y for y in available_years if y >= 1973]
        print(f"   将处理 {len(years)} 个年份")
        cleaned_df = loader.process_multiple_years(years)
        if cleaned_df is not None:
            filename = f'weather_1973_2019_{time_suffix}_{detrend_suffix}_{discrete_suffix}.csv'
            loader.save_processed_data(cleaned_df, filename)
            loader.generate_summary_statistics(cleaned_df)

    elif choice == "4":
        print("\n📝 处理全部历史数据（1901-2019，目标站点）...")
        print(f"   ⚠️  警告：全部数据量很大，可能需要较长时间")
        confirm = input("   确认继续？(yes/no) [no]: ").strip().lower()
        if confirm == "yes":
            cleaned_df = loader.process_multiple_years(available_years)
            if cleaned_df is not None:
                filename = f'weather_1901_2019_{time_suffix}_{detrend_suffix}_{discrete_suffix}.csv'
                loader.save_processed_data(cleaned_df, filename)
                loader.generate_summary_statistics(cleaned_df)
        else:
            print("   ✗ 已取消处理全部数据")

    print("\n" + "=" * 80)
    print("✅ 数据处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
