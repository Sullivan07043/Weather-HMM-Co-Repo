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
import warnings
warnings.filterwarnings('ignore')


class GSODDataLoader:
    """NOAA GSOD 数据加载和处理器"""
    
    def __init__(self, data_root=None, output_dir=None, n_bins=5, discretize=True):
        """
        初始化数据加载器
        
        Args:
            data_root: 数据根目录路径
            output_dir: 输出目录路径
            n_bins: 离散化的组数（默认5组，所有连续特征使用相同组数）
            discretize: 是否对连续特征进行归一化和离散化（默认True）
        """
        if data_root is None:
            # 自动检测数据路径
            current_file = Path(__file__).resolve()
            proj_root = current_file.parent.parent.parent
            data_root = proj_root / "kaggle_data" / "datasets" / "noaa" / \
                       "noaa-global-surface-summary-of-the-day" / "versions" / "2"
        
        self.data_root = Path(data_root)
        self.gsod_dir = self.data_root / "gsod_all_years"
        self.station_info_path = self.data_root / "isd-history.csv"
        
        if output_dir is None:
            output_dir = Path(__file__).parent / "processed"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 离散化参数
        self.n_bins = n_bins
        self.discretize = discretize
        
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
    
    def load_station_metadata(self):
        """
        加载站点元数据
        
        Returns:
            DataFrame: 包含站点信息的数据框
        """
        print("📋 加载站点元数据...")
        df = pd.read_csv(self.station_info_path)
        
        # 清理列名
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # 创建 site_id (USAF-WBAN)
        df['site_id'] = df['USAF'].astype(str).str.zfill(6) + '-' + \
                        df['WBAN'].astype(str).str.zfill(5)
        
        print(f"   ✅ 加载了 {len(df)} 个站点的元数据")
        return df
    
    def parse_gsod_file(self, filepath):
        """
        解析单个 GSOD 文件（使用固定宽度格式）
        
        Args:
            filepath: 文件路径（可以是 .gz 或 .op 文件）
            
        Returns:
            DataFrame: 解析后的数据
        """
        try:
            # GSOD 数据格式定义（基于官方文档）
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

            names = ['STN---', 'WBAN', 'YEARMODA', 'TEMP', 'TEMP_COUNT', 'DEWP', 'DEWP_COUNT',
                     'SLP', 'SLP_COUNT', 'STP', 'STP_COUNT', 'VISIB', 'VISIB_COUNT',
                     'WDSP', 'WDSP_COUNT', 'MXSPD', 'GUST', 'MAX', 'MAX_FLAG',
                     'MIN', 'MIN_FLAG', 'PRCP', 'PRCP_FLAG', 'SNDP', 'FRSHTT']
            
            # 根据文件扩展名选择打开方式
            if str(filepath).endswith('.gz'):
                with gzip.open(filepath, 'rt') as f:
                    df = pd.read_fwf(f, colspecs=colspecs, names=names, skiprows=1)
            else:
                df = pd.read_fwf(filepath, colspecs=colspecs, names=names, skiprows=1)
            
            return df
        except Exception as e:
            print(f"   ⚠️  解析文件失败 {filepath}: {e}")
            return None
    
    def process_year_data(self, year, max_stations=None):
        """
        处理指定年份的数据
        
        Args:
            year: 年份（如 2015）
            max_stations: 最大处理站点数（用于测试，None表示处理全部）
            
        Returns:
            DataFrame: 处理后的数据
        """
        print(f"\n📅 处理 {year} 年数据...")
        
        tar_path = self.gsod_dir / f"gsod_{year}.tar"
        if not tar_path.exists():
            print(f"   ❌ 文件不存在: {tar_path}")
            return None
        
        all_data = []
        
        # 提取并处理 tar 文件
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            
            # 限制处理的站点数量（用于测试）
            if max_stations:
                members = members[:max_stations]
            
            print(f"   处理 {len(members)} 个站点文件...")
            
            for member in tqdm(members, desc=f"   解析 {year}"):
                if not member.name.endswith('.op.gz') and not member.name.endswith('.op'):
                    continue
                
                # 提取文件
                f = tar.extractfile(member)
                if f is None:
                    continue
                
                # 读取内容
                try:
                    # GSOD 数据格式定义（基于官方文档）
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

                    names = ['STN---', 'WBAN', 'YEARMODA', 'TEMP', 'TEMP_COUNT', 'DEWP', 'DEWP_COUNT',
                             'SLP', 'SLP_COUNT', 'STP', 'STP_COUNT', 'VISIB', 'VISIB_COUNT',
                             'WDSP', 'WDSP_COUNT', 'MXSPD', 'GUST', 'MAX', 'MAX_FLAG',
                             'MIN', 'MIN_FLAG', 'PRCP', 'PRCP_FLAG', 'SNDP', 'FRSHTT']
                    
                    if member.name.endswith('.gz'):
                        content = gzip.decompress(f.read()).decode('utf-8')
                    else:
                        content = f.read().decode('utf-8')
                    
                    # 解析为 DataFrame (使用固定宽度格式)
                    from io import StringIO
                    df = pd.read_fwf(StringIO(content), colspecs=colspecs, names=names, skiprows=1)
                    
                    if len(df) > 0:
                        all_data.append(df)
                
                except Exception as e:
                    continue
        
        if not all_data:
            print(f"   ❌ 没有成功解析任何数据")
            return None
        
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"   ✅ 成功处理 {len(combined_df)} 行数据，来自 {len(all_data)} 个站点")
        
        return combined_df
    
    def clean_and_transform(self, df):
        """
        清洗和转换数据
        
        Args:
            df: 原始数据框
            
        Returns:
            DataFrame: 清洗后的数据框
        """
        print("\n🧹 清洗和转换数据...")
        
        # 创建 site_id
        df['site_id'] = df['STN---'].astype(str).str.zfill(6) + '-' + \
                        df['WBAN'].astype(str).str.zfill(5)
        
        # 转换日期格式（处理可能的异常值）
        df['YEARMODA'] = df['YEARMODA'].astype(str).str.zfill(8)
        df['date'] = pd.to_datetime(df['YEARMODA'], format='%Y%m%d', errors='coerce')
        
        # 删除日期无效的行
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            print(f"   ⚠️  删除 {invalid_dates} 行无效日期数据")
            df = df.dropna(subset=['date'])
        
        # 处理缺失值
        for col, missing_val in self.missing_values.items():
            if col in df.columns:
                df[col] = df[col].replace(missing_val, np.nan)
        
        # 解析 FRSHTT 标记（天气条件编码）
        # FRSHTT 是一个6位数字，每位代表一种天气现象的有无
        if 'FRSHTT' in df.columns:
            frshtt_str = df['FRSHTT'].astype(str).str.zfill(6)
            df['fog'] = frshtt_str.str[0].astype(int)           # 雾
            df['rain'] = frshtt_str.str[1].astype(int)          # 雨/细雨
            df['snow'] = frshtt_str.str[2].astype(int)          # 雪/冰粒
            df['hail'] = frshtt_str.str[3].astype(int)          # 冰雹
            df['thunder'] = frshtt_str.str[4].astype(int)       # 雷暴
            df['tornado'] = frshtt_str.str[5].astype(int)       # 龙卷风
        
        # 选择并重命名特征列
        feature_cols = {
            'site_id': 'site_id',
            'date': 'date',
            'TEMP': 'mean_temp',          # 平均温度
            'DEWP': 'dew_point',          # 露点
            'SLP': 'sea_level_pressure',  # 海平面气压
            'STP': 'station_pressure',    # 站点气压
            'VISIB': 'visibility',        # 能见度
            'WDSP': 'wind_speed',         # 平均风速
            'MXSPD': 'max_wind_speed',    # 最大风速
            'GUST': 'wind_gust',          # 阵风
            'MAX': 'max_temp',            # 最高温度
            'MIN': 'min_temp',            # 最低温度
            'PRCP': 'precipitation',      # 降水量
            'SNDP': 'snow_depth',         # 雪深
            'fog': 'fog',
            'rain': 'rain',
            'snow': 'snow',
            'hail': 'hail',
            'thunder': 'thunder',
            'tornado': 'tornado'
        }
        
        # 只保留存在的列
        available_cols = {k: v for k, v in feature_cols.items() if k in df.columns}
        df_cleaned = df[list(available_cols.keys())].rename(columns=available_cols)
        
        # 按 site_id 和 date 排序（重要！符合README要求）
        df_cleaned = df_cleaned.sort_values(['site_id', 'date']).reset_index(drop=True)
        
        print(f"   ✅ 清洗完成，保留 {len(df_cleaned)} 行，{len(df_cleaned.columns)} 列特征")
        
        # 如果启用离散化，进行归一化和分组
        if self.discretize:
            df_cleaned = self.normalize_and_discretize(df_cleaned)
        
        return df_cleaned
    
    def normalize_and_discretize(self, df):
        """
        对连续特征进行归一化和离散化
        
        Args:
            df: 清洗后的数据框
            
        Returns:
            DataFrame: 归一化和离散化后的数据框
        """
        print(f"\n🔢 归一化和离散化连续特征...")
        print(f"   离散化组数: {self.n_bins} 组")
        
        df_processed = df.copy()
        
        # 存储归一化参数（用于文档记录）
        normalization_info = {}
        
        for feature in self.continuous_features:
            if feature not in df_processed.columns:
                continue
            
            # 获取非缺失数据
            valid_mask = df_processed[feature].notna()
            valid_data = df_processed.loc[valid_mask, feature]
            
            if len(valid_data) == 0:
                print(f"   ⚠️  {feature}: 全部缺失，跳过")
                continue
            
            # 1. 归一化到 [0, 1]
            min_val = valid_data.min()
            max_val = valid_data.max()
            
            if max_val > min_val:
                normalized = (valid_data - min_val) / (max_val - min_val)
                
                # 2. 离散化为 n_bins 组（使用等宽分箱）
                # 组标签: 0, 1, 2, ..., n_bins-1
                discretized = pd.cut(normalized, 
                                    bins=self.n_bins, 
                                    labels=range(self.n_bins),
                                    include_lowest=True)
                
                # 转换为整数
                discretized = discretized.astype(float)  # 先转float以处理NaN
                
                # 保存原始列（作为 feature_raw）
                df_processed[f'{feature}_raw'] = df_processed[feature]
                
                # 更新主列为离散化后的值
                df_processed.loc[valid_mask, feature] = discretized
                
                # 记录归一化信息
                normalization_info[feature] = {
                    'min': min_val,
                    'max': max_val,
                    'bins': self.n_bins,
                    'unique_values': len(valid_data.unique())
                }
                
                print(f"   ✅ {feature:25s}: [{min_val:8.2f}, {max_val:8.2f}] → [{0}, {self.n_bins-1}]")
            else:
                print(f"   ⚠️  {feature}: 值范围为0，跳过离散化")
        
        # 保存归一化信息到文件
        info_path = self.output_dir / "normalization_info.txt"
        with open(info_path, 'w') as f:
            f.write("连续特征归一化和离散化信息\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"离散化组数: {self.n_bins}\n")
            f.write(f"离散化方法: 等宽分箱 (equal-width binning)\n")
            f.write(f"组标签: 0, 1, 2, ..., {self.n_bins-1}\n\n")
            
            for feature, info in normalization_info.items():
                f.write(f"\n{feature}:\n")
                f.write(f"  原始范围: [{info['min']:.2f}, {info['max']:.2f}]\n")
                f.write(f"  唯一值数: {info['unique_values']}\n")
                f.write(f"  离散化后: {info['bins']} 组 (0-{info['bins']-1})\n")
        
        print(f"\n   📄 归一化信息已保存到: {info_path}")
        print(f"   ✅ 离散化完成: {len(normalization_info)} 个连续特征")
        
        return df_processed
    
    def save_processed_data(self, df, filename='processed_weather_data.csv'):
        """
        保存处理后的数据
        
        Args:
            df: 处理后的数据框
            filename: 输出文件名
        """
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"\n💾 数据已保存到: {output_path}")
        print(f"   - 行数: {len(df)}")
        print(f"   - 列数: {len(df.columns)}")
        print(f"   - 站点数: {df['site_id'].nunique()}")
        print(f"   - 日期范围: {df['date'].min()} 到 {df['date'].max()}")
        
        # 如果启用了离散化，显示相关信息
        if self.discretize:
            raw_cols = [c for c in df.columns if c.endswith('_raw')]
            print(f"   - 离散化特征数: {len(self.continuous_features)}")
            print(f"   - 离散化组数: {self.n_bins}")
            print(f"   - 原始值列（*_raw）: {len(raw_cols)} 个")
    
    def process_multiple_years(self, years, max_stations_per_year=None):
        """
        处理多个年份的数据
        
        Args:
            years: 年份列表
            max_stations_per_year: 每年最大处理站点数
            
        Returns:
            DataFrame: 合并后的数据
        """
        all_years_data = []
        
        for year in years:
            df = self.process_year_data(year, max_stations=max_stations_per_year)
            if df is not None:
                all_years_data.append(df)
        
        if not all_years_data:
            print("❌ 没有成功处理任何年份的数据")
            return None
        
        # 合并所有年份
        print(f"\n🔗 合并 {len(all_years_data)} 个年份的数据...")
        combined_df = pd.concat(all_years_data, ignore_index=True)
        
        # 清洗和转换
        cleaned_df = self.clean_and_transform(combined_df)
        
        return cleaned_df
    
    def generate_summary_statistics(self, df):
        """
        生成数据摘要统计
        
        Args:
            df: 处理后的数据框
        """
        print("\n📊 数据摘要统计:")
        print("=" * 60)
        
        print(f"\n基本信息:")
        print(f"  - 总行数: {len(df):,}")
        print(f"  - 总站点数: {df['site_id'].nunique():,}")
        print(f"  - 日期范围: {df['date'].min()} 到 {df['date'].max()}")
        
        # 每个站点的平均观测天数
        obs_per_site = df.groupby('site_id').size()
        print(f"\n每个站点的观测天数:")
        print(f"  - 平均: {obs_per_site.mean():.1f} 天")
        print(f"  - 中位数: {obs_per_site.median():.1f} 天")
        print(f"  - 最小: {obs_per_site.min()} 天")
        print(f"  - 最大: {obs_per_site.max()} 天")
        
        # 特征缺失率
        print(f"\n特征缺失率:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_rate = df[col].isna().sum() / len(df) * 100
            print(f"  - {col}: {missing_rate:.2f}%")
        
        print("=" * 60)


def main():
    """主函数：处理数据集中所有年份的数据"""
    
    print("🌤️  NOAA GSOD 数据处理工具")
    print("=" * 60)
    
    # 询问是否离散化
    print("\n是否对连续特征进行归一化和离散化？")
    print("1. 是 - 归一化到[0,1]并离散化为N组（推荐用于HMM）")
    print("2. 否 - 保持原始连续值")
    
    discretize_choice = input("\n请选择 (1/2) [默认: 1]: ").strip() or "1"
    discretize = (discretize_choice == "1")
    
    n_bins = 5  # 默认5组
    if discretize:
        n_bins_input = input(f"离散化组数 (3-10) [默认: 5]: ").strip()
        if n_bins_input.isdigit():
            n_bins = int(n_bins_input)
            n_bins = max(3, min(10, n_bins))  # 限制在3-10之间
        print(f"✅ 将使用 {n_bins} 组进行离散化")
    else:
        print("✅ 将保持连续值")
    
    # 创建加载器实例
    loader = GSODDataLoader(n_bins=n_bins, discretize=discretize)
    
    # 加载站点元数据
    station_metadata = loader.load_station_metadata()
    
    # 自动检测所有可用年份
    print("\n📅 检测可用年份...")
    available_years = []
    for tar_file in sorted(loader.gsod_dir.glob("gsod_*.tar")):
        year = int(tar_file.stem.split('_')[1])
        available_years.append(year)
    
    print(f"   找到 {len(available_years)} 个年份: {available_years[0]} - {available_years[-1]}")
    
    # 处理数据选项
    print("\n选项: 处理哪些年份的数据？")
    print("1. 快速测试（2015年，前50个站点）")
    print("2. 单年完整（2015年，所有站点）")
    print("3. 近期数据（1973-2019年，推荐：数据更完整）")
    print("4. 全部数据（1901-2019年，所有站点）")
    
    choice = input(f"\n请选择 (1/2/3/4) [默认: 3]: ").strip() or "3"
    
    if choice == "1":
        # 快速测试：2015年，前50个站点
        print("\n🧪 快速测试模式...")
        df = loader.process_year_data(2015, max_stations=50)
        if df is not None:
            cleaned_df = loader.clean_and_transform(df)
            loader.save_processed_data(cleaned_df, 'weather_data_2015_sample.csv')
            loader.generate_summary_statistics(cleaned_df)
    
    elif choice == "2":
        # 单年完整
        print("\n📊 处理单年完整数据...")
        df = loader.process_year_data(2015)
        if df is not None:
            cleaned_df = loader.clean_and_transform(df)
            loader.save_processed_data(cleaned_df, 'weather_data_2015_full.csv')
            loader.generate_summary_statistics(cleaned_df)
    
    elif choice == "3":
        # 近期数据（1973年后数据更完整）
        print("\n🌍 处理近期数据（1973-2019）...")
        print("   ⏰ 这可能需要较长时间，请耐心等待...")
        years = [y for y in available_years if y >= 1973]
        print(f"   处理 {len(years)} 个年份")
        cleaned_df = loader.process_multiple_years(years)
        if cleaned_df is not None:
            loader.save_processed_data(cleaned_df, 'weather_data_1973_2019_full.csv')
            loader.generate_summary_statistics(cleaned_df)
    
    elif choice == "4":
        # 全部数据
        print("\n🌎 处理全部历史数据（1901-2019）...")
        print("   ⚠️  警告：这将处理超过100年的数据，可能需要数小时！")
        confirm = input("   确认继续？(yes/no) [no]: ").strip().lower()
        
        if confirm == "yes":
            print(f"   处理 {len(available_years)} 个年份")
            cleaned_df = loader.process_multiple_years(available_years)
            if cleaned_df is not None:
                loader.save_processed_data(cleaned_df, 'weather_data_1901_2019_full.csv')
                loader.generate_summary_statistics(cleaned_df)
        else:
            print("   已取消处理全部数据")
    
    print("\n✅ 数据处理完成！")


if __name__ == "__main__":
    main()
