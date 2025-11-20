import os
import kagglehub

# 设置下载路径到 Proj 目录
# 获取 Proj 目录的绝对路径（当前文件在 Weather-HMM-Co-Repo/data/download.py）
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
download_dir = os.path.join(proj_root, "kaggle_data")

# 创建下载目录
os.makedirs(download_dir, exist_ok=True)

# 设置 kagglehub 缓存目录环境变量
# kagglehub 会使用 KAGGLEHUB_CACHE 环境变量来设置下载目录
os.environ['KAGGLEHUB_CACHE'] = download_dir

print(f"📦 正在下载数据集: noaa/noaa-global-surface-summary-of-the-day")
print(f"📁 保存到: {download_dir}")

# 使用 kagglehub 下载数据集
path = kagglehub.dataset_download("noaa/noaa-global-surface-summary-of-the-day")

print(f"\n✅ 数据集已下载")
print(f"📂 数据路径: {os.path.abspath(path)}")