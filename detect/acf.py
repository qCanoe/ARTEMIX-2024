import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from tqdm import tqdm

# 读取数据
file_path = 'original_data/原始BLUR交易记录.csv'
data = pd.read_csv(file_path)

# 转换日期时间列
data['BLOCK_TIMESTAMP'] = pd.to_datetime(data['BLOCK_TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')

# 提取日期和小时信息
data['datetime'] = data['BLOCK_TIMESTAMP'].dt.floor('h')  # 使用小写 'h'，避免FutureWarning

# 计算傅里叶变换
def calculate_fft(transactions):
    if len(transactions) > 1:  # 确保有足够的数据点进行计算
        transactions = transactions.to_numpy()  # 转换为numpy数组
        transactions = transactions - transactions.mean()  # 去除均值
        yf = fft(transactions)
        xf = fftfreq(len(transactions), d=1)  # d=1 表示时间间隔为1小时
        return xf, np.abs(yf)
    else:
        return np.array([np.nan]), np.array([np.nan])  # 如果数据点不足，返回NaN

# 判断周期性
def is_significant_periodic(xf, yf, threshold=20):  # 提高阈值到20
    peak_indices = np.where(yf > threshold)[0]
    return len(peak_indices) > 0

# 处理每个地址
def process_address(address):
    address_data = data[(data['SELLER_ADDRESS'] == address) | (data['BUYER_ADDRESS'] == address)]
    hourly_transactions = address_data.groupby('datetime').size()
    xf, yf = calculate_fft(hourly_transactions)
    is_periodic = is_significant_periodic(xf, yf)
    return (address, xf, yf, is_periodic)

# 获取所有唯一地址
unique_addresses = pd.concat([data['SELLER_ADDRESS'], data['BUYER_ADDRESS']]).unique()

# 处理所有地址并显示进度条
results = []
for address in tqdm(unique_addresses, desc="Processing addresses"):
    results.append(process_address(address))

# 分离结果
suspicious_addresses = [address for address, xf, yf, is_periodic in results if is_periodic]
fft_results = [(address, xf, yf) for address, xf, yf, is_periodic in results]

# 保存可疑地址
output_file = 'qyy/txt/significant_periodic_addresses.txt'
with open(output_file, 'w') as f:
    for address in suspicious_addresses:
        f.write(f"{address}\n")

# 保存每个地址及其频谱值
fft_output_file = 'qyy/txt/address_fft_values.txt'
with open(fft_output_file, 'w') as f:
    for address, xf, yf in fft_results:
        xf_str = ','.join(map(str, xf))
        yf_str = ','.join(map(str, yf))
        f.write(f"{address}: xf({xf_str}), yf({yf_str})\n")

print(f"Detected {len(suspicious_addresses)} suspicious addresses with significant periodic trading patterns. Saved to {output_file}.")
print(f"Saved FFT values for each address to {fft_output_file}.")
