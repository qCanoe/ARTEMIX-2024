import pandas as pd
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

# 输入文件路径
file_path = 'original_data/原始BLUR交易记录.csv'
data = pd.read_csv(file_path)

# 日期时间列的转换
data['BLOCK_TIMESTAMP'] = pd.to_datetime(data['BLOCK_TIMESTAMP'])

# 合并买卖地址
all_addresses = pd.concat([data[['BLOCK_TIMESTAMP', 'SELLER_ADDRESS']], data[['BLOCK_TIMESTAMP', 'BUYER_ADDRESS']].rename(columns={'BUYER_ADDRESS': 'SELLER_ADDRESS'})])
all_addresses = all_addresses.rename(columns={'SELLER_ADDRESS': 'ADDRESS'})

# 提取日期信息并计算每日交易次数
all_addresses['day'] = all_addresses['BLOCK_TIMESTAMP'].dt.date

# 计算每小时交易次数
all_addresses['hour'] = all_addresses['BLOCK_TIMESTAMP'].dt.hour
hourly_transactions = all_addresses.groupby(['ADDRESS', 'day', 'hour']).size().reset_index(name='hourly_transaction_count')

# 计算特定时间段（午夜到凌晨）的交易次数
midnight_transactions = all_addresses[all_addresses['BLOCK_TIMESTAMP'].dt.hour.isin(range(0, 6))]
midnight_transaction_count = midnight_transactions.groupby(['ADDRESS', 'day']).size().reset_index(name='midnight_transaction_count')

# 计算每日交易次数
daily_transactions = all_addresses.groupby(['ADDRESS', 'day']).size().reset_index(name='daily_transaction_count')

# 合并所有特征
features = daily_transactions.merge(hourly_transactions.groupby('ADDRESS')['hourly_transaction_count'].max().reset_index(), on='ADDRESS', how='left')
features = features.merge(midnight_transaction_count.groupby('ADDRESS')['midnight_transaction_count'].sum().reset_index(), on='ADDRESS', how='left')
features.fillna(0, inplace=True)

# Isolation Forest参数
contamination_rate = 0.01  # 异常点比例
random_state_value = 42    # 随机数生成器种子

# 使用 tqdm 显示进度条
print("Training Isolation Forest...")
anomaly_detector = IsolationForest(contamination=contamination_rate, random_state=random_state_value)
features['anomaly_score'] = anomaly_detector.fit_predict(features[['daily_transaction_count', 'hourly_transaction_count', 'midnight_transaction_count']])

# 提取可疑地址并保存结果
suspicious_addresses = features[features['anomaly_score'] == -1]['ADDRESS'].unique()

# 生成输出文件名，带时间戳
output_file = f'qyy/txt/isolation_nodes.txt'
with open(output_file, 'w') as f:
    for address in suspicious_addresses:
        f.write(f"{address}\n")

print(f"Detected {len(suspicious_addresses)} suspicious addresses. Saved to {output_file}.")
