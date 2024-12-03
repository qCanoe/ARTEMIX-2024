import pandas as pd
import numpy as np
import torch
import os

# 读取数据
df = pd.read_csv('original_data/原始BLUR交易记录.csv')

# 读取 true_nodes
with open('qyy/txt/True_nodes.txt', 'r') as f:
    true_nodes = set(f.read().splitlines())

# 转换时间格式
df['BLOCK_TIMESTAMP'] = pd.to_datetime(df['BLOCK_TIMESTAMP'])

# 计算买入和卖出时间
df['buy_time'] = df.groupby(['TOKENID', 'BUYER_ADDRESS'])['BLOCK_TIMESTAMP'].transform('min')
df['sell_time'] = df.groupby(['TOKENID', 'SELLER_ADDRESS'])['BLOCK_TIMESTAMP'].transform('max')

# 计算持有时间
df['holding_time'] = (df['sell_time'] - df['buy_time']).dt.days

# 过滤掉持有时间小于等于 0 的记录
df = df[df['holding_time'] > 0]

# 计算平均持有时间
avg_holding_time = df.groupby('BUYER_ADDRESS')['holding_time'].mean().reset_index()
avg_holding_time.columns = ['ADDRESS', 'avg_holding_time']

# 合并平均持有时间数据
df = df.merge(avg_holding_time, left_on='BUYER_ADDRESS', right_on='ADDRESS', how='left')

# 读取地址顺序
address_order = pd.read_csv('original_data/每个地址被标记wash的次数.csv', header=None)
address_order = address_order[0].dropna().unique().tolist()

# 创建一个字典以将地址映射到其平均持有时间
address_to_avg_holding_time = avg_holding_time.set_index('ADDRESS')['avg_holding_time'].to_dict()

# 根据地址顺序创建 tensor
avg_holding_time_tensor = torch.tensor([address_to_avg_holding_time.get(address, 0) for address in address_order])

# 检查 tensor 大小
print(f"Tensor size: {avg_holding_time_tensor.size()}")

# 确保 tensor 大小是 203101
if avg_holding_time_tensor.size(0) == 203101:
    print("The tensor size is correct.")
else:
    print("The tensor size is incorrect.")

# 对 tensor 进行归一化
avg_holding_time_tensor_normalized = (avg_holding_time_tensor - avg_holding_time_tensor.min()) / (avg_holding_time_tensor.max() - avg_holding_time_tensor.min())

# 确保目录存在
os.makedirs('qyy/tensor', exist_ok=True)

# 保存归一化后的 tensor
torch.save(avg_holding_time_tensor_normalized, 'qyy/tensor/avg_holding_time_tensor.pt')

print("Normalized tensor saved successfully.")
