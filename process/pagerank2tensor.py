import torch
import pandas as pd
import numpy as np

# 读取pagerank_results.txt文件
pagerank_file = 'qyy/txt/pagerank_results.txt'
pagerank_data = {}
with open(pagerank_file, 'r') as f:
    for line in f:
        address, score = line.strip().split(' -- ')
        pagerank_data[address] = float(score)

# 读取每个地址被标记wash的次数.csv文件
csv_file = 'original_data/每个地址被标记wash的次数.csv'
address_df = pd.read_csv(csv_file, header=None)

# 创建一个大小为203101的tensor
tensor_size = 203101
pagerank_tensor = torch.zeros(tensor_size)

# 将pagerank得分对应到csv文件的地址位置
address_to_index = {address: idx for idx, address in enumerate(address_df[0])}
for address, score in pagerank_data.items():
    if address in address_to_index:
        index = address_to_index[address] - 1
        if 0 <= index < tensor_size:
            pagerank_tensor[index] = score

# 归一化tensor
pagerank_tensor = (pagerank_tensor - pagerank_tensor.min()) / (pagerank_tensor.max() - pagerank_tensor.min())

# 保存tensor
output_file = 'normalized_pagerank_tensor.pt'
torch.save(pagerank_tensor, output_file)

print(f'Tensor saved to {output_file}')
