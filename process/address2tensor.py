import pandas as pd
import torch

# 定义文件路径
txt_file_path = r'reciprocal_transaction_nodes.txt'
csv_file_path = 'original_data/每个地址被标记wash的次数.csv'
tensor_size = 203101

# 从txt文件中加载地址
with open(txt_file_path, 'r', encoding='utf-8') as file:
    txt_addresses = file.read().splitlines()

# 从csv文件中加载地址
csv_addresses = pd.read_csv(csv_file_path, header=None, usecols=[0], encoding='utf-8').iloc[:, 0].tolist()

# 创建tensor
tensor = torch.zeros(tensor_size, dtype=torch.int32)

# 标记tensor中的位置
address_to_index = {address: idx for idx, address in enumerate(csv_addresses)}

for address in txt_addresses:
    if address in address_to_index:
        index = address_to_index[address]
        tensor[index] = 1

# 保存tensor到.pt文件
output_file_path = 'qyy/tensor/test2.pt'
torch.save(tensor, output_file_path)

print(f'Tensor saved to {output_file_path}')
