import torch
import numpy as np

# 加载数据
data = torch.load(r'qyy\tensor\features.pt')

# 查看数据类型
print("Data type:", type(data))

# 处理不同的数据格式
if isinstance(data, torch.Tensor):
    print("Shape of tensor:", data.shape)
    print("Data:", data)
    np.savetxt('tensor_data.txt', data.numpy())

elif isinstance(data, dict):
    for key, value in data.items():
        print(f"\nKey: {key}")
        print(f"Value: {value}")
        if isinstance(value, torch.Tensor):
            print(f"Shape of tensor '{key}':", value.shape)
            np.savetxt(f'tensor_{key}_data.txt', value.numpy())

elif isinstance(data, list):
    for i, item in enumerate(data):
        print(f"\nItem {i}: {item}")
        if isinstance(item, torch.Tensor):
            print(f"Shape of tensor {i}:", item.shape)
            np.savetxt(f'tensor_{i}_data.txt', item.numpy())
else:
    print("Unsupported data format")
