import torch

# 定义文件路径
file_path = r'qyy/txt/final_community_detection_results_with_index.txt'

# 初始化一个字典来存储地址索引和社区结果
community_results = {}

# 读取文件内容
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if '--' in line:
            index, community = line.split('--')
            index = index.strip()
            community = int(community.strip())
            # 只处理有效的数字索引
            if index.isdigit():
                index = int(index)
                if 0 <= index < 203101:
                    community_results[index] = community

# 初始化tensor，所有元素设为-1表示未分类
num_addresses = 203101  # 根据上下文，这是地址总数
community_tensor = torch.full((num_addresses,), -1, dtype=torch.int32)

# 更新tensor中对应索引的社区结果
for index, community in community_results.items():
    community_tensor[index] = community

# 保存tensor
output_path = 'qyy/tensor/community_results_new4.pt'
torch.save(community_tensor, output_path)

print(f'Community results tensor saved to {output_path}')
