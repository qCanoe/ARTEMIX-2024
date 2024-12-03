import pandas as pd
import networkx as nx
from tqdm import tqdm

# 读取原始交易数据
data = pd.read_csv('original_data/原始BLUR交易记录.csv')

# 筛选出sale和bid_won交易
sale_data = data[data['EVENT_TYPE'] == 'sale']
bid_won_data = data[data['EVENT_TYPE'] == 'bid_won']

# 创建一个空的图
G = nx.Graph()

# 添加sale交易的边
for index, row in sale_data.iterrows():
    G.add_edge(row['SELLER_ADDRESS'], row['BUYER_ADDRESS'], type='sale')

# 添加bid_won交易的边
for index, row in bid_won_data.iterrows():
    G.add_edge(row['SELLER_ADDRESS'], row['BUYER_ADDRESS'], type='bid_won')

# 创建一个字典来存储每个节点来回交易的邻居节点数目
reciprocal_neighbors_count = {}

# 遍历每个节点，显示进度条
for node in tqdm(G.nodes, desc="Processing nodes"):
    # 获取节点的所有邻居
    neighbors = list(G.neighbors(node))
    reciprocal_neighbors = 0
    
    # 检查每个邻居是否有来回交易
    for neighbor in neighbors:
        if G.has_edge(node, neighbor) and G.has_edge(neighbor, node):
            edges = G.get_edge_data(node, neighbor)
            if ('sale' in edges.values() and 'bid_won' in edges.values()):
                reciprocal_neighbors += 1
    
    reciprocal_neighbors_count[node] = reciprocal_neighbors

# 筛选邻居数目大于2的节点
filtered_addresses = [address for address, count in reciprocal_neighbors_count.items() if count > 2]

# 将结果保存到txt文件中
with open('filtered_star_addresses.txt', 'w') as f:
    for address in filtered_addresses:
        f.write(f"{address}\n : {reciprocal_neighbors_count[address]}\n")

print(f"Filtered addresses saved to filtered_addresses.txt")
