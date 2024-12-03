import networkx as nx

# 读取GraphML文件
G = nx.read_graphml('data/trans_graph.graphml')

# 计算PageRank值
pagerank_values = nx.pagerank(G, alpha=0.85)

# 将结果保存为txt文件
with open('pagerank_results.txt', 'w') as file:
    for node, value in pagerank_values.items():
        file.write(f"{node} -- {value}\n")
