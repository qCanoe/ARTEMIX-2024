import networkx as nx

G = nx.read_graphml("data/trans_graph.graphml")

# 找到所有source和target相同的节点
loop_nodes = set()
for node in G.nodes():
    if G.has_edge(node, node):
        loop_nodes.add(node)

with open("qyy/loop_nodes.txt", "w") as f:
    for node in loop_nodes:
        f.write(f"{node}\n")

print(f"Found {len(loop_nodes)} loop nodes. Saved to data/loop_nodes.txt")
