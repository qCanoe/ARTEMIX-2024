# %%
from gensim.models import Word2Vec
import random
import networkx as nx
import numpy as np
import torch
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

trans_graph_path = './ARTEMIS-WWW2024/data/trans_graph.graphml'
y = torch.load('./ARTEMIS-WWW2024/data/y.pt')
G = nx.read_graphml(trans_graph_path)

# %%
# 定义DeepWalk模型
class DeepWalk:
    def __init__(self, graph, dimensions=16, walk_length=5, num_walks=20, workers=1):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.walks = list(self.generate_walks())

    def generate_walks(self):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(walk_length=self.walk_length, start_node=node))
        return walks

    def random_walk(self, walk_length, start_node):
        walk = [str(start_node)]
        for _ in range(walk_length - 1):
            cur = walk[-1]
            cur_neighbors = list(self.graph.neighbors(cur))
            if len(cur_neighbors) > 0:
                walk.append(str(random.choice(cur_neighbors)))
        return walk

    def fit(self):
        return Word2Vec(self.walks, vector_size=self.dimensions, window=10, min_count=1, sg=1, workers=self.workers)

# %%
# 检查是否已经生成了随机游走
walks_path = './ARTEMIS-WWW2024/data/deepwalk_random_walks.pkl'
if os.path.exists(walks_path):
    with open(walks_path, 'rb') as f:
        walks = pickle.load(f)
    model = Word2Vec.load("./ARTEMIS-WWW2024/data/deepwalk_model.model")
else:
    # 生成随机游走
    deepwalk = DeepWalk(G, dimensions=16, walk_length=5, num_walks=20, workers=6)
    walks = deepwalk.walks
    # 保存随机游走
    with open(walks_path, 'wb') as f:
        pickle.dump(walks, f)
    # 训练模型
    model = deepwalk.fit()
    model.save("./ARTEMIS-WWW2024/data/deepwalk_model.model")

# %%
# 生成节点嵌入
embeddings = []
for node in G.nodes():
    if str(node) in model.wv:
        embeddings.append(model.wv[str(node)])
    else:
        embeddings.append(np.zeros(model.vector_size))

embeddings = np.array(embeddings[:203101])

# %%

# 使用训练和测试掩码划分数据
train_mask = np.zeros(y.shape[0], dtype=np.bool_)
test_mask = np.zeros(y.shape[0], dtype=np.bool_)
train_test_split_num = int(y.shape[0] * 0.75)
train_index = random.sample(range(y.shape[0]), train_test_split_num)
test_index = list(set(range(y.shape[0])) - set(train_index))
train_mask[train_index] = True
test_mask[test_index] = True
print("train node num: ", train_mask.sum())
print("test node num: ", test_mask.sum())
print("true data percentage in train data: ", y[train_mask].sum() / len(y[train_mask]))
print("true data percentage in test data: ", y[test_mask].sum() / len(y[test_mask]))
X_train, X_test = embeddings[train_mask], embeddings[test_mask]
y_train, y_test = y.numpy()[train_mask], y.numpy()[test_mask]

# 使用逻辑回归进行分类
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# 预测测试集上的标签
y_pred = classifier.predict(X_test)


# print macro precision, recall, f1
print(classification_report(y_test, y_pred, output_dict=True)['1'])
# %%
