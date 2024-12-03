# %%
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import random
import networkx as nx
import numpy as np
import torch
import os
import pickle

trans_graph_path = './ARTEMIS-WWW2024/data/trans_graph.graphml'
y = torch.load('./ARTEMIS-WWW2024/data/y.pt')
G = nx.read_graphml(trans_graph_path)

walks_path = './ARTEMIS-WWW2024/data/random_walks.pkl'
if os.path.exists(walks_path):
    node2vec = Node2Vec(G, dimensions=16, walk_length=5, num_walks=20, workers=6)
    with open(walks_path, 'rb') as f:
        walks = pickle.load(f)
else:
    node2vec = Node2Vec(G, dimensions=16, walk_length=5, num_walks=20, workers=6)
    walks = node2vec.walks
    with open(walks_path, 'wb') as f:
        pickle.dump(walks, f)


model = node2vec.fit(window=10, min_count=1, batch_words=4)
embeddings = []
for node in G.nodes():
    if str(node) in model.wv:
        embeddings.append(model.wv[str(node)])

embeddings = np.array(embeddings[:203101])

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


classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred, output_dict=True)['1'])