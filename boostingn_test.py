import torch
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 加载可疑结果的tensor文件
tensor1 = torch.load('qyy/tensor/loop_nodes.pt').numpy()
tensor2 = torch.load('qyy/tensor/short_cycles.pt').numpy()
tensor3 = torch.load('qyy/tensor/nft_loop.txt.pt').numpy()
tensor4 = torch.load('qyy/tensor/back_forth.pt').numpy()
tensor = torch.load('qyy/tensor/star_patterns_nodes.pt').numpy()

tensor5 = torch.load('qyy/tensor/isolation_forest.pt').numpy()
tensor6 = torch.load('qyy/tensor/fft.pt').numpy()

tensor7 = torch.load('qyy/tensor/features.pt').numpy()

# 加载 PageRank 结果
pagerank_tensor = torch.load('qyy/tensor/normalized_pagerank_tensor.pt').numpy()

# 加载社区检测结果
community_tensor = torch.load('qyy/tensor/community_results_new.pt').numpy().reshape(-1, 1)


# 进行独热编码
encoder = OneHotEncoder(categories='auto')
community_encoded = encoder.fit_transform(community_tensor).toarray()

# 加载正确标签数据
y_true = torch.load('data/y.pt').numpy()

# 构建特征矩阵，包括PageRank结果
X = np.hstack((tensor1.reshape(-1, 1),
               tensor2.reshape(-1, 1),
               tensor3.reshape(-1, 1),
               tensor4.reshape(-1, 1),
               tensor.reshape(-1, 1),
               tensor5.reshape(-1, 1),
               tensor6.reshape(-1, 1),
               pagerank_tensor.reshape(-1, 1),
               tensor7.reshape(-1, 1),
                community_encoded
               ))

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

# 定义Boosting模型
base_clf = DecisionTreeClassifier(max_depth=2, min_samples_split=5)
boost_clf = AdaBoostClassifier(estimator=base_clf, algorithm='SAMME', n_estimators=200, learning_rate=0.1, random_state=42)

# 训练Boosting模型
boost_clf.fit(X_train, y_train)

# 预测
y_pred = boost_clf.predict(X_test)

# 计算评估结果
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Final Results:')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')
