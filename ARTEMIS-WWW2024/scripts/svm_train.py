import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import torch

features_tensor = torch.load('./ARTEMIS-WWW2024/data/node_basic_features.pt')
labels_tensor = torch.load('./ARTEMIS-WWW2024/data/y.pt')

features = features_tensor.numpy()
labels = labels_tensor.numpy()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.1)

    params = {
        'C': 0.8,
        'kernel': 'linear',
        'gamma': 'scale',
    }

    svc = SVC(**params)

    # 训练模型
    svc.fit(X_train, y_train, sample_weight=[1 if i == 0 else 2 for i in y_train])

    # 预测测试集
    y_pred = svc.predict(X_test)

    # 输出分类报告
    report = classification_report(y_test, y_pred)
    print(report)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(f"Precision: {report_dict['1']['precision']:.3f}")
    print(f"Recall: {report_dict['1']['recall']:.3f}")
    print(f"F1-score: {report_dict['1']['f1-score']:.3f}")
