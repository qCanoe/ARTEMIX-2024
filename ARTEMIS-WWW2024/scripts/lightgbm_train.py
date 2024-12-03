import pandas as pd
import lightgbm as lgb
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


X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.1)
params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': 10,
        'learning_rate': 0.01,
        'n_estimators': 200,
    }
lgb_classifier = lgb.LGBMClassifier(**params)
lgb_classifier.fit(X_train, y_train)
y_pred = lgb_classifier.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
report_dict = classification_report(y_test, y_pred, output_dict=True)
print(f"Precision: {report_dict['1']['precision']:.3f}")
print(f"Recall: {report_dict['1']['recall']:.3f}")
print(f"F1-score: {report_dict['1']['f1-score']:.3f}")
