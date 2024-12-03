import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report

# 加载tensor数据
nft_purchases_tensor = torch.load('qyy/tensor/nft_purchases_tensor.pt')
interactions_tensor = torch.load('qyy/tensor/interactions_tensor.pt')
avg_holding_time_tensor = torch.load('qyy/tensor/avg_holding_time_tensor.pt')
active_days_tensor = torch.load('qyy/tensor/active_days_tensor.pt')

# 将tensor转换为numpy数组
X_nft_purchases = nft_purchases_tensor.numpy()
X_interactions = interactions_tensor.numpy()
X_avg_holding_time = avg_holding_time_tensor.numpy()
X_active_days = active_days_tensor.numpy()

# 合并所有特征
X = np.vstack((X_nft_purchases, X_interactions, X_avg_holding_time, X_active_days)).T

# 加载标签数据
y = torch.load('data/y.pt').numpy()

# 交叉验证设置
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 训练随机森林模型并进行交叉验证预测
rf = RandomForestClassifier(n_estimators=100, random_state=42)
y_pred = cross_val_predict(rf, X, y, cv=cv)

# 输出最终的预测tensor
y_pred_tensor = torch.tensor(y_pred)

# 评估结果
report = classification_report(y, y_pred)
print("Classification Report:")
print(report)

# 保存预测结果tensor
torch.save(y_pred_tensor, 'qyy/tensor/y_pred_tensor.pt')

print("Prediction tensor saved successfully.")
print(f"Prediction tensor size: {y_pred_tensor.size()}")
