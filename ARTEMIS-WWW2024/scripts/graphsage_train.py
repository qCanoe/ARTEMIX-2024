# %%
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
import numpy as np
import random
import sys
sys.path.append('../')
from model.graphsage import GraphSAGENet
from sklearn.metrics import precision_score, recall_score, f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

market_manu_features = torch.load('./WWW 2024/market_manu_features.pt').to(device)
edge_index = torch.load('./WWW 2024/edge_index.pt').to(device)
base_edge_features = torch.load('./WWW 2024/base_edge_features.pt').to(device)
nft_multimodal_bmbedding_features = torch.load('./WWW 2024/nft_multimodal_bmbedding_features.pt').to(device)
y = torch.load('./WWW 2024/y.pt').to(device)

edge_features = torch.cat([base_edge_features, nft_multimodal_bmbedding_features], dim=1)

# %%

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

data = Data(x=market_manu_features, y=y,
            edge_index=edge_index, edge_attr=edge_features,
            train_mask=train_mask, test_mask=test_mask).to(device)

model = GraphSAGENet().to(device)

num_pos = data.y[data.train_mask].sum().item()
num_neg = data.train_mask.sum().item() - num_pos
class_weights = torch.tensor([4], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

model.train() 
patience = 20 
best_loss = float('inf')
patience_counter = 0

for epoch in range(100000):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
    loss.backward() 
    optimizer.step()

    if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.5f}")
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[data.train_mask], data.y[data.train_mask].float())
                val_predictions = torch.sigmoid(val_out)
                val_pred_binary = (val_predictions >= 0.5).int()
                val_pred = val_pred_binary[data.train_mask].cpu()
                y_val = data.y[data.train_mask].int().cpu()
                val_precision = precision_score(y_val.numpy(), val_pred.numpy(), zero_division=1)
                val_recall = recall_score(y_val.numpy(), val_pred.numpy())
                val_f1 = f1_score(y_val.numpy(), val_pred.numpy())
                print(f"Validation - Precision: {val_precision:.3f}, Recall: {val_recall:.3f}, F1 Score: {val_f1:.3f}, Loss: {val_loss.item():.5f}")
            model.train()

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == patience:
                    print("Stopping early due to lack of improvement on the validation set.")
                    break



model.eval()
with torch.no_grad():
    predictions = model(data.x, data.edge_index)
    predictions = torch.sigmoid(predictions)
    pred_binary = (predictions >= 0.5).int()
    pred_test = pred_binary[data.test_mask].cpu()
    y_test = data.y[data.test_mask].int().cpu()
    precision = precision_score(y_test.numpy(), pred_test.numpy(), zero_division=1)
    recall = recall_score(y_test.numpy(), pred_test.numpy())
    f1 = f1_score(y_test.numpy(), pred_test.numpy())
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")