# %%
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
import numpy as np
import random
import sys
sys.path.append('../')
from model.artemis import ArtemisNet
from model.artemis_components.NeighborSamplerByNFT import NeighborSamplerbyNFT
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.data import DataLoader


# 检查是否有可用的CUDA设备，如果有，使用第一个可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据并移动到GPU（如果可用）
node_basic_features = torch.load('./WWW 2024/node_basic_features.pt').to(device)
node_advanced_features = torch.load('./WWW 2024/node_advanced_features.pt').to(device)
edge_index = torch.load('./WWW 2024/edge_index.pt').to(device)
base_edge_features = torch.load('./WWW 2024/base_edge_features.pt').to(device)
nft_multimodal_bmbedding_features = torch.load('./WWW 2024/nft_multimodal_bmbedding_features.pt').to(device)
y = torch.load('./WWW 2024/y.pt').to(device)
node_sample_prob = torch.load('./WWW 2024/node_sample_prob.pt')
node_sample_prob = node_sample_prob / node_sample_prob.sum()

node_features = torch.cat([node_basic_features, node_advanced_features], dim=1)
edge_features = torch.cat([base_edge_features, nft_multimodal_bmbedding_features], dim=1)

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

data = Data(x=node_basic_features, y=y,
            edge_index=edge_index, edge_attr=edge_features,
            train_mask=train_mask, test_mask=test_mask).to(device)


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler

import os

# 定义日志文件的路径
log_path = "./WWW 2024/artemis_model_log.txt"

if os.path.exists(log_path):
    os.remove(log_path)

with open(log_path, "a") as log_file:
    log_file.write("Epoch, Average Loss, Average Accuracy, Average Precision, Average Recall, Average F1 Score\n")


for run in range(5):
    print(f"Starting run {run+1}...\n")
    with open(log_path, "a") as log_file:

        model = ArtemisNet(data.x.shape[1], data.edge_attr.shape[1], 32).to(device)

        # 计算类权重
        num_pos = data.y[data.train_mask].sum().item()
        num_neg = data.train_mask.sum().item() - num_pos
        class_weights = torch.tensor([1], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)


        sizes = [8, 1, 1]
        edge_sampler = NeighborSamplerbyNFT(edge_index=data.edge_index, sizes=sizes, edge_attr=data.edge_attr, prob_vector=node_sample_prob)
        patience = 10
        best_loss = float('inf')
        patience_counter = 0

        train_nodes = torch.where(torch.from_numpy(data.train_mask))[0]  # Keep train_nodes on CPU


        labels = data.y[data.train_mask].cpu().numpy()

        class_counts = np.bincount(labels)
        weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        sample_weights = weights[labels]

        # WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=False)

        patience = 10
        best_loss = float('inf')
        patience_counter = 0

        best_model = None
        best_f1 = 0.0

        for epoch in range(100):
            model.train()
            total_loss = 0
            total_accuracy = 0
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            batch_count = 0

            for subset_nodes in DataLoader(train_nodes, batch_size=256, sampler=sampler):
                batch_size, n_id, adjs = edge_sampler.sample(subset_nodes)
                n_id = n_id.to(device)

                optimizer.zero_grad()

                edge_index_0, e_id_0, size_0 = adjs[0].edge_index, adjs[0].e_id, adjs[0].size
                edge_attr_0 = data.edge_attr[e_id_0].to(device)
                edge_index_1, _, size_1 = adjs[1].edge_index, adjs[1].e_id, adjs[1].size
                edge_index_2, _, size_2 = adjs[2].edge_index, adjs[2].e_id, adjs[2].size

                out = model(data.x[n_id], (edge_index_0.to(device), edge_index_1.to(device), edge_index_2.to(device)), (edge_attr_0, None, None))

                loss = criterion(out, data.y[n_id].float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                predictions = torch.sigmoid(out)
                pred_binary = (predictions >= 0.5).int()
                pred = pred_binary.cpu()
                y = data.y[n_id].int().cpu()

                total_accuracy += accuracy_score(y.numpy(), pred.numpy())
                total_precision += precision_score(y.numpy(), pred.numpy(), zero_division=1)
                total_recall += recall_score(y.numpy(), pred.numpy())
                total_f1 += f1_score(y.numpy(), pred.numpy())

                batch_count += 1

            avg_loss = total_loss / batch_count
            avg_accuracy = total_accuracy / batch_count
            avg_precision = total_precision / batch_count
            avg_recall = total_recall / batch_count
            avg_f1 = total_f1 / batch_count

            print(f"Epoch {epoch} | Average Loss: {avg_loss:.5f} | Average Accuracy: {avg_accuracy:.3f} | "
                f"Average Precision: {avg_precision:.3f} | Average Recall: {avg_recall:.3f} | "
                f"Average F1 Score: {avg_f1:.3f}")
            log_file.write(f"{epoch}, {avg_loss:.5f}, {avg_accuracy:.3f}, {avg_precision:.3f}, {avg_recall:.3f}, {avg_f1:.3f}\n")

            # Early stopping logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == patience:
                    print("Stopping early due to lack of improvement on the validation set.")
                    break

            model.eval()
            test_nodes = torch.where(torch.from_numpy(data.test_mask))[0]
            with torch.no_grad():
                batch_size, n_id, adjs = edge_sampler.sample(test_nodes)
                n_id = n_id.to(device)

                edge_index_0, e_id_0, size_0 = adjs[0].edge_index, adjs[0].e_id, adjs[0].size
                edge_attr_0 = data.edge_attr[e_id_0].to(device)
                edge_index_1, _, size_1 = adjs[1].edge_index, adjs[1].e_id, adjs[1].size
                edge_index_2, _, size_2 = adjs[2].edge_index, adjs[2].e_id, adjs[2].size

                out = model(data.x[n_id], (edge_index_0.to(device), edge_index_1.to(device), edge_index_2.to(device)), (edge_attr_0, None, None))

                # 计算指标
                predictions = torch.sigmoid(out)
                pred_binary = (predictions >= 0.5).int()
                pred = pred_binary.cpu()
                y = data.y[n_id].int().cpu()
                accuracy = accuracy_score(y.numpy(), pred.numpy())
                precision = precision_score(y.numpy(), pred.numpy(), zero_division=1)
                recall = recall_score(y.numpy(), pred.numpy())
                f1 = f1_score(y.numpy(), pred.numpy())

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model.state_dict().copy()


        model.load_state_dict(best_model)
        model.eval()
        test_nodes = torch.where(torch.from_numpy(data.test_mask))[0]
        with torch.no_grad():
            batch_size, n_id, adjs = edge_sampler.sample(test_nodes)
            n_id = n_id.to(device)

            edge_index_0, e_id_0, size_0 = adjs[0].edge_index, adjs[0].e_id, adjs[0].size
            edge_attr_0 = data.edge_attr[e_id_0].to(device)
            edge_index_1, _, size_1 = adjs[1].edge_index, adjs[1].e_id, adjs[1].size
            edge_index_2, _, size_2 = adjs[2].edge_index, adjs[2].e_id, adjs[2].size

            out = model(data.x[n_id], (edge_index_0.to(device), edge_index_1.to(device), edge_index_2.to(device)), (edge_attr_0, None, None))

            # 计算指标
            predictions = torch.sigmoid(out)
            pred_binary = (predictions >= 0.5).int()
            pred = pred_binary.cpu()
            y = data.y[n_id].int().cpu()
            accuracy = accuracy_score(y.numpy(), pred.numpy())
            precision = precision_score(y.numpy(), pred.numpy(), zero_division=1)
            recall = recall_score(y.numpy(), pred.numpy())
            f1 = f1_score(y.numpy(), pred.numpy())

            print(f"Test - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")
            log_file.write(f"Test - Run {run+1}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}\n\n")

        print(f"Run {run+1} completed.\n")