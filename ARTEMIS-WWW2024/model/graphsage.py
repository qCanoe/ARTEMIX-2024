import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

class GraphSAGENet(torch.nn.Module):
    def __init__(self):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(11, 32)
        self.bn1 = BatchNorm(32)
        self.conv2 = SAGEConv(32, 32)
        self.bn2 = BatchNorm(32)
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp(x)
        return x.squeeze()
