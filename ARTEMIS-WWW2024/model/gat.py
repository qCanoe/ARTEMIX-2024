import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

class GATNet(torch.nn.Module):
    def __init__(self, in_node_channels, hidden_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_node_channels, hidden_channels, heads=8, dropout=0.6)  # GAT层，8个注意力头
        self.bn1 = BatchNorm(hidden_channels * 8)  # 批量归一化，考虑到多头合并
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=1, concat=False, dropout=0.6)  # 第二个GAT层，1个注意力头
        self.bn2 = BatchNorm(hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels + in_node_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout正则化
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index):
        inital_embedding = x
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)  # GAT通常使用ELU激活函数
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.cat([x, inital_embedding], dim=1)
        x = self.mlp(x)
        return x.squeeze()