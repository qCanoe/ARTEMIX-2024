import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm, MLP

class GINNet(torch.nn.Module):
    def __init__(self, in_node_channels, hidden_channels):
        super(GINNet, self).__init__()
        mlp1 = MLP([in_node_channels, hidden_channels, hidden_channels])
        mlp2 = MLP([hidden_channels, hidden_channels, hidden_channels])

        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.mlp(x)
        return x.squeeze()