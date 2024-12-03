import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.nn import MessagePassing

class ArtemisFirstLayerConv(MessagePassing):
    def __init__(self, in_node_channels, in_edge_channels, out_channels, aggr='mean'):
        super(ArtemisFirstLayerConv, self).__init__(aggr=aggr)
        self.lin = nn.Linear(in_node_channels + in_edge_channels, out_channels)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.lin(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out):
        return F.relu(aggr_out)

class ArtemisNet(nn.Module):
    def __init__(self, in_node_channels, in_edge_channels, hidden_channels):
        super(ArtemisNet, self).__init__()
        self.conv1 = ArtemisFirstLayerConv(in_node_channels, in_edge_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels + in_node_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index_tuple, edge_attr_tuple):
        edge_index_0, edge_index_1, edge_index_2 = edge_index_tuple
        edge_attr_0, _, _ = edge_attr_tuple

        # First layer with residual connection
        inital_embedding = x
        x = self.conv1(x, edge_index_0, edge_attr_0)
        x = F.relu(self.bn1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # Second layer with residual connection
        x = self.conv2(x, edge_index_1)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # Third layer with residual connection
        x = self.conv3(x, edge_index_2)
        x = F.relu(self.bn3(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # Apply MLP to the final output
        x = torch.cat([x, inital_embedding], dim=1)
        x = self.mlp(x)
        return x.squeeze()