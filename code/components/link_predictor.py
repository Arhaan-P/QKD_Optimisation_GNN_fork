import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, TransformerConv

class AdvancedQKDLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, edge_attr_channels, hidden_channels, dropout_rate):
        super().__init__()

        self.conv1 = TransformerConv(in_channels, hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels)

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_attr_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )

        self.link_predictor = torch.nn.Sequential(
            torch.nn.Linear(3 * hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        edge_features = self.edge_mlp(edge_attr)
        
        return x, edge_features

    def decode(self, z, edge_features, edge_label_index):
        src, dst = edge_label_index

        if edge_features.size(0) != edge_label_index.size(1):
            edge_features = edge_features.mean(dim=0, keepdim=True).repeat(edge_label_index.size(1), 1)

        node_features = torch.cat([
            z[src],
            z[dst],
            edge_features
        ], dim=-1)

        return self.link_predictor(node_features).squeeze(-1)