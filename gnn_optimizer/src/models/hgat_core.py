import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear

class RecurrentHGAT(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, metadata):
        super().__init__()

        self.hidden_channels = hidden_channels
        
        # 1. Input Encoders (Feature Projection) 
        # Project varying input sizes to a common hidden dimension
        self.encoder_dict = nn.ModuleDict()
        self.encoder_dict['intersection'] = Linear(-1, hidden_channels)
        self.encoder_dict['lane'] = Linear(-1, hidden_channels)

        # 2. Spatial GNN Layers
        # Layer 1: Aggregates info from neighbors using Attention
        self.conv1 = HeteroConv({
            ('lane', 'part_of', 'intersection'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('intersection', 'adjacent_to', 'intersection'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False),
            ('lane', 'feeds_into', 'lane'): GATConv((-1, -1), hidden_channels, heads=num_heads, add_self_loops=False)
        }, aggr='sum')
        
        # Layers will be defined in future commits
        
    def forward(self, x_dict, edge_index_dict, hidden_state=None):
        # Forward logic will be implemented later
        pass