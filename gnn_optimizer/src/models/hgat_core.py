import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear

# --- IMPORT THE NEW MODULES ---
from src.config import ModelConfig
from src.models.policy_head import TrafficPolicyHead
from src.models.uncertainty import BayesianDropout

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

        # 3. Temporal Recurrent Layer.
        # GRU Cell to remember history.
        gnn_out_dim = hidden_channels * num_heads
        self.gru = nn.GRUCell(gnn_out_dim, hidden_channels)

        # 4. Uncertainty Module (The New Class)
        self.mc_dropout = BayesianDropout(p=ModelConfig.DROPOUT_RATE)

        # 5. Actor-Critic Head
        self.policy_head = TrafficPolicyHead(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, hidden_state=None):
        # 1. Encode Raw Features
        x_dict_encoded = {}
        for node_type, x in x_dict.items():
            x_dict_encoded[node_type] = F.relu(self.encoder_dict[node_type](x))

        # 2. Spatial Processing (GNN)
        x_dict_out = self.conv1(x_dict_encoded, edge_index_dict)
        
        # Apply Activation & The Custom Dropout
        x_dict_out = {k: self.mc_dropout(F.relu(v)) for k, v in x_dict_out.items()}
        
        # 3. Temporal Processing (GRU)
        intersection_embeddings = x_dict_out['intersection']
        
        if hidden_state is None:
            hidden_state = torch.zeros_like(intersection_embeddings[:, :self.hidden_channels])

        # Update memory
        new_hidden_state = self.gru(intersection_embeddings, hidden_state)
        
        # 4. Decision Making (Actor & Critic)
        action_logits, state_value = self.policy_head(new_hidden_state)
        
        return action_logits, state_value, new_hidden_state