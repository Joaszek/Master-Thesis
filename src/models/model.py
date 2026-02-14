import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, SAGEConv, BatchNorm, global_mean_pool, global_max_pool
from torch_scatter import scatter_add, scatter_max, scatter_mean


class EdgeProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, edge_attr):
        return self.proj(edge_attr)


class AttentionPooling(nn.Module):
    """
    Attention-based graph-level pooling.
    Learns importance weights for each node.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, batch):
        """
        Args:
            x: [num_nodes, hidden_dim] node features
            batch: [num_nodes] batch assignment

        Returns:
            [batch_size, hidden_dim] pooled features
        """
        # Compute attention scores
        att_scores = self.att(x)  # [num_nodes, 1]

        # Apply softmax per graph (subtract max for numerical stability)
        max_scores, _ = scatter_max(att_scores, batch, dim=0)  # [num_graphs, 1]
        att_scores = att_scores - max_scores[batch]             # broadcast back to nodes
        att_weights = torch.exp(att_scores)

        # Normalize per graph
        sum_weights = scatter_add(att_weights, batch, dim=0)[batch]
        att_weights = att_weights / (sum_weights + 1e-8)

        # Weighted sum
        weighted_x = x * att_weights
        out = scatter_add(weighted_x, batch, dim=0)

        return out


class GATv2Block(nn.Module):
    def __init__(self, in_dim, out_dim, heads, edge_dim, dropout):
        super().__init__()
        assert out_dim % heads == 0

        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True
        )

        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SAGEBlock(nn.Module):
    """Standard GraphSAGE block — no edge features."""
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.conv = SAGEConv(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SAGEEdgeBlock(nn.Module):
    """GraphSAGE block with edge features aggregated and concatenated to node features."""
    def __init__(self, in_dim, edge_dim, out_dim, dropout):
        super().__init__()
        self.conv = SAGEConv(in_dim + edge_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # Aggregate edge features to target nodes via mean
        target_nodes = edge_index[1]
        agg_edge = scatter_mean(edge_attr, target_nodes, dim=0, dim_size=x.size(0))
        x_cat = torch.cat([x, agg_edge], dim=-1)

        x_out = self.conv(x_cat, edge_index)
        x_out = self.norm(x_out)
        x_out = self.relu(x_out)
        x_out = self.dropout(x_out)
        return x_out


class EllipticGNN(nn.Module):
    """
    Full GNN model: [Edge Projection →] Conv layers → Pooling → MLP Classifier

    Supports three conv_type architectures:
      - "gatv2":     GATv2 with native edge feature support
      - "sage":      Standard GraphSAGE (no edge features)
      - "sage_edge": GraphSAGE with edge features aggregated to nodes

    Args:
        node_feat_dim:  input node feature dimension
        edge_feat_dim:  input edge feature dimension
        hidden_dim:     hidden layer width (128 default)
        num_layers:     number of conv layers (3 default)
        heads:          multi-head attention heads (4 default, only for gatv2)
        edge_proj_dim:  edge feature projection dimension (32 default)
        num_classes:    output classes (2 = binary)
        dropout:        dropout rate (0.3 default)
        conv_type:      "gatv2", "sage", or "sage_edge"
    """

    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers,
                 heads, edge_proj_dim, num_classes, dropout, conv_type="gatv2"):
        super().__init__()
        assert num_layers >= 1

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type

        # Edge projection — only needed for gatv2 and sage_edge
        self.uses_edge_features = conv_type in ("gatv2", "sage_edge")
        if self.uses_edge_features:
            self.edge_proj = EdgeProjection(edge_feat_dim, edge_proj_dim)
        else:
            self.edge_proj = None

        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Build conv layers based on architecture
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gatv2":
                assert hidden_dim % heads == 0
                self.conv_layers.append(
                    GATv2Block(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        heads=heads,
                        edge_dim=edge_proj_dim,
                        dropout=dropout
                    )
                )
            elif conv_type == "sage":
                self.conv_layers.append(
                    SAGEBlock(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        dropout=dropout
                    )
                )
            elif conv_type == "sage_edge":
                self.conv_layers.append(
                    SAGEEdgeBlock(
                        in_dim=hidden_dim,
                        edge_dim=edge_proj_dim,
                        out_dim=hidden_dim,
                        dropout=dropout
                    )
                )
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

        # Triple pooling: attention + mean + max
        self.att_pool = AttentionPooling(hidden_dim)
        self.readout_dim = hidden_dim * 3

        # Deeper classifier with BatchNorm for better generalization
        self.classifier = nn.Sequential(
            nn.Linear(self.readout_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),

            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # Project edge features if needed
        if self.uses_edge_features:
            edge_attr = self.edge_proj(edge_attr)

        x = self.input_proj(x)

        for conv_block in self.conv_layers:
            x_new = conv_block(x, edge_index, edge_attr)
            x = x + x_new  # residual connection

        # Triple pooling: attention + mean + max
        x_att = self.att_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat((x_att, x_mean, x_max), dim=1)

        logits = self.classifier(x_pooled)

        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
