"""Blueprint-conditioned graph VAE (Section 3.3 of the paper).

Encoder   two message-passing layers (d_feat -> 128 -> 64), ReLU after each, dropout 0.3
          after the first, then linear maps to mu and log sigma^2 (d_z = 31 per node).
Sampling  Z = mu + eps * sigma
Decoder   A_hat = sigmoid(Z Z^T)

Layer types: GCN (uses |blueprint edge weights|), GraphSAGE (structure only), GAT (4 heads
concatenated in layer 1, one head in layer 2; blueprint edge weights enter as edge features).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

CONVS = ("gcn", "sage", "gat")


class BlueprintVAE(nn.Module):
    def __init__(self, conv: str, d_feat: int, hidden_dim: int = 128, out_dim: int = 64,
                 latent_dim: int = 31, dropout: float = 0.3, gat_heads: int = 4):
        super().__init__()
        if conv not in CONVS:
            raise ValueError(f"conv must be one of {CONVS}")
        self.conv = conv
        self.dropout = dropout
        if conv == "gcn":
            self.l1, self.l2 = GCNConv(d_feat, hidden_dim), GCNConv(hidden_dim, out_dim)
        elif conv == "sage":
            self.l1, self.l2 = SAGEConv(d_feat, hidden_dim), SAGEConv(hidden_dim, out_dim)
        else:
            self.l1 = GATConv(d_feat, hidden_dim // gat_heads, heads=gat_heads, edge_dim=1)
            self.l2 = GATConv(hidden_dim, out_dim, heads=1, edge_dim=1)
        self.to_mu = nn.Linear(out_dim, latent_dim)
        self.to_logvar = nn.Linear(out_dim, latent_dim)

    def _gnn(self, layer, x, edge_index, edge_weight):
        if self.conv == "gcn":
            # GCN normalizes by weighted degree, which must be positive: use |correlation|
            return layer(x, edge_index, edge_weight.abs())
        if self.conv == "gat":
            return layer(x, edge_index, edge_attr=edge_weight.unsqueeze(1))
        return layer(x, edge_index)

    def encode(self, x, edge_index, edge_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self._gnn(self.l1, x, edge_index, edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self._gnn(self.l2, h, edge_index, edge_weight))
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x, edge_index, edge_weight, batch_size: int, sample: bool = True):
        """Return decoder logits (B,N,N) (so A_hat = sigmoid(logits)), mu and logvar (B,N,d_z)."""
        mu, logvar = self.encode(x, edge_index, edge_weight)
        z = self.reparameterize(mu, logvar, sample)
        n = z.shape[0] // batch_size
        z = z.view(batch_size, n, -1)
        logits = z @ z.transpose(1, 2)
        return logits, mu.view(batch_size, n, -1), logvar.view(batch_size, n, -1)


def vae_loss(logits: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float):
    """L = L_BCE + beta * L_KL, both averaged over the graphs of the batch.

    L_BCE = mean over all N^2 entries of the binary cross-entropy (Eq. 7),
    L_KL  = -1/(2N) sum_i sum_j (1 + log s2 - mu^2 - s2) (Eq. 8).
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(1, 2))
    n = mu.shape[1]
    kl = -0.5 / n * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1, 2))
    return (bce + beta * kl).mean(), bce.mean(), kl.mean()
