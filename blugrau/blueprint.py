"""Graph blueprint (Section 3.2 of the paper), vectorized over a batch of graphs.

Node description   f_i = [ m_i^(1..K), y, eta_i ]       K centralities, label, Z noise values
Blueprint edges    (A_b)_ij = Pearson(f_i, f_j)          correlation between node descriptions
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def node_descriptions(centralities: torch.Tensor, labels: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Concatenate centralities (B,N,K), labels (B,) and noise (B,N,Z) into (B,N,K+1+Z)."""
    B, N, _ = centralities.shape
    y = labels.to(centralities.dtype).view(B, 1, 1).expand(B, N, 1)
    return torch.cat([centralities, y, noise], dim=2)


def pearson_rows(F: torch.Tensor) -> torch.Tensor:
    """Pearson correlation between all pairs of rows of every matrix in (B, N, D) -> (B, N, N).

    Rows with zero variance get correlation 0 with every other row.
    """
    Fc = F - F.mean(dim=2, keepdim=True)
    cov = Fc @ Fc.transpose(1, 2)
    std = Fc.pow(2).sum(dim=2).sqrt()
    denom = std.unsqueeze(2) * std.unsqueeze(1)
    corr = cov / denom.clamp_min(1e-12)
    return torch.nan_to_num(corr, nan=0.0)


def make_blueprint(centralities: torch.Tensor, labels: torch.Tensor, noise_dim: int = 20,
                   generator: Optional[torch.Generator] = None, noise_std: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build node descriptions F (B,N,d_feat) and blueprint edges A_b (B,N,N) with fresh N(0, std^2 I) noise."""
    B, N, _ = centralities.shape
    noise = noise_std * torch.randn(B, N, noise_dim, generator=generator, device=centralities.device,
                                    dtype=centralities.dtype)
    F = node_descriptions(centralities, labels, noise)
    return F, pearson_rows(F)


def batch_graph(F: torch.Tensor, A_b: torch.Tensor):
    """Flatten a batch of fully connected blueprints for PyG layers.

    Returns node features (B*N, d), edge_index (2, B*N*(N-1)) without self-loops, and the
    blueprint edge weights (Pearson correlations) of shape (B*N*(N-1),).
    """
    B, N, d = F.shape
    device = F.device
    src, dst = torch.where(~torch.eye(N, dtype=torch.bool, device=device))
    offsets = (torch.arange(B, device=device) * N).view(B, 1)
    edge_index = torch.stack([(src.unsqueeze(0) + offsets).reshape(-1), (dst.unsqueeze(0) + offsets).reshape(-1)])
    w = A_b[:, src, dst].reshape(-1)
    return F.reshape(B * N, d), edge_index, w
