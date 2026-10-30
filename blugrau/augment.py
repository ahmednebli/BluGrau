"""Generation of augmented graphs (Section 3.5, Algorithm 1 of the paper)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .blueprint import batch_graph, make_blueprint
from .config import VAEConfig
from .model import BlueprintVAE


@torch.no_grad()
def generate_from_blueprints(model: BlueprintVAE, centralities: np.ndarray, labels: np.ndarray,
                             cfg: VAEConfig, device: torch.device, batch_size: int = 32,
                             seed: int = 0, zero_diagonal: bool = False) -> np.ndarray:
    """Decode one new graph per row of ``centralities``/``labels`` with fresh blueprint noise.

    The latent code is sampled from the posterior (dropout is off). Returns (n, N, N) in (0, 1).
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    cent = torch.as_tensor(centralities, dtype=torch.float32)
    lab = torch.as_tensor(labels, dtype=torch.float32)
    out = []
    for start in range(0, len(cent), batch_size):
        c, y = cent[start:start + batch_size].to(device), lab[start:start + batch_size].to(device)
        F, A_b = make_blueprint(c, y, cfg.noise_dim, gen, cfg.noise_std)
        x, ei, ew = batch_graph(F, A_b)
        logits, _, _ = model(x, ei, ew, c.shape[0], sample=True)
        A_hat = torch.sigmoid(logits)
        if zero_diagonal:
            A_hat = A_hat * (1 - torch.eye(A_hat.shape[1], device=device))
        out.append(A_hat.cpu().numpy())
    return np.concatenate(out, axis=0)


def augment(model: BlueprintVAE, centralities: np.ndarray, labels: np.ndarray, n_per_label: int,
            cfg: VAEConfig, device: torch.device, seed: int = 0, zero_diagonal: bool = False
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ``n_per_label`` graphs for every class.

    For each new graph a training graph of the class is drawn uniformly with replacement; its
    blueprint is rebuilt with fresh noise and decoded. The paper does not specify how source
    graphs are chosen, so this is an implementation choice.
    """
    rng = np.random.default_rng(seed)
    graphs, out_labels = [], []
    for c in np.unique(labels):
        pool = np.where(labels == c)[0]
        src = rng.choice(pool, size=n_per_label, replace=True)
        gen = generate_from_blueprints(model, centralities[src], labels[src], cfg, device,
                                       seed=int(seed * 1000 + c), zero_diagonal=zero_diagonal)
        graphs.append(gen)
        out_labels.append(np.full(n_per_label, c, dtype=int))
    return np.concatenate(graphs), np.concatenate(out_labels)
