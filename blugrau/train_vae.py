"""Training of the blueprint VAE (Section 3.4 and Implementation Details of the paper)."""
from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from .blueprint import batch_graph, make_blueprint
from .config import VAEConfig
from .model import BlueprintVAE, vae_loss


def _step_inputs(cent, labels, cfg: VAEConfig, generator=None):
    F, A_b = make_blueprint(cent, labels, cfg.noise_dim, generator, cfg.noise_std)
    return batch_graph(F, A_b)


def build_model(cfg: VAEConfig, num_centralities: int) -> BlueprintVAE:
    d_feat = num_centralities + 1 + cfg.noise_dim
    return BlueprintVAE(cfg.conv, d_feat, cfg.hidden_dim, cfg.out_dim, cfg.latent_dim, cfg.dropout, cfg.gat_heads)


@torch.no_grad()
def _validate(model, cent, labels, target, cfg, device, seed: int = 0) -> float:
    """Mean reconstruction (BCE) loss on the validation graphs, with fixed noise for stable monitoring."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    for start in range(0, len(cent), cfg.batch_size):
        sl = slice(start, start + cfg.batch_size)
        c, y, t = cent[sl].to(device), labels[sl].to(device), target[sl].to(device)
        x, ei, ew = _step_inputs(c, y, cfg, gen)
        logits, mu, logvar = model(x, ei, ew, c.shape[0])
        _, bce, _ = vae_loss(logits, t, mu, logvar, 0.0)
        total += bce.item() * c.shape[0]
        count += c.shape[0]
    return total / count


def fit_vae(train: Tuple[np.ndarray, np.ndarray, np.ndarray], val: Tuple[np.ndarray, np.ndarray, np.ndarray],
            cfg: VAEConfig, device: torch.device, verbose: bool = True) -> Tuple[BlueprintVAE, List[Dict]]:
    """Train with KL annealing and early stopping on the validation reconstruction loss.

    ``train`` and ``val`` are tuples ``(centralities (G,N,K), labels (G,), adjacency in [0,1] (G,N,N))``.
    Returns the best model (lowest validation loss) and the training history.
    """
    tc, ty, tt = (torch.as_tensor(a, dtype=torch.float32) for a in train)
    vc, vy, vt = (torch.as_tensor(a, dtype=torch.float32) for a in val)
    model = build_model(cfg, tc.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(cfg.max_epochs):
        beta = cfg.beta_max * min(1.0, epoch / cfg.beta_anneal_epochs)
        model.train()
        perm = torch.randperm(len(tc))
        tot, n_seen = 0.0, 0
        for start in range(0, len(tc), cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            c, y, t = tc[idx].to(device), ty[idx].to(device), tt[idx].to(device)
            x, ei, ew = _step_inputs(c, y, cfg)
            logits, mu, logvar = model(x, ei, ew, c.shape[0])
            loss, _, _ = vae_loss(logits, t, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * c.shape[0]
            n_seen += c.shape[0]
        val_loss = _validate(model, vc, vy, vt, cfg, device)
        if not np.isfinite(val_loss) or not np.isfinite(tot):
            raise RuntimeError(f"non-finite loss at epoch {epoch + 1}; check the inputs (NaN/inf in centralities?)")
        history.append({"epoch": epoch + 1, "beta": beta, "train_loss": tot / n_seen, "val_recon": val_loss})
        if verbose and (epoch % 10 == 0 or epoch == cfg.max_epochs - 1):
            print(f"  [{cfg.conv}] epoch {epoch + 1:3d} beta={beta:.1e} train={tot / n_seen:.4f} val_recon={val_loss:.4f}")
        if val_loss < best_val - 1e-6:
            best_val, best_epoch, best_state = val_loss, epoch + 1, copy.deepcopy(model.state_dict())
        elif epoch + 1 - best_epoch >= cfg.patience:
            if verbose:
                print(f"  [{cfg.conv}] early stop at epoch {epoch + 1} (best {best_epoch}, val {best_val:.4f})")
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def vae_config_dict(cfg: VAEConfig) -> Dict:
    return asdict(cfg)
