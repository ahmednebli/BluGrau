"""Downstream classifiers on flattened adjacency matrices: MLP and linear SVM (Section 4)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import LinearSVC
from torch import nn

from .config import ClassifierConfig


def _flatten(A: np.ndarray) -> np.ndarray:
    return A.reshape(len(A), -1).astype(np.float32)


def _metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (score > threshold).astype(int)
    auc = roc_auc_score(y_true, score) if len(np.unique(y_true)) == 2 else float("nan")
    return {"acc": float(accuracy_score(y_true, pred)), "auc": float(auc)}


def make_mlp(in_dim: int, hidden) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def run_mlp(train: Tuple[np.ndarray, np.ndarray], val: Tuple[np.ndarray, np.ndarray],
            test: Tuple[np.ndarray, np.ndarray], cfg: ClassifierConfig, device: torch.device,
            seed: int = 42) -> Dict[str, float]:
    """Train the MLP (Adam) and report test accuracy / AUC of the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    Xtr, ytr = torch.as_tensor(_flatten(train[0])), torch.as_tensor(train[1], dtype=torch.float32)
    Xva, yva = torch.as_tensor(_flatten(val[0])).to(device), torch.as_tensor(val[1], dtype=torch.float32).to(device)
    Xte = torch.as_tensor(_flatten(test[0])).to(device)
    model = make_mlp(Xtr.shape[1], cfg.mlp_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.mlp_lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best, best_state = float("inf"), None
    for _ in range(cfg.mlp_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), cfg.mlp_batch_size):
            idx = perm[s:s + cfg.mlp_batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss_fn(model(xb).squeeze(1), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva).squeeze(1), yva).item()
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(Xte).squeeze(1)).cpu().numpy()
    return _metrics(test[1], prob, 0.5)


def run_svm(train: Tuple[np.ndarray, np.ndarray], test: Tuple[np.ndarray, np.ndarray],
            cfg: ClassifierConfig, seed: int = 42) -> Dict[str, float]:
    """Linear SVM; AUC is computed from the signed decision function."""
    clf = LinearSVC(C=cfg.svm_c, dual="auto", max_iter=20000, random_state=seed)
    clf.fit(_flatten(train[0]), train[1])
    return _metrics(test[1], clf.decision_function(_flatten(test[0])), 0.0)
