#!/usr/bin/env python
"""Train one BluGrAu model on the training split and write an augmented graph set.

Output: <out> (.npz) with arrays ``adjacency`` (n, N, N) in [0, 1] and ``labels`` (n,).
Adjacency values are on the min-max scale fitted on the training graphs (stored as ``lo``, ``hi``).
"""
import argparse

import numpy as np

from blugrau.augment import augment
from blugrau.config import CENTRALITIES, VAEConfig
from blugrau.centrality import compute_centralities
from blugrau.data import MinMaxScaler, load_dataset, make_split
from blugrau.train_vae import fit_vae
from blugrau.utils import get_device, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conv", choices=["gcn", "sage", "gat"], default="gat")
    p.add_argument("--n-per-label", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vae-epochs", type=int, default=VAEConfig.max_epochs)
    p.add_argument("--zero-diagonal", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    set_seed(a.seed)
    dev = get_device(a.device)
    A, y = load_dataset(a.data_dir)
    split = make_split(y, a.seed)
    scaler = MinMaxScaler().fit(A[split.train])
    An = scaler.transform(A)
    cfg = VAEConfig(conv=a.conv, max_epochs=a.vae_epochs)
    idx = np.concatenate([split.train, split.val])
    cent = np.zeros((len(A), A.shape[1], len(CENTRALITIES)))
    cent[idx] = compute_centralities(An[idx], CENTRALITIES, a.n_jobs)

    model, _ = fit_vae((cent[split.train], y[split.train], An[split.train]),
                       (cent[split.val], y[split.val], An[split.val]), cfg, dev)
    adj, lab = augment(model, cent[split.train], y[split.train], a.n_per_label, cfg, dev,
                       seed=a.seed, zero_diagonal=a.zero_diagonal)
    np.savez(a.out, adjacency=adj, labels=lab, **scaler.state())
    print(f"wrote {len(adj)} graphs to {a.out}")


if __name__ == "__main__":
    main()
