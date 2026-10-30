#!/usr/bin/env python
"""Run the full experiment of the paper on one dataset.

Baseline (no augmentation) versus BluGrAu with GCN, GraphSAGE and GAT encoders, at 1000 and
2000 generated graphs per label, evaluated with an MLP and a linear SVM. Also writes the
topology comparison and the example graphs used by make_figures.py.

Outputs in --out-dir: results.csv / results.md, topology.csv, examples.npz, config.json,
vae_<conv>.pt and vae_history_<conv>.csv.
"""
import argparse

from blugrau.config import CENTRALITIES, ClassifierConfig, ExperimentConfig, VAEConfig
from blugrau.experiment import run_experiment


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with adjacency.npy and labels.npy")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--convs", nargs="+", default=["gcn", "sage", "gat"], choices=["gcn", "sage", "gat"])
    p.add_argument("--aug-sizes", nargs="+", type=int, default=[1000, 2000])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify", action="store_true", help="stratified train/test split (paper: not stated)")
    p.add_argument("--vae-epochs", type=int, default=VAEConfig.max_epochs)
    p.add_argument("--mlp-epochs", type=int, default=ClassifierConfig().mlp_epochs)
    p.add_argument("--noise-std", type=float, default=VAEConfig.noise_std, help="std of the blueprint noise")
    p.add_argument("--centralities", nargs="+", default=list(CENTRALITIES))
    p.add_argument("--zero-diagonal", action="store_true", help="zero the diagonal of generated graphs")
    p.add_argument("--n-jobs", type=int, default=1, help="processes for centrality computation")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    run_experiment(
        a.data_dir, a.out_dir, a.convs,
        vae_cfg=VAEConfig(max_epochs=a.vae_epochs, noise_std=a.noise_std),
        cls_cfg=ClassifierConfig(mlp_epochs=a.mlp_epochs),
        exp_cfg=ExperimentConfig(seed=a.seed, stratify=a.stratify, aug_sizes=tuple(a.aug_sizes)),
        centralities=a.centralities, device=a.device, n_jobs=a.n_jobs, zero_diagonal=a.zero_diagonal)


if __name__ == "__main__":
    main()
