"""End-to-end experiment of the paper: baseline vs. BluGrAu augmentation with GCN / GraphSAGE / GAT."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from .augment import augment, generate_from_blueprints
from .centrality import compute_centralities
from .classify import run_mlp, run_svm
from .config import CENTRALITIES, ClassifierConfig, ExperimentConfig, VAEConfig
from .data import MinMaxScaler, load_dataset, make_split
from .topology import compare
from .train_vae import fit_vae
from .utils import get_device, set_seed


def _first_n_per_label(graphs: np.ndarray, labels: np.ndarray, n: int):
    keep = np.concatenate([np.where(labels == c)[0][:n] for c in np.unique(labels)])
    return graphs[keep], labels[keep]


def _eval_row(name: str, n: int, train, val, test, ccfg: ClassifierConfig, device, seed: int) -> Dict:
    mlp = run_mlp(train, val, test, ccfg, device, seed)
    svm = run_svm(train, test, ccfg, seed)
    row = {"augmenter": name, "n_per_label": n, "mlp_acc": mlp["acc"], "mlp_auc": mlp["auc"],
           "svm_acc": svm["acc"], "svm_auc": svm["auc"]}
    print("  " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    return row


def run_experiment(data_dir: str, out_dir: str, convs: Sequence[str] = ("gcn", "sage", "gat"),
                   vae_cfg: VAEConfig | None = None, cls_cfg: ClassifierConfig | None = None,
                   exp_cfg: ExperimentConfig | None = None, centralities: Sequence[str] = CENTRALITIES,
                   device: str | None = None, n_jobs: int = 1, zero_diagonal: bool = False) -> pd.DataFrame:
    vae_cfg, cls_cfg, exp_cfg = vae_cfg or VAEConfig(), cls_cfg or ClassifierConfig(), exp_cfg or ExperimentConfig()
    os.makedirs(out_dir, exist_ok=True)
    dev = get_device(device)
    set_seed(exp_cfg.seed)

    A, y = load_dataset(data_dir)
    G, N, _ = A.shape
    split = make_split(y, exp_cfg.seed, exp_cfg.test_size, vae_cfg.val_fraction, exp_cfg.stratify)
    print(f"graphs={G} nodes={N} train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    scaler = MinMaxScaler().fit(A[split.train])
    An = scaler.transform(A)

    # Centralities are needed only for graphs that feed the VAE (train and validation).
    cache = os.path.join(out_dir, "centralities.npy")
    fit_idx = np.concatenate([split.train, split.val])
    if os.path.exists(cache):
        cent = np.load(cache)
    else:
        print(f"computing {len(centralities)} centralities for {len(fit_idx)} graphs ...")
        cent = np.zeros((G, N, len(centralities)))
        cent[fit_idx] = compute_centralities(An[fit_idx], centralities, n_jobs)
        np.save(cache, cent)

    train_set = (An[split.train], y[split.train])
    val_set, test_set = (An[split.val], y[split.val]), (An[split.test], y[split.test])

    rows: List[Dict] = []
    print("baseline (no augmentation)")
    rows.append(_eval_row("No Aug.", 0, train_set, val_set, test_set, cls_cfg, dev, exp_cfg.seed))

    n_max = max(exp_cfg.aug_sizes)
    examples = {"reference": An[split.train][0]}
    topo_tables = []
    rng = np.random.default_rng(exp_cfg.seed)
    for conv in convs:
        print(f"BluGrAu with {conv.upper()} encoder")
        cfg = replace(vae_cfg, conv=conv)
        set_seed(exp_cfg.seed)
        model, history = fit_vae(
            (cent[split.train], y[split.train], An[split.train]),
            (cent[split.val], y[split.val], An[split.val]), cfg, dev)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, f"vae_history_{conv}.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(out_dir, f"vae_{conv}.pt"))

        gen, gen_y = augment(model, cent[split.train], y[split.train], n_max, cfg, dev,
                             seed=exp_cfg.seed, zero_diagonal=zero_diagonal)
        for n in sorted(exp_cfg.aug_sizes):
            g, gl = _first_n_per_label(gen, gen_y, n)
            aug_train = (np.concatenate([train_set[0], g]), np.concatenate([train_set[1], gl]))
            rows.append(_eval_row(conv.upper() if conv != "sage" else "GraphSAGE", n, aug_train, val_set, test_set,
                                  cls_cfg, dev, exp_cfg.seed))

        # topology: matched numbers of original and generated graphs
        n_match = min(exp_cfg.topology_n, len(train_set[0]), len(gen))
        sel = rng.choice(len(gen), size=n_match, replace=False)
        tab = compare(train_set[0][:n_match], gen[sel], n_jobs=n_jobs)
        tab.insert(0, "augmenter", conv)
        topo_tables.append(tab)

        examples[f"pred_{conv}"] = generate_from_blueprints(
            model, cent[split.train][:1], y[split.train][:1], cfg, dev, seed=exp_cfg.seed,
            zero_diagonal=zero_diagonal)[0]

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    pd.concat(topo_tables).to_csv(os.path.join(out_dir, "topology.csv"), index=False)
    np.savez(os.path.join(out_dir, "examples.npz"), **examples)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({"vae": asdict(vae_cfg), "classifier": asdict(cls_cfg), "experiment": asdict(exp_cfg),
                   "centralities": list(centralities), "scaler": scaler.state(),
                   "split_sizes": {"train": len(split.train), "val": len(split.val), "test": len(split.test)}},
                  fh, indent=2, default=list)
    _write_markdown(results, os.path.join(out_dir, "results.md"))
    print(results.round(3).to_string(index=False))
    return results


def _write_markdown(df: pd.DataFrame, path: str) -> None:
    head = "| Augmenter | n per label | MLP Acc | MLP AUC | SVM Acc | SVM AUC |\n|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {r.augmenter} | {r.n_per_label} | {r.mlp_acc:.2f} | {r.mlp_auc:.2f} | {r.svm_acc:.2f} | {r.svm_auc:.2f} |\n"
        for r in df.itertuples())
    with open(path, "w") as fh:
        fh.write(head + body)
