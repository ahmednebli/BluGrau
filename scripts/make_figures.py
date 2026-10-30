#!/usr/bin/env python
"""Draw the topology and qualitative figures from the outputs of run_experiment.py.

Reads  <out-dir>/topology.csv and <out-dir>/examples.npz
Writes <out-dir>/fig_topology.pdf and <out-dir>/fig_qualitative.pdf
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REAL, AUG = "#0072B2", "#D55E00"
NAMES = {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}


def topology_figure(out_dir: str) -> None:
    df = pd.read_csv(os.path.join(out_dir, "topology.csv"))
    augmenters = list(dict.fromkeys(df["augmenter"]))
    measures = list(dict.fromkeys(df["measure"]))
    fig, axes = plt.subplots(len(augmenters), len(measures), figsize=(1.6 * len(measures), 1.7 * len(augmenters) + 0.5),
                             squeeze=False)
    for r, aug in enumerate(augmenters):
        for c, m in enumerate(measures):
            row = df[(df.augmenter == aug) & (df.measure == m)].iloc[0]
            ax = axes[r, c]
            ax.bar([0], [row.original_mean], 0.7, color=REAL, yerr=[row.original_se], capsize=2.5)
            ax.bar([1], [row.generated_mean], 0.7, color=AUG, yerr=[row.generated_se], capsize=2.5)
            top = max(row.original_mean + row.original_se, row.generated_mean + row.generated_se)
            ax.set_ylim(0, top * 1.4)
            for x, v, s in ((0, row.original_mean, row.original_se), (1, row.generated_mean, row.generated_se)):
                ax.text(x, v + s + top * 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
            ax.text(0.5, 0.97, f"{row.relative_diff_pct:+.0f}%", transform=ax.transAxes, ha="center", va="top",
                    fontsize=7, fontweight="bold")
            ax.set_xticks([])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(m.capitalize(), fontsize=8, fontweight="bold")
            if c == 0:
                ax.set_ylabel(NAMES.get(aug, aug), fontsize=8)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=REAL), plt.Rectangle((0, 0), 1, 1, color=AUG)],
               ["Original graphs", "BluGrAu graphs"], loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out_dir, "fig_topology.pdf"))


def qualitative_figure(out_dir: str) -> None:
    ex = np.load(os.path.join(out_dir, "examples.npz"))
    ref = ex["reference"]
    preds = {k[5:]: ex[k] for k in ex.files if k.startswith("pred_")}
    cols = 1 + len(preds)
    fig, axes = plt.subplots(2, cols, figsize=(1.6 * cols, 3.4), squeeze=False)
    axes[0, 0].imshow(ref, cmap="viridis", vmin=0, vmax=1)
    axes[0, 0].set_title("Original", fontsize=8)
    axes[1, 0].axis("off")
    for j, (name, pred) in enumerate(preds.items(), start=1):
        axes[0, j].imshow(pred, cmap="viridis", vmin=0, vmax=1)
        axes[0, j].set_title(NAMES.get(name, name), fontsize=8)
        axes[1, j].imshow(np.abs(pred - ref), cmap="viridis", vmin=0, vmax=1)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    axes[1, 0].text(0.5, 0.5, "Bottom row:\n|generated - original|", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_qualitative.pdf"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    topology_figure(a.out_dir)
    qualitative_figure(a.out_dir)
    print(f"figures written to {a.out_dir}")


if __name__ == "__main__":
    main()
