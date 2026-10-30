#!/usr/bin/env python
"""Write a small random two-class dataset (adjacency.npy, labels.npy) to test the pipeline.

Class 1 has slightly stronger connections inside a block of regions. No neuroscience meaning.
"""
import argparse

import numpy as np

from blugrau.data import save_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--graphs", type=int, default=40)
    p.add_argument("--nodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    labels = np.arange(a.graphs) % 2
    mats = []
    for y in labels:
        m = rng.random((a.nodes, a.nodes)) * (rng.random((a.nodes, a.nodes)) < 0.4)
        m[: a.nodes // 3, : a.nodes // 3] *= 1.0 + 0.8 * y
        m = (m + m.T) / 2
        np.fill_diagonal(m, 0)
        mats.append(m)
    save_dataset(np.array(mats), labels, a.out_dir)
    print(f"wrote {a.graphs} graphs with {a.nodes} nodes to {a.out_dir}")


if __name__ == "__main__":
    main()
