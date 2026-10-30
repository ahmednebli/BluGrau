"""Dataset loading, splitting and normalization.

On disk a dataset is a folder with two files::

    adjacency.npy   float array (num_graphs, N, N), one connectome per subject
    labels.npy      int array   (num_graphs,), binary class labels

``prepare_data.py`` creates them from the per-subject CSV layout of the sex cohort or
from arbitrary arrays (for example the cannabis cohort exported from Grenol-Net).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------- IO
def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    adjacency = np.load(os.path.join(data_dir, "adjacency.npy")).astype(np.float64)
    labels = np.load(os.path.join(data_dir, "labels.npy")).astype(int)
    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError(f"adjacency.npy must have shape (G, N, N), got {adjacency.shape}")
    if len(labels) != len(adjacency):
        raise ValueError("adjacency.npy and labels.npy have different lengths")
    return adjacency, labels


def save_dataset(adjacency: np.ndarray, labels: np.ndarray, data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "adjacency.npy"), adjacency)
    np.save(os.path.join(data_dir, "labels.npy"), labels)


def from_sex_cohort_csv(raw_dir: str, positive: str = "male") -> Tuple[np.ndarray, np.ndarray]:
    """Read the sex-classification cohort.

    ``raw_dir/clinics.csv`` (separator ';', column ``gender``) and
    ``raw_dir/subjects/S001.csv, S002.csv, ...`` (headerless N x N matrices). Row ``i``
    of ``clinics.csv`` belongs to ``S{i+1:03}.csv``. Label 1 = ``positive`` gender.
    """
    clinics = pd.read_csv(os.path.join(raw_dir, "clinics.csv"), sep=";")
    mats, labels = [], []
    for idx, row in clinics.iterrows():
        path = os.path.join(raw_dir, "subjects", f"S{idx + 1:03}.csv")
        mats.append(pd.read_csv(path, header=None).values)
        labels.append(int(row["gender"] == positive))
    return np.array(mats, dtype=np.float64), np.array(labels, dtype=int)


# ------------------------------------------------------------------------ splitting
@dataclass
class Split:
    train: np.ndarray   # indices used to fit everything (VAE and classifiers)
    val: np.ndarray     # unaugmented validation indices (early stopping / model selection)
    test: np.ndarray    # unaugmented test indices


def make_split(labels: np.ndarray, seed: int = 42, test_size: float = 0.2,
               val_fraction: float = 0.15, stratify: bool = False) -> Split:
    """80/20 train-test split, then ``val_fraction`` of the training part held out for validation."""
    idx = np.arange(len(labels))
    strat = labels if stratify else None
    train_all, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    strat_tr = labels[train_all] if stratify else None
    train, val = train_test_split(train_all, test_size=val_fraction, random_state=seed, stratify=strat_tr)
    return Split(train=np.sort(train), val=np.sort(val), test=np.sort(test))


# ----------------------------------------------------------------------- normalizing
class MinMaxScaler:
    """Global min-max scaling of adjacency entries to [0, 1], fitted on training graphs only."""

    def fit(self, adjacency: np.ndarray) -> "MinMaxScaler":
        self.lo = float(adjacency.min())
        self.hi = float(adjacency.max())
        if self.hi <= self.lo:
            raise ValueError("cannot scale a constant adjacency array")
        return self

    def transform(self, adjacency: np.ndarray) -> np.ndarray:
        return np.clip((adjacency - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def state(self) -> Dict[str, float]:
        return {"lo": self.lo, "hi": self.hi}
