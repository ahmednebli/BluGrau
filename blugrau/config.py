"""Default hyperparameters, taken from the paper (Sections 3 and 4)."""
from dataclasses import dataclass, field
from typing import List, Tuple

# Ten centrality measures that form the node roles (K = 10). The paper names degree,
# betweenness, closeness, eigenvector centrality, PageRank, HITS and (for evaluation) Katz.
# The remaining slots are filled with harmonic and load centrality; edit this list to
# match the exact set used in the paper.
CENTRALITIES: Tuple[str, ...] = (
    "strength", "betweenness", "closeness", "harmonic", "eigenvector",
    "pagerank", "hits_hub", "hits_authority", "katz", "load",
)

# Centralities compared between original and generated graphs (Figure 2 of the paper).
TOPOLOGY_MEASURES: Tuple[str, ...] = ("betweenness", "closeness", "pagerank", "katz")


@dataclass
class VAEConfig:
    conv: str = "gat"              # gcn | sage | gat
    noise_dim: int = 20            # Z: noise entries per node in the blueprint
    hidden_dim: int = 128          # first GNN layer
    out_dim: int = 64              # second GNN layer
    latent_dim: int = 31           # d_z per node
    dropout: float = 0.3           # after the first GNN layer
    gat_heads: int = 4             # first GAT layer (second layer uses one head)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 16
    max_epochs: int = 200
    patience: int = 20             # early stopping on validation reconstruction loss
    val_fraction: float = 0.15     # of the training graphs
    beta_max: float = 1e-3         # KL weight after annealing
    beta_anneal_epochs: int = 50   # linear 0 -> beta_max
    noise_std: float = 1.0         # std of the Gaussian blueprint noise (paper: Gaussian, scale not stated)


@dataclass
class ClassifierConfig:
    mlp_hidden: List[int] = field(default_factory=lambda: [512, 256, 64, 8])
    mlp_lr: float = 0.01
    mlp_epochs: int = 100          # not stated in the paper
    mlp_batch_size: int = 32       # not stated in the paper
    svm_c: float = 1.0


@dataclass
class ExperimentConfig:
    seed: int = 42
    test_size: float = 0.2
    stratify: bool = False
    aug_sizes: Tuple[int, ...] = (1000, 2000)   # generated graphs per label
    topology_n: int = 70                         # graphs per group in the topology comparison
