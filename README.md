# BluGrAu: Blueprint Graph Augmentation for Brain Graphs

Reference implementation of **BluGrAu**, a data augmentation method for brain graphs (connectomes) that varies connection weights while keeping the topological role of every brain region fixed.

1. **Graph blueprint.** Each region is described by ten centrality measures, the class label and random noise. Two regions are connected in the blueprint by the Pearson correlation of their descriptions.
2. **Blueprint-conditioned graph VAE.** A two-layer GNN encoder (GCN, GraphSAGE or GAT) maps the blueprint to a latent code per node, and an inner-product decoder turns the code into a weighted adjacency matrix.
3. **Augmentation.** Fresh noise gives a new blueprint of the same input graph, so a single graph yields arbitrarily many new graphs that carry its label.

The code follows the paper section by section. Every choice that the paper leaves open is listed in [`docs/implementation_notes.md`](docs/implementation_notes.md), together with points that deserve checking on real data.

> **Status.** The pipeline is covered by unit tests and was run end to end on synthetic data. It has not been run on the two real cohorts, so the numbers of the paper have not been reproduced by this repository yet. Read the section "Checks on real data" in the implementation notes before reporting results.

## Installation

```bash
git clone <your-repository-url>
cd BluGrAu
python -m venv .venv && source .venv/bin/activate
pip install torch            # choose the build for your platform: https://pytorch.org
pip install -e .             # installs the remaining dependencies
pytest -q                    # optional: fast unit tests
```

## Data

A dataset is a folder with `adjacency.npy` (shape `G x N x N`) and `labels.npy` (shape `G`, binary).

```bash
# Sex cohort: clinics.csv (separator ';', column "gender") and subjects/S001.csv, S002.csv, ...
python scripts/prepare_data.py --format sex-csv --raw-dir raw/sex --out-dir data/sex

# Any other cohort, for example the cannabis cohort exported from Grenol-Net (35 x 35 matrices)
python scripts/prepare_data.py --format arrays --adjacency cannabis_adj.npy --labels cannabis_labels.npy \
       --out-dir data/cannabis
```

No data are included. To try the code, create a random dataset:

```bash
python scripts/make_synthetic_data.py --out-dir data/synthetic --graphs 40 --nodes 20
```

## Run the experiment of the paper

```bash
python scripts/run_experiment.py --data-dir data/sex --out-dir results/sex
python scripts/make_figures.py --out-dir results/sex
```

This trains one VAE per encoder (GCN, GraphSAGE, GAT), generates 1000 and 2000 graphs per label, trains an MLP and a linear SVM on the original plus generated graphs, and compares centralities of original and generated graphs. Useful options:

| Option | Meaning |
|---|---|
| `--convs gcn sage gat` | encoders to evaluate |
| `--aug-sizes 1000 2000` | generated graphs per label |
| `--seed 42` | seed of the 80/20 split and of all training runs |
| `--stratify` | stratified train/test split (the paper does not state it) |
| `--noise-std 1.0` | standard deviation of the blueprint noise |
| `--centralities ...` | override the ten centrality measures |
| `--zero-diagonal` | set the diagonal of generated graphs to zero |
| `--n-jobs 8` | processes for centrality computation |
| `--vae-epochs`, `--mlp-epochs` | shorten runs for testing |

Outputs in `--out-dir`:

| File | Content |
|---|---|
| `results.csv`, `results.md` | accuracy and AUC of the MLP and the SVM for the baseline and every augmenter and size |
| `topology.csv` | mean and standard error of betweenness, closeness, PageRank and Katz centrality for original and generated graphs, with the relative difference |
| `examples.npz` | one original graph and the graph generated from its blueprint by each encoder |
| `vae_<conv>.pt`, `vae_history_<conv>.csv` | trained weights and training curves |
| `centralities.npy`, `config.json` | cached centralities and all settings used |
| `fig_topology.pdf`, `fig_qualitative.pdf` | figures (after `make_figures.py`) |

To generate an augmented set without the evaluation:

```bash
python scripts/augment.py --data-dir data/sex --out aug_gat.npz --conv gat --n-per-label 1000
```

## Default hyperparameters

All defaults live in `blugrau/config.py` and follow the paper.

| Component | Setting |
|---|---|
| Blueprint | K = 10 centralities, label, Z = 20 Gaussian noise values per node, d_feat = 31, Pearson edges |
| Encoder | two layers, 31 to 128 to 64, ReLU, dropout 0.3 after the first layer; GAT: 4 heads (concatenated), then 1 head |
| Latent | 31 dimensions per node, reparameterization trick |
| Decoder | sigmoid of the inner product of latent codes |
| Loss | entry-wise BCE plus beta times KL, beta annealed linearly from 0 to 1e-3 over 50 epochs |
| Optimization | Adam, learning rate 1e-3, weight decay 1e-5, batch size 16, at most 200 epochs |
| Early stopping | patience 20 on the validation reconstruction loss, 15% of the training graphs held out |
| Classifiers | MLP on flattened matrices with hidden layers 512, 256, 64, 8 and one output unit, Adam with learning rate 0.01; linear SVM |
| Split | random 80/20 train/test split with seed 42; validation and test graphs are never augmented |

## Repository structure

```
blugrau/
  config.py       default hyperparameters (paper values)
  data.py         dataset IO, train/val/test split, min-max scaling
  centrality.py   ten centrality measures on weighted directed graphs
  blueprint.py    node descriptions, Pearson blueprint edges, batching for PyG
  model.py        blueprint VAE (GCN / GraphSAGE / GAT encoders) and the loss
  train_vae.py    training with KL annealing and early stopping
  augment.py      generation of augmented graphs (Algorithm 1)
  classify.py     MLP and linear SVM
  topology.py     centrality comparison of original and generated graphs
  experiment.py   end-to-end experiment
scripts/          command-line entry points
tests/            unit tests
docs/             implementation notes
```

## Citation

```bibtex
@misc{sahebghadam2026blugrau,
  title  = {BluGrAu: Blueprint Graph Augmentation for Topology-Aware Brain Graph Classification},
  author = {Sahebghadam, Helia and Ben Dhifallah, Montassar and Touihri, Mariem and Nebli, Ahmed},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Add a license file before publishing the repository.
