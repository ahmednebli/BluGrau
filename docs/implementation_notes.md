# Implementation notes

## 1. Paper to code

| Paper | Code |
|---|---|
| 3.1 Notation, adjacency normalized to [0, 1] | `data.MinMaxScaler`, fitted on training graphs and clipped |
| 3.2 Node roles, ten centralities | `centrality.compute_centralities`, `config.CENTRALITIES` |
| 3.2 Node description, Eq. 1 | `blueprint.node_descriptions` |
| 3.2 Blueprint edges, Eq. 2 | `blueprint.pearson_rows` |
| 3.3 Encoder, Eq. 3 | `model.BlueprintVAE.encode` |
| 3.3 Latent distribution and sampling, Eq. 4 and 5 | `model.BlueprintVAE.to_mu`, `to_logvar`, `reparameterize` |
| 3.3 Decoder, Eq. 6 | `model.BlueprintVAE.forward` (logits = ZZ^T, A_hat = sigmoid) |
| 3.4 Objective, Eq. 7 and 8, KL annealing | `model.vae_loss`, `train_vae.fit_vae` |
| 3.5 Algorithm 1 | `train_vae.fit_vae`, `augment.augment` |
| 4.2 Downstream evaluation | `classify.run_mlp`, `classify.run_svm`, `experiment.run_experiment` |
| 5.2 Topology preservation | `topology.compare`, `scripts/make_figures.py` |
| 5.3 Qualitative inspection | `examples.npz`, `scripts/make_figures.py` |

## 2. Choices the paper does not specify

These are decisions made in the code. If the experiments of the paper used something else, change the corresponding line and note it in the paper.

1. **The ten centralities.** The paper names degree, betweenness, closeness, eigenvector centrality, PageRank, HITS and (for evaluation) Katz. The default list is strength (weighted degree), betweenness, closeness, harmonic, eigenvector, PageRank, HITS hub, HITS authority, Katz and load. Harmonic and load centrality are placeholders for the two measures that the paper does not name. Edit `config.CENTRALITIES`.
2. **Graph interpretation for centralities.** Matrices are weighted directed graphs, negative entries and the diagonal are dropped, and for path-based measures (betweenness, closeness, harmonic, load) the edge length is the inverse of the weight, because weights are connection strengths.
3. **Centrality scaling.** Every centrality is min-max normalized across the nodes of each graph. Without a common scale, measures with large values would dominate the Pearson correlation.
4. **Blueprint noise.** Standard normal noise scaled by `noise_std` (default 1.0), drawn anew for every blueprint, including at every training step.
5. **Edge weights in the encoder.** GCN uses the absolute value of the correlation, because its degree normalization needs positive weights. GAT receives the signed correlation as an edge feature. GraphSAGE ignores edge weights, so for it the blueprint only supplies a fully connected structure.
6. **Choice of source graphs.** For every generated graph, a training graph of the requested class is drawn uniformly with replacement. The smaller augmentation size uses the first n generated graphs of each class from the larger draw.
7. **Generation.** The latent code is sampled from the posterior, dropout is off, and the diagonal of the output is kept (use `--zero-diagonal` to remove it).
8. **Split and validation.** `train_test_split` with seed 42, not stratified by default. 15% of the training graphs form the validation set that is used for VAE early stopping and for choosing the MLP epoch.
9. **Classifier details not given.** MLP: 100 epochs, batch size 32, epoch with the lowest validation loss. SVM: `LinearSVC` with C = 1, AUC from the signed decision function. Accuracy uses a threshold of 0.5 on the MLP probability and 0 on the SVM score.
10. **Topology evaluation.** Each centrality is normalized per graph as in item 3 and averaged over nodes. Seventy original training graphs are compared with seventy randomly chosen generated graphs. Katz centrality uses alpha = 0.5 divided by the spectral radius of the graph.

## 3. Checks on real data

The code was tested on synthetic data only. Four properties of the method as described deserve a look when the real cohorts are run, because they can limit what the model reproduces.

1. **Decoder capacity.** The decoder returns `sigmoid(ZZ^T)`. A Gram matrix cannot be negative on average, and its diagonal is non-negative, so the output diagonal is at least 0.5 and the mean logit is at least zero. Real connectomes have a zero diagonal and many weak entries. On the synthetic data, the validation reconstruction loss plateaued near 0.72, while a constant predictor reaches 0.36. Whether this also happens on the real cohorts is open. If the reconstruction is poor, `--zero-diagonal` fixes the diagonal only at generation time.
2. **Noise scale against centralities.** The centralities lie in [0, 1] after normalization, while the default noise has unit variance and 20 dimensions. The Pearson correlation between node descriptions can then be dominated by noise, which weakens the topological signal in the blueprint. Try `--noise-std` values below 1 and compare topology and classification results.
3. **Figures of the paper.** The reconstruction examples in the paper show a dark diagonal and a near-exact copy of the original. A decoder of the form `sigmoid(ZZ^T)` with a noise-dominated input would not be expected to do this. If the figures were produced by a different architecture, the code and the paper text need to be reconciled before the repository is cited as the source of those figures.
4. **Data leakage.** Centralities, scaling and the VAE use training graphs only. For the longitudinal cannabis cohort, make sure that both time points of a subject fall on the same side of the split, because the split in this code is by graph, not by subject.
