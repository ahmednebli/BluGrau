"""Fast checks of the core components (run with ``pytest``)."""
import numpy as np
import torch

from blugrau.augment import augment
from blugrau.blueprint import batch_graph, make_blueprint, pearson_rows
from blugrau.centrality import compute_centralities
from blugrau.config import CENTRALITIES, VAEConfig
from blugrau.data import MinMaxScaler, make_split
from blugrau.model import BlueprintVAE, vae_loss
from blugrau.train_vae import build_model, fit_vae


def _toy(g=12, n=10, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((g, n, n)) * (rng.random((g, n, n)) < 0.5)
    a = (a + a.transpose(0, 2, 1)) / 2
    for m in a:
        np.fill_diagonal(m, 0)
    return a, np.arange(g) % 2


def test_pearson_matches_numpy():
    F = torch.randn(2, 6, 9)
    got = pearson_rows(F)[1].numpy()
    ref = np.corrcoef(F[1].numpy())
    assert np.allclose(got, ref, atol=1e-5)


def test_centralities_shape_and_range():
    a, _ = _toy()
    c = compute_centralities(a[:3], CENTRALITIES)
    assert c.shape == (3, 10, len(CENTRALITIES))
    assert np.isfinite(c).all() and c.min() >= -1e-9 and c.max() <= 1 + 1e-9


def test_blueprint_dimensions():
    cent = torch.rand(4, 10, 10)
    F, Ab = make_blueprint(cent, torch.tensor([0, 1, 0, 1]), noise_dim=20)
    assert F.shape == (4, 10, 31) and Ab.shape == (4, 10, 10)
    x, ei, ew = batch_graph(F, Ab)
    assert x.shape == (40, 31) and ei.shape == (2, 4 * 10 * 9) and ew.shape == (4 * 10 * 9,)


def test_vae_forward_and_loss_for_all_encoders():
    cent = torch.rand(3, 10, 10)
    for conv in ("gcn", "sage", "gat"):
        model = BlueprintVAE(conv, d_feat=31)
        F, Ab = make_blueprint(cent, torch.tensor([0, 1, 0]))
        logits, mu, logvar = model(*batch_graph(F, Ab), batch_size=3)
        assert logits.shape == (3, 10, 10) and mu.shape == (3, 10, 31)
        loss, bce, kl = vae_loss(logits, torch.rand(3, 10, 10), mu, logvar, beta=1e-3)
        assert torch.isfinite(loss)


def test_fit_and_augment():
    a, y = _toy()
    split = make_split(y)
    An = MinMaxScaler().fit(a[split.train]).transform(a)
    cent = compute_centralities(An, CENTRALITIES)
    cfg = VAEConfig(conv="gcn", max_epochs=3)
    model, hist = fit_vae((cent[split.train], y[split.train], An[split.train]),
                          (cent[split.val], y[split.val], An[split.val]), cfg, torch.device("cpu"), verbose=False)
    assert len(hist) >= 1
    adj, lab = augment(model, cent[split.train], y[split.train], 5, cfg, torch.device("cpu"))
    assert adj.shape == (10, 10, 10) and set(lab) == {0, 1}
    assert adj.min() >= 0 and adj.max() <= 1
