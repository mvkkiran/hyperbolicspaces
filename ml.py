"""Machine learning *on top of* the embeddings — Euclidean vs hyperbolic twins.

1. k-nearest-neighbours with the geometry's own distance.
2. Multinomial logistic regression:
   - Euclidean: ordinary linear layer (decision boundaries = straight lines).
   - Hyperbolic: Ganea, Bécigneul & Hofmann (2018) "hyperbolic MLR" — the
     decision boundaries are hyperbolic hyperplanes (circular arcs in the
     Poincaré disk). Each class k has a point p_k on the ball and a tangent
     direction a_k; the logit is the signed hyperbolic distance of x to the
     boundary through p_k orthogonal to a_k (times a scale).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from hyperbolic import EPS, expmap0, lambda_x, mobius_add

torch.set_default_dtype(torch.float64)


def _splits(labels: np.ndarray, idx: np.ndarray, n_repeats: int, test_size: float, seed: int):
    y = labels[idx]
    counts = np.bincount(y)
    n_test = max(1, int(round(test_size * len(idx))))
    stratify_ok = counts.min() >= 2 and n_test >= len(counts) and (len(idx) - n_test) >= len(counts)
    if stratify_ok:
        ss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
        yield from ss.split(np.zeros(len(idx)), y)
    else:
        ss = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
        yield from ss.split(np.zeros(len(idx)))


# ---------------------------------------------------------------------------
# k-NN
# ---------------------------------------------------------------------------
def knn_accuracy(
    D: np.ndarray, labels: np.ndarray, idx: np.ndarray,
    k: int = 3, n_repeats: int = 20, test_size: float = 0.3, seed: int = 0,
) -> tuple[float, float]:
    """Mean ± std test accuracy of k-NN using a precomputed distance matrix."""
    accs = []
    for tr, te in _splits(labels, idx, n_repeats, test_size, seed):
        tr_i, te_i = idx[tr], idx[te]
        kk = min(k, len(tr_i))
        clf = KNeighborsClassifier(n_neighbors=kk, metric="precomputed")
        clf.fit(D[np.ix_(tr_i, tr_i)], labels[tr_i])
        pred = clf.predict(D[np.ix_(te_i, tr_i)])
        accs.append(float((pred == labels[te_i]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


# ---------------------------------------------------------------------------
# Multinomial logistic regression
# ---------------------------------------------------------------------------
class EuclideanMLR(torch.nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.lin = torch.nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class HyperbolicMLR(torch.nn.Module):
    """Ganea et al. 2018, eq. (25), curvature −1.

    logit_k(x) = λ_{p_k} ‖a_k‖ · asinh( 2 ⟨(−p_k) ⊕ x, a_k⟩ /
                                        ((1 − ‖(−p_k) ⊕ x‖²) ‖a_k‖) )
    p_k is kept on the ball by parametrising it through expmap0.
    """

    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.p_tangent = torch.nn.Parameter(torch.randn(n_classes, dim) * 0.1)
        self.a = torch.nn.Parameter(torch.randn(n_classes, dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = expmap0(self.p_tangent)                       # (K, d)
        xs = mobius_add(-p[None, :, :], x[:, None, :])    # (N, K, d)
        a = self.a[None, :, :]
        an = a.norm(dim=-1).clamp_min(EPS)                # (1, K)
        num = 2 * (xs * a).sum(-1)                        # (N, K)
        den = (1 - (xs * xs).sum(-1)).clamp_min(EPS) * an
        lam = lambda_x(p).squeeze(-1)[None, :]            # (1, K)
        return lam * an * torch.asinh(num / den)


def train_mlr(
    X: np.ndarray, labels: np.ndarray, train_idx: np.ndarray,
    geometry: str, n_classes: int, epochs: int = 300, lr: float = 0.05, seed: int = 0,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = (HyperbolicMLR if geometry == "hyperbolic" else EuclideanMLR)(X.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.as_tensor(X[train_idx])
    yt = torch.as_tensor(labels[train_idx], dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(xt), yt)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    return model(torch.as_tensor(X)).argmax(-1).numpy()


def mlr_accuracy(
    X: np.ndarray, labels: np.ndarray, idx: np.ndarray, geometry: str, n_classes: int,
    n_repeats: int = 10, test_size: float = 0.3, epochs: int = 300, seed: int = 0,
) -> tuple[float, float]:
    accs = []
    for r, (tr, te) in enumerate(_splits(labels, idx, n_repeats, test_size, seed)):
        model = train_mlr(X, labels, idx[tr], geometry, n_classes, epochs=epochs, seed=seed + r)
        accs.append(float((predict(model, X[idx[te]]) == labels[idx[te]]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def decision_grid(model: torch.nn.Module, geometry: str, extent: float, res: int = 150):
    """Class predicted at every point of a 2-D grid (for drawing regions).
    Points outside the unit disk are masked for the hyperbolic model."""
    xs = np.linspace(-extent, extent, res)
    gx, gy = np.meshgrid(xs, xs)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    if geometry == "hyperbolic":
        inside = np.linalg.norm(pts, axis=1) < 0.995
        z = np.full(len(pts), np.nan)
        z[inside] = predict(model, pts[inside])
    else:
        z = predict(model, pts).astype(float)
    return xs, z.reshape(res, res)
