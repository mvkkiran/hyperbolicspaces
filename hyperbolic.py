"""Poincaré-ball geometry, embedding training, Sarkar's construction, metrics.

Everything is in float64: hyperbolic coordinates near the boundary of the
disk need a lot of precision (see the "precision" discussion in the notes).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import spearmanr

torch.set_default_dtype(torch.float64)

EPS = 1e-12
MAX_NORM = 1.0 - 1e-7


# ---------------------------------------------------------------------------
# Poincaré ball primitives (curvature -1)
# ---------------------------------------------------------------------------
def project(x: torch.Tensor) -> torch.Tensor:
    """Pull points that escaped the ball back inside."""
    n = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return torch.where(n > MAX_NORM, x / n * MAX_NORM, x)


def mobius_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x ⊕ y — the hyperbolic replacement for vector addition."""
    xy = (x * y).sum(-1, keepdim=True)
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / den.clamp_min(EPS)


def poincare_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Geodesic distance; broadcasts over leading dims."""
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    diff2 = ((x - y) ** 2).sum(-1)
    arg = 1 + 2 * diff2 / ((1 - x2).clamp_min(EPS) * (1 - y2).clamp_min(EPS))
    return torch.acosh(arg.clamp_min(1.0 + 1e-15))


def lambda_x(x: torch.Tensor) -> torch.Tensor:
    """Conformal factor: how much the metric is stretched at x."""
    return 2 / (1 - (x * x).sum(-1, keepdim=True)).clamp_min(EPS)


def expmap(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Walk from x along tangent vector v for |v| units of hyperbolic length."""
    vn = v.norm(dim=-1, keepdim=True).clamp_min(EPS)
    second = torch.tanh(lambda_x(x) * vn / 2) * v / vn
    return project(mobius_add(x, second))


def logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Inverse of expmap: the tangent vector at x pointing to y."""
    u = mobius_add(-x, y)
    un = u.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return (2 / lambda_x(x)) * torch.atanh(un.clamp_max(MAX_NORM)) * u / un


def expmap0(v: torch.Tensor) -> torch.Tensor:
    vn = v.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return project(torch.tanh(vn) * v / vn)


def logmap0(y: torch.Tensor) -> torch.Tensor:
    yn = y.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return torch.atanh(yn.clamp_max(MAX_NORM)) * y / yn


def pairwise_dist(X: torch.Tensor, geometry: str) -> torch.Tensor:
    if geometry == "hyperbolic":
        return poincare_dist(X[:, None, :], X[None, :, :])
    return torch.cdist(X, X)


def geodesic_points(x: np.ndarray, y: np.ndarray, n: int = 40) -> np.ndarray:
    """Points along the hyperbolic straight line from x to y (for drawing)."""
    xt = torch.as_tensor(x)[None, :]
    yt = torch.as_tensor(y)[None, :]
    v = logmap(xt, yt)
    t = torch.linspace(0, 1, n)[:, None]
    pts = expmap(xt.expand(n, -1), t * v)
    return pts.numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _stress_loss(D_emb: torch.Tensor, D_target: torch.Tensor) -> torch.Tensor:
    """Relative stress: average squared relative error over all pairs.

    Minimising this directly asks "can this geometry reproduce the tree
    distances?" — exactly what distortion measures.
    """
    iu = torch.triu_indices(D_emb.shape[0], D_emb.shape[0], offset=1)
    d, t = D_emb[iu[0], iu[1]], D_target[iu[0], iu[1]]
    return (((d - t) / t) ** 2).mean()


def _link_loss(
    X: torch.Tensor, edges: torch.Tensor, non_nbrs: list[np.ndarray],
    n_neg: int, geometry: str, rng: np.random.Generator,
) -> torch.Tensor:
    """Nickel & Kiela (2017) loss: for each edge (u,v), make v the closest
    among {v} ∪ random negatives — a softmax over negative distances."""
    u, v = edges[:, 0], edges[:, 1]
    negs = torch.as_tensor(
        np.stack([rng.choice(non_nbrs[int(i)], n_neg, replace=True) for i in u])
    )
    cand = torch.cat([v[:, None], negs], dim=1)                 # (E, 1+n_neg)
    xu = X[u][:, None, :]
    xc = X[cand]
    if geometry == "hyperbolic":
        d = poincare_dist(xu, xc)
    else:
        d = (xu - xc).norm(dim=-1)
    logits = -d
    return torch.nn.functional.cross_entropy(
        logits, torch.zeros(len(u), dtype=torch.long)
    )


def center_on(emb: np.ndarray, geometry: str, i: int) -> np.ndarray:
    """Apply the isometry that moves point i to the origin (distances unchanged).

    Both geometries are homogeneous: nothing in the objective pins the root to
    the centre, so we move it there afterwards to make 'radius' comparable.
    """
    if geometry == "hyperbolic":
        X = torch.as_tensor(emb)
        return mobius_add(-X[i][None, :], X).numpy()
    return emb - emb[i]


def train_embedding(
    dist: np.ndarray,
    geometry: str = "hyperbolic",
    dim: int = 2,
    loss: str = "stress",
    epochs: int = 400,
    lr: float | None = None,
    tau: float = 1.0,
    n_neg: int = 10,
    seed: int = 0,
    optimizer: str = "adam",
    root: int | None = 0,
) -> tuple[np.ndarray, list[float]]:
    """Train an embedding of a graph given its shortest-path distance matrix.

    geometry: 'euclidean' or 'hyperbolic' (Poincaré ball).
    loss: 'stress' (match tree distances scaled by tau) or 'link' (Nickel &
    Kiela ranking loss with negative sampling).
    optimizer (hyperbolic only — Euclidean always uses Adam):
      'adam' — points are parametrised as expmap0(θ) with θ a free Euclidean
               vector; Adam runs on θ. Same optimizer as the Euclidean side.
      'rsgd' — Riemannian SGD directly on the ball: the Euclidean gradient is
               rescaled by (1-|x|^2)^2/4 and the point is projected back
               inside after every step (Nickel & Kiela 2017).
    root: if given, the result is re-centred so this node sits at the origin.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    D_target = torch.as_tensor(dist * tau)
    adj = dist == 1
    edges = torch.as_tensor(np.argwhere(adj))                    # both directions
    non_nbrs = [np.where(~adj[i] & (np.arange(n) != i))[0] for i in range(n)]

    use_rsgd = geometry == "hyperbolic" and optimizer == "rsgd"
    theta = torch.nn.Parameter((torch.rand(n, dim) - 0.5) * 2e-3)
    if lr is None:
        if use_rsgd:
            lr = 20.0 if loss == "stress" else 1.0
        else:
            lr = 0.05
    opt = None if use_rsgd else torch.optim.Adam([theta], lr=lr)

    def points() -> torch.Tensor:
        if geometry == "hyperbolic" and not use_rsgd:
            return expmap0(theta)
        return theta

    history: list[float] = []
    for ep in range(epochs):
        X = points()
        if loss == "stress":
            L = _stress_loss(pairwise_dist(X, geometry), D_target)
        else:
            L = _link_loss(X, edges, non_nbrs, n_neg, geometry, rng)
        if opt is not None:
            opt.zero_grad()
            L.backward()
            opt.step()
        else:
            if theta.grad is not None:
                theta.grad.zero_()
            L.backward()
            with torch.no_grad():
                step = lr * (0.1 if ep < epochs // 10 else 1.0)       # burn-in
                scale = ((1 - (theta * theta).sum(-1, keepdim=True)) ** 2) / 4
                theta.data = project(theta.data - step * scale * theta.grad)
        history.append(L.item())
    emb = points().detach().numpy()
    if root is not None:
        emb = center_on(emb, geometry, root)
    return emb, history


# ---------------------------------------------------------------------------
# Sarkar's combinatorial construction (2-D, no training)
# ---------------------------------------------------------------------------
def sarkar_embedding(parent: dict[int, int | None], root: int, tau: float = 1.0) -> np.ndarray:
    """Embed a tree in the Poincaré disk *exactly* by construction.

    Root at the origin; each node's children are spread evenly on a circle of
    hyperbolic radius tau around it, in the angular gap left by the parent.
    Longer edges (larger tau) → lower distortion but more precision needed.
    """
    children: dict[int, list[int]] = {}
    for c, p in parent.items():
        if p is not None:
            children.setdefault(p, []).append(c)
    n = len(parent)
    emb = np.zeros((n, 2))
    r = math.tanh(tau / 2)  # Euclidean radius of a point at hyperbolic distance tau from 0

    def place(node: int, parent_node: int | None) -> None:
        kids = children.get(node, [])
        if not kids:
            return
        x = torch.as_tensor(emb[node])[None, :]
        if parent_node is None:
            theta0, k = 0.0, len(kids)
            angles = [theta0 + 2 * math.pi * i / k for i in range(k)]
        else:
            # move `node` to the origin, see where the parent lands
            p = mobius_add(-x, torch.as_tensor(emb[parent_node])[None, :])[0].numpy()
            theta0 = math.atan2(p[1], p[0])
            k = len(kids) + 1  # parent occupies one slot
            angles = [theta0 + 2 * math.pi * (i + 1) / k for i in range(len(kids))]
        for kid, a in zip(kids, angles):
            local = torch.tensor([[r * math.cos(a), r * math.sin(a)]])
            emb[kid] = mobius_add(x, local)[0].numpy()  # move back
        for kid in kids:
            place(kid, node)

    place(root, None)
    return emb


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def embedding_distances(emb: np.ndarray, geometry: str) -> np.ndarray:
    return pairwise_dist(torch.as_tensor(emb), geometry).numpy()


def mean_average_precision(D_emb: np.ndarray, dist: np.ndarray) -> float:
    """Reconstruction MAP: for each node, are its graph neighbours the nearest
    points in the embedding? (local quality, 1.0 = perfect)."""
    n = dist.shape[0]
    aps = []
    for i in range(n):
        nbrs = np.where(dist[i] == 1)[0]
        others = np.array([j for j in range(n) if j != i])
        order = others[np.argsort(D_emb[i, others], kind="stable")]
        hits, ap = 0, 0.0
        for rank, j in enumerate(order, start=1):
            if j in nbrs:
                hits += 1
                ap += hits / rank
        aps.append(ap / len(nbrs))
    return float(np.mean(aps))


def distortion(D_emb: np.ndarray, dist: np.ndarray) -> tuple[float, float]:
    """Average relative error |s·d_emb − d_graph| / d_graph over all pairs,
    after choosing the best global scale s (global quality, 0 = perfect)."""
    iu = np.triu_indices(dist.shape[0], 1)
    d, t = D_emb[iu], dist[iu]
    s = float((d * t).sum() / max((d * d).sum(), EPS))
    return float(np.mean(np.abs(s * d - t) / t)), s


def rank_correlation(D_emb: np.ndarray, dist: np.ndarray) -> float:
    iu = np.triu_indices(dist.shape[0], 1)
    return float(spearmanr(D_emb[iu], dist[iu]).correlation)


def depth_norm_correlation(emb: np.ndarray, depth: np.ndarray) -> float:
    """Does distance from the origin track depth in the hierarchy?"""
    return float(spearmanr(np.linalg.norm(emb, axis=1), depth).correlation)


def bits_needed(emb: np.ndarray) -> float:
    """Rough bits of precision needed to tell the outermost point from the
    boundary: −log2(1 − |x|_max)."""
    m = float(np.max(np.linalg.norm(emb, axis=1)))
    return -math.log2(max(1 - m, 1e-300))


def evaluate(emb: np.ndarray, geometry: str, dist: np.ndarray, depth: np.ndarray) -> dict:
    D = embedding_distances(emb, geometry)
    dis, scale = distortion(D, dist)
    out = {
        "MAP (neighbours ranked first)": mean_average_precision(D, dist),
        "Distortion (avg. relative error)": dis,
        "Rank correlation with tree distance": rank_correlation(D, dist),
        "Depth ↔ radius correlation": depth_norm_correlation(emb, depth),
    }
    if geometry == "hyperbolic":
        out["Precision needed (bits)"] = bits_needed(emb)
    return out
