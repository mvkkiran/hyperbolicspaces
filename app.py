"""Euclidean vs hyperbolic embeddings — a hands-on tour.

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import hyperbolic as hyp
import ml
import viz
from data import DEFAULT_TEXT, parse_hierarchy, short

st.set_page_config(page_title="Euclidean vs Hyperbolic embeddings", layout="wide", page_icon="🌀")

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# cached workers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _train(dist: np.ndarray, geometry: str, dim: int, loss: str, epochs: int, tau: float, seed: int, optimizer: str):
    return hyp.train_embedding(dist, geometry=geometry, dim=dim, loss=loss, epochs=epochs, tau=tau, seed=seed,
                               optimizer=optimizer)


@st.cache_data(show_spinner=False)
def _sweep(dist: np.ndarray, depth: np.ndarray, dims: tuple[int, ...], loss: str, epochs: int, tau: float, seed: int,
           optimizer: str):
    rows = []
    for geom in ("euclidean", "hyperbolic"):
        for d in dims:
            emb, _ = hyp.train_embedding(dist, geometry=geom, dim=d, loss=loss, epochs=epochs, tau=tau, seed=seed,
                                         optimizer=optimizer)
            m = hyp.evaluate(emb, geom, dist, depth)
            rows.append({"geometry": geom, "dim": d,
                         "MAP": m["MAP (neighbours ranked first)"],
                         "Distortion": m["Distortion (avg. relative error)"]})
    return rows


def plot(fig):
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# sidebar: data + training knobs
# ---------------------------------------------------------------------------
st.sidebar.title("🌀 Settings")
with st.sidebar.expander("Hierarchy of statements (editable)", expanded=False):
    text = st.text_area("Two spaces = one level deeper", DEFAULT_TEXT, height=400, label_visibility="collapsed")
try:
    H = parse_hierarchy(text)
except ValueError as e:
    st.sidebar.error(str(e))
    st.stop()

st.sidebar.markdown("**Training**")
loss = st.sidebar.radio(
    "Objective (same for both geometries)",
    ["stress", "link"], format_func=lambda s: {"stress": "match tree distances (stress)",
                                               "link": "rank neighbours first (Nickel & Kiela)"}[s],
    help="Stress: minimise relative error between embedding and tree distances. "
         "Link: softmax ranking loss with negative sampling, as in the Poincaré-embeddings paper.")
tau = st.sidebar.slider("Edge length τ (hyperbolic units per hop)", 0.3, 3.0, 1.0, 0.1,
                        help="Only matters for the stress objective and Sarkar's construction. "
                             "Longer edges push points toward the boundary: lower distortion, more precision needed.")
optimizer = st.sidebar.radio(
    "Hyperbolic optimizer", ["adam", "rsgd"],
    format_func=lambda s: {"adam": "Adam in tangent space (same as Euclidean side)",
                           "rsgd": "Riemannian SGD on the ball (Nickel & Kiela)"}[s],
    help="Adam: points are expmap0(θ) and Adam updates θ — identical optimizer for both geometries. "
         "RSGD: rescale the gradient by (1−|x|²)²/4 and project back into the ball after each step.")
epochs = st.sidebar.slider("Epochs", 100, 1500, 500, 100)
seed = st.sidebar.number_input("Seed", 0, 999, 0)
group_depth = st.sidebar.slider("Colour points by ancestor at depth", 1, int(H.depth.max()), min(2, int(H.depth.max())))
show_labels = st.sidebar.checkbox("Show text labels on plots", True)

st.title("Euclidean vs. hyperbolic embeddings — hands on")
st.caption(
    "Same statements, same graph, same objective, same 2 dimensions. "
    "Only the *geometry* changes. Based on the Stanford DAWN posts on hyperbolic embeddings."
)

tab_data, tab_words, tab_embed, tab_dims, tab_ml, tab_learn = st.tabs(
    ["1 · The data", "2 · Words vs meaning", "3 · Embed in 2-D", "4 · Dimensions", "5 · ML on top", "6 · Learn"]
)

# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("A hierarchy of English statements")
    st.markdown(
        f"**{H.n} statements**, depth {H.depth.max()}. Each child is a *more specific* version of its parent "
        "(a golden retriever **is a** dog **is a** mammal **is an** animal …). "
        "Hierarchies like this are everywhere: taxonomies, org charts, file systems, product catalogues, "
        "ontologies such as WordNet or MeSH. The **graph distance** (number of hops) is our ground truth for "
        "“how related are these two statements?”"
    )
    plot(viz.tree_figure(H, group_depth))
    with st.expander("Why is this hard for ordinary (Euclidean) embeddings?"):
        st.markdown(
            """
            A tree with branching factor *b* has about *bᵈ* nodes at depth *d*: the number of things grows
            **exponentially** as you go deeper. In Euclidean space the amount of room at distance *r* from a
            point grows only **polynomially** (a circle's circumference is 2πr, a sphere's surface is 4πr²).
            So deep, bushy trees literally run out of room: siblings get squashed together, or cousins from
            different branches end up as close as siblings. Adding dimensions helps a little but the distortion
            still grows with the size of the tree.

            In **hyperbolic** space the circumference of a circle grows like *eʳ* — exponentially, just like the tree.
            There is always more room near the boundary, so every branch gets its own corner.
            """
        )

# ---------------------------------------------------------------------------
with tab_words:
    st.subheader("Word overlap is not meaning")
    st.markdown(
        "A bag-of-words / TF-IDF model (the simplest “relevancy” signal, and a decent proxy for what a purely "
        "lexical search does) scores two statements as similar when they **share words**. "
        "Pick a statement and compare its nearest neighbours by word overlap with its true neighbours in the hierarchy."
    )
    vec = TfidfVectorizer(stop_words="english").fit(H.statements)
    S = cosine_similarity(vec.transform(H.statements))
    np.fill_diagonal(S, -1)

    default_i = next((i for i, s in enumerate(H.statements) if "whale" in s), 0)
    pick = st.selectbox("Statement", range(H.n), index=default_i, format_func=lambda i: H.statements[i])
    c1, c2 = st.columns(2)
    k = 4
    with c1:
        st.markdown("**Nearest by word overlap (TF-IDF cosine)**")
        order = np.argsort(-S[pick])[:k]
        st.table(pd.DataFrame({"statement": [H.statements[j] for j in order],
                               "similarity": [f"{S[pick, j]:.2f}" for j in order],
                               "hops away in tree": [int(H.dist[pick, j]) for j in order]}))
    with c2:
        st.markdown("**Nearest in the hierarchy (graph distance)**")
        order = np.argsort(H.dist[pick] + np.arange(H.n) * 1e-6)[1:k + 1]
        st.table(pd.DataFrame({"statement": [H.statements[j] for j in order],
                               "hops away in tree": [int(H.dist[pick, j]) for j in order]}))
    iu = np.triu_indices(H.n, 1)
    from scipy.stats import spearmanr
    rho = spearmanr(-S[iu], H.dist[iu]).correlation
    st.metric("Rank correlation between word-overlap *distance* and tree distance", f"{rho:.2f}",
              help="1.0 would mean word overlap perfectly recovers the hierarchy. The embeddings in the next tab "
                   "reach 0.9+ because they are trained on the structure, not the surface words.")
    st.info(
        "The whale, the penguin and the shark all **swim in the ocean**, so word overlap says they belong together. "
        "The hierarchy says a whale is a *mammal*, closer to a dog than to a shark. Likewise the airplane and the "
        "sparrow both *fly in the sky*, and the chihuahua, the laptop and the phone all *fit in a bag*. "
        "Context and semantics live in the **relations**, not in the words — which is why we embed the graph."
    )

# ---------------------------------------------------------------------------
with tab_embed:
    st.subheader("The same graph, embedded in 2-D Euclidean space vs. the 2-D Poincaré disk")
    with st.spinner("Training both embeddings…"):
        emb_e, hist_e = _train(H.dist, "euclidean", 2, loss, epochs, tau, int(seed), optimizer)
        emb_h, hist_h = _train(H.dist, "hyperbolic", 2, loss, epochs, tau, int(seed), optimizer)
    st.caption("Both embeddings are re-centred so the root statement sits at the origin "
               "(an isometry in each geometry — no distance changes).")
    hl = st.multiselect("Highlight statements", range(H.n), default=[],
                        format_func=lambda i: short(H.statements[i]))
    c1, c2 = st.columns(2)
    with c1:
        plot(viz.embedding_figure(H, emb_e, "euclidean", "Euclidean plane ℝ²", group_depth, hl, show_labels))
    with c2:
        plot(viz.embedding_figure(H, emb_h, "hyperbolic", "Poincaré disk 𝔹² (dotted circle = infinity)",
                                  group_depth, hl, show_labels))

    me = hyp.evaluate(emb_e, "euclidean", H.dist, H.depth)
    mh = hyp.evaluate(emb_h, "hyperbolic", H.dist, H.depth)
    df = pd.DataFrame({"Euclidean 2-D": me, "Hyperbolic 2-D": mh}).reindex(
        ["MAP (neighbours ranked first)", "Distortion (avg. relative error)",
         "Rank correlation with tree distance", "Depth ↔ radius correlation", "Precision needed (bits)"])
    st.dataframe(df.style.format("{:.3f}", na_rep="—"), width="stretch")
    with st.expander("How to read these numbers"):
        st.markdown(
            """
            * **MAP** (↑, max 1): for each statement, are its *direct* neighbours in the hierarchy the closest
              points in the embedding? A local, ranking-style measure — what matters for search/recommendation.
            * **Distortion** (↓, min 0): average relative error between embedding distance and tree distance over
              *all pairs*, after the best global rescaling. A global measure — what matters if you want distances to
              mean something.
            * **Rank correlation** (↑): Spearman ρ between embedding distances and hop counts over all pairs.
            * **Depth ↔ radius** (↑): in a good hyperbolic embedding the root sits at the centre and depth grows with
              distance from the origin — the hierarchy level is *readable from the norm*. Euclidean embeddings have no
              such privileged centre.
            * **Precision** (hyperbolic only): −log₂(1 − |x|max). Points near the boundary need many bits; above ~24
              you can no longer trust float32, above ~53 not even float64. This is the hidden cost of hyperbolic space.
            """
        )

    c1, c2 = st.columns(2)
    De = hyp.embedding_distances(emb_e, "euclidean")
    Dh = hyp.embedding_distances(emb_h, "hyperbolic")
    with c1:
        plot(viz.distance_scatter(De, H.dist, hyp.distortion(De, H.dist)[1], "Euclidean: tree distance vs embedding distance"))
        plot(viz.depth_radius_figure(H, emb_e, "Euclidean: depth vs distance from origin"))
    with c2:
        plot(viz.distance_scatter(Dh, H.dist, hyp.distortion(Dh, H.dist)[1], "Hyperbolic: tree distance vs embedding distance"))
        plot(viz.depth_radius_figure(H, emb_h, "Hyperbolic: depth vs distance from origin"))

    with st.expander("Training curves"):
        plot(viz.loss_figure({"Euclidean (Adam)": hist_e,
                              "Hyperbolic (" + ("Adam in tangent space" if optimizer == "adam" else "Riemannian SGD") + ")": hist_h}))
        if optimizer == "rsgd":
            st.caption(
                "Hyperbolic points are updated with **Riemannian SGD**: the Euclidean gradient is rescaled by "
                "(1 − |x|²)²/4 (steps shrink near the boundary where the space is stretched) and the point is "
                "projected back inside the disk. The first 10 % of epochs use a smaller step (burn-in). "
                "Note it needs a much larger nominal learning rate than Adam because that rescaling is ≤ ¼."
            )
        else:
            st.caption(
                "Hyperbolic points are written as **expmap₀(θ)** — θ is an ordinary vector in the flat tangent space at "
                "the origin, and the exponential map wraps it onto the disk. Adam then optimises θ exactly as it does the "
                "Euclidean coordinates. This is the simplest way to reuse Euclidean optimisers in a curved space."
            )

    st.markdown("### Compare two statements")
    c1, c2 = st.columns(2)
    a = c1.selectbox("A", range(H.n), index=next((i for i, s in enumerate(H.statements) if "whale" in s), 0),
                     format_func=lambda i: H.statements[i])
    b = c2.selectbox("B", range(H.n), index=next((i for i, s in enumerate(H.statements) if "shark" in s), 1),
                     format_func=lambda i: H.statements[i])
    se, sh = hyp.distortion(De, H.dist)[1], hyp.distortion(Dh, H.dist)[1]
    st.table(pd.DataFrame({
        "tree (hops)": [int(H.dist[a, b])],
        "Euclidean (rescaled)": [f"{De[a, b] * se:.2f}"],
        "hyperbolic (rescaled)": [f"{Dh[a, b] * sh:.2f}"],
        "word overlap": [f"{S[a, b]:.2f}"],
    }, index=["distance A↔B"]))

    st.markdown("### Bonus: no training at all — Sarkar's combinatorial construction")
    st.markdown(
        "Hyperbolic space is such a natural home for trees that you can embed one **exactly, by hand, in linear time**: "
        "put the root at the centre, spread its children evenly on a circle of hyperbolic radius τ, then recursively do "
        "the same for every child in the angular gap left by its parent. The only knob is τ. "
        "This is what the Stanford group used to hit MAP 0.989 on WordNet in just 2 dimensions."
    )
    emb_s = hyp.sarkar_embedding(H.parent, H.root, tau)
    ms = hyp.evaluate(emb_s, "hyperbolic", H.dist, H.depth)
    c1, c2 = st.columns([3, 2])
    with c1:
        plot(viz.embedding_figure(H, emb_s, "hyperbolic", f"Sarkar construction, τ = {tau}", group_depth, hl, show_labels))
    with c2:
        st.dataframe(pd.DataFrame({"Sarkar 2-D": ms}).style.format("{:.3f}"), width="stretch")
        st.caption(
            "Slide τ in the sidebar. Larger τ → distortion falls toward 0 but the *bits needed* climb: "
            "the price of pushing points toward the boundary. The paper proves this trade-off is unavoidable."
        )

# ---------------------------------------------------------------------------
with tab_dims:
    st.subheader("Does adding dimensions rescue Euclidean space?")
    st.markdown(
        "The headline result from the Poincaré-embeddings paper: on WordNet nouns, a **5-D hyperbolic** embedding "
        "(MAP 0.82) beat a **200-D Euclidean** one (MAP 0.17). Let's reproduce the shape of that curve on our tiny tree."
    )
    dims = (2, 3, 5, 10, 20)
    with st.spinner("Training 10 embeddings…"):
        rows = _sweep(H.dist, H.depth, dims, loss, min(epochs, 500), tau, int(seed), optimizer)
    c1, c2 = st.columns(2)
    with c1:
        plot(viz.sweep_figure(rows, "MAP"))
    with c2:
        plot(viz.sweep_figure(rows, "Distortion"))
    st.dataframe(pd.DataFrame(rows).pivot(index="dim", columns="geometry", values=["MAP", "Distortion"])
                 .style.format("{:.3f}"), width="stretch")
    st.markdown(
        "Our tree is small (a few dozen nodes), so Euclidean space eventually catches up — but notice **how many** "
        "dimensions it needs to match what hyperbolic space does in 2. On real taxonomies with 10⁴–10⁵ nodes the "
        "gap never closes: Euclidean distortion grows with the size of the tree, hyperbolic distortion does not."
    )

# ---------------------------------------------------------------------------
with tab_ml:
    st.subheader("Machine learning in a curved space")
    st.markdown(
        """
        Once you have hyperbolic vectors, can you still *do* machine learning? Mostly yes, but every tool that
        quietly assumes Euclidean geometry has to be rebuilt:

        | Euclidean tool | What breaks | Hyperbolic replacement |
        |---|---|---|
        | `x + y` | there is no vector addition | Möbius addition `x ⊕ y` |
        | straight line | geodesics are circular arcs | exp / log maps between the ball and its tangent spaces |
        | gradient step `x − η∇` | would step outside the disk | Riemannian SGD: rescale ∇ by (1 − ‖x‖²)²/4, then project |
        | mean of points | the arithmetic mean leaves the manifold | Fréchet / Einstein midpoint |
        | linear classifier `⟨a, x⟩ + b` | hyperplanes are not flat | hyperbolic MLR: signed distance to a *hyperbolic* hyperplane (Ganea et al. 2018) |
        | k-NN / k-means | need a distance | just use the Poincaré distance ✔ |

        Below we train the **same two classifiers** on the 2-D Euclidean and 2-D hyperbolic embeddings from tab 3
        (same objective, same epochs) and ask them to predict a statement's **category** — its ancestor at a chosen
        depth. Train/test splits are repeated many times; we report the mean accuracy.
        """
    )
    label_depth = st.slider("Category = ancestor at depth", 1, max(1, int(H.depth.max()) - 1), min(2, max(1, int(H.depth.max()) - 1)),
                            help="Depth 1 = coarse (living thing / machine), depth 2 = finer (animal / plant / vehicle / computer).")
    labels, eligible, names = H.labels_at_depth(label_depth)
    idx = np.array(eligible)
    n_classes = len(names)
    st.caption(f"{len(idx)} statements in {n_classes} classes: " + ", ".join(f"**{n}**" for n in names))

    c1, c2, c3 = st.columns(3)
    k = c1.slider("k for k-NN", 1, 7, 3)
    test_size = c2.slider("Test fraction", 0.2, 0.6, 0.4, 0.05)
    n_rep = c3.slider("Repeated splits", 5, 50, 20, 5)

    emb_e, _ = _train(H.dist, "euclidean", 2, loss, epochs, tau, int(seed), optimizer)
    emb_h, _ = _train(H.dist, "hyperbolic", 2, loss, epochs, tau, int(seed), optimizer)
    De = hyp.embedding_distances(emb_e, "euclidean")
    Dh = hyp.embedding_distances(emb_h, "hyperbolic")

    with st.spinner("Training classifiers…"):
        knn_e = ml.knn_accuracy(De, labels, idx, k, n_rep, test_size, int(seed))
        knn_h = ml.knn_accuracy(Dh, labels, idx, k, n_rep, test_size, int(seed))
        mlr_e = ml.mlr_accuracy(emb_e, labels, idx, "euclidean", n_classes, min(n_rep, 10), test_size, seed=int(seed))
        mlr_h = ml.mlr_accuracy(emb_h, labels, idx, "hyperbolic", n_classes, min(n_rep, 10), test_size, seed=int(seed))
    res = pd.DataFrame({
        "Euclidean 2-D": [f"{knn_e[0]:.2f} ± {knn_e[1]:.2f}", f"{mlr_e[0]:.2f} ± {mlr_e[1]:.2f}"],
        "Hyperbolic 2-D": [f"{knn_h[0]:.2f} ± {knn_h[1]:.2f}", f"{mlr_h[0]:.2f} ± {mlr_h[1]:.2f}"],
    }, index=[f"{k}-NN accuracy", "Logistic regression accuracy"])
    st.table(res)

    st.markdown("#### Decision regions of the two logistic regressions (one split)")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(idx))
    n_te = max(1, int(round(test_size * len(idx))))
    tr_idx, te_idx = idx[perm[n_te:]], idx[perm[:n_te]]
    model_e = ml.train_mlr(emb_e, labels, tr_idx, "euclidean", n_classes, seed=int(seed))
    model_h = ml.train_mlr(emb_h, labels, tr_idx, "hyperbolic", n_classes, seed=int(seed))
    xs_e, z_e = ml.decision_grid(model_e, "euclidean", float(np.abs(emb_e).max()) * 1.15)
    xs_h, z_h = ml.decision_grid(model_h, "hyperbolic", 1.0)
    acc_e = float((ml.predict(model_e, emb_e[te_idx]) == labels[te_idx]).mean())
    acc_h = float((ml.predict(model_h, emb_h[te_idx]) == labels[te_idx]).mean())
    c1, c2 = st.columns(2)
    with c1:
        plot(viz.embedding_figure(H, emb_e, "euclidean", f"Euclidean MLR — straight boundaries · test acc {acc_e:.2f}",
                                  label_depth, list(te_idx), show_labels, (xs_e, z_e, n_classes)))
    with c2:
        plot(viz.embedding_figure(H, emb_h, "hyperbolic", f"Hyperbolic MLR — arc boundaries · test acc {acc_h:.2f}",
                                  label_depth, list(te_idx), show_labels, (xs_h, z_h, n_classes)))
    st.caption("Points with a black outline are the held-out test statements for this split. "
               "Colours of points = true class; shaded regions = predicted class.")
    st.markdown(
        """
        **What to look for.** In the Poincaré disk each sub-tree occupies its own *sector* fanning out from the
        centre, so a hyperbolic hyperplane (an arc meeting the boundary at right angles) separates categories
        cleanly — and the same arc keeps working for deeper descendants you have never seen, because they land
        further out in the same sector. In the Euclidean plot the sub-trees are tangled by lack of room, so
        straight lines cut through them.

        Beyond these two: hyperbolic neural networks (Ganea et al.), hyperbolic graph convolutions (HGCN),
        hyperbolic attention, hyperbolic image embeddings for hierarchical labels, and
        **mixed-curvature product spaces** (Gu et al.) that let a model use a hyperbolic factor for the tree-like
        part of the data and a Euclidean or spherical factor for the rest. See tab 6.
        """
    )

# ---------------------------------------------------------------------------
with tab_learn:
    notes = HERE / "research_notes.md"
    if notes.exists():
        st.markdown(notes.read_text())
    else:
        st.warning("research_notes.md not found.")
