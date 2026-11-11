# Euclidean vs. hyperbolic embeddings — hands on

A small Streamlit app that makes the ideas in the Stanford DAWN posts
([Hyperbolic embeddings with the right amount of hyperbole](https://dawn.cs.stanford.edu/news/hyperbolic-embeddings-hopefully-right-amount-hyperbole)
and [Into the wild: machine learning in non-Euclidean spaces](https://dawn.cs.stanford.edu/news/wild-machine-learning-non-euclidean-spaces))
concrete, using a hierarchy of plain English statements instead of equations.

## Run

```bash
cd /Users/krishm20/PycharmProjects/Experiments
source venv/bin/activate
cd hyperbolic_embeddings
streamlit run app.py
```

## What's inside

| File | Purpose |
|---|---|
| `data.py` | The editable hierarchy of statements (indented text → tree), labels, layout |
| `hyperbolic.py` | Poincaré-ball maths (distance, Möbius addition, exp/log maps), Riemannian SGD training, Sarkar's combinatorial construction, metrics (MAP, distortion, rank correlation, depth↔radius, precision bits) |
| `ml.py` | k-NN and multinomial logistic regression in both geometries (hyperbolic MLR after Ganea et al. 2018) |
| `viz.py` | Plotly figures (tree, Poincaré disk with geodesic edges, decision regions, sweeps) |
| `app.py` | The Streamlit app (6 tabs) |
| `research_notes.md` | Plain-English digest of the two posts and the papers behind them |

## The experiment

1. **Data** – ~30 statements where each child is a more specific version of its parent
   ("golden retriever" → "dog" → "mammal" → "animal"). Hop count in the tree is the
   ground-truth notion of relatedness. The sentences contain deliberate word traps
   ("swims in the ocean" appears under mammal, bird and fish).
2. **Words vs meaning** – TF-IDF similarity is fooled by the traps; the hierarchy is not.
3. **Embed in 2-D** – the *same* graph, *same* objective, *same* dimension, trained in
   ℝ² (Adam) and in the Poincaré disk 𝔹² (Riemannian SGD). Compare MAP, distortion,
   rank correlation and whether depth is readable from the radius. Also Sarkar's
   training-free construction and its precision cost.
4. **Dimensions** – how many Euclidean dimensions it takes to catch up with 2 hyperbolic ones.
5. **ML on top** – k-NN and logistic regression in both geometries, with decision regions
   drawn on the disk (hyperbolic "hyperplanes" are arcs).
6. **Learn** – the research notes.
