# Hyperbolic Embeddings: Practitioner Notes

Sources: two Stanford DAWN blog posts, "Hyperbolic Embeddings with a Hopefully Right Amount of Hyperbole" (De Sa, Gu, Ré, Sala, 2018) and "Into the Wild: Machine Learning in Non-Euclidean Spaces" (Sala, Chami, Wolf, Gu, Gunel, Ré, 2019), plus the papers they cite. Full citations are at the end.

## 1. Why trees do not fit in flat (Euclidean) space

A tree with branching factor `b` has about `b^d` nodes at depth `d`. The number of nodes grows **exponentially** with depth. If you place the root at the origin and want every level to be one unit further out, the nodes at depth `d` all have to live on a sphere of radius `d`.

The problem is that a Euclidean ball of radius `r` has volume proportional to `r^n`. That is **polynomial** growth. Room grows slowly; the tree grows fast. Past a few levels you simply run out of space to keep siblings and cousins apart while also keeping the distances from the root correct. Adding dimensions helps a little, but only polynomially, so it never catches up with the exponential tree.

A second way to see it (from post 1): for a parent `O` with children `x` and `y`, the graph says `d(x,y) = d(x,O) + d(O,y)`. In Euclidean space that only holds if `x` and `y` are on exactly opposite sides of `O`, which a third child cannot also be. The gap is unavoidable.

**Distortion** is the metric for this failure. For every pair of nodes, compare the embedding distance to the true graph distance; distortion is the average relative error, `|d_embed - d_graph| / d_graph`. Zero means the embedding reproduces the tree exactly. Euclidean distortion on trees grows with tree size no matter how many dimensions you spend, because the mismatch above is baked into the geometry, not the dimension count.

## 2. What hyperbolic space is, intuitively

Hyperbolic space is a space of constant negative curvature. The useful picture is the **Poincaré disk**: every point lives inside the unit circle. The twist is how distance is measured: a ruler shrinks as it moves toward the boundary, so two points that look close to the edge on the page are actually far apart. Distance from the origin goes to infinity as you approach the circle. In short, **the disk has infinite room near the edge**.

Consequences that matter for embeddings:

- Volume of a hyperbolic ball grows **exponentially** with radius, matching trees exactly.
- Shortest paths (geodesics) are arcs that curve **toward the origin**. Leaf to leaf is cheapest by swinging inward toward the shared ancestor, which is exactly the "path through the parent" behaviour of trees.
- Angles match Euclidean angles (the model is conformal), so cosine-style similarity still makes sense.

Post 2 adds the complementary case: data full of cycles fits **spherical** (positive curvature) space better. Geometry should match structure.

## 3. Practical results from post 1 (the embedding post)

**Dimensions.** Nickel and Kiela (2017) showed that on the WordNet noun hierarchy, a 5-D Poincaré embedding reaches reconstruction MAP 0.823 while a 200-D Euclidean embedding manages only 0.168. Sala et al. (2018) then showed a **2-D** hyperbolic embedding from their combinatorial method reaches MAP **0.989**, beating Nickel and Kiela's 200-D Poincaré result (0.87). Two numbers per node beat two hundred.

**The combinatorial (Sarkar) construction.** Instead of training, build the embedding directly:

1. If the data is a general graph, first approximate it by a tree (a well-studied problem with existing algorithms).
2. Put the root at the origin. Place its children evenly on a circle around it. Recurse: for each child, place its children on a circle around it, spread out in the angular wedge pointing away from the parent.

One knob, the edge-scaling radius, controls quality: longer edges separate subtrees more and drive distortion toward zero. The method is linear-time, works in any dimension, and can warm-start a gradient-based embedder. Forests are handled by linking components with one very long edge.

**The precision catch.** Pushing distortion down pushes points toward the boundary, where norms look like 0.9999999. Storing a coordinate `a` costs roughly `-log2(1 - |a|)` bits. In the post's 31-node examples, a binary tree at distortion 0.01 needed about 23 bits; three long chains of the same size needed about 50 bits, which is beyond float32. The required precision depends on the longest path in the tree (you cannot control it) and the distortion you want (you can). Hyperbolic space loves **short, bushy** trees and struggles with **long chains**. More dimensions reduce the bits needed, so there is a genuine precision vs dimension vs fidelity tradeoff; the authors prove it is unavoidable.

**Two metrics, two goals.**

- **MAP (mean average precision)** is a *local* measure: for each node, rank all other nodes by embedding distance and check whether its true graph neighbours come first. 1.0 means every neighbourhood is perfectly recovered. Good for retrieval-style use.
- **Distortion** is a *global* measure: are all pairwise distances right, including far-apart pairs? Good for anything that uses actual distance values.

An embedding can have perfect MAP and poor distortion (right ordering, wrong scale), so pick the metric that matches your downstream use.

**Also from post 1:** h-MDS, a hyperbolic analogue of multidimensional scaling, recovers points from a distance matrix without iteration; the PyTorch trainer minimises *squared* hyperbolic distance because the gradient of plain distance has a singularity. Pretrained embeddings for WordNet, Wikidata and MusicBrainz were released in GloVe format.

## 4. Practical points from post 2 (doing ML in curved space)

**What breaks.** Almost everything in standard ML quietly assumes a vector space:

- There is no vector addition. You cannot add a gradient to a point and expect to stay in the space.
- There are no straight lines, so linear layers, hyperplane classifiers and linear interpolation have no direct meaning.
- The "mean" is not the arithmetic average. The analogue (the Karcher or Fréchet mean) is an optimisation problem and can be ill-defined, for example for antipodal points on a sphere.

**What fixes it.** The posts lean on a small toolkit from differential geometry:

- **Tangent space:** at every point there is a flat local approximation, a regular vector space where you can do ordinary linear algebra.
- **Exp map / log map:** exp takes a tangent vector and "walks" it along a geodesic onto the manifold; log does the reverse. These are the bridge between flat computations and the curved space.
- **Riemannian SGD** (Bonnabel): compute the Euclidean gradient, rescale it by the local metric, and step along the manifold instead of adding. In the Poincaré ball the rescaling is a scalar, `(1 - ||x||^2)^2 / 4`, followed by a projection that keeps points strictly inside the ball. Riemannian Adam (Bécigneul and Ganea) and SVRG variants exist.
- **Möbius (gyrovector) operations** (Ganea, Bécigneul, Hofmann 2018): closed-form analogues of addition (⊕) and matrix-vector multiplication (⊗) that stay inside the Poincaré ball, giving hyperbolic feed-forward layers, GRUs and a hyperbolic **multinomial logistic regression** whose decision boundaries are geodesic hypersurfaces instead of hyperplanes. Hyperbolic GCN (Chami et al. 2019) applies the aggregation in the tangent space at the origin, moving data in and out with log and exp.
- **Hyperbolic SVM** (Cho et al. 2019): large-margin classification reduces to a convex problem in the hyperboloid model, with a kernel extension for nonlinear boundaries.

**Product spaces / mixed curvature** (Gu et al. 2019). Real data is rarely a pure tree. Take a product of hyperbolic, Euclidean and spherical factors; squared distance is the sum of squared distances in each factor, so training is no harder than in one space. The "signature" (how many copies of each, with what dimensions) can be chosen by a heuristic. In the post's table, a product `H^5 x S^5` had the lowest distortion on Power Grid (0.032) and Facebook (0.040), pure `H^10` won on CS PhDs, and `(S^5)^2` won on city distances, each beating `R^10`.

**Data that benefits.** Taxonomies and ontologies (WordNet), knowledge graphs with hierarchical relations (hyperbolic KG rotations), tree-like citation and social networks (PubMed, Facebook), phylogenies, word hypernymy and textual entailment (Tifrea et al.; Ganea et al.). Images with hierarchical label sets are a natural fit for hyperbolic classifiers, though neither post reports image experiments (see note at the end).

## 5. Hyperbolic vs Euclidean: a quick checklist

Prefer **hyperbolic** when:

- The data is a tree or close to one (low Gromov hyperbolicity, few cycles, clear parent/child or is-a relations).
- The hierarchy is **bushy and shallow** rather than a few very long chains.
- You want very few dimensions (2 to 10) for storage, visualisation or interpretability.
- The downstream task is ranking, nearest-neighbour or hierarchy reconstruction (MAP-style).
- You can tolerate float64 or careful numerics, or use the hyperboloid model for training stability.

Prefer **Euclidean** (or a product space) when:

- The structure is grid-like, dense, or full of cycles; consider spherical factors for cycles.
- Your pipeline needs plain linear algebra, off-the-shelf layers and libraries without Riemannian optimisers.
- Depth is tiny and dimensions are cheap; the gain will be small.
- You need exact global distances along long paths, where hyperbolic precision costs explode.

Not sure? Fit `R^n`, `H^n` and a small product space and compare distortion and MAP on held-out pairs.

## Citations

1. De Sa, Gu, Ré, Sala. "Hyperbolic Embeddings with a Hopefully Right Amount of Hyperbole." Stanford DAWN blog, 2018. https://dawn.cs.stanford.edu/news/hyperbolic-embeddings-hopefully-right-amount-hyperbole
2. Sala, Chami, Wolf, Gu, Gunel, Ré. "Into the Wild: Machine Learning in Non-Euclidean Spaces." Stanford DAWN blog, 2019. https://dawn.cs.stanford.edu/news/wild-machine-learning-non-euclidean-spaces
3. Nickel, Kiela. "Poincaré Embeddings for Learning Hierarchical Representations." NeurIPS 2017. https://arxiv.org/abs/1705.08039
4. Sala, De Sa, Gu, Ré. "Representation Tradeoffs for Hyperbolic Embeddings." ICML 2018. https://arxiv.org/abs/1804.03329
5. Sarkar. "Low Distortion Delaunay Embedding of Trees in Hyperbolic Plane." Graph Drawing 2011. https://homepages.inf.ed.ac.uk/rsarkar/papers/HyperbolicDelaunayFull.pdf
6. Ganea, Bécigneul, Hofmann. "Hyperbolic Neural Networks." NeurIPS 2018. https://arxiv.org/abs/1805.09112
7. Gu, Sala, Gunel, Ré. "Learning Mixed-Curvature Representations in Product Spaces." ICLR 2019. https://openreview.net/pdf?id=HJxeWnCcF7
8. Cho, DeMeo, Peng, Berger. "Large-Margin Classification in Hyperbolic Space." AISTATS 2019. https://arxiv.org/abs/1806.00437
9. Bonnabel. "Stochastic Gradient Descent on Riemannian Manifolds." IEEE TAC 2013. https://arxiv.org/abs/1111.5280
10. Code: https://github.com/HazyResearch/hyperbolics ; pretrained embeddings: https://hypere.stanford.edu/

*Note on sourcing: the claims about hyperbolic SVM come from Cho et al., not from the DAWN posts, which do not mention SVMs or hyperbolic MLR by name. The "images with hierarchical labels" use case is an inference from the hyperbolic-classifier literature (e.g. Khrulkov et al. 2019, cited in post 2), not a result reported in either post.*
