# Network edge weights: adjacency, distance, and the `edge_weight` option

This note explains how COMPASS turns the PCA **adjacency** matrix into the edge
weights used by the network analysis, why the semantics are subtle, and what the
optional `edge_weight` config key does. It exists to support a team decision on
which edge-weight convention to adopt.

## The two stages

The value on each graph edge is produced in **two separate stages**. Keeping them
distinct is the key to reading the code correctly.

### Stage 1 — build the ADJACENCY matrix (`descriptors/pca.py`)

1. PCA reduces each residue's profile (its rows across the `GC`, `INTERACTIONS`
   and `COMMPROP` matrices) to a 2-D point (`perform_pca`, `n_components=2`).
2. `calc_chunk_distances` / `calc_adjacency_matrix` compute the **Euclidean
   distance in that PCA space** between residues *i* and *j* (stored only when
   `> threshold = 0.3`, else 0). At this point large value = **far apart** in
   profile space.
3. `run_pca` then does `adj_mat = 1 - adj_mat` — flipping distance into a
   **similarity** — and `save_matrix(..., norm=True)` normalizes it to `[0, 1]`.

**Result:** the stored `*_ADJACENCY.mat` is `normalize(1 - PCA_distance)`.

> **High adjacency ⇒ residues are *close* in PCA space ⇒ they have *similar*
> `[GC, INTERACTIONS, COMMPROP]` profiles.**
>
> Note this is a *profile similarity between two residues' overall interaction
> fingerprints*, **not** a direct pairwise coupling strength between *i* and *j*.

The `1 - x` here is a deliberate, monotonic-decreasing transform (conceptually in
the same family as `1/x` or `-log(x)`), applied to the **distance** to define the
similarity.

### Stage 2 — turn adjacency into an edge weight (`network/graph_constructor.py`)

`build_graph_from_matrices` adds an edge for every residue pair that is both
spatially close (`MINDIST < Graph cutoff`, default 5 Å) **and** has
`adjacency > 0`. The **edge set is the spatial-contact graph**; the
`edge_weight` mode only changes the numeric weight stored on those edges, via
`_edge_cost`:

| `edge_weight` | edge weight formula | meaning of the weight |
|---|---|---|
| `adjacency` *(default)* | `weight = adjacency`         | the similarity itself |
| `inverse`               | `weight = 1 / adjacency`     | inverse similarity    |
| `neglog`                | `weight = -log(adjacency)`   | surprisal-like        |

The raw similarity is always stored on the edge as `adjacency` too, so it can be
inspected or re-transformed later without rebuilding.

## Why the weight direction matters

NetworkX treats the edge `weight` as a **distance / cost**: `betweenness_centrality`,
`closeness_centrality`, and shortest-/alternative-path finding all **minimize**
total weight. So the mode changes *which* edges a "short" path prefers:

| mode | cost(adj=0.9) | cost(adj=0.1) | shortest paths prefer |
|---|---|---|---|
| `adjacency` (default) | 0.900 | 0.100 | **low-adjacency** (dissimilar-profile) edges |
| `inverse` | 1.111 | 10.000 | high-adjacency (similar-profile) edges |
| `neglog`  | 0.105 |  2.303 | high-adjacency (similar-profile) edges |

Because the default feeds the **similarity directly as a distance**, high-adjacency
edges are treated as *far* and are avoided by shortest paths. The Stage-1 `1 - x`
does **not** offset this: it defines the similarity, but the similarity is then
used as a cost without being re-inverted. `inverse` / `neglog` perform that
re-inversion at Stage 2.

Whether paths *should* prefer similar- or dissimilar-profile residues is a
**modeling choice**, not a bug — hence the toggle rather than a forced change.

## What each mode affects (and does not)

- **Affected** (use the weight as a distance): `betweenness`, `closeness`,
  shortest paths, alternative paths.
- **Not affected**: Leiden **communities** and **cliques** are computed on the
  *unweighted* graph, so they are identical across all three modes. The edge set
  itself is also identical across modes.

## Usage

The key is optional; omitting it preserves the original behavior exactly.

```ini
[distance cutoffs]
Graph = 5
Cliques = 10
edge_weight = adjacency    # adjacency (default) | inverse | neglog
```

## Caveats

- `neglog` gives an edge with `adjacency = 1` a weight of `0` (zero-cost). Valid
  for Dijkstra (non-negative), just be aware such edges are "free".
- `inverse` sends very small adjacencies to very large weights.
- Artificial edges added by `ensure_graph_connectivity` (only when the graph is
  disconnected) carry no explicit weight; NetworkX then treats them as `1`,
  whose relative scale differs between modes. This is independent of the choice.

## Pointers

- Stage 1: `src/compass/descriptors/pca.py` (`run_pca`, `calc_adjacency_matrix`)
- Stage 2: `src/compass/network/graph_constructor.py` (`_edge_cost`,
  `build_graph_from_matrices`)
- Config: `src/compass/descriptors/config.py` (`optional_params`,
  `allowed_edge_weights`)
