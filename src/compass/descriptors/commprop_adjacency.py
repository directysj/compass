import os

import numpy as np


def commprop_adjacency(
        mat_dict,
        out_dir,
        title,
        norm=True,
        prec=4,
        save=True
):
    """
    COMPASS 2.0 adjacency: COMMPROP (communication propensity) as the sole,
    fully transparent edge-weight source.

    Why COMMPROP alone
    ------------------
    The previous adjacency fused several descriptors (CP, interaction frequency,
    COMMPROP) through PCA, which introduced a dimensionality-reduction black box.
    Benchmarking showed COMMPROP already recovers the communication signal on its
    own (the fused adjacency correlated with COMMPROP at rho~0.86, and on ASBench
    COMMPROP was the single best observable for allosteric-site recovery, with the
    fusion adding no orthogonal signal). So the adjacency IS COMMPROP, made
    explicit and reproducible.

    What this matrix means
    ----------------------
    The returned matrix is the *coupling* (affinity) A_ij in [0, 1]: higher =
    stronger dynamic communication between residues i and j. It is the object used
    downstream for label generation and, after contact gating (MINDIST) in
    build_graph_from_matrices, becomes the weighted edge set of the network.

    Graph-theory convention used downstream
    ---------------------------------------
    The coupling A_ij is NOT used directly as a shortest-path cost. A high coupling
    must map to a SHORT distance so that Dijkstra/betweenness/closeness find the
    strongest-communication path, not the weakest. Interpreting A_ij as a per-edge
    transmission efficiency, the efficiency along a path is the product of its edge
    couplings, and maximizing that product is equivalent to minimizing the sum of
    -log(A_ij). Hence, the graph edge distance is d_ij = -log(A_ij) (see
    GraphConstructor.build_graph_from_matrices). This keeps the saved ADJACENCY.mat
    an interpretable coupling matrix while giving the network a principled,
    additive distance.

    Parameters
    ----------
    mat_dict : dict
        Descriptor matrices; must contain mat_dict["COMMPROP"]["data"] as an
        (n, n) array (already inverted so that high = strong communication).
    out_dir, title : str
        Output directory and job title; the matrix is written to
        {out_dir}/matrices/{title}_ADJACENCY.mat (COMPASS 1.0 convention).
    norm : bool
        Min-max normalize the finite entries to [0, 1].
    prec : int
        Decimal precision for the saved matrix.
    save : bool
        Write the matrix to disk.

    Returns
    -------
    numpy.ndarray
        The (n, n) coupling/adjacency matrix in [0, 1] with a unit diagonal.
    """
    A = np.asarray(mat_dict["COMMPROP"]["data"], dtype=float)
    n = A.shape[0]
    assert A.shape == (n, n), f"COMMPROP must be square, got {A.shape}"

    A = 0.5 * (A + A.T)                       # enforce symmetry (guards asymmetric input)

    if norm:                                  # min-max over finite entries -> [0, 1]
        finite = np.isfinite(A)
        lo, hi = A[finite].min(), A[finite].max()
        A = (A - lo) / (hi - lo) if hi > lo else np.zeros_like(A)

    np.fill_diagonal(A, 1.0)                  # self-adjacency = 1 (COMPASS 1.0 convention)

    if save:
        mdir = os.path.join(out_dir, "matrices")
        os.makedirs(mdir, exist_ok=True)
        out_name = os.path.join(mdir, f"{title}_ADJACENCY.mat")
        np.savetxt(out_name, A, fmt=f"%.{prec}f")

    return A
