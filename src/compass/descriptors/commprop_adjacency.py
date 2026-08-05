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
    COMPASS 2.0 adjacency: COMMPROP as the sole edge-weight source.

    Rationale: across ~2800 proteins the PCA fusion collapsed to COMMPROP
    (rho~0.86 with the fused adjacency), and on ASBench COMMPROP was the single
    observable that best recovered allosteric sites, with the fusion adding no
    orthogonal signal. So the adjacency IS COMMPROP, made explicit.

    Topology (contact gating via MINDIST) stays in build_graph_from_matrices.

    Returns the (n, n) [0,1] adjacency and, if save=True, writes
    {out_dir}/matrices/{title}_ADJACENCY.mat in the COMPASS 1.0 convention.
    """
    A = np.asarray(mat_dict["COMMPROP"]["data"], dtype=float)
    n = A.shape[0]
    assert A.shape == (n, n), f"COMMPROP must be square, got {A.shape}"

    A = 0.5 * (A + A.T)                      # enforce symmetry (guards asymmetric input)

    if norm:                                 # min-max over finite entries -> [0,1]
        finite = np.isfinite(A)
        lo, hi = A[finite].min(), A[finite].max()
        A = (A - lo) / (hi - lo) if hi > lo else np.zeros_like(A)

    np.fill_diagonal(A, 1.0)             # self-adjacency = 1 (COMPASS 1.0 convention)

    if save:
        import os
        mdir = os.path.join(out_dir, "matrices")
        os.makedirs(mdir, exist_ok=True)
        out_name = os.path.join(mdir, f"{title}_ADJACENCY.mat")
        np.savetxt(out_name, A, fmt=f"%.{prec}f")

    return A