# Created by gonzalezroy at 6/17/24
"""
Functions related to the calculation of geometric descriptors
"""
import os.path
import time
from os.path import join

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numba import njit

import compass.descriptors.topo_traj as tt


@njit(parallel=False)
def calc_dist(atom1_coords, atom2_coords):
    """
    Computes the Euclidean distance between two atoms in a molecule

    Args:
        atom1_coords: 3D coordinate array of the first atom
        atom2_coords: 3D coordinate array of the second atom

    Returns:
        float: the Euclidean distance between the two atoms
    """
    dx = atom1_coords[0] - atom2_coords[0]
    dy = atom1_coords[1] - atom2_coords[1]
    dz = atom1_coords[2] - atom2_coords[2]
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


@njit(parallel=False)
def calc_min_dist(coords1, coords2):
    """
    Get the minimumm distance between two sets of coordinates

    Args:
        coords1: coordinates of the first residue
        coords2: coordinates of the second residue

    Returns:
        The minimum distance between two sets of coordinates
    """
    diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
    dist_squared = np.sum(diff ** 2, axis=2)

    return np.sqrt(np.min(dist_squared))

@njit(parallel=False)
def calc_angles(coords_d, coords_h, coords_a):
    """
    Calculate all pairwise D-H-A angles

    Args:
        coords_d: donor coordinates, shape (n1, 3)
        coords_h: hydrogen coordinates, shape (n1, 3)
        coords_a: acceptor coordinates, shape (n2, 3)

    Returns:
        angles: array of shape (n1, n2) with angles in degrees
    """
    # Vectors from H to D and from H to A (matching your original: d - h, a - h)
    # vec_dh: (n1, 1, 3) for broadcasting
    vec_dh = coords_d[:, np.newaxis, :] - coords_h[:, np.newaxis, :]  # D - H
    # vec_ah: (n1, n2, 3)
    vec_ah = coords_a[np.newaxis, :, :] - coords_h[:, np.newaxis, :]  # A - H

    # Compute dot products: sum over last axis
    dot_product = np.sum(vec_dh * vec_ah, axis=2)  # shape: (n1, n2)

    # Compute magnitudes
    dh_norm = np.sqrt(np.sum(vec_dh ** 2, axis=2))  # shape: (n1, 1)
    ah_norm = np.sqrt(np.sum(vec_ah ** 2, axis=2))  # shape: (n1, n2)

    # Compute cosine and clip to avoid numerical errors from floating point
    cos_angle = dot_product / (dh_norm * ah_norm + 1e-10)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Convert to degrees
    angles = np.rad2deg(np.arccos(cos_angle))

    return angles

@njit(parallel=False)
def find_sb(frame_coords, oxy_i, nitro_j, cut_off):
    """
    Find a single salt bridge between two residues

    Args:
        frame_coords: 3D coordinates of the frame
        oxy_i: oxygen atom index of the first residue
        nitro_j: nitrogen atom index of the second residue
        cut_off: cutoff distance

    Returns:
        bool: True if a salt bridge is found, False otherwise
    """
    # Get coordinates for all oxygen and nitrogen atoms
    coords1 = frame_coords[oxy_i]  # shape: (n1, 3)
    coords2 = frame_coords[nitro_j]  # shape: (n2, 3)

    # Compute all pairwise distances using broadcasting
    # coords1[:, np.newaxis, :] has shape (n1, 1, 3)
    # coords2[np.newaxis, :, :] has shape (1, n2, 3)
    # diff has shape (n1, n2, 3)
    diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]

    # Calculate squared distances: (n1, n2)
    dist_squared = np.sum(diff ** 2, axis=2)

    # Check if any distance is below cut off threshold
    return np.any(dist_squared < cut_off * cut_off)


@njit(parallel=False)
def find_hb(frame_coords, donors_i, hydros_i, acceptors_j, da_cut, ha_cut, dha_cut):
    """
    Find a single hydrogen bond between two residues

    Args:
        frame_coords: 3D coordinates of the frame
        donors_i: donor atom indices of the first residue
        hydros_i: hydrogen atom indices of the first residue
        acceptors_j: acceptor atom indices of the second residue
        da_cut: distance cutoff for the donor-acceptor distance
        ha_cut: distance cutoff for the hydrogen-acceptor distance
        dha_cut: angle cutoff for the donor-hydrogen-acceptor angle

    Returns:
        bool: True if a hydrogen bond is found, False otherwise
    """
    # Get all coordinates at once
    coords_d = frame_coords[donors_i]  # shape: (n1, 3)
    coords_h = frame_coords[hydros_i]  # shape: (n1, 3)
    coords_a = frame_coords[acceptors_j]  # shape: (n2, 3)

    # Compute all pairwise donor-acceptor distances
    # Broadcasting: (n1, 1, 3) - (1, n2, 3) = (n1, n2, 3)
    diff_da = coords_d[:, np.newaxis, :] - coords_a[np.newaxis, :, :]
    dist_da = np.sqrt(np.sum(diff_da ** 2, axis=2))  # shape: (n1, n2)

    # Filter pairs that pass donor-acceptor distance cutoff
    da_mask = dist_da < da_cut  # shape: (n1, n2)

    # Early exit if no pairs pass first cutoff
    if not np.any(da_mask):
        return False

    # Compute all pairwise hydrogen-acceptor distances
    diff_ha = coords_h[:, np.newaxis, :] - coords_a[np.newaxis, :, :]
    dist_ha = np.sqrt(np.sum(diff_ha ** 2, axis=2))  # shape: (n1, n2)

    # Filter pairs that pass both distance cutoffs
    ha_mask = dist_ha < ha_cut
    combined_mask = da_mask & ha_mask  # shape: (n1, n2)

    # Early exit if no pairs pass both distance cutoffs
    if not np.any(combined_mask):
        return False

    # Compute angles only for pairs that passed distance filters
    # Vectorized angle calculation for all pairs
    angles = calc_angles(coords_d, coords_h, coords_a)  # shape: (n1, n2)

    # Check if any pair satisfies all criteria
    return np.any(combined_mask & (angles > dha_cut))

def save_matrix(arr, n, out_name, norm=False, prec=2):
    """
    Save a matrix to a file

    Args:
        arr: array to save
        n: number of columns
        missing: missing indices
        out_name: output file name
        norm: normalize the matrix?
        diag: fill the diagonal with 1?
        prec: precision of the values to save

    Returns:
        matrix_name: name of the saved matrix
    """
    # Convert to matrix if needed
    matrix = tt.to_matrix(arr, n) if len(arr.shape) == 1 else arr

    # Normalize if requested
    if norm:
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        matrix = (matrix - min_val) / (max_val - min_val)

    # Save matrix
    np.savetxt(out_name, matrix, fmt=f"%.{prec}f")
    return matrix


def get_matrix_name(out_dir, title, suffix):
    """
    Generate a matrix name

    Args:
        out_dir: output directory
        title: title of the matrix
        suffix: suffix of the matrix

    Returns:
        matrix_name: name of the matrix
    """
    matrix_dir = join(out_dir, "matrices")
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir, exist_ok=True)
    return join(out_dir, 'matrices', f"{title}_{suffix}.mat")


def plot_matrix(matrix, matrix_title, output_name):
    """
    Plot the Generalized Correlation matrix.

    Args:
        matrix: Generalized Correlation matrix
        matrix_title: title of the matrix
        output_name: output name for the plot
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(matrix, cmap="jet")
    plt.title(matrix_title)
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    plt.savefig(output_name)
    plt.close()


def process_matrices(arg, n, calphas, ave_min_dist, occ_nb, cp, occ_sb, occ_hb,
                     occ_int, mi, gc, first_timer, ):
    # Declare missing residues
    # cp_miss = [i for i, x in enumerate(calphas) if calphas[x] == -1]

    # Declare matrices to process
    matrices = {
        "MINDIST": {"data": ave_min_dist, "norm": False,
                    "prec": 4, "title": "Pairwise Minimum Distances"},

        "NONBOND": {"data": occ_nb, "norm": False, "prec": 4,
                    "title": "Non-Bonded Interactions"},

        "SALTBRIDGES": {"data": occ_sb, "norm": False, "prec": 4,
                        "title": "Salt Bridges"},

        "HBONDS": {"data": occ_hb, "norm": False, "prec": 4,
                   "title": "Hydrogen Bonds"},

        "INTERACTIONS": {"data": occ_int, "norm": False, "prec": 2,
                         "title": "Interactions"},

        "COMMPROP": {"data": cp, "norm": True, "prec": 4,
                     "title": "Communication Properties"},

        "MI": {"data": mi, "norm": True, "prec": 4,
               "title": "Mutual Information"},

        "GC": {"data": gc, "norm": True, "prec": 4,
               "title": "Generalized Correlation"},
    }

    # Process matrices
    matrices_name = {}
    for matrix in matrices:
        # Get matrix data
        data = matrices[matrix]["data"]
        # miss_list = matrices[matrix]["miss"]
        normalize = matrices[matrix]["norm"]
        precision = matrices[matrix]["prec"]

        # Save matrices
        matrix_name = get_matrix_name(arg.out_dir, arg.title, matrix)
        matrices_name.update({matrix: matrix_name})
        matrix_data = save_matrix(data, n, matrix_name, norm=normalize, prec=precision)
        matrices[matrix].update({"data": matrix_data})

        # Plot matrices
        plot_name = matrix_name.replace(".mat", ".png")
        matrix_title = matrices[matrix]["title"]
        plot_matrix(matrix_data, matrix_title, plot_name)

    saving_time = round(time.time() - first_timer, 2)
    print(f" ⏱️  Until saving & plotting matrices: {saving_time} s")
    return matrices, matrices_name

