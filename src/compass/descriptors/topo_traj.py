# Created by gonzalezroy at 6/6/24
"""Manage common operations as well as those related to topo and traj files"""
import json
import time
from collections import defaultdict
from os.path import basename, join

import mdtraj as md
import numpy as np
from numba import njit
from numba.typed.typeddict import Dict
from numba.core import types


def prepare_datastructures(arg, first_timer):
    """
    Prepare datastructures for the calculation of descriptors

    Args:
        arg: namespace with the arguments
        first_timer: first timer to measure the time

    Returns:
        mini_traj: the first chunk of the trajectory
        trajs: list of trajectories
        resids_to_atoms: numba Dict of each residue's indices
        resids_to_noh: numba Dict of each residue's indices without hydrogens
        calphas: numba Dict of each residue's C-alpha indices
        oxy: numba Dict of each residue's oxygen indices
        nitro: numba Dict of each residue's nitrogen indices
        donors: numba Dict of each residue's donor indices
        hydros: numba Dict of each residue's hydrogen indices
        acceptors: numba Dict of each residue's acceptor indices
        corr_indices: list of indices of atoms to be considered for correlation
    """
    # Produce mapping files
    # mapping = Mapping(arg.out_dir)
    # map_file, renumbered_pdb = mapping.pre_processing(arg.topo)
    # mapping.post_processing(map_file, renumbered_pdb)

    # Load trajectory
    trajs = arg.traj.split()
    mini_traj = next(md.iterload(trajs[0], top=arg.topo, chunk=1))
    full_topo = mini_traj.topology.to_dataframe()[0]
    map_file = join(arg.out_dir, 'mapping_file.txt')
    # Produce mapping files
    # remap_toppology(arg.topo, mini_traj, arg.out_dir)

    # Indices of residues in the load trajectory and equivalence
    resids_to_atoms, resids_to_noh, internal_equiv = get_resids_indices(mini_traj)
    # print(resids_to_atoms, "resids_to_atoms in topo_traj")
    raw = {y: x for x in resids_to_atoms for y in resids_to_atoms[x]}
    atoms_to_resids = pydict_to_numbadict(raw)

    # Atom selections indices for descriptors calculation
    calphas = get_calpha_p_indices(mini_traj, atoms_to_resids, map_file=map_file)
    oxy, nitro = get_sb_indices(full_topo, atoms_to_resids)
    donors, hydros, acceptors = get_dha_indices(mini_traj, arg.heavies, atoms_to_resids)
    # Correlation atoms = the CA / C5' ATOM index of each residue, ordered by
    # residue index r so the MI/GC rows line up with every other matrix (which
    # are all indexed in resids_to_atoms order). Taken from calphas, which is now
    # keyed by residue index. The old code used calphas.keys() (0..n-1), slicing
    # the first n atoms instead of the CA atoms.
    corr_indices = [int(calphas[r]) for r in range(len(calphas))]

    prep_time = round(time.time() - first_timer, 2)
    print(f" 📋 System details: number of trajectories are {len(trajs)}")
    print(f" 📋 System details: number of residues are {len(calphas)}")
    print(f" ⏱️  Until datastructures prepared: {prep_time} s")

    return (mini_traj, trajs, resids_to_atoms, resids_to_noh, calphas, oxy, nitro,
            donors, hydros, acceptors, corr_indices)


def get_xyz_chunks(trajs, topo, chunk_size=500):
    """
    Load chunks of xyz coordinates from a list of trajectories

    Args:
        trajs: list of trajectories
        topo: system topology
        chunk_size: size of the chunk to load

    Returns:
        chunk.xyz: chunk of xyz coordinates
    """
    for traj in trajs:
        chunks = md.iterload(traj, top=topo, chunk=chunk_size)
        for chunk in chunks:
            yield chunk.xyz


# Nucleic-acid residue-name patterns (DNA / RNA, incl. 5'/3' terminal variants).
_DNA_RESNAME = "(resname =~ '(5|3)?D([ATGC]){1}(3|5)?$')"
_RNA_RESNAME = "(resname =~ '(3|5)?R?([AUGC]){1}(3|5)?$')"


def _backbone_anchor_indices(trajectory):
    """
    Indices of the per-residue anchor atoms that define which residues COMPASS
    treats as polymer: protein alpha-carbons (name CA) and nucleic C5' carbons.

    A residue is analysed iff it owns one of these anchors, so waters, ions,
    lipids, and ligands are excluded -- COMPASS runs on protein and nucleic-acid
    atoms ONLY. The element guard (an anchor must be carbon) stops a non-polymer
    atom that merely shares the name -- e.g. a calcium ion named 'CA' -- from
    being mistaken for an alpha-carbon.

    Args:
        trajectory: trajectory loaded in mdtraj format

    Returns:
        Sorted numpy array of anchor atom indices (topology-wide, 0-based).
    """
    top = trajectory.topology
    ca_atoms = top.select("name CA")
    p_atoms = top.select(
        f'({_DNA_RESNAME} or {_RNA_RESNAME}) and name "C5\'"')
    raw = np.concatenate((ca_atoms, p_atoms)).astype(int)
    anchors = [i for i in raw
               if getattr(top.atom(int(i)).element, "symbol", None) == "C"]
    return np.sort(np.asarray(anchors, dtype=int))


def get_resids_indices(trajectory):
    """
    Get indices of residues in the loaded trajectory, keeping ONLY protein and
    nucleic-acid residues while excluding water, ions, lipids, and ligands.

    Args:
        trajectory: trajectory loaded in mdtraj format
    Returns:
        res_ind_numba: numba Dict of each residue's all atoms indices
        res_ind_noh_numba: numba Dict of each residue's non-hydrogen atom indices
        babel_dict: the equivalence between the original resid numbering and
                   the 0-based numbering used internally
    """
    # Parse the topological information
    df = trajectory.topology.to_dataframe()[0]

    # Per-residue anchor atoms (protein CA / nucleic C5'); residues without an
    # anchor -- water, ions, lipids, ligands -- are dropped entirely.
    anchor_atoms = _backbone_anchor_indices(trajectory)
    anchor_set = set(int(i) for i in anchor_atoms)

    # Group EVERY atom by residue key first, so groupby(...).indices yields the
    # real (topology-wide) atom indices. Grouping a pre-filtered frame instead
    # returns positions WITHIN the subset -- wrong atom indices as soon as the
    # system contains any non-protein atoms.
    group_all = df.groupby(["chainID", "resSeq", "segmentID"]).indices

    # Keep only residues that own an anchor atom (protein / nucleic).
    group_by_index = {
        key: idx for key, idx in group_all.items()
        if anchor_set.intersection(idx.tolist())
    }
    if not group_by_index:
        raise ValueError(
            "No protein or nucleic-acid residues found in the topology; "
            "nothing for COMPASS to analyse.")

    # Create non-hydrogen version
    group_by_index_noh = {}
    for key, values in group_by_index.items():
        noh = values[df.loc[values, "element"] != "H"]
        group_by_index_noh[key] = noh

    # Create babel dictionaries
    babel_dict = {i: x for i, x in enumerate(group_by_index)}

    # Transform to zero-based indices dictionaries
    res_ind_zero = {i: group_by_index[x] for i, x in enumerate(group_by_index)}
    res_ind_noh = {i: group_by_index_noh[x] for i, x in
                   enumerate(group_by_index_noh)}

    # Convert to numba dictionaries
    res_ind_numba = pydict_to_numbadict(res_ind_zero)
    res_ind_noh_numba = pydict_to_numbadict(res_ind_noh)

    return res_ind_numba, res_ind_noh_numba, babel_dict


'''
def get_resids_indices(trajectory):
    """
    Get indices of residues in the load trajectory

    Args:
        trajectory: trajectory loaded in mdtraj format

    Returns:
        res_ind_numba: numba Dict of each residue's indices
        babel_dict: the equivalence between the original resid numbering and
                    the 0-based used internally
    """
    # Parse the topological information
    df = trajectory.topology.to_dataframe()[0]
    group_by_index = df.groupby(["chainID", "resSeq", "segmentID"]).indices
    group_by_index_noh = {}
    for key in group_by_index:
        values = group_by_index[key]
        noh = values[df.loc[values, "element"] != "H"]
        group_by_index_noh[key] = noh

    babel_dict = {i: x for i, x in enumerate(group_by_index)}
    babel_dict_noh = {i: x for i, x in enumerate(group_by_index_noh)}

    # Transform to numba-dict
    res_ind_zero = {i: group_by_index[x] for i, x in enumerate(group_by_index)}
    res_ind_noh = {i: group_by_index_noh[x] for i, x in
                   enumerate(group_by_index_noh)}

    res_ind_numba = pydict_to_numbadict(res_ind_zero)
    res_ind_noh_numba = pydict_to_numbadict(res_ind_noh)
    return res_ind_numba, res_ind_noh_numba, babel_dict

'''


def get_corr_indices(trajectory, map_file):
    """
    Get atomic indices for correlation calculation

    Args:
        trajectory: trajectory loaded in mdtraj format

    Returns:
        all_atoms: indices of all atoms to be considered for correlation
    """

    # Per-residue anchor atoms (protein CA / nucleic C5'), carbon-guarded and
    # identical to the set used by get_resids_indices so counts stay in sync.
    all_atoms = _backbone_anchor_indices(trajectory)

    # Write atom details to the specified map_file
    with open(map_file, 'w') as file:
        for idx in all_atoms:
            atom = trajectory.topology.atom(int(idx))

            # Writing atom details to file
            file.write(f"Atom Index: {idx}, Atom Name: {atom.name}, "
                       f"Residue Name: {atom.residue.name}, Residue Index: {atom.residue.index}, "
                       f"Residue Number: {atom.residue}, chain id:{atom.residue.chain.chain_id}\n")
    return np.asarray(all_atoms, dtype=np.int32)


def get_calpha_p_indices(trajectory, atoms_to_resids, map_file, numba=True):
    """
    Get atomic indices for C-alpha atoms

    Args:
        trajectory: trajectory loaded in mdtraj format
        atoms_to_resids: dict mapping atoms indices to residues indices
        numba: whether to return a numba dict or a regular dict

    Returns:
        alphas: indices of C-alpha atoms
    """
    # Per-residue anchor atoms (protein CA / nucleic C5'), carbon-guarded and
    # identical to the set used by get_resids_indices / get_corr_indices.
    all_atoms = _backbone_anchor_indices(trajectory)
    n_resids = len(all_atoms)
    calphas_p_raw = get_corr_indices(trajectory, map_file)
    # print(np.shape(calphas_p_raw),n_resids)
    # calphas[r] = the CA/C5' ATOM index of residue r, where r is the residue
    # index used by resids_to_atoms / resids_to_noh (groupby order). We key each
    # anchor by atoms_to_resids[anchor] instead of by its position in the sorted
    # anchor list, so this stays aligned with resids_to_noh even if atom order
    # and residue-enumeration order differ (non-monotonic resSeq, insertion
    # codes, string-sorted chain IDs). The value is the atom index because it is
    # indexed straight into frame_coords downstream; the previous code stored the
    # residue index r itself, so frame_coords[r] read the wrong atom.
    calphas_p = {}
    for a in calphas_p_raw:
        calphas_p[int(atoms_to_resids[int(a)])] = int(a)

    if len(calphas_p) != n_resids:
        raise ValueError("\nThe number of calphas + P atoms is different from"
                         " the number of residues")

    if numba:
        alphas = pydict_to_numbadict(calphas_p)
    else:
        alphas = calphas_p
    return alphas


def save_atom_mapping(trajectory, calphas, out_path):
    """
    Write the canonical node -> residue mapping used by the network stage.

    Keyed by residue index r (0..n-1) in the SAME order as every descriptor
    matrix (resids_to_atoms / groupby order), because it is built from `calphas`
    (residue index -> CA/C5' atom index). This replaces re-deriving the mapping
    from the PDB in atom-index order, which could misalign labels with the matrix
    indices when atom order != residue-enumeration order. Values match
    ReadFiles.atom_mapping: [residue_name, atom_name, resSeq, chain_id].

    Args:
        trajectory: MDTraj trajectory whose topology owns the calpha atoms
        calphas: dict {residue_index: CA/C5' atom index} (numba or plain)
        out_path: JSON file to write

    Returns:
        mapping: the dict written (str keys, list values), for convenience
    """
    top = trajectory.topology
    mapping = {}
    for r in range(len(calphas)):
        atom = top.atom(int(calphas[r]))
        residue = atom.residue
        chain_id = residue.chain.chain_id \
            if residue.chain.chain_id is not None else ''
        mapping[str(r)] = [residue.name, atom.name, int(residue.resSeq),
                           chain_id]
    with open(out_path, 'w') as f:
        json.dump(mapping, f)
    return mapping


def get_sb_indices(topo_df, atoms_to_resids):
    """
    Get atomic indices for salt bridges calculation

    Args:
        topo_df: topology dataframe as returned by MDTraj
        atoms_to_resids: dict mapping atoms indices to residues indices

    Returns:
        o_indices: indices of selected oxygen atoms (see VMD definitions)
        n_indices: indices of selected nitrogen atoms (see VMD definitions)

    """
    # Macro definitions
    sel_O1 = topo_df.resName.isin(["ASP", "GLU"])
    sel_O2 = topo_df.element == "O"
    sel_N1 = topo_df.resName.isin(["ARG", "HIS", "LYS", "HSP"])
    sel_N2 = topo_df.element == "N"

    # Get indices of selected atoms, restricted to tracked (protein/nucleic)
    # residues so a non-polymer residue sharing one of these names cannot leak
    # in and raise a KeyError on the lookups below.
    valid_atoms = set(atoms_to_resids.keys())
    o_indices = [x for x in topo_df[sel_O1 & sel_O2].index if x in valid_atoms]
    n_indices = [x for x in topo_df[sel_N1 & sel_N2].index if x in valid_atoms]

    # Process the oxygen indices to a numba dict
    oxy_raw1 = defaultdict(list)
    [oxy_raw1[atoms_to_resids[x]].append(x) for x in o_indices]
    oxy_raw3 = {x: np.asarray(oxy_raw1[x], dtype=np.int32) for x in oxy_raw1}
    oxy = pydict_to_numbadict(oxy_raw3)

    # Process the nitrogen indices to a numba dict
    nitro_raw1 = defaultdict(list)
    [nitro_raw1[atoms_to_resids[x]].append(x) for x in n_indices]
    nitro_raw3 = {x: np.asarray(nitro_raw1[x], dtype=np.int32) for x in
                  nitro_raw1}
    nitro = pydict_to_numbadict(nitro_raw3)
    return oxy, nitro


def get_dha_indices(trajectory, heavies_elements, atoms_to_resids):
    """
    Get 0-based indices of donors, hydrogens, and acceptors in an MDTraj traj

    Args:
        trajectory: MDTraj trajectory object
        heavies_elements: name of elements considered as heavies
        atoms_to_resids: dict mapping atoms indices to residues indices

    Returns:
        donors: indices of donor atoms (N or O bonded to H)
        hydros: indices of hydrogen atoms (H bonded to N or O)
        heavies: indices of heavy atoms (N or O)
    """
    # Get heavies and hydrogen indices
    df, bonds = trajectory.topology.to_dataframe()
    all_hydrogens = set(df[df.element == "H"].index)

    a_raw1 = set(np.where(df.element.isin(heavies_elements))[0])

    # Keep only heavy atoms that belong to tracked residues (proteins and
    # nucleic acids). Solvated / membrane systems (e.g. GPCRMD) contain waters,
    # lipids, ions, and ligands whose N/O/S atoms are NOT in atoms_to_resids;
    # looking them up below raised a bare numba KeyError. Restricting the heavy
    # set here also filters the donors and hydrogens, since both are derived
    # from bonds to these atoms and a bonded H shares the heavy atom's residue.
    valid_atoms = set(atoms_to_resids.keys())
    a_raw1 &= valid_atoms

    # Find D-H indices
    h_raw1 = []
    d_raw1 = []
    for values in bonds:
        at1 = int(values[0])
        at2 = int(values[1])
        if (at1 in all_hydrogens) and (at2 in a_raw1):
            h_raw1.append(at1)
            d_raw1.append(at2)
        elif (at2 in all_hydrogens) and (at1 in a_raw1):
            h_raw1.append(at2)
            d_raw1.append(at1)
        else:
            continue

    # Process the indices of donors to a numba dict
    d_raw = defaultdict(list)
    [d_raw[atoms_to_resids[x]].append(x) for x in d_raw1]
    d_raw3 = {x: np.asarray(d_raw[x], dtype=np.int32) for x in d_raw}
    donors = pydict_to_numbadict(d_raw3)

    # Process the indices of hydrogens to a numba dict
    h_raw = defaultdict(list)
    [h_raw[atoms_to_resids[x]].append(x) for x in h_raw1]
    h_raw3 = {x: np.asarray(h_raw[x], dtype=np.int32) for x in h_raw}
    hydros = pydict_to_numbadict(h_raw3)

    # Process the indices of acceptors to a numba dict
    a_raw = defaultdict(list)
    [a_raw[atoms_to_resids[x]].append(x) for x in a_raw1]
    a_raw3 = {x: np.asarray(a_raw[x], dtype=np.int32) for x in a_raw}
    acceptors = pydict_to_numbadict(a_raw3)
    return donors, hydros, acceptors


@njit(parallel=False)
def dict_get(dico, key):
    """
    Get the value of a key in a dictionary or return None if the key is not in
    the dictionary

    Args:
        dico: dictionary to search
        key: key to search

    Returns:
        value: value of the key in the dictionary or None if the key is not in
               the dictionary
    """
    try:
        value = dico[key]
        return value
    except:
        return np.empty(0, dtype=np.int32)

def _infer_numba_type(value):
    """Map a Python/numpy value to a numba type usable in Dict.empty()."""
    if isinstance(value, np.ndarray):
        if value.dtype == np.int32:   return types.int32[:]
        if value.dtype == np.int64:   return types.int64[:]
        if value.dtype == np.float32: return types.float32[:]
        if value.dtype == np.float64: return types.float64[:]
        raise TypeError(f"Unsupported ndarray dtype: {value.dtype}")
    if isinstance(value, (bool, np.bool_)):     return types.boolean
    if isinstance(value, (int, np.integer)):    return types.int64
    if isinstance(value, (float, np.floating)): return types.float64
    if isinstance(value, str):                  return types.unicode_type
    raise TypeError(f"Unsupported value type: {type(value)}")

def pydict_to_numbadict(py_dict, key_type=None, value_type=None):
    """
    Convert a Python dict into a numba.typed.Dict with an explicit signature.

    Compatible with numba >= 0.59, which no longer lazily infers element types
    from ``Dict().update({k: v})``. Types are inferred from the first item
    when not supplied; an empty input dict requires explicit ``key_type``
    and ``value_type``.
    """
    if not py_dict:
        if key_type is None or value_type is None:
            raise ValueError(
                "pydict_to_numbadict: cannot infer types from an empty dict; "
                "pass key_type and value_type explicitly."
            )
        return Dict.empty(key_type=key_type, value_type=value_type)

    first_key = next(iter(py_dict))
    first_val = py_dict[first_key]
    if key_type   is None: key_type   = _infer_numba_type(first_key)
    if value_type is None: value_type = _infer_numba_type(first_val)

    numba_dict = Dict.empty(key_type=key_type, value_type=value_type)
    for k, v in py_dict.items():
        numba_dict[k] = v
    return numba_dict


def to_matrix(one_dim_array, n, nested=False):
    """
    Converts a one-dimensional array of size = N * (N-1) / 2, into the
    equivalent N * N matrix

    Args:
        one_dim_array: one-dimensional array
        n: number of col / row in the square matrix

    Returns:
        matrix: N * N symmetrycal matrix
    """
    matrix = np.zeros((n, n))
    k = 0

    if nested:
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = one_dim_array[k][0]
                k += 1
    else:
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = one_dim_array[k]
                k += 1

    matrix += matrix.T
    return matrix


class Mapping:
    # todo: simiplify with prody or another pdb parser as explicit line
    #  handling in PDB can be tricky even if standard format exists
    """
    @Sneha's class for residue renumbering and mapping operations on PDB files.
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir

    def pre_processing(self, input_pdb):
        # todo: handle topology formats others than PDB
        """
        Renumbers the residues in a PDB file sequentially from 1, changing all chain identifiers to 'A'.
        Outputs a renumbered PDB file and a map file that records the original and new residue numbers and chains.

        Parameters:
        input_pdb (str): Path to the input PDB file.
        """
        # Define file paths for the renumbered PDB file and the map file
        renumbered_pdb_raw = input_pdb.replace(".pdb", "_renumbered.pdb")
        renumbered_pdb = join(self.out_dir, basename(renumbered_pdb_raw))
        map_file_raw = input_pdb.replace(".pdb", "_map.txt")
        map_file = join(self.out_dir, basename(map_file_raw))

        # Open input PDB file for reading, renumbered PDB file and map file for writing
        with (
            open(input_pdb, "r") as infile,
            open(renumbered_pdb, "w") as outfile,
            open(map_file, "w") as mapfile
        ):
            # Initialize variables for residue renumbering and mapping
            current_residue_number = 0
            residue_map = {}
            last_residue_id = None

            # Loop through each line in the input PDB file
            for line in infile:
                if line.startswith(("ATOM", "HETATM")):
                    # Extract chain ID, old residue number, and residue name
                    chain_id = line[21]
                    old_residue_number = line[22:26].strip()
                    res_name = line[17:20].strip()
                    residue_id = (chain_id, old_residue_number, res_name)

                    # Check if it's a new residue
                    if residue_id != last_residue_id:
                        current_residue_number += 1
                        last_residue_id = residue_id
                        residue_map[(chain_id, old_residue_number)] = (
                            current_residue_number,
                            "A",
                        )

                    # Write renumbered line with new chain ID 'A'
                    new_line = (
                            line[:21]
                            + "A"
                            + str(current_residue_number).rjust(4)
                            + line[26:]
                    )
                    outfile.write(new_line)
                else:
                    # Write non-ATOM/HETATM lines as they are
                    outfile.write(line)

            # Write the residue map to the map file
            for (chain_id, old_number), (
                    new_number, new_chain) in residue_map.items():
                mapfile.write(
                    f"{chain_id} {old_number} {new_chain} {new_number}\n")

        # Print confirmation messages
        # print(f"Renumbered PDB file saved as: {renumbered_pdb}")
        # print(f"Residue mapping file saved as: {map_file}")
        return map_file, renumbered_pdb

    def post_processing(self, map_file, renumbered_pdb):
        """
        Restores the original residue numbering and chain identifiers in a renumbered PDB file using the map file.

        Parameters:
        map_file (str): Path to the map file containing the original and new residue numbers and chains.
        renumbered_pdb (str): Path to the renumbered PDB file.
        """
        # Define file path for the restored PDB file
        original_pdb = renumbered_pdb.replace("_renumbered.pdb",
                                              "_restored.pdb")

        # Initialize a dictionary to store the residue mapping information
        residue_map = {}

        # Read the map file and populate the residue mapping dictionary
        with open(map_file, "r") as mapfile:
            for line in mapfile:
                original_chain, old_number, new_chain, new_number = line.split()
                residue_map[(new_chain, new_number)] = (
                    original_chain, old_number)

        # Open renumbered PDB file for reading and restored PDB file for writing
        with open(renumbered_pdb, "r") as infile, open(original_pdb,
                                                       "w") as outfile:
            # Loop through each line in the renumbered PDB file
            for line in infile:
                if line.startswith(("ATOM", "HETATM")):
                    # Extract new chain ID and new residue number
                    new_chain = line[21]
                    new_number = line[22:26].strip()
                    original_chain, old_number = residue_map[
                        (new_chain, new_number)]

                    # Write restored line with original chain ID and residue number
                    new_line = (
                            line[:21] + original_chain + old_number.rjust(
                        4) + line[26:]
                    )
                    outfile.write(new_line)
                else:
                    # Write non-ATOM/HETATM lines as they are
                    outfile.write(line)

        # Print confirmation message
        # print(f"Restored PDB file saved as: {original_pdb}")

# =============================================================================
#
# =============================================================================
# import mdtraj as md
# import prody as prd
#
# load topology and trajectory
# topo = '/home/rglez/RoyHub/compass/data/MDs/nucleosome_full_2c/1kx5_dry.pdb'
# traj = '/home/rglez/RoyHub/compass/data/MDs/nucleosome_full_2c/nuc-prot-trim.dcd'
# out_dir = '/home/rglez/RoyHub/compass/data/outputs/nucleosome_full_2c'
