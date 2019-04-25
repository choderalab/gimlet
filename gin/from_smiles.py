"""
from_smiles.py

Operations for read smiles string.

MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center
and Authors

Authors:
Yuanqing Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# =============================================================================
# imports
# =============================================================================
# dependencies
import tensorflow as tf
tf.enable_eager_execution()

# packages
from gin.molecule import *

# =============================================================================
# CONSTANTS
# =============================================================================
"""
ATOMS = [
    # common stuff,
    # w/ or w/o aromacity
    'C',
    'c',
    'N',
    'n',
    'O',
    'o',
    'S',
    's',

    # slightly uncommon
    'B',
    'b',
    'P',
    'p',

    # halogens
    'F',
    'Cl',
    'Br',

    # chiral stuff
    '[C@H]',
    '[C@@H]',

    # stuff with charges
    '[F-]',
    '[Cl-]',
    '[Br-]',
    '[Na+]',
    '[K+]',
    '[OH-]',
    '[NH4+]',
    '[H+]',

    # NOTE:
    # due to the consideration of speed, we don't support more exotic atoms
]
"""

ORGANIC_ATOMS = [
    'C',
    'N',
    'O',
    'S',
    'P',
    'F',
    'R', # Br
    'L', # Cl
]

N_ORGANIC_ATOMS = len(ORGANIC_ATOMS)
ORGANIC_ATOMS_IDXS = list(range(N_ORGANIC_ATOMS))

TOPOLOGY_MARKS = [
    '=',
    '#',
    '(',
    ')',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
]

ALL_TOPOLOGY_REGEX_STR = '|'.join(TOPOLOGY_MARKS)
ALL_ORGANIC_ATOMS_STR = '|'.join(ORGANIC_ATOMS)

# =============================================================================
# utility functions
# =============================================================================
@tf.contrib.eager.defun
def smiles_to_organic_topological_molecule(smiles):
    """ Decode a SMILES string to a molecule object.

    Organic atoms:
    [C, N, O, S, P, F, Cl, Br]

    Corresponding indices:
    [0, 1, 2, 3, 4, 5, 6, 7]

    Parameters
    ----------
    smiles : str,
        smiles representation of a molecule.

    Returns
    -------
    molecule : molecule.Molecule object.
    """
    # initialize a molecule
    mol = molecule.Molecule()

    # ==========================
    # get rid of the longer bits
    # ==========================
    # 'Br' to 'R'
    smiles = tf.strings.regex_replace(
        smiles, 'Br', 'R')

    # 'Cl' to 'L'
    smiles = tf.strings.regex_replace(
        smiles, 'Cl', 'L')

    # remove chiral stuff
    smiles = tf.strings.regex_replace(
        smiles, '\[C@H\]|\[C@@H\]', 'C')

    smiles_atoms_only = tf.strings.regex_replace(
        smiles,
        ALL_TOPOLOGY_REGEX_STR,
        '')

    # =================================
    # translate atoms notations to idxs
    # =================================
    # carbon
    # aromatic or not
    smiles_atoms_only = tf.strings.regex_replace(
        smiles_atoms_only,
        'C|c',
        '0')

    # nitrogen
    # aromatic or not
    smiles_atoms_only = tf.strings.regex_replace(
        smiles_atoms_only,
        'N|n',
        '1')

    # oxygen
    # aromatic or not
    smiles_atoms_only = tf.strings.regex_replace(
        smiles_atoms_only,
        'O|o',
        '2')

    # sulfur
    # aromatic or not
    smiles_atoms_only = tf.strings.regex_replace(
        smiles_atoms_only,
        'S|s',
        '3')

    # phosphorus
    # aromatic or not
    smiles_atoms_only = tf.strings.regex_replace(
        smiles_atoms_only,
        'P|p', # NOTE: although not common, adding this doesn't hurt speed
        '4')

    # fluorine
    smiles_atoms_only = tf.strings.regex_replace(
        smiles_atoms_only,
        'F',
        '5')

    # chlorine
    smiles_atoms_only = tf.strings.regex_replace(
        smiles_atoms_only,
        'L',
        '6')

    # bromine
    smiles_atoms_only = tf.strings.regex_replace(
        smiles_atoms_only,
        'R',
        '7')

    # get it into int
    atoms = tf.string_split(
        [smiles_atoms_only],
        '').values

    atoms = tf.strings.to_number(
        atoms,
        tf.int64)

    # ===================
    # handle the topology
    # ===================
    # initialize the adjacency map
    adjacency_map = tf.eye()

    smiles_topology_only = tf.strings.regex_replace(
        smiles,
        ALL_ORGANIC_ATOMS_STR,
        '0')

    smiles_topology_only = tf.string_split(
        [smiles_topology_only],
        '').values

    topology_idxs = tf.reshape(
        tf.where(
            tf.not_equal(
                smiles_topology_only,
                '0')),
        [-1])

    topology_chrs = tf.gather(
        topology_idxs,
        smiles_topology_only)

    # map the topology idxs onto the atoms
    topology_idxs = topology_idxs\
        - tf.range(topology_idxs.shape[0], dtype=tf.int64)
