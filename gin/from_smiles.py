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
# from gin.molecule import *

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
    '\(',
    '\)',
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
N_ATOM_COUNTER_STR = '|'.join(
    TOPOLOGY_MARKS + [
    'l',
    'r',
    '\[',
    '\]',
    '@'
])



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

    NOTE: to speed things up, this is the minimalistic function to
          parse a small molecule, with no flags and assertions whatsoever.
          we assume the validity of the smiles string.

    Parameters
    ----------
    smiles : str,
        smiles representation of a molecule.

    Returns
    -------
    molecule : molecule.Molecule object.
    """
    with tf.init_scope(): # register the variables
        # it is still eager environment here
        # we need to get:
        #     - number of atoms
        #     - number of brackets

        # get the total length
        length = tf.strings.length(smiles)

        # get the number of atoms
        n_atoms = tf.strings.length(
            tf.strings.regex_replace(
                smiles,
                N_ATOM_COUNTER_STR,
                ''))

        # initialize the adjacency map
        # the adjaceny map, by default,
        # (a.k.a. where no topology characters presented,)
        # should be an upper triangular matrix $A$,
        # with
        #
        # $$
        # A_{ij} = \begin{cases}
        # 1, i = j + 1;
        # 0, \mathtt{elsewhere}.
        # \end{cases}
        # $$
        #
        # for speed,
        # we achieve this by
        # deleting the first column and the last row
        # of an identity matrix with one more column & row

        adjacency_map = tf.Variable(
            tf.eye(
                n_atoms + 1,
                dtype=tf.float32)[1:, :-1]) # to enable aromatic bonds

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

    # get rid of all the topology chrs
    # in order to get the atoms
    smiles_atoms_only = tf.strings.regex_replace(
        smiles,
        ALL_TOPOLOGY_REGEX_STR,
        '')

    smiles_aromatic = tf.strings.regex_replace(
        smiles,
        'c|n|o|p',
        'A')

    # =================================
    # translate atoms notations to idxs
    # =================================
    # TODO: further optimize this bit
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

    # split it in order to convert to tf.Tensor with dtype=tf.int64
    atoms = tf.string_split(
        [smiles_atoms_only],
        '').values # NOTE: add values here because this is suppose to be sparse

    atoms = tf.strings.to_number(
        atoms,
        tf.int64) # NOTE: int64 to be compatible

    # ==============================
    # handle the topology characters
    # ==============================

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
        smiles_topology_only,
        topology_idxs)

    # map the topology idxs onto the atoms
    topology_idxs = topology_idxs\
        - tf.range(
            tf.cast(
                tf.shape(topology_idxs)[0],
                dtype=tf.int64),
            dtype=tf.int64)\
        - tf.constant(
            1,
            dtype=tf.int64)

    # ===========
    # bond orders
    # ===========
    #
    # NOTE: do this first allow us to tolerate the rings connected
    #       to multiple bonds
    # double bonds and triple bonds
    # modify the bond order to two or three
    double_bond_idxs = tf.reshape(
        tf.where(
            tf.equal(
                topology_chrs,
                '=')),
        [-1])

    triple_bond_idxs = tf.reshape(
        tf.where(
            tf.equal(
                topology_chrs,
                '#')),
        [-1])

    # update double bonds
    adjacency_map.scatter_nd_update(
        tf.transpose(
            tf.concat(
                [
                    tf.expand_dims(
                        double_bond_idxs,
                        0),
                    tf.expand_dims(
                        double_bond_idxs + 1,
                        0)
                ],
                axis=0)),
        tf.ones_like(
            double_bond_idxs,
            dtype=tf.float32) * 2)

    # update triple bonds
    adjacency_map.scatter_nd_update(
        tf.transpose(
            tf.concat(
                [
                    tf.expand_dims(
                        triple_bond_idxs,
                        0),
                tf.expand_dims(
                    triple_bond_idxs + 1,
                    0)
                ],
                axis=0)),
        tf.ones_like(
            triple_bond_idxs,
            dtype=tf.float32) * 3)

    # ========
    # branches
    # ========
    # side chains are marked with brackets.
    # two things happen when a pair of brackets is present
    # - the connection between the right bracket and the atom right to it
    #   is dropped
    # - a bond is formed between the right bracket and the atom left to the
    #   left bracket

    # init a queue
    bracket_queue = tf.constant([], dtype=tf.int64)

    # get a large matrix to record the pairs
    bracket_pairs = tf.constant(
        [[0, 0]],
        tf.int64)

    first_right = tf.constant(True)
    def if_left(idx, bracket_queue, bracket_pairs):
        # if a certain position
        # it is a left bracket
        # put that bracket in the queue
        # NOTE: queue is not supported in graph
        # bracket_queue.enqueue(idx)
        bracket_queue = tf.concat(
            [
                bracket_queue,
                [idx]
            ],
            axis=0)

        return idx, bracket_queue, bracket_pairs

    def if_right(idx, bracket_queue, bracket_pairs,
            topology_idxs=topology_idxs,
            ):
        # if at a certain position
        # it is a right bracket
        # we need to do the following things
        #   - get a left bracket out of the queue
        #   - modify the adjaceny matrix
        right_idx = topology_idxs[idx]
        left_idx = topology_idxs[bracket_queue[-1]]
        bracket_queue = bracket_queue[:-1]
        bracket_pairs = tf.concat(
            [
                bracket_pairs,
                tf.expand_dims(
                    [left_idx, right_idx],
                    axis=0)
            ],
            axis=0)

        return idx, bracket_queue, bracket_pairs

    def loop_body(idx, bracket_queue, bracket_pairs,
            topology_chrs=topology_chrs):
        # get the flag
        chr_flag = topology_chrs[idx]

        # if left
        idx, bracket_queue, bracket_pairs = tf.cond(
            tf.equal(
                chr_flag,
                '('),

            # if chr_flag == '('
            lambda: if_left(idx, bracket_queue, bracket_pairs),

            # else:
            lambda: (idx, bracket_queue, bracket_pairs))

        # if right
        idx, bracket_queue, bracket_pairs = tf.cond(
            tf.equal(
                chr_flag,
                ')'),

            # if chr_flag == '('
            lambda: if_right(idx, bracket_queue, bracket_pairs),

            # else:
            lambda: (idx, bracket_queue, bracket_pairs))

        # increment
        return idx+1, bracket_queue, bracket_pairs

    # get rid of the first row
    bracket_pairs = bracket_pairs[1:, ]

    # while loop
    max_iter = tf.shape(topology_idxs)[0]
    idx = tf.constant(0)
    _, _, bracket_pairs = tf.while_loop(
        # condition
        lambda idx, _0, _1: tf.less(idx, max_iter),

        # body
        lambda idx, bracket_queue, bracket_pairs: \
            loop_body(idx, bracket_queue, bracket_pairs),

        # var
        [idx, bracket_queue, bracket_pairs],

        shape_invariants = [
            idx.get_shape(),
            tf.TensorShape((None, )),
            tf.TensorShape((None, 2))
        ])


    left_bracket_idxs = bracket_pairs[:, 0]
    right_bracket_idxs = bracket_pairs[:, 1]

    # NOTE: it is impossible for a bracket to be at the end of
    #       the smiles string
    current_bond_idxs = tf.transpose(
        tf.concat(
            [
                tf.expand_dims(
                    right_bracket_idxs,
                    0),
                tf.expand_dims(
                    right_bracket_idxs + 1,
                    0)
            ],
            axis=0))

    current_bond_order = tf.gather_nd(
        adjacency_map,
        current_bond_idxs)

    new_bond_idxs = tf.transpose(
        tf.concat(
            [
                tf.expand_dims(
                    left_bracket_idxs,
                    0),
                tf.expand_dims(
                    right_bracket_idxs + 1,
                    0)
            ],
            axis=0))

    # drop the connection between right bracket and the atom right to it
    adjacency_map.scatter_nd_update(
        current_bond_idxs,
        tf.zeros_like(current_bond_order)
    )

    # connect the atom right of the right bracket to the atom left of
    # the left bracket
    adjacency_map.scatter_nd_update(
        new_bond_idxs,
        tf.ones_like(current_bond_order))

    # =====
    # rings
    # =====
    #
    # NOTE: to speed things up,
    #       we allow a maximum of 5 rings in a molecule,
    #       this is already enough for the whole ZINC database
    #
    # when a digit appears in the SMILES string
    # a bond (sinlge bond) is formed between
    # two labeled atoms

    # while loop
    idx = 0
    max_iter = 5
    connection_chrs = tf.constant([
        '1',
        '2',
        '3',
        '4',
        '5',
    ], dtype=tf.string)

    def loop_body(idx):
        # get the connection character
        connection_chr = connection_chrs[idx]

        # search for this in the SMILES string
        # NOTE: we assume that there are two of them,
        #       again, unapologetically, we don't put any assertion here
        connection_idxs = tf.reshape(
            tf.where(
                tf.equal(
                    topology_chrs,
                    connection_chr)),
            [-1])

        tf.cond(
            tf.greater(
                tf.shape(connection_idxs)[0],
                0),

            # if connection_idxs.shape[0] > 0
            # NOTE: the bond closing the ring is always single bond
            lambda: adjacency_map[
                connection_idxs[0], connection_idxs[1]].assign(
                    tf.constant(1, dtype=tf.float32)),

            # else:
            lambda: tf.constant(0, tf.float32))


        # increment
        return idx + 1

    # NOTE:
    #    tf.while_loop sometimes throws bugs here

    for idx in range(max_iter):
         loop_body(idx)

    # ===========
    # aromaticity
    # ===========
    #
    # hard code aromaticity:
    #   where the aromatic atoms are coded as lower case letters
    # update the aromatic bond orders to half bonds
    smiles_aromatic = tf.string_split(
        [smiles_aromatic],
        '').values

    aromatic_idxs = tf.reshape(
        tf.where(
            tf.equal(
                smiles_aromatic,
                'A')),
        [-1])


    # we change the bond order of the aromatic atoms
    # to 1.5
    # now all the aromatic atoms should still be adjacent to each other
    current_bond_idxs = tf.transpose(
        tf.concat(
            [
                tf.expand_dims(
                    aromatic_idxs,
                    0),
                tf.expand_dims(
                    aromatic_idxs+1,
                    0)
            ],
            axis=0))

    current_bond_order = tf.gather(
        adjacency_map,
        current_bond_idxs)

    modified_bond_order = tf.where( # if
        tf.greater_equal(
            current_bond_order,
            tf.constant(1, dtype=tf.float32)),

        # if current_bond_order >= 1:
        current_bond_order - 0.5,

        # else:
        # if there is no bond right now, then we don't do anything
        current_bond_order)



    return atoms, adjacency_map
