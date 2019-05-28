"""
MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center,
Weill Cornell Medical College, Nicea Research, and Authors

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
import tensorflow as tf
# tf.enable_eager_execution()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
N_CPUS = multiprocessing.cpu_count()

# =============================================================================
# constants
# =============================================================================

TRANSLATION = {
    0: 'C',
    1: 'N',
    2: 'O',
    3: 'S',
    4: 'P',
    5: 'F',
    6: 'Cl',
    7: 'Br',
    8: 'I',
    9: 'H'
}

idx = 0

# =============================================================================
# module functions
# =============================================================================
def write_one_sdf(*mol):
    """ Write molecule to SDF file.

    """
    global idx
    mol = mol[0]

    lines = tf.constant(
        [
            [str(idx)],
            ['   gin-%s' % tf.timestamp().numpy()],
            [' ']
        ],
        dtype=tf.string)


    atoms = mol[0]
    adjacency_map = mol[1]
    n_atoms = tf.shape(atoms)[0]
    # here we toloerate the situation where the user feed the system
    # a topology-only molecule

    if len(mol) == 2:
        coordinates = tf.zeros((n_atoms, 3), dtype=tf.float32)

    else:
        coordinates = mol[2]

    atom_chunk = tf.concat(
        [
            tf.strings.as_string(
                coordinates,
                precision=4,
                scientific=False,
                width=9,
                fill=' '),
            tf.expand_dims(
                tf.map_fn(
                    lambda x: TRANSLATION[int(x.numpy())],
                    atoms,
                    tf.string),
                1)
        ],
        axis=1)

    # find the positions at which there is a bond
    is_bond = tf.greater(
        adjacency_map,
        tf.constant(0, dtype=tf.float32))

    # dirty stuff to get the bond indices to update
    all_idxs_x, all_idxs_y = tf.meshgrid(
        tf.range(tf.cast(n_atoms, tf.int64), dtype=tf.int64),
        tf.range(tf.cast(n_atoms, tf.int64), dtype=tf.int64))

    all_idxs_stack = tf.stack(
        [
            all_idxs_y,
            all_idxs_x
        ],
        axis=2)

    # get the bond indices
    bond_idxs = tf.boolean_mask(
        all_idxs_stack,
        is_bond)

    # TODO: right now this is not correct, we cannot simply
    # round up the bond orders
    bond_orders = tf.cast(
        tf.math.round(
            tf.gather_nd(
                adjacency_map,
                bond_idxs)),
        tf.int64)

    bond_chunk = tf.concat(
        [
            tf.strings.as_string(
                bond_idxs + 1,
                width=3,
                fill=' '),
            tf.strings.as_string(
                tf.expand_dims(
                    bond_orders,
                    1),
                width=3,
                fill=' ')
        ],
        axis=1)

    n_bonds = tf.shape(bond_idxs)[0]

    lines = tf.concat(
        [
            lines,
            [[' %s %s  0     0  0  0  0  0  0999 V2000' %\
            (
                n_atoms.numpy(),
                n_bonds.numpy()
            )]],
            tf.strings.reduce_join(
                tf.concat(
                    [
                        atom_chunk,
                        tf.tile(
                            [['  0  0  0  0  0  0  0  0  0  0  0  0']],
                            [n_atoms, 1]),
                    ],
                    axis=1),
                axis=1,
                separator=' ',
                keepdims=True),

            tf.strings.reduce_join(
                tf.concat(
                    [
                        bond_chunk,
                        tf.tile(
                            [['  0  0  0  0']],
                            [n_bonds, 1])
                    ],
                    axis=1),
                axis=1,
                separator='',
                keepdims=True),
            [['M  END']],
            [['$$$$']]
        ],
        axis=0)

    idx += 1
    return lines

def write_sdf(mols, file_path):
    """ Write multiple molecules in one data object.

    """

    lines = tf.concat(
        [
            write_one_sdf(mol) for mol in mols
        ],
        axis=0)

    tf.io.write_file(
        file_path,
        tf.reshape(
            tf.strings.reduce_join(
                lines,
                axis=0,
                separator='\n'),
            []))
