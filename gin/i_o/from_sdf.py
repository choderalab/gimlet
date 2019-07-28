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
# imports
# =============================================================================
TRANSLATION = {
    b'C': 0,
    b'N': 1,
    b'O': 2,
    b'S': 3,
    b'P': 4,
    b'F': 5,
    b'Cl': 6,
    b'Br': 7,
    b'I': 8,
    b'H': 9
}

# =============================================================================
# utility functions
# =============================================================================
def to_ds(file_path, has_charge=False):
    """

    Organic atoms:
    [C, N, O, S, P, F, Cl, Br, I, H]

    Corresponding indices:
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    # read the file
    text = tf.io.read_file(
        file_path)

    lines = tf.strings.split(
        tf.expand_dims(text, 0),
        '\n').values

    lines = tf.boolean_mask(
        lines,

        tf.logical_not(
            tf.strings.regex_full_match(
                lines,
                '.*RAD.*')))

    # get the starts and the ends
    starts = tf.strings.regex_full_match(
        lines,
        '.*V2000.*')

    ends = tf.strings.regex_full_match(
        lines,
        '.*END.*')

    starts_idxs = tf.boolean_mask(
        tf.range(
            tf.cast(
                starts.shape[0],
                tf.int64),
            dtype=tf.int64),
        starts)

    ends_idxs = tf.boolean_mask(
        tf.range(
            tf.cast(
                ends.shape[0],
                tf.int64),
            dtype=tf.int64),
        ends) - tf.constant(1, dtype=tf.int64)

    mol_chunks = tf.concat(
        [
            tf.expand_dims(
                starts_idxs,
                1),
            tf.expand_dims(
                ends_idxs,
                1)
        ],
        axis=1)

    def read_one_mol(idx, has_charge=has_charge, lines=lines):
        mol_chunk = mol_chunks[idx]
        start = mol_chunk[0]
        end = mol_chunk[1]

        # process the head line to get the number of the atoms and bonds
        head = lines[start]
        head = tf.strings.split(tf.expand_dims(head, 0), ' ').values
        head = tf.boolean_mask(
            head,
            tf.logical_not(
                tf.equal(
                    head, '')))

        n_atoms = tf.strings.to_number(
            head[0],
            tf.int64)

        n_bonds = tf.strings.to_number(
            head[1],
            tf.int64)

        # get the lines for atoms and bonds
        atoms_lines = tf.slice(
            lines,
            tf.expand_dims(start+1, 0),
            tf.expand_dims(n_atoms, 0))

        bonds_lines = tf.slice(
            lines,
            tf.expand_dims(start+n_atoms+1, 0),
            tf.expand_dims(n_bonds, 0))

        charges = tf.zeros((n_atoms, ), dtype=tf.float32)

        if has_charge:
            # check if there is any total charge annotated in the system
            has_total_charge = tf.reduce_any(
                tf.strings.regex_full_match(
                    tf.slice(
                        lines,
                        tf.expand_dims(start+1, 0),
                        tf.expand_dims(end-start, 0)),
                    '.*CHG.*'))

            if has_total_charge:
                charge_lines = tf.slice(
                    lines,
                    tf.expand_dims(start+n_atoms+n_bonds+2, 0),
                    tf.expand_dims(2*n_atoms, 0))

            else:
                charge_lines = tf.slice(
                    lines,
                    tf.expand_dims(start+n_atoms+n_bonds+1, 0),
                    tf.expand_dims(2*n_atoms, 0))

            charge_lines = tf.gather(
                charge_lines,
                tf.add(
                    tf.math.multiply(
                        tf.constant(2, dtype=tf.int64),
                        tf.range(
                            n_atoms,
                            dtype=tf.int64)),
                    tf.constant(1, dtype=tf.int64)))

            charges = tf.strings.to_number(
                tf.reshape(
                    charge_lines,
                    [-1]),
                tf.float32)

        # process atom lines
        atoms_lines = tf.strings.split(atoms_lines, ' ').values
        atoms_lines = tf.reshape(
            tf.boolean_mask(
                atoms_lines,
                tf.logical_not(
                    tf.equal(
                        atoms_lines,
                        ''))),
            [n_atoms, -1])

        coordinates = tf.strings.to_number(
            atoms_lines[:, :3],
            tf.float32)


        atoms = tf.cast(
            tf.map_fn(
                lambda x: TRANSLATION[x.numpy()],
                atoms_lines[:, 3],
                tf.int32),
            tf.int64)

        # process bond lines
        bonds_lines = tf.strings.split(bonds_lines, ' ').values
        bonds_lines = tf.reshape(
            tf.boolean_mask(
                bonds_lines,
                tf.logical_not(
                    tf.equal(
                        bonds_lines,
                        ''))),
            [n_bonds, -1])

        bond_idxs = tf.strings.to_number(
            bonds_lines[:, :2],
            tf.int64) - 1

        bond_orders = tf.strings.to_number(
            bonds_lines[:, 2],
            tf.float32)

        adjacency_map = tf.Variable(
            tf.zeros((n_atoms, n_atoms),
            dtype=tf.float32))

        adjacency_map = adjacency_map.scatter_nd_update(
            bond_idxs,
            bond_orders)

        adjacency_map = adjacency_map.read_value()

        return atoms, adjacency_map, coordinates, charges

    ds = tf.data.Dataset.from_tensor_slices(
        tf.range(
            tf.cast(
                mol_chunks.shape[0],
                tf.int64),
            dtype=tf.int64))

    ds = ds.map(lambda x: tf.py_function(
        read_one_mol,
        [x],
        [tf.int64, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=4*N_CPUS)

    return ds
