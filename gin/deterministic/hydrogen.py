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
import gin
from gin.deterministic import typing

# =============================================================================
# utility functions
# =============================================================================
def add_hydrogen(mol):
    """ Add hydrogen to the molecule.

    Parameters
    ----------
    mol : gin.molecule.Molecule object
    """
    # get the current atoms and adjacency map
    atoms = mol.atoms
    adjacency_map = mol.adjacency_map
    adjacency_map_full = tf.transpose(adjacency_map) + adjacency_map

    # type the atoms
    atom_types = typing.Typing(mol)

    # calculate the number of hydrogens added to heavy atoms
    # the heavy atoms with one hydrogen
    has_one = tf.reduce_any(
        tf.reshape(
            tf.concat(
                [
                    # sp3 carbon
                    tf.logical_and(
                        atom_types.is_carbon,
                        tf.logical_and(
                            atom_types.is_sp3,
                            atom_types.is_connected_to_3_heavy)),

                    # sp2 carbon
                    tf.logical_and(
                        atom_types.is_carbon,
                        tf.logical_and(
                            atom_types.is_sp2,
                            atom_types.is_connected_to_2_heavy)),

                    # sp1 carbon
                    tf.logical_and(
                        atom_types.is_carbon,
                        tf.logical_and(
                            atom_types.is_sp1,
                            atom_types.is_connected_to_1_heavy)),

                    # sp3 nitrogen or phosphorus
                    tf.logical_and(
                        tf.logical_or(
                            atom_types.is_nitrogen,
                            atom_types.is_phosphorus),
                        tf.logical_and(
                            atom_types.is_sp3,
                            atom_types.is_connected_to_2_heavy)),

                    # sp2 nitrogen or phosphorus
                    tf.logical_and(
                        tf.logical_or(
                            atom_types.is_nitrogen,
                            atom_types.is_phosphorus),
                        tf.logical_and(
                            atom_types.is_sp2,
                            atom_types.is_connected_to_1_heavy)),

                    # sp3 oxygen or sulfur
                    tf.logical_and(
                        tf.logical_or(
                            atom_types.is_oxygen,
                            atom_types.is_sulfur),
                        atom_types.is_sp3)

                ],
                axis=0),
            [None, tf.shape(atoms)[0]]),
        axis=0)

    one_idxs = tf.boolean_mask(
        tf.range(
            tf.shape(atoms[0]),
            dtype=tf.int64),
        has_one)

    # the heavy atoms with two hydrogens
    has_two = tf.reduce_any(
        tf.reshape(
            tf.concat(
                [
                    # sp3 carbon
                    tf.logical_and(
                        atom_types.is_carbon,
                        tf.logical_and(
                            atom_types.is_sp3,
                            atom_types.is_connected_to_2_heavy)),

                    # sp2 carbon
                    tf.logical_and(
                        atom_types.is_carbon,
                        tf.logical_and(
                            atom_types.is_sp2,
                            atom_types.is_connected_to_1_heavy)),

                    # sp3 nitrogen or phosphorus
                    tf.logical_and(
                        tf.logical_or(
                            atom_types.is_nitrogen,
                            atom_types.is_phosphorus),
                        tf.logical_and(
                            atom_types.is_sp3,
                            atom_types.is_connected_to_1_heavy))

                ]),
            [None, tf.shape(atoms)[0]]),
        axis=0)

    two_idxs = tf.boolean_mask(
        tf.range(
            tf.shape(atoms[0]),
            dtype=tf.int64),
        has_two)

    # the heavy atoms with three hydrogens
    has_three = tf.logical_and(
        atom_types.is_carbon,
        tf.logical_and(
            atom_types.is_sp3,
            atom_types.is_connected_to_1_heavy))

    three_idxs = tf.boolean_mask(
        tf.range(
            tf.shape(atoms[0]),
            dtype=tf.int64))

    # build the adjacency map blocks to be appended to the map
    # TODO: figure out a way to do this more efficiently

    # init
    hydrogen_block = tf.transpose(
        tf.expand_dims(
            (tf.ones((tf.shape(atoms)[0], ),
                dtype=tf.float32)))) \
        * tf.constant(-1, dypte=tf.float32)

    def one_body(idx, hydrogen_block):
        # grab idx to update
        one_idx = one_idxs[idx]

        # get the new line to append
        new_line = tf.transpose(
            tf.expand_dims(
                tf.where(
                    tf.equal(
                        tf.range(
                            tf.shape(atoms[0]),
                            dtype=tf.float32),
                        one_idx),

                    # if is one_idx
                    tf.constant(1, dtype=tf.float32),

                    # else:
                    tf.constant(0, dtype=tf.float32)),
                axis=0))

        # append it to the matrix
        hydrogen_block = tf.concat(
            [
                hydrogen_block,
                new_line # do this once
            ],
            axis=1)

        return idx + 1, hydrogen_block

    def two_body(idx, hydrogen_block):
        # grab idx to update
        two_idx = two_idxs[idx]

        # get the new line to append
        new_line = tf.transpose(
            tf.expand_dims(
                tf.where(
                    tf.equal(
                        tf.range(
                            tf.shape(atoms[0]),
                            dtype=tf.float32),
                        one_idx),

                    # if is one_idx
                    tf.constant(1, dtype=tf.float32),

                    # else:
                    tf.constant(0, dtype=tf.float32)),
                axis=0))

        # append it to the matrix
        hydrogen_block = tf.concat(
            [
                hydrogen_block,
                new_line, # do this two times
                new_line
            ],
            axis=1)

        return idx + 1, hydrogen_block

    def three_body(idx, hydrogen_block):
        # grab idx to update
        two_idx = two_idxs[idx]

        # get the new line to append
        new_line = tf.transpose(
            tf.expand_dims(
                tf.where(
                    tf.equal(
                        tf.range(
                            tf.shape(atoms[0]),
                            dtype=tf.float32),
                        one_idx),

                    # if is one_idx
                    tf.constant(1, dtype=tf.float32),

                    # else:
                    tf.constant(0, dtype=tf.float32)),
                axis=0))

        # append it to the matrix
        hydrogen_block = tf.concat(
            [
                hydrogen_block,
                new_line, # do this three times
                new_line,
                new_line
            ],
            axis=1)

        return idx + 1, hydrogen_block

    # execute all the functions
    # one
    idx = tf.constant(0, dtype=tf.int64)
    idx, hydrogen_block = tf.while_loop(
        # condition
        lambda idx, _: tf.less(
            idx,
            tf.count_nonzero(has_one)),

        # body
        one_body,

        # var
        [idx, hydrogen_block],

        # invar
        shape_invariants=[
            idx.get_shape(),
            tf.TensorShape([
                tf.shape(atoms[0]),
                None])])

    # execute all the functions
    # two
    idx = tf.constant(0, dtype=tf.int64)
    idx, hydrogen_block = tf.while_loop(
        # condition
        lambda idx, _: tf.less(
            idx,
            tf.count_nonzero(has_one)),

        # body
        one_body,

        # var
        [idx, hydrogen_block],

        # invar
        shape_invariants=[
            idx.get_shape(),
            tf.TensorShape([
                tf.shape(atoms[0]),
                None])])

    # execute all the functions
    idx = tf.constant(0, dtype=tf.int64)
    idx, hydrogen_block = tf.while_loop(
        # condition
        lambda idx, _: tf.less(
            idx,
            tf.count_nonzero(has_one)),

        # body
        one_body,

        # var
        [idx, hydrogen_block],

        # invar
        shape_invariants=[
            idx.get_shape(),
            tf.TensorShape([
                tf.shape(atoms[0]),
                None])])

    # get rid of the first block
    hydrogen_block = hydrogen_block[:, :1]

    # get the total number of hydrogen
    n_hydrogen = tf.shape(hydrogen_block)[1]

    # modify the attributes of molecules
    atoms = tf.concat(
        [
            atom,
            8 * tf.ones((n_hydrogen, ), dtype=tf.int64)
        ],
        axis=0)

    adjacency_map = tf.concat(
        [
            adjacency_map,
            hydrogen_block
        ],
        axis=0)

    return gin.molecule.Molecule(
        atoms,
        adjacency_map)
