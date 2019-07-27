"""
conformer.py

Generate conformers for MD calculation using distance geometry.
Code adopted from:
https://github.com/openbabel/openbabel/blob/master/src/distgeom.cpp


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
import math
# import tensorflow_probability as tfp

# =============================================================================
# constants
# =============================================================================
BOND_ENERGY_THRS = 0.5
DIST12_TOL = tf.constant(0.01, dtype=tf.float32)
DIST13_TOL = tf.constant(0.03, dtype=tf.float32)
DIST14_TOL = tf.constant(0.05, dtype=tf.float32)
DIST15_TOL = tf.constant(0.07, dtype=tf.float32)

# =============================================================================
# utility functions
# =============================================================================
# @tf.function
def set_12_bounds(upper, lower):
    """ Floyd algorithm, as was implemented here:
    doi: 10.1002/0470845015.cda018

    for k in range(n_atoms):
        for i in range(n_atoms - 1):
            for j in (i + 1, n_atoms):
                if upper[i, j] > upper[i, k] + upper[k, j]:
                    upper[i, j].assign(upper[i, k] + upper[k, j])
                if lower[i, j] < lower[i, k] - upper[k, j]:
                    lower[i, j].assign(lower[i, k] - upper[k, j])
                elif lower[i, j] < lower[j, k] - upper[k, i]:
                    lower[i, j].assign(lower[j, k] - upper[k, i])

    """

    # NOTE: here TensorFlow will optimize the graph
    n_atoms = upper.shape[0]

    def inner_loop(upper, lower, k, i, j):
        upper[i, j].assign(
            tf.cond(
                tf.greater(
                    upper[i, j],
                    upper[i, k] + upper[k, j]),

                lambda: (upper[i, k] + upper[k, j] + DIST12_TOL),

                lambda: upper[i, j]))

        upper[j, i].assign(upper[i, j])

        lower[i, j].assign(
            tf.cond(
                tf.less(
                    lower[i, j],
                    lower[i, k] - upper[k, j]),

                lambda: lower[i, k] - upper[k, j] - DIST12_TOL,

                lambda: lower[i, j]))

        lower[j, i].assign(lower[i, j])

        lower[i, j].assign(
            tf.cond(
                tf.less(
                    lower[i, j],
                    lower[j, k] - upper[k, i]),

                lambda: lower[j, k] - upper[k, i] - DIST12_TOL,

                lambda: lower[i, j]))

        lower[j, i].assign(lower[i, j])

        return upper, lower, k, i, j+1

    def middle_loop(upper, lower, k, i):
        j = i + 1
        tf.while_loop(
            lambda upper, lower, k, i, j: tf.less(j, n_atoms),
            inner_loop,
            [upper, lower, k, i, j])
        return upper, lower, k, i+1

    def outer_loop(upper, lower, k):
        i = 0
        tf.while_loop(
            lambda upper, lower, k, i: tf.less(i, n_atoms - 1),
            middle_loop,
            [upper, lower, k, i])
        return upper, lower, k+1

    k = 0
    upper, lower, k = tf.while_loop(
        lambda upper, lower, k: tf.less(k, n_atoms),
        outer_loop,
        [upper, lower, k])

    return upper, lower

def set_13_bounds(upper, lower, adjacency_map,
        typing_assignment, forcefield):
    """ Calculate the 1-3 bounds based on cosine relationships.

    """

    n_atoms = tf.shape(upper, tf.int64)[0]

    # get the full adjacency_map
    adjacency_map_full = tf.transpose(adjacency_map) + adjacency_map

    # init the angles idxs to be all negative ones
    angle_idxs = tf.constant([[-1, -1, -1]], dtype=tf.int64)

    def process_one_atom_if_there_is_angle(idx, angle_idxs,
            adjacency_map_full=adjacency_map_full):

        # get all the connection indices
        connection_idxs = tf.where(
            tf.greater(
                adjacency_map_full[idx, :],
                tf.constant(0, dtype=tf.float32)))

        # get the number of connections
        n_connections = tf.shape(connection_idxs)[0]

        # get the combinations from these connection indices
        connection_combinations = tf.gather_nd(
            tf.stack(
                tf.meshgrid(
                    connection_idxs,
                    connection_idxs),
                axis=2),
            tf.where(
                tf.greater(
                    tf.linalg.band_part(
                        tf.ones(
                            (
                                n_connections,
                                n_connections
                            ),
                            dtype=tf.int64),
                        0, -1),
                    tf.constant(0, dtype=tf.int64))))

        connection_combinations = tf.boolean_mask(
            connection_combinations,
            tf.greater(
                connection_combinations[:, 0] \
                 - connection_combinations[:, 1],
                tf.constant(0, dtype=tf.int64)))

        angle_idxs = tf.concat(
            [
                angle_idxs,
                tf.concat(
                    [
                        tf.expand_dims(
                            connection_combinations[:, 0],
                            1),
                        tf.expand_dims(
                            idx * tf.ones(
                                (tf.shape(connection_combinations)[0], ),
                                dtype=tf.int64),
                            1),
                        tf.expand_dims(
                            connection_combinations[:, 1],
                            1)
                    ],
                    axis=1)
            ],
            axis=0)

        return idx + 1, angle_idxs

    def process_one_atom(idx, angle_idxs,
            adjacency_map_full=adjacency_map_full):

        if tf.less(
            tf.math.count_nonzero(adjacency_map_full[idx, :]),
            tf.constant(1, dtype=tf.int64)):
            return idx+1, angle_idxs

        else:
            return process_one_atom_if_there_is_angle(idx, angle_idxs)

    idx = tf.constant(0, dtype=tf.int64)
    # use while loop to update the indices forming the angles
    idx, angle_idxs = tf.while_loop(
        # condition
        lambda idx, angle_idxs: tf.less(idx, n_atoms),

        process_one_atom,

        [idx, angle_idxs],

        shape_invariants=[
            idx.get_shape(),
            tf.TensorShape((None, 3))])

    # discard the first row
    angle_idxs = angle_idxs[1:, ]

    # discard the angles where there is a bond between atom1 and atom3
    # (n_angles, ) Boolean
    is_not_bond_13 = tf.equal(
        tf.gather_nd(
            adjacency_map_full,
            tf.concat(
                [
                    tf.expand_dims(angle_idxs[:, 0], 1),
                    tf.expand_dims(angle_idxs[:, 2], 1)
                ],
                axis=1)),
        tf.constant(0, dtype=tf.float32))

    angle_idxs = tf.boolean_mask(
        angle_idxs,
        is_not_bond_13)

    # get the optimal angles from the forcefield
    angle_specs = tf.map_fn(
        lambda angle: tf.convert_to_tensor(
                forcefield.get_angle(
                    int(
                        tf.gather(
                            typing_assignment, angle[0]).numpy()),
                    int(
                        tf.gather(
                            typing_assignment, angle[1]).numpy()),
                    int(
                        tf.gather(
                            typing_assignment, angle[2]).numpy()))),
        angle_idxs,
        dtype=tf.float32)

    optimal_angles = angle_specs[:, 0]

    # $$
    # BC = \sqrt{AB ^ 2 + AC ^ 2 - 2 * AB * BC * cos(A)}
    # $$
    upper = tf.tensor_scatter_nd_update(
        upper,
        tf.concat(
            [
                tf.expand_dims(angle_idxs[:, 0], 1),
                tf.expand_dims(angle_idxs[:, 2], 1)
            ],
            axis=1),
        tf.sqrt(
            tf.math.reduce_sum(
                [
                    tf.pow(
                        tf.gather_nd(
                            upper,
                            tf.concat(
                                [
                                    tf.expand_dims(angle_idxs[:, 0], 1),
                                    tf.expand_dims(angle_idxs[:, 1], 1)
                                ],
                                axis=1)) - DIST12_TOL,
                        2),

                    tf.pow(
                        tf.gather_nd(
                            upper,
                            tf.concat(
                                [
                                    tf.expand_dims(angle_idxs[:, 2], 1),
                                    tf.expand_dims(angle_idxs[:, 1], 1)
                                ],
                                axis=1)) - DIST12_TOL,
                        2),

                    tf.constant(-2, dtype=tf.float32) \
                        * (tf.gather_nd(
                            upper,
                            tf.concat(
                                [
                                    tf.expand_dims(angle_idxs[:, 0], 1),
                                    tf.expand_dims(angle_idxs[:, 1], 1)
                                ],
                                axis=1)) - DIST12_TOL) \
                        * (tf.gather_nd(
                            upper,
                            tf.concat(
                                [
                                    tf.expand_dims(angle_idxs[:, 2], 1),
                                    tf.expand_dims(angle_idxs[:, 1], 1)
                                ],
                                axis=1)) - DIST12_TOL) \
                        * tf.math.cos(optimal_angles)
                ],
                axis=0
            )) + DIST12_TOL)

    upper = tf.linalg.band_part(
        upper,
        0, -1) + \
        tf.transpose(
            tf.linalg.band_part(
                upper,
                0, -1))

    lower = tf.tensor_scatter_nd_update(
        lower,
        tf.concat(
            [
                tf.expand_dims(angle_idxs[:, 0], 1),
                tf.expand_dims(angle_idxs[:, 2], 1)
            ],
            axis=1),
        tf.sqrt(
            tf.math.reduce_sum(
                [
                    tf.pow(
                        tf.gather_nd(
                            lower,
                            tf.concat(
                                [
                                    tf.expand_dims(angle_idxs[:, 0], 1),
                                    tf.expand_dims(angle_idxs[:, 1], 1)
                                ],
                                axis=1)) + DIST12_TOL,
                        2),

                    tf.pow(
                        tf.gather_nd(
                            lower,
                            tf.concat(
                                [
                                    tf.expand_dims(angle_idxs[:, 2], 1),
                                    tf.expand_dims(angle_idxs[:, 1], 1)
                                ],
                                axis=1)) + DIST12_TOL,
                        2),

                    tf.constant(-2, dtype=tf.float32) \
                        * (tf.gather_nd(
                            lower,
                            tf.concat(
                                [
                                    tf.expand_dims(angle_idxs[:, 0], 1),
                                    tf.expand_dims(angle_idxs[:, 1], 1)
                                ],
                                axis=1)) + DIST12_TOL) \
                        * (tf.gather_nd(
                            lower,
                            tf.concat(
                                [
                                    tf.expand_dims(angle_idxs[:, 2], 1),
                                    tf.expand_dims(angle_idxs[:, 1], 1)
                                ],
                                axis=1)) + DIST12_TOL) \
                        * tf.math.cos(optimal_angles)
                ],
                axis=0
            )) - DIST12_TOL)

    lower = tf.linalg.band_part(
        lower,
        0, -1) + \
        tf.transpose(
            tf.linalg.band_part(
                lower,
                0, -1))

    return upper, lower

# @tf.function
def embed(distance_matrix):
    """ EMBED algorithm, as was implemented here:
    10.1002/0470845015.cda018
    """
    n_atoms = tf.shape(distance_matrix, tf.int64)[0]

    # $$
    # d_{io}^2 = \frac{1}{N}\sum_{i=1}^{N}d_{ij}^2
    # - \frac{1}{N^2}\sum_{j=2}^N \sum_{k=2}{j-1}d_{jk}^2
    # $$

    d_o_2 = tf.reduce_mean(
        tf.pow(
            distance_matrix,
            2),
        axis=0) \
        - tf.math.divide(
            tf.reduce_sum(
                tf.pow(
                    distance_matrix,
                    2)),
            2 * tf.pow(
                tf.cast(n_atoms, tf.float32),
                2))

    # $$
    # g_ij = (d_{io}^2 + d_{jo}^2 - d_{ij}^2)
    # $$

    g = tf.math.divide(
        tf.tile(
            tf.expand_dims(
                d_o_2,
                0),
            [n_atoms, 1]) \
        + tf.tile(
            tf.expand_dims(
                d_o_2,
                1),
            [1, n_atoms]) \
        - tf.pow(
            distance_matrix,
            2),
        2)


    e, v = tf.linalg.eigh(g)
    e = tf.math.sqrt(tf.tile(tf.expand_dims(e[-3:], 0), [n_atoms, 1]))
    v = v[:, -3:]

    x = v * e

    return x

# =============================================================================
# module class
# =============================================================================
class Conformers(object):
    """ A molecule system that could be calculated under distance geometry.

    """
    def __init__(self, mol, forcefield, typing):
        self.mol = mol
        self.n_atoms = tf.shape(mol[0])[0]
        self.atoms = mol[0]
        self.adjacency_map = mol[1]
        self.forcefield = forcefield
        self.typing = typing

    def get_conformers_from_distance_geometry(self, n_samples):
        """ Get the equilibrium bond length as initial bond length.

        """
        # find the positions at which there is a bond
        is_bond = tf.greater(
            self.adjacency_map,
            tf.constant(0, dtype=tf.float32))

        # dirty stuff to get the bond indices to update
        all_idxs_x, all_idxs_y = tf.meshgrid(
            tf.range(tf.cast(self.n_atoms, tf.int64), dtype=tf.int64),
            tf.range(tf.cast(self.n_atoms, tf.int64), dtype=tf.int64))

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

        # get the types
        typing_assignment = self.typing(self.mol).get_assignment()

        bond_specs = tf.map_fn(
            lambda bond: tf.convert_to_tensor(
                    self.forcefield.get_bond(
                        int(tf.gather(typing_assignment, bond[0]).numpy()),
                        int(tf.gather(typing_assignment, bond[1]).numpy()))),
            bond_idxs,
            dtype=tf.float32)


        # TODO:
        # figure out how to get the upper and lower boundary of the bonds
        delta_x = tf.math.sqrt(
            tf.math.divide(
                tf.constant(2 * BOND_ENERGY_THRS, tf.float32),
                bond_specs[:, 1]))

        # get the lower and upper bond
        bonds_upper_bound = bond_specs[:, 0] + delta_x
        bonds_lower_bound = bond_specs[:, 0] - delta_x

        # put the constrains into the matrix
        upper_bound = tf.Variable(10 * tf.ones_like(self.mol[1]))
        lower_bound = tf.Variable(0.05 * tf.ones_like(self.mol[1]))

        upper_bound.scatter_nd_update(
            bond_idxs,
            bonds_upper_bound)

        lower_bound.scatter_nd_update(
            bond_idxs,
            bonds_lower_bound)

        upper_bound.scatter_nd_update(
            tf.reverse(bond_idxs, axis=[1]),
            bonds_upper_bound)

        lower_bound.scatter_nd_update(
            tf.reverse(bond_idxs, axis=[1]),
            bonds_lower_bound)

        upper_bound.scatter_nd_update(
            tf.tile(
                tf.expand_dims(
                    tf.range(
                        tf.cast(
                            self.n_atoms,
                            tf.int64),
                        dtype=tf.int64),
                    1),
                [1, 2]),
            tf.zeros((self.n_atoms, ), dtype=tf.float32))

        lower_bound.scatter_nd_update(
            tf.tile(
                tf.expand_dims(
                    tf.range(
                        tf.cast(
                            self.n_atoms,
                            tf.int64),
                        dtype=tf.int64),
                    1),
                [1, 2]),
            tf.zeros((self.n_atoms, ), dtype=tf.float32))

        upper_bound, lower_bound = set_12_bounds(upper_bound, lower_bound)
        upper_bound, lower_bound = set_13_bounds(upper_bound, lower_bound,
            self.adjacency_map, typing_assignment, self.forcefield)

        # NOTE: the following code is commented out because
        # because we don't want to use tfp for now
        # sample from a uniform distribution

        # distance_matrix_distribution = tfp.distributions.Uniform(
        #     low=lower_bound,
        #     high=upper_bound)

        # distance_matrices = distance_matrix_distribution.sample(n_samples)

        distance_matrices = tf.tile(
            tf.expand_dims(
                lower_bound,
                0),
            [n_samples, 1, 1]) \
            + tf.random.uniform(
                shape=(n_samples, self.n_atoms, self.n_atoms),
                minval=0.,
                maxval=1.,
                dtype=tf.float32) \
            * tf.tile(
                tf.expand_dims(
                    upper_bound - lower_bound,
                    0),
                [n_samples, 1, 1])

        distance_matrices = tf.linalg.band_part(distance_matrices, 0, -1)
        distance_matrices = tf.transpose(distance_matrices, perm=[0, 2, 1]) \
            + distance_matrices

        conformers = 10 * tf.map_fn(
            embed,
            distance_matrices)

        return conformers
