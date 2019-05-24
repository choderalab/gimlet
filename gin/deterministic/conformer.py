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

# =============================================================================
# constants
# =============================================================================
BOND_ENERGY_THRS = 500

# =============================================================================
# utility functions
# =============================================================================
# @tf.contrib.eager.defun
def floyd(n_atoms, upper, lower):
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

    def inner_loop(upper, lower, k, i, j):
        upper[i, j].assign(
            tf.cond(
                tf.greater(
                    upper[i, j],
                    upper[i, k] + upper[k, j]),

                lambda: (upper[i, k] + upper[k, j]),

                lambda: upper[i, j]))

        lower[i, j].assign(
            tf.cond(
                tf.less(
                    lower[i, j],
                    lower[i, k] - upper[k, j]),

                lambda: lower[i, k] - upper[k, j],

                lambda: lower[i, j]))

        lower[i, j].assign(
            tf.cond(
                tf.less(
                    lower[i, j],
                    lower[j, k] - upper[k, i]),

                lambda: lower[j, k] - upper[k, i],

                lambda: lower[i, j]))


        return upper, lower, k, i, j+1

    def middle_loop(upper, lower, k, i):
        j = i + 1
        upper, lower, k, i, j = tf.while_loop(
            lambda upper, lower, k, i, j: tf.less(j, n_atoms),
            inner_loop,
            [upper, lower, k, i, j])
        return upper, lower, k, i+1

    def outer_loop(upper, lower, k):
        i = 0
        upper, lower, k, i = tf.while_loop(
            lambda upper, lower, k, i: tf.less(i, n_atoms - 1),
            middle_loop,
            [upper, lower, k, i])
        return upper, lower, k+1

    k = 0
    upper, lower, k = tf.while_loop(
        lambda upper, lower, k: tf.less(k, n_atoms),
        outer_loop,
        [upper, lower, k])

    # only keep the upper part
    upper = tf.linalg.band_part(upper, 0, -1)
    lower = tf.linalg.band_part(lower, 0, -1)

    upper = tf.transpose(upper) + upper
    lower = tf.transpose(lower) + lower

    return upper, lower

# @tf.contrib.eager.defun
def embed(n_atoms, distance_matrix):
    """ EMBED algorithm, as was implemented here:
    10.1002/0470845015.cda018
    """
    # $$
    # d_{io}^2 = \frac{1}{N}\sum_{i=1}^{N}d_{ij}^2
    # - \frac{1}{N^2}\sum_{j=2}^N \sum_{k=2}{j-1}d_{jk}^2
    # $$

    d_o_2 = tf.div(
        tf.reduce_sum(
            tf.pow(
                distance_matrix,
                2),
            axis=0),
        tf.cast(n_atoms, tf.float32)) \
        - tf.div(
            tf.reduce_sum(
                tf.pow(
                    tf.linalg.band_part(
                        distance_matrix,
                        0, -1),
                    2)),
            tf.pow(
                tf.cast(n_atoms, tf.float32), 2))

    # $$
    # g_ij = (d_{io}^2 + d_{jo}^2 - d_{ij}^2)
    # $$

    g = tf.div(
        tf.tile(
            tf.expand_dims(
                d_o_2,
                0),
            [n_atoms, 1]) \
            + tf.tile(
                tf.expand_dims(
                    d_o_2,
                    1),
                [1, n_atoms])
            - tf.pow(
                distance_matrix,
                2),
        2)

    # x = tf.linalg.sqrtm(g)
    e, v = tf.linalg.eigh(g)
    e = tf.tile(tf.expand_dims(e[:3], 0), [n_atoms, 1])
    v = v[:, :3]

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

        # get the specs of the bond
        bond_specs = tf.py_func(
            lambda *bonds: tf.convert_to_tensor(
                [
                    self.forcefield.get_bond(
                        int(tf.gather(typing_assignment, bond[0]).numpy()),
                        int(tf.gather(typing_assignment, bond[1]).numpy())) \
                    for bond in bonds
                ]),
            bond_idxs,
            [tf.float32])

        # TODO:
        # figure out how to get the upper and lower boundary of the bonds
        delta_x = tf.math.sqrt(
            tf.div(
                tf.constant(2 * BOND_ENERGY_THRS, tf.float32),
                bond_specs[:, 1]))

        # get the lower and upper bond
        bonds_upper_bound = bond_specs[:, 0] + delta_x
        bonds_lower_bound = bond_specs[:, 0] - delta_x

        # put the constrains into the matrix
        upper_bound = tf.Variable(10 * tf.ones_like(self.mol[1]))
        lower_bound = tf.Variable(0.05 * tf.ones_like(self.mol[1]))

        upper_bound = tf.scatter_nd_update(
            upper_bound,
            bond_idxs,
            bonds_upper_bound)

        lower_bound = tf.scatter_nd_update(
            lower_bound,
            bond_idxs,
            bonds_lower_bound)

        upper_bound = tf.scatter_nd_update(
            upper_bound,
            tf.reverse(bond_idxs, axis=[1]),
            bonds_upper_bound)

        lower_bound = tf.scatter_nd_update(
            lower_bound,
            tf.reverse(bond_idxs, axis=[1]),
            bonds_lower_bound)

        upper_bound = tf.scatter_nd_update(
            upper_bound,
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

        lower_bound = tf.scatter_nd_update(
            lower_bound,
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

        upper_bound, lower_bound = floyd(self.n_atoms, upper_bound, lower_bound)
        # sample from a uniform distribution
        distance_matrix_distribution = tf.distributions.Uniform(
            low=lower_bound,
            high=upper_bound)

        distance_matrices = distance_matrix_distribution.sample(n_samples)
        distance_matrices = tf.linalg.band_part(distance_matrices, 0, -1)
        distance_matrices = tf.transpose(distance_matrices, perm=[0, 2, 1]) \
            + distance_matrices

        conformers = tf.map_fn(
            lambda x: embed(self.n_atoms, x),
            distance_matrices)

        return conformers