"""
MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center,
Weill Cornell Medical College, and Authors

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
import gin

def alkane_energy(atoms, adjacency_map, coordinates):

    bond_idxs, angle_idxs, torsion_idxs, one_four_idxs, nonbonded_idxs = gin.deterministic.mm.indices.get_idxs(
        adjacency_map)

    bond_atoms = tf.gather(atoms, bond_idxs)
    angle_atoms = tf.gather(atoms, angle_idxs)
    one_four_atoms = tf.gather(atoms, one_four_idxs)
    nonbonded_atoms = tf.gather(atoms, nonbonded_idxs)

    bond_k = tf.where(
        tf.logical_and(
            tf.equal(
                bond_atoms[:, 0],
                tf.constant(0, dtype=tf.int64)),
            tf.equal(
                bond_atoms[:, 1],
                tf.constant(0, dtype=tf.int64))),
        253634.08,
        282252.64)

    bond_l = tf.where(
        tf.logical_and(
            tf.equal(
                bond_atoms[:, 0],
                tf.constant(0, dtype=tf.int64)),
            tf.equal(
                bond_atoms[:, 1],
                tf.constant(0, dtype=tf.int64))),
        0.1535,
        0.1092)

    angle_k = tf.where(

        tf.logical_and(
            tf.equal(
                angle_atoms[:, 0],
                tf.constant(9, dtype=tf.int64)),
            tf.equal(
                angle_atoms[:, 2],
                tf.constant(9, dtype=tf.int64))),

        tf.constant(329.95024, dtype=tf.float32),

        tf.where(
            tf.logical_and(
                tf.equal(
                    angle_atoms[:, 0],
                    tf.constant(0, dtype=tf.int64)),
                tf.equal(
                    angle_atoms[:, 2],
                    tf.constant(0, dtype=tf.int64))),

            tf.constant(528.94128, dtype=tf.float32),
            tf.constant(388.02416)))

    angle_l = tf.where(

        tf.logical_and(
            tf.equal(
                angle_atoms[:, 0],
                tf.constant(9, dtype=tf.int64)),
            tf.equal(
                angle_atoms[:, 2],
                tf.constant(9, dtype=tf.int64))),

        tf.constant(1.89106424454, dtype=tf.float32),

        tf.where(
            tf.logical_and(
                tf.equal(
                    angle_atoms[:, 0],
                    tf.constant(0, dtype=tf.int64)),
                tf.equal(
                    angle_atoms[:, 2],
                    tf.constant(0, dtype=tf.int64))),

            tf.constant(1.93085775148, dtype=tf.float32),
            tf.constant(1.92073484182, dtype=tf.float32)))

    sigma = tf.where(
        tf.equal(
            atoms,
            tf.constant(0, dtype=tf.int64)),
        tf.constant(0.339966950842, dtype=tf.float32),
        tf.constant(0.264953278775, dtype=tf.float32))

    epsilon = tf.where(
        tf.equal(
            atoms,
            tf.constant(0, dtype=tf.int64)),
        tf.constant(0.4577296, dtype=tf.float32),
        tf.constant(0.0656888, dtype=tf.float32))

    sigma_pair = tf.math.multiply(
        tf.constant(0.5, dtype=tf.float32),
        tf.math.add(
            tf.tile(
                tf.reshape(
                    sigma,
                    [-1, 1]),
                [1, tf.shape(sigma, tf.int64)[0]]),
            tf.tile(
                tf.reshape(
                    sigma,
                    [1, -1]),
                [tf.shape(sigma, tf.int64)[0], 1])))

    epsilon_pair = tf.math.sqrt(
        tf.math.multiply(
            tf.tile(
                tf.reshape(
                    epsilon,
                    [-1, 1]),
                [1, tf.shape(epsilon, tf.int64)[0]]),
            tf.tile(
                tf.reshape(
                    epsilon,
                    [1, -1]),
                [tf.shape(epsilon, tf.int64)[0], 1])))

    bond_energy = gin.deterministic.mm.energy.bond(
        gin.deterministic.mm.geometry.get_distances(bond_idxs, coordinates),
        bond_k,
        bond_l)

    angle_energy = gin.deterministic.mm.energy.angle(
        gin.deterministic.mm.geometry.get_angles(angle_idxs, coordinates),
        angle_k,
        angle_l)

    one_four_energy = 0.5 * gin.deterministic.mm.energy.lj(
        gin.deterministic.mm.geometry.get_distances(one_four_idxs, coordinates),
        tf.gather_nd(sigma_pair, one_four_idxs),
        tf.gather_nd(epsilon_pair, one_four_idxs))

    nonbonded_energy = gin.deterministic.mm.energy.lj(
        gin.deterministic.mm.geometry.get_distances(nonbonded_idxs, coordinates),
        tf.gather_nd(sigma_pair, nonbonded_idxs),
        tf.gather_nd(epsilon_pair, nonbonded_idxs))

    return bond_energy, angle_energy, one_four_energy, nonbonded_energy
