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

# =============================================================================
# module functions
# =============================================================================
def get_idxs(adjacency_map):

    # bond idxs are where adjacency matrix is greater than zero.
    bond_idxs =  tf.where(
        tf.greater(
            tf.linalg.band_part(
                adjacency_map,
                0, -1),
            tf.constant(0, dtype=tf.float32)))

    # get the bond idxs in both order
    _bond_idxs = tf.concat(
        [
            bond_idxs,
            tf.reverse(
                bond_idxs,
                axis=[1])
        ],
        axis=0)

    # get the number of _directed_ bonds
    _n_bonds = tf.shape(_bond_idxs, tf.int64)[0]

    # enumerate all bond pairs
    # (n_bond * n_bond, 4)
    bond_pairs = tf.reshape(
        tf.concat(
            [
                tf.tile(
                    tf.expand_dims(
                        _bond_idxs,
                        axis=0),
                    [_n_bonds, 1, 1]),
                tf.tile(
                    tf.expand_dims(
                        _bond_idxs,
                        axis=1),
                    [1, _n_bonds, 1])
            ],
            axis=2),
        [-1, 4])

    # angles are where two _directed bonds share one _inner_ atom
    angle_idxs = tf.gather(
        tf.boolean_mask(
            bond_pairs,
            tf.logical_and(
                tf.equal(
                    bond_pairs[:, 1],
                    bond_pairs[:, 2]),
                    tf.less(
                        bond_pairs[:, 0],
                        bond_pairs[:, 3]))),
        [0, 1, 3],
        axis=1)

    # get the angle pairs in both order
    _angle_idxs = tf.concat(
        [
            angle_idxs,
            tf.reverse(
                angle_idxs,
                axis=[1])
        ],
        axis=0)

    # get the number of _directed_ angles
    _n_angles = tf.shape(_angle_idxs, tf.int64)[0]

    # enumerate all bond pairs
    # (n_angles * n_angles, 6)
    angle_pairs = tf.reshape(
        tf.concat(
            [
                tf.tile(
                    tf.expand_dims(
                        _angle_idxs,
                        axis=0),
                    [_n_angles, 1, 1]),
                tf.tile(
                    tf.expand_dims(
                        _angle_idxs,
                        axis=1),
                    [1, _n_angles, 1])
            ],
            axis=2),
        [-1, 6])

    # angles are where two _directed bonds share one _inner_ atom
    torsion_idxs = tf.gather(
        tf.boolean_mask(
            angle_pairs,
            tf.logical_and(
                tf.logical_and(
                    tf.equal(
                        angle_pairs[:, 1],
                        angle_pairs[:, 3]),
                    tf.equal(
                        angle_pairs[:, 2],
                        angle_pairs[:, 4])),
                tf.less(
                    angle_pairs[:, 0],
                    angle_pairs[:, 5]))),
        [0, 1, 2, 5],
        axis=1)

    # one four idxs are just the two ends of torsion idxs
    one_four_idxs = tf.gather(
        torsion_idxs,
        [0, 3])

    # nonbonded idxs are those that cannot be connected by
    # 1-, 2-, and 3-walks
    adjacency_map_full = tf.math.add(
        adjacency_map,
        tf.transpose(
            adjacency_map))

    nonbonded_idxs = tf.where(
        tf.reduce_sum(
            [
                # 1-walk
                adjacency_map_full,

                # 2-walk
                tf.matmul(
                    adjacency_map_full,
                    adjacency_map_full),

                # 3-walk
                tf.matmul(
                    adjacency_map_full,
                    tf.matmul(
                        adjacency_map_full,
                        adjacency_map_full))
            ],
            axis=0))

    return bond_idxs, angle_idxs, torsion_idxs, one_four_idxs, nonbonded_idxs
