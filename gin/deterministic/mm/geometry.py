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
# utility functions
# =============================================================================
def get_distance_matrix(coordinates):
    """ Calculate the distance matrix from coordinates.

    $$
    D_{ij}^2 = <X_i, X_i> - 2<X_i, X_j> + <X_j, X_j>

    Parameters
    ----------
    coordinates: tf.Tensor, shape=(n_atoms, 3)

    """
    X_2 = tf.reduce_sum(
        tf.math.square(
            coordinates),
        axis=2,
        keepdims=True)

    return tf.math.sqrt(
        tf.nn.relu(
            X_2 - 2 * tf.matmul(
                coordinates,
                tf.transpose(coordinates, [0, 2, 1])) \
                + tf.transpose(X_2, [0, 2, 1])))


# =============================================================================
# module functions
# =============================================================================
def get_distances(idxs, coordinates):
    """ Get the distances between nodes given coordinates.

    """
    # get the distance matrix
    distance_matrix = get_distance_matrix(
        coordinates)

    # calculate the distances
    distances = tf.transpose(
        tf.gather_nd(
            tf.transpose( # put the batch dimension as the last dimension
                distance_matrix,
                [1, 2, 0]),
            idxs),
        [1, 0])

    return distances


def get_angles(angle_idxs, coordinates, return_cos=False):
    """ Calculate angles from a set of indices and coordinates.

    """
    # get the coordinates of the atoms forming the angle
    # (batch_size, n_angles, 3, 3)
    angle_coordinates = tf.gather(
        coordinates,
        angle_idxs,
        axis=1)

    # (batcn_angles, 3)
    angle_left = angle_coordinates[:, :, 1, :] \
        - angle_coordinates[:, :, 0, :]

    # (n_angles, 3)
    angle_right = angle_coordinates[:, :, 1, :] \
        - angle_coordinates[:, :, 2, :]

    # (n_batch, n_angles, )
    angles = tf.math.atan2(
        tf.norm(
            tf.linalg.cross(
                angle_left,
                angle_right),
            axis=2),
        tf.reduce_sum(
            tf.multiply(
                angle_left,
                angle_right),
            axis=2))

    return angles

def get_torsions(torsion_idxs, coordinates, return_cos=False):
    """ Calculate the dihedrals based on coordinates and the indices of
    the torsions.

    Parameters
    ----------
    coordinates: tf.Tensor, shape=(n_atoms, 3)
    torsion_idxs: # TODO: describe
    """
    # get the coordinates of the atoms forming the dihedral
    # (batch_size, n_torsions, 4, 3)
    torsion_idxs = tf.gather(
        coordinates,
        torsion_idxs,
        axis=1)

    # (batch_size, n_torsions, 3)
    normal_left = tf.linalg.cross(
        torsion_idxs[:, :, 1, :] - torsion_idxs[:, :, 0, :],
        torsion_idxs[:, :, 1, :] - torsion_idxs[:, :, 2, :])

    # (batch_size, n_torsions, 3)
    normal_right = tf.linalg.cross(
        torsion_idxs[:, :, 2, :] - torsion_idxs[:, :, 3, :],
        torsion_idxs[:, :, 2, :] - torsion_idxs[:, :, 1, :])

    # (batch_size, n_torsions, )
    dihedrals = tf.math.atan2(
        tf.norm(
            tf.linalg.cross(
                normal_left,
                normal_right),
            axis=2),
        tf.reduce_sum(
            tf.multiply(
                normal_left,
                normal_right),
            axis=2))

    return dihedrals
