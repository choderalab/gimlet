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

# =============================================================================
# utility functions
# =============================================================================
def featurize_atoms(
        atoms,
        adjacency_map,
        element=True,
        hybridization=True,
        aromaticity=True):
    """ Featurize a small molecule to be fed into training.

    """
    # grab the typing object
    typing = gin.deterministic.typing.Typing([atoms, adjacency_map])

    # init the feature vector
    feature = tf.expand_dims(
        tf.zeros_like(atoms, dtype=tf.float32),
        1)

    if element == True:
        feature = tf.concat(
            [
                feature,
                tf.expand_dims(
                    tf.where(
                        typing.is_carbon,
                        tf.ones_like(atoms, dtype=tf.float32),
                        tf.zeros_like(atoms, dtype=tf.float32)),
                    1)
            ],
            axis=1)

        feature = tf.concat(
            [
                feature,
                tf.expand_dims(
                    tf.where(
                        typing.is_nitrogen,
                        tf.ones_like(atoms, dtype=tf.float32),
                        tf.zeros_like(atoms, dtype=tf.float32)),
                    1)
            ],
            axis=1)

        feature = tf.concat(
            [
                feature,
                tf.expand_dims(
                    tf.where(
                        typing.is_oxygen,
                        tf.ones_like(atoms, dtype=tf.float32),
                        tf.zeros_like(atoms, dtype=tf.float32)),
                    1)
            ],
            axis=1)

        feature = tf.concat(
            [
                feature,
                tf.expand_dims(
                    tf.where(
                        typing.is_sulfur,
                        tf.ones_like(atoms, dtype=tf.float32),
                        tf.zeros_like(atoms, dtype=tf.float32)),
                    1)
            ],
            axis=1)

        feature = tf.concat(
            [
                feature,
                tf.expand_dims(
                    tf.where(
                        typing.is_flourine,
                        tf.ones_like(atoms, dtype=tf.float32),
                        tf.zeros_like(atoms, dtype=tf.float32)),
                    1)
            ],
            axis=1)

        feature = tf.concat(
            [
                feature,
                tf.expand_dims(
                    tf.where(
                        typing.is_chlorine,
                        tf.ones_like(atoms, dtype=tf.float32),
                        tf.zeros_like(atoms, dtype=tf.float32)),
                    1)
            ],
            axis=1)

        feature = tf.concat(
            [
                feature,
                tf.expand_dims(
                    tf.where(
                        typing.is_iodine,
                        tf.ones_like(atoms, dtype=tf.float32),
                        tf.zeros_like(atoms, dtype=tf.float32)),
                    1)
            ],
            axis=1)

    if hybridization == True:
        feature = tf.concat(
            [
                feature,
                tf.expand_dims(
                    tf.where(
                        typing.is_sp1,
                        tf.ones_like(atoms, dtype=tf.float32),
                        tf.zeros_like(atoms, dtype=tf.float32)),
                    1)
            ],
            axis=1)

        feature = tf.concat(
            [
                feature,
                tf.expand_dims(
                    tf.where(
                        typing.is_sp2,
                        tf.ones_like(atoms, dtype=tf.float32),
                        tf.zeros_like(atoms, dtype=tf.float32)),
                    1)
            ],
            axis=1)

        feature = tf.concat(
            [
                feature,
                tf.expand_dims(
                    tf.where(
                        typing.is_sp3,
                        tf.ones_like(atoms, dtype=tf.float32),
                        tf.zeros_like(atoms, dtype=tf.float32)),
                    1)
            ],
            axis=1)

    if aromaticity == True:
        feature = tf.concat(
            [
                feature,
                tf.expand_dims(
                    tf.where(
                        typing.is_aromatic,
                        tf.ones_like(atoms, dtype=tf.float32),
                        tf.zeros_like(atoms, dtype=tf.float32)),
                    1)
            ],
            axis=1)

    feature = feature[:, 1:]
    return feature
