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
class Featurizer(object):
    """ Provide featurization for atoms in small molecules.

    """
    def __init__(self, mol):
        self.mol = mol
        self.atoms = mol[0]
        self.adjacency_map = mol[1]
        self.n_atoms = mol[0].shape[0]
        self.typing = gin.deterministic.typing.Typing(mol)

    def one_hot_atom_type(self):
        # (n_atoms, 8)
        return tf.one_hot(self.atoms, 8)

    def hybridization(self):
        # (n_atoms, 3)
        return tf.where(
            tf.transpose(
                tf.concat(
                    [
                        tf.expand_dims(
                            self.typing.is_sp1,
                            0),
                        tf.expand_dims(
                            self.typing.is_sp2,
                            0),
                        tf.expand_dims(
                            self.typing.is_sp3,
                            0)
                    ],
                    0)),
            tf.ones((self.n_atoms, 3), dtype=tf.float32),

            tf.zeros((self.n_atoms, 3), dtype=tf.float32))

    def aromaticity(self):
        # (n_atoms, 1)
        return tf.where(
            self.typing.is_aromatic,
            self.ones((self.n_atoms, 1), dtype=tf.float32),
            self.zeros((self.n_atoms), 1), dtype=tf.float32)

    def valence(self):
        return tf.count_nonzero(
            tf.transpose(self.adjacency_map) + self.adjacency_map,
            0)

    def electron_affinity(self):
        pass
