"""
gc_architectures.py

Ready-made novel gc architectures.

MIT License

Copyright (c) 2019 Chodera lab // Memorial Sloan Kettering Cancer Center
and Authors

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
import gin
import tonic
import tensorflow as tf
tf.enable_eager_execution()


def kearnes_2016():
    """ arXiv:1603.00856
    """
    class GCTyping(gin.deterministic.typing.TypingBase):
        def __init__(self):
            super(GCTyping, self).__init__() # TODO: parameter `mol` unfilled

        def atom_type(self):
            return tf.one_hot() # TODO: does this require an argument?

    return gin.probabilistic.gn.GraphNet(
        f_e=tf.keras.layers.Dense(16),

        f_v=tf.keras.layers.Lambda(
            lambda x: tf.keras.layers.Dense(16)(tf.one_hot(x, 8))),

        f_u=(lambda x, y: tf.zeros((1, 16), dtype=tf.float32)),

        phi_e=tonic.nets.for_gn.ConcatenateThenFullyConnect((16, 'elu', 16, 'sigmoid')),

        phi_v=tonic.nets.for_gn.ConcatenateThenFullyConnect((16, 'elu', 16, 'sigmoid')),

        phi_u=tonic.nets.for_gn.ConcatenateThenFullyConnect((16, 'elu', 16, 'sigmoid')),

        rho_e_v=(lambda h_e, atom_is_connected_to_bonds: tf.reduce_sum(
            tf.where(
                tf.tile(
                    tf.expand_dims(
                        atom_is_connected_to_bonds,
                        2),
                    [1, 1, h_e.shape[1]]),
                tf.tile(
                    tf.expand_dims(
                        h_e,
                        0),
                    [
                        atom_is_connected_to_bonds.shape[0], # n_atoms
                        1,
                        1
                    ]),
                tf.zeros((
                    atom_is_connected_to_bonds.shape[0],
                    h_e.shape[0],
                    h_e.shape[1]))),
            axis=1)),

        rho_e_u=(lambda x: tf.expand_dims(tf.reduce_sum(x, axis=0), 0)),

        rho_v_u=(lambda x: tf.expand_dims(tf.reduce_sum(x, axis=0), 0)),

        f_r=f_r((4, 'tanh', 4, 'elu', 1)),

        repeat=3)
