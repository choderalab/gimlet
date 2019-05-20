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
tf.enable_eager_execution()

class MoleculeDistanceGeometrySystem(gin.molecule.Molecule):
    """ A molecule system that could be calculated under distance geometry.

    """
    def __init__(self, forcefield):
        super(MoleculeDistanceGeometrySystem, self).__init__()
        self.forcefield = forcefield

    def get_typing(self, typing_object):
        """ Get the atom typing corresponding to the specific forcefield.

        Parameters
        ----------
        typing_object : gin.typing.Typing obejct

        """


    def get_bond_constraint(self):
        """ Get the equilibrium bond length as initial bond length.

        """
        # find the positions at which there is a bond
        is_bond = tf.greater(
            self.adjacency_map,
            tf.constant(0, dtype=tf.float32))

        # dirty stuff to get the bond indices to update
        all_idxs_x, all_idxs_y = tf.meshgrid(
            tf.range(self.n_atoms, dtype=tf.int64),
            tf.range(self.n_atoms, dtype=tf.int64))

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
