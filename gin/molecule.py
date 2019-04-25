"""
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
import tensorflow as tf
tf.enable_eager_execution

# =============================================================================
# utility classes
# =============================================================================
class Molecule(object):
    """ An object signifying a molecule.

    Attributes
    atoms : tf.Tensor, shape = (n_atoms, ), dtype = tf.int64,
        each entry is the index of one atom
    bonds : tf.Tensor, shape = (n_bonds, 3), dtype = tf.int64,
        each entry is (atom0, atom1, bond_order)
    """

    def __init__(
        self,
        include_charges=False,
        include_chiralities=False):

        # initialize atoms and bonds object
        self._atoms = tf.Variable([], dtype=tf.int64)
        self._bonds = tf.Variable([[]], d type=tf.int64)

        if include_charges == True:
            self._charges = tf.Variable([], dtype=tf.int64)

        if include_chiralities == True:
            self._chiralities = tf.Variable([], dtype=tf.int64)

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        self._atoms = atoms

    @property
    def bonds(self):
        return self._bonds

    @property
    def charges(self):
        return self._charges

    @property
    def chiralities(self):
        return self._chiralities

    def is_valid(self):
        return True
