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
tf.enable_eager_execution

# =============================================================================
# module classes
# =============================================================================
class Molecule(object):
    """ A base object signifying a molecule.

    Attributes
    ----------
    atoms : tf.Tensor, shape = (n_atoms, ), dtype = tf.int64,
        each entry is the index of one atom.
    adjacency_map : tf.Tensor, shape = (n_atoms, n_atoms, ), dtype = tf.float32,
        each entry $A_{ij}$ denotes the bond order between atom $i$ and $j$.
    """

    def __init__(
            self,
            atoms=None,
            adjacency_map=None):

        # initialize atoms and bonds object
        self._atoms = atoms
        self._adjacency_map = adjacency_map

    @property
    def atoms(self):
        return self._atoms

    @property
    def adjacency_map(self):
        return self._adjacency_map

    @atoms.setter
    def atoms(self, _atoms):
        self._atoms = _atoms

    @adjacency_map.setter
    def adjacency_map(self, _adjacency_map):
        self._adjacency_map = _adjacency_map

    @property
    def as_list(self):
        return [self._atoms, self._adjacency_map]
