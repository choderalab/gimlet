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
import gin.deterministic.forcefield

# =============================================================================
# utility functions
# =============================================================================
def topological_molecule_to_3d_conformation_distance_geometry(
        mol,
        forcefiled='gaff'):
    """ Generate the 3d conformation of a small molecule based on its
    adjacency map using distance geometry.


    """


# =============================================================================
# module classes
# =============================================================================
class SingleMoleculeMechanicsSystem(object):
    """
    A single molecule system that could be calculated in MD calculation.
    """
    def __init__(
            self,
            mol
            coordinates=None,
            typing=gin.deterministic.typing.TypingGAFF,
            forcefield=gin.deterministic.forcefields.gaff):

        self.mol = mol
        self.atoms = mol[0]
        self.adjacency_map = mol[1]
        self.coordinates = coordinates
        self.typing = typing
        self.forcefield = forcefield


        if type(self.coordinates) == type(None):
            self.coordinates = gin.conformer(
                mol,
                self.forcefield,
                self.typing).get_conformers_from_distance_geometry(1)[0]

    def get_bonds(self):
        """ Get the config of all the bonds in the system.

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

        self.bond_idxs = bond_idxs
        self.bond_k = bond_specs[:, 0]
        self.bond_length = bond_specs[:, 1]


    def get_angle_params(self):
        """ Get all the angles in the system.

        """
