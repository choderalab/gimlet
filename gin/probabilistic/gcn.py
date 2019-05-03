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

import gin.molecule

# =============================================================================
# utility classes
# =============================================================================
class GCN(tf.keras.Model):
    """ A group of functions trainable under back-propagation to update atoms
    and molecules based on their neighbors and global attributes.

    Structures adopted from:
    arXiv:1806.01261v3

    Attributes
    ----------
    phi_e : function,
        applied per edge, with arguments $(e_k, v_r_k, v_s_k, u)$, and returns
        $e_k'$.
    rho_e_v : function,
        applied to $E'_i$, and aggregates the edge updates updates for edges
        that project to vertex $i$, into $\bar{e_i'}, which will be used in the
        next step's node update.
    phi_v : function,
        applied to each node $i$, to compute an updated node attribute, $v_i'$,.
    rho_e_u : function.
        applied to $E'$, and aggregates all edge updates, into $\bar{e'}$,
        which will then be used in the next step's global update.
    rho_v_u : function,
        applied to $V'$, and aggregates all node updates, into $\bar{v'}$,
        which will then be used in the next step's global update.
    phi_u : function,
        applied once per graph, and computes and update for the global
        attribute, $u'$.

    """

    def __init__(
            self,
            phi_e: lambda x: x,
            rho_e_v: lambda x: x,
            phi_v: lambda x: x,
            rho_e_u: lambda x: x,
            rho_v_u: lambda x: x,
            phi_u: lambda x: x):

        super(GCN, self).__init__()
        self.phi_e = phi_e
        self.rho_e_v = rho_e_v
        self.phi_v = phi_v
        self.rho_e_u = rho_e_u
        self.phi_u = phi_u

    @tf.contrib.eager.defun
    def propagate(
            self,
            molecules,
            molecule_attributes_shape=(1, ),
            atom_attributes_shape=(1, )):

        """ Propagate the message between nodes and edges.

        Parameters
        ----------
        molecules : a list of molecules to be
        """
        with tf.init_scope():
            # get the number of the molecules
            n_molecules = len(molecules)

        # define the function needed for propagate one molecule
        def propagate_one(idx, molecule_attributes, atom_attributes,
                molecules=molecules):
            # get the specific molecule
            mol = molecules[idx]

            # get the attributes of the molecule
            atoms = mol.atoms
            adjacency_map = mol.adjacency_map


        # for loop
        idx = tf.constant(0, dtype=tf.int64)

        idx, molecule_attributes, atom_attributes = tf.while_loop(
            # while idx < n_molecules
            tf.less(
                idx,
                tf.cast(
                    n_molecules,
                    tf.int64)),

            # loop body
            lambda molecule_attributes, atom_attributes: propagate_one(
                molecule_attributes, atom_attributes),

            # loop var
            [idx, molecule_attributes, atom_attributes],

            shape_invariants=[
                idx.get_shape(),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None])],

            parallel_iterations=n_molecules)
