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

import gin.molecule

# =============================================================================
# module classes
# =============================================================================
class GraphNet(tf.keras.Model):
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

            # building blocks for GCN
            # update function
            phi_e=lambda *x: x[0],
            phi_u=lambda *x: x[0],
            phi_v=lambda *x: x[0],

            # aggregate functions, default to be sum
            rho_e_v=(lambda h_e, atom_is_connected_to_bonds: tf.reduce_sum(
                tf.where( # here we grab the edges connected to nodes
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

            # readout phase
            f_r=lambda *x:x[0],

            # featurization
            f_e=lambda x:x,
            f_v=lambda x:x,
            f_u=lambda x:x,

            repeat=3):

        super(GraphNet, self).__init__()
        self.phi_e = phi_e
        self.rho_e_v = rho_e_v
        self.phi_v = phi_v
        self.rho_e_u = rho_e_u
        self.rho_v_u = rho_v_u
        self.phi_u = phi_u
        self.f_r = f_r
        self.f_e = f_e
        self.f_v = f_v
        self.f_u = f_u
        self.repeat = repeat

    @tf.function
    def _call(
            self,
            mol, # note that the molecules here could be featurized
            repeat=3):

        """ Propagate between nodes and edges.

        Parameters
        ----------
        molecules : a list of molecules to be
        """

        # get the attributes of the molecule
        adjacency_map = mol[1]
        atoms = mol[0]
        adjacency_map_full = adjacency_map \
            + tf.transpose(adjacency_map)
        n_atoms = tf.cast(tf.shape(atoms)[0], tf.int64)

        # (n_atoms, n_atoms, 2)
        all_idxs_stack = tf.stack(
            tf.meshgrid(
                tf.range(n_atoms, dtype=tf.int64),
                tf.range(n_atoms, dtype=tf.int64)),
            axis=2)

        # (n_atoms, n_atoms, 2) # boolean
        is_bond = tf.greater(
            adjacency_map,
            tf.constant(0, dtype=tf.float32))

        # (n_bonds, 2)
        bond_idxs = tf.boolean_mask(
            all_idxs_stack,
            is_bond)

        n_bonds = tf.cast(tf.shape(bond_idxs)[0], tf.int64)

        # (n_bonds, n_atoms)
        bond_is_connected_to_atoms = tf.logical_or(
            tf.equal(
                tf.tile(
                    tf.expand_dims(
                        tf.range(n_atoms),
                        0),
                    [n_bonds, 1]),
                tf.tile(
                    tf.expand_dims(
                        bond_idxs[:,0],
                        1),
                    [1, n_atoms])),

            tf.equal(
                tf.tile(
                    tf.expand_dims(
                        tf.range(n_atoms),
                        0),
                    [n_bonds, 1]),
                tf.tile(
                    tf.expand_dims(
                        bond_idxs[:,1],
                        1),
                    [1, n_atoms])))


        # (n_atoms, n_bonds)
        atom_is_connected_to_bonds = tf.transpose(
            bond_is_connected_to_atoms)

        # (n_bonds, )
        # NOTE: here we use the same boolean mask as before, so they
        #       should be following the same order
        bond_orders = tf.boolean_mask(
            adjacency_map,
            is_bond)

        # initialize the hidden layers
        # (n_bonds, ...)
        h_e = self.f_e(tf.expand_dims(bond_orders, 1))
        h_e_0 = h_e
        h_e_history = tf.expand_dims(h_e_0, 1)

        # (n_atoms, ...)
        h_v = self.f_v(atoms)
        h_v_0 = h_v
        h_v_history = tf.expand_dims(h_v_0, 1)

        # (...)
        h_u = self.f_u(atoms, adjacency_map)
        h_u_0 = h_u
        h_u_history = tf.expand_dims(h_u_0, 1)

        def propagate_one_time(
            iter_idx,
            h_e, h_v, h_u,
            h_e_history, h_v_history, h_u_history):
            # update $ e'_k $
            # $$
            # e'_k = \phi^e (e_k, v_{rk}, v_{sk}, u)
            # $$

            '''
            # (n_bonds, d_v)
            h_left = tf.gather(
                h_v,
                bond_idxs[:, 0])

            # (n_bonds, d_v)
            h_right = tf.gather(
                h_v,
                bond_idxs[:, 1])
            '''

            h_left_and_right = tf.boolean_mask(
                tf.tile( # (n_bonds, n_atoms, d_v)
                    tf.expand_dims(
                        h_v,
                        0),
                    [n_bonds, 1, 1]),
                bond_is_connected_to_atoms)

            h_left, h_right = tf.split(h_left_and_right, 2)

            # (n_bonds, d_e)
            h_e = self.phi_e(h_e, h_e_0, h_left, h_right,
                tf.tile( # repeat global attribute to the number of bonds
                    h_u,
                    [n_bonds, 1]))

            h_e_history = tf.concat(
                [
                    h_e_history,
                    tf.expand_dims(
                        h_e,
                        1)
                ],
                axis=1)

            # aggregate $ \bar{e_i'} $
            # $$
            # \bar{e_i'} = \rho^{e \rightarrow v} (E'_i)
            # $$

            # (n_atoms, d_e)
            h_e_bar_i = self.rho_e_v(h_e, atom_is_connected_to_bonds)

            # update $ v'_i $
            # $$
            # v'_i = phi^v (\bar{e_i}, v_i, u)
            # $$

            # (n_atoms, d_v)
            h_v = self.phi_v(h_v, h_v_0, h_e_bar_i,
                tf.tile(
                    h_u,
                    [n_atoms, 1]))

            h_v_history = tf.concat(
                [
                    h_v_history,
                    tf.expand_dims(
                        h_v,
                        1)
                ],
                axis=1)

            # aggregate $ \bar{e'} $
            # $$
            # \bar{e'} = \rhp^{e \rightarrow u} (E')
            # $$

            # (...)
            h_e_bar = self.rho_e_u(h_e)

            # aggregate $ \bar{v'} $
            # $$
            # \bar{v'} = \rho^{v \rightarrow u} (V')
            # $$

            # (...)
            h_v_bar = self.rho_v_u(h_v)

            # update $ u' $
            # $$
            # u' = \phi^u (\bar{e'}, \bar{v'}, u)
            # $$

            # (...)
            h_u = self.phi_u(h_u, h_u_0, h_e_bar, h_v_bar)

            h_u_history = tf.concat(
                [
                    h_u_history,
                    tf.expand_dims(
                        h_u,
                        1)
                ],
                axis=1)

            return (
                iter_idx + 1,
                h_e, h_v, h_u,
                h_e_history, h_v_history, h_u_history)

        # use while loop to execute the graph multiple times
        iter_idx = tf.constant(0, dtype=tf.int64)

        iter_idx, h_e, h_v, h_u, h_e_history, h_v_history, h_u_history \
            = tf.while_loop(
            # condition
            lambda \
                iter_idx, h_e, h_v, h_u, h_e_history, h_v_history, h_u_history:\
                    tf.less(iter_idx, self.repeat),

            # loop body
            propagate_one_time,

            # loop vars
            [
                iter_idx,
                h_e, h_v, h_u,
                h_e_history, h_v_history, h_u_history
            ],

            # shape_invariants
            shape_invariants = [
                iter_idx,
                h_e.get_shape(),
                h_v.get_shape(),
                h_u.get_shape(),
                tf.TensorShape((None, h_e.shape[0], h_e.shape[1])),
                tf.TensorShape((None, h_v.shape[0], h_v.shape[1])),
                tf.TensorShape((None, h_u.shape[0], h_u.shape[1])),
                ])

        y_bar = self.f_r(
            h_e, h_v, h_u,
            h_e_history, h_v_history, h_u_history)

        return y_bar

    def call(self, molecules, repeat=3):
        return self._call(molecules, repeat=repeat)

    def switch(self, to_test=True):
        for fn in [self.rho_e_u, self.rho_e_v, self.rho_v_u]:
            if hasattr(fn, 'switch'):
                fn.switch()
