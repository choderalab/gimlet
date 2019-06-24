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
                        [1, 1, tf.shape(h_e)[1]]),
                    tf.tile(
                        tf.expand_dims(
                            h_e,
                            0),
                        [
                            tf.shape(atom_is_connected_to_bonds)[0], # n_atoms
                            1,
                            1
                        ]),
                    tf.zeros((
                        tf.shape(atom_is_connected_to_bonds)[0],
                        tf.shape(h_e)[0],
                        tf.shape(h_e)[1]))),
                axis=1)),

            rho_e_u=(lambda h_e, bond_in_mol: tf.reduce_sum(
                tf.multiply(
                    tf.tile(
                        tf.expand_dims(
                            tf.where( # (n_bonds, n_mols)
                                tf.boolean_mask(
                                    bond_in_mol,
                                    tf.reduce_any(
                                        bond_in_mol,
                                        axis=1),
                                    axis=0),
                                tf.ones_like(
                                    tf.boolean_mask(
                                        bond_in_mol,
                                        tf.reduce_any(
                                            bond_in_mol,
                                            axis=1),
                                        axis=0),
                                    dtype=tf.float32),
                                tf.zeros_like(
                                    tf.boolean_mask(
                                        bond_in_mol,
                                        tf.reduce_any(
                                            bond_in_mol,
                                            axis=1),
                                        axis=0),
                                    dtype=tf.float32)),
                            2),
                        [1, 1, h_e.shape[1]]),
                    tf.tile( # (n_bonds, n_mols, d_e)
                        tf.expand_dims(
                            h_e, # (n_bonds, d_e)
                            1),
                        [1, bond_in_mol.shape[1], 1])),
                axis=0)),

            rho_v_u=(lambda h_v, atom_in_mol: tf.reduce_sum(
                tf.multiply(
                    tf.tile(
                        tf.expand_dims(
                            tf.where( # (n_bonds, n_mols)
                                atom_in_mol,
                                tf.ones_like(
                                    atom_in_mol,
                                    dtype=tf.float32),
                                tf.zeros_like(
                                    atom_in_mol,
                                    dtype=tf.float32)),
                            2),
                        [1, 1, h_v.shape[1]]),
                    tf.tile( # (n_bonds, n_mols, d_e)
                        tf.expand_dims(
                            h_v, # (n_bonds, d_e)
                            1),
                        [1, atom_in_mol.shape[1], 1])),
                axis=0)),

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

    # @tf.function
    def _call(
            self,
            atoms, # NOTE: here there could be more than one mol
            adjacency_map,
            atom_in_mol=False, # (n_atoms, )
            bond_in_mol=False, # (n_bonds, )
            batched_attr_mask=False,
            repeat=3):
        """ More general __call__ method.

        """

        # get the attributes of the molecule
        # adjacency_map = mol[1]
        # atoms = mol[0]
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

        if tf.logical_not(tf.reduce_any(atom_in_mol)):
            atom_in_mol = tf.tile(
                [[True]],
                [n_atoms, 1])

        if tf.logical_not(tf.reduce_any(bond_in_mol)):
            bond_in_mol = tf.tile(
                [[True]],
                [n_bonds, 1])

        if tf.logical_not(tf.reduce_all(batched_attr_mask)):
            batched_attr_mask = tf.constant([[True]])

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
        d_e = tf.shape(h_e, tf.int64)[1]

        # (n_atoms, ...)
        h_v = self.f_v(atoms)
        h_v_0 = h_v
        h_v_history = tf.expand_dims(h_v_0, 1)
        d_v = tf.shape(h_v, tf.int64)[1]

        # (n_mols, ...)
        # NOTE: here $h_u$ could have more than one first dimensions
        h_u = self.f_u(atoms, adjacency_map)
        h_u_0 = h_u
        h_u_history = tf.expand_dims(h_u_0, 1)
        d_u = tf.shape(h_u, tf.int64)[1]

        n_mols = tf.shape(h_u, tf.int64)[0]

        def propagate_one_time(
                iter_idx,
                h_e, h_v, h_u,
                h_e_history, h_v_history, h_u_history,
                atom_in_mol=atom_in_mol, # (n_atoms, n_mols)
                bond_in_mol=bond_in_mol # (n_bonds, n_mols)
            ):

            # update $ e'_k $
            # $$
            # e'_k = \phi^e (e_k, v_{rk}, v_{sk}, u)
            # $$

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
                tf.reduce_sum(
                    tf.boolean_mask(
                        tf.tile(
                            tf.expand_dims(
                                h_u, # (n_mols, d_u)
                                0), # (1, n_mols, d_u)
                            [h_e.shape[0], 1, 1]),
                        tf.boolean_mask(
                            bond_in_mol,
                            tf.reduce_any(
                                bond_in_mol,
                                axis=1),
                            axis=0)),
                    1,
                    keepdims=True))

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
            h_v = self.phi_v(
                h_v, # (n_atoms, d_v)
                h_v_0, # (n_atoms, d_v)
                h_e_bar_i, # (n_atoms, d_v)
                tf.reduce_sum(
                    tf.where(
                        tf.tile(
                            tf.expand_dims(
                                atom_in_mol,
                                2),
                            [1, 1, h_u.shape[1]]),
                        tf.tile(
                            tf.expand_dims(
                                h_u,
                                0),
                            [n_atoms, 1, 1]),
                        tf.zeros_like(
                            tf.tile(
                                tf.expand_dims(
                                    h_u,
                                    0),
                                [n_atoms, 1, 1]))),
                    axis=1))

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
            # (n_mols, d_e)
            h_e_bar = self.rho_e_u(h_e, bond_in_mol)

            # aggregate $ \bar{v'} $
            # $$
            # \bar{v'} = \rho^{v \rightarrow u} (V')
            # $$
            # (n_mols, d_v)
            h_v_bar = self.rho_v_u(h_v, atom_in_mol)

            # update $ u' $
            # $$
            # u' = \phi^u (\bar{e'}, \bar{v'}, u)
            # $$
            # (n_mols, d_u)
            h_u = self.phi_u(
                h_u,
                h_u_0,
                h_e_bar,
                h_v_bar)

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
                iter_idx, \
                h_e, h_v, h_u, h_e_history, \
                h_v_history, h_u_history:\
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
                iter_idx.get_shape(),
                h_e.get_shape(),
                h_v.get_shape(),
                h_u.get_shape(),
                tf.TensorShape((h_e.shape[0], None, h_e.shape[1])),
                tf.TensorShape((h_v.shape[0], None, h_v.shape[1])),
                tf.TensorShape((h_u.shape[0], None, h_u.shape[1]))
                ])

        y_bar = self.f_r(
            h_e, h_v, h_u,
            h_e_history, h_v_history, h_u_history,
            atom_in_mol, bond_in_mol)

        return y_bar

    @tf.function
    def _call_batch(
            self,
            mol, # NOTE: here there could be more than one mol
            atom_in_mol=None, # (n_atoms, )
            bond_in_mol=None, # (n_bonds, )
            repeat=3):
        raise NotImplementedError

    @staticmethod
    def batch(
            mols_with_attributes,
            inner_batch_size=128,
            outer_batch_size=None):
        """ Group molecules into batches.

        Parameters
        ----------
        mols : list
            molecules to be batched
        outer_batch_size : int
        inner_batch_size : int

        Returns
        -------
        batched_atoms
        batched_adjacency_map

        """

        # @tf.function # decorators not available in Alpha stage of tf 2.0
        def _batch(
            mols_with_attributes=mols_with_attributes,
            inner_batch_size=inner_batch_size,
            outer_batch_size=outer_batch_size):

            # init the sum of batch size
            atom_idx = tf.constant(0, dtype=tf.int64)
            bond_idx = tf.constant(0, dtype=tf.int64)
            mol_idx = tf.constant(0, dtype=tf.int64)

            batched_atoms_cache = tf.constant(
                -1,
                shape=[inner_batch_size,],
                dtype=tf.int64)

            batched_adjacency_map_cache = tf.constant(
                -1,
                shape=[inner_batch_size, inner_batch_size],
                dtype=tf.float32)

            batched_atom_in_mol_cache = tf.constant(
                False,
                shape=[inner_batch_size, inner_batch_size//4])

            batched_bond_in_mol_cache = tf.constant(
                False,
                shape=[
                        4 * inner_batch_size,
                        inner_batch_size // 4,
                    ]) # safe choice

            batched_attr_cache = tf.tile(
                tf.expand_dims(
                    tf.constant(-1, dtype=tf.float32),
                    0),
                [inner_batch_size])

            batched_attr_mask_cache = tf.tile(
                tf.expand_dims(
                    tf.constant(True),
                    0),
                [inner_batch_size])

            batched_atoms = tf.expand_dims(
                batched_atoms_cache, 0)

            batched_adjacency_map = tf.expand_dims(
                batched_adjacency_map_cache, 0)

            batched_atom_in_mol = tf.expand_dims(
                batched_atom_in_mol_cache, 0)

            batched_bond_in_mol= tf.expand_dims(
                batched_bond_in_mol_cache, 0)

            batched_attr = tf.expand_dims(
                batched_attr_cache, 0)

            batched_attr_mask = tf.expand_dims(
                batched_attr_mask_cache, 0)

            # loop through mols
            for atoms, adjacency_map, attr in mols_with_attributes:
                n_atoms = tf.shape(atoms, tf.int64)[0]
                n_bonds = tf.math.count_nonzero(adjacency_map)

                if tf.greater(
                    atom_idx + n_atoms,
                    inner_batch_size):

                    # put cache into variables
                    batched_atoms = tf.concat(
                        [
                            batched_atoms,
                            tf.expand_dims(batched_atoms_cache, 0)
                        ],
                        axis=0)

                    batched_adjacency_map = tf.concat(
                        [
                            batched_adjacency_map,
                            tf.expand_dims(batched_adjacency_map_cache, 0)
                        ],
                        axis=0)

                    batched_atom_in_mol = tf.concat(
                        [
                            batched_atom_in_mol,
                            tf.expand_dims(batched_atom_in_mol_cache, 0)
                        ],
                        axis=0)

                    batched_bond_in_mol = tf.concat(
                        [
                            batched_bond_in_mol,
                            tf.expand_dims(batched_bond_in_mol_cache, 0)
                        ],
                        axis=0)

                    batched_attr = tf.concat(
                        [
                            batched_attr,
                            tf.expand_dims(batched_attr_cache, 0)
                        ],
                        axis=0)

                    batched_attr_mask = tf.concat(
                        [
                            batched_attr_mask,
                            tf.expand_dims(batched_attr_mask_cache, 0)
                        ],
                        axis=0)

                    # use this one extra molecule to initiate the next batch
                    batched_atoms_cache = tf.concat(
                        [
                            atoms,
                            tf.tile(
                                tf.constant(
                                    -1,
                                    shape=(1,),
                                    dtype=tf.int64),
                                [inner_batch_size - n_atoms])
                        ],
                        axis=0)

                    batched_adjacency_map_cache = tf.pad(
                        adjacency_map,
                        [
                            [0, inner_batch_size - n_atoms],
                            [0, inner_batch_size - n_atoms]
                        ])

                    batched_atom_in_mol_cache = tf.pad(
                        tf.tile(
                            [[True]],
                            [n_atoms, 1]),
                        [
                            [0, inner_batch_size - n_atoms],
                            [0, inner_batch_size//4 - 1]
                        ],
                        constant_values=False)

                    batched_bond_in_mol_cache = tf.pad(
                        tf.tile(
                            [[True]],
                            [n_bonds, 1]),
                        [
                            [0, 4 * inner_batch_size - n_bonds],
                            [0, inner_batch_size//4 - 1]
                        ],
                        constant_values=False)

                    batched_attr_cache = tf.concat(
                        [
                            [attr],
                            tf.tile(
                                tf.constant(
                                    -1,
                                    shape=(1,),
                                    dtype=tf.float32),
                                [inner_batch_size - 1])
                        ],
                        axis=0)

                    batched_attr_mask_cache = tf.concat(
                        [
                            [True],
                            tf.tile(
                                tf.constant(
                                    False,
                                    shape=(1,)),
                            [inner_batch_size -1])
                        ],
                        axis=0)

                    # re-init counter
                    atom_idx = tf.constant(n_atoms, dtype=tf.int64)
                    bond_idx = tf.constant(n_bonds, dtype=tf.int64)
                    mol_idx = tf.constant(1, dtype=tf.int64)

                else:
                    # get the mask to update atoms and bonds
                    one_d_atom_mask = tf.logical_and(
                        tf.greater_equal(
                            tf.range(
                                inner_batch_size,
                                dtype=tf.int64),
                            atom_idx),
                        tf.less(
                            tf.range(
                                inner_batch_size,
                                dtype=tf.int64),
                            atom_idx + n_atoms))

                    two_d_atom_mask = tf.logical_and(
                        tf.tile(
                            tf.expand_dims(
                                one_d_atom_mask,
                                0),
                            [inner_batch_size, 1]),
                        tf.tile(
                            tf.expand_dims(
                                one_d_atom_mask,
                                1),
                            [1, inner_batch_size]))

                    batched_atoms_cache = tf.where(
                        # cond
                        one_d_atom_mask,

                        # where True
                        tf.concat(
                            [
                                tf.tile(
                                    [tf.constant(-1, dtype=tf.int64)],
                                    [atom_idx]),
                                atoms,
                                tf.tile(
                                    [tf.constant(-1, dtype=tf.int64)],
                                    [inner_batch_size - atom_idx - n_atoms])
                            ],
                            axis=0),

                        # where False
                        batched_atoms_cache)

                    batched_adjacency_map_cache = tf.where(
                        # cond
                        two_d_atom_mask,

                        # where True
                        tf.pad(
                            adjacency_map,
                            [
                                [
                                    atom_idx,
                                    inner_batch_size - atom_idx - n_atoms
                                ],
                                [
                                    atom_idx,
                                    inner_batch_size - atom_idx - n_atoms
                                ]
                            ],
                            constant_values=0),

                        # where False
                        batched_adjacency_map_cache)

                    batched_atom_in_mol_cache = tf.logical_or(
                        tf.pad(
                            tf.tile(
                                [[True]],
                                [n_atoms, 1]),
                            [
                                [
                                    atom_idx,
                                    inner_batch_size - atom_idx - n_atoms
                                ],
                                [
                                    mol_idx,
                                    inner_batch_size//4 - mol_idx - 1
                                ]
                            ],
                            constant_values=False),
                        batched_atom_in_mol_cache)

                    batched_bond_in_mol_cache = tf.logical_or(
                        tf.pad(
                            tf.tile(
                                [[True]],
                                [n_bonds, 1]),
                            [
                                [
                                    bond_idx,
                                    4 * inner_batch_size - bond_idx - n_bonds
                                ],
                                [
                                    mol_idx,
                                    inner_batch_size//4 - mol_idx - 1
                                ]
                            ],
                            constant_values=False),
                        batched_bond_in_mol_cache)

                    batched_attr_cache = tf.where(
                        tf.equal(
                            tf.range(
                                inner_batch_size,
                                dtype=tf.int64),
                            mol_idx),
                        tf.ones(
                            (inner_batch_size,),
                            dtype=tf.float32),
                        batched_attr_cache)

                    batched_attr_mask_cache = tf.logical_or(
                        batched_attr_mask_cache,
                        tf.equal(
                            tf.range(
                                inner_batch_size,
                                dtype=tf.int64),
                            mol_idx))

                    atom_idx = atom_idx + n_atoms
                    bond_idx = bond_idx + n_bonds
                    mol_idx = mol_idx + 1

                    assert tf.equal(
                        bond_idx,
                        tf.math.count_nonzero(batched_bond_in_mol_cache))

            return (
                batched_atoms,
                batched_adjacency_map,
                batched_atom_in_mol,
                batched_bond_in_mol,
                batched_attr,
                batched_attr_mask)

        inner_ds = tf.data.Dataset.from_tensor_slices(_batch())
        inner_ds = inner_ds.skip(1)

        if type(outer_batch_size) == type(None):
            return inner_ds

        else:
            outer_ds = inner_ds.batch(outer_batch_size, True)
            return outer_ds

    def call(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def switch(self, to_test=True):
        for fn in [self.rho_e_u, self.rho_e_v, self.rho_v_u]:
            if hasattr(fn, 'switch'):
                fn.switch()
