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

# =============================================================================
# utility functions
# =============================================================================

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
                        [1, 1, tf.shape(h_e)[1]]),
                    tf.tile( # (n_bonds, n_mols, d_e)
                        tf.expand_dims(
                            h_e, # (n_bonds, d_e)
                            1),
                        [1, tf.shape(bond_in_mol)[1], 1])),
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
                        [1, 1, tf.shape(h_v)[1]]),
                    tf.tile( # (n_bonds, n_mols, d_e)
                        tf.expand_dims(
                            h_v, # (n_bonds, d_e)
                            1),
                        [1, tf.shape(atom_in_mol)[1], 1])),
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

    @tf.function
    def _call(
            self,
            atoms, # NOTE: here there could be more than one mol
            adjacency_map,
            atom_in_mol=False, # (n_atoms, )
            bond_in_mol=False, # (n_bonds, )
            batched_attr_in_mol=False,
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

        # grab atoms that are at the two ends of a bond
        # (n_bonds, 2)
        left_idxs = bond_idxs[:, 0]

        # (n_bonds, 2)
        right_idxs = bond_idxs[:, 1]

        n_bonds = tf.cast(tf.shape(bond_idxs)[0], tf.int64)

        if tf.logical_not(tf.reduce_any(atom_in_mol)):
            atom_in_mol = tf.tile(
                [[True]],
                [n_atoms, 1])

        if tf.logical_not(tf.reduce_any(bond_in_mol)):
            bond_in_mol = tf.tile(
                [[True]],
                [n_bonds, 1])

        if tf.logical_not(tf.reduce_any(batched_attr_in_mol)):
            batched_attr_in_mol = tf.constant([[True]])

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
        h_u = self.f_u(atoms, adjacency_map, batched_attr_in_mol)
        h_u_0 = h_u
        h_u_history = tf.expand_dims(h_u_0, 1)
        d_u = tf.shape(h_u, tf.int64)[1]
        n_mols = tf.shape(h_u, tf.int64)[0]

        # specify what we know about the shape of the mask
        bond_in_mol.set_shape([None, None])
        atom_in_mol.set_shape([None, None])

        # trim the extra `False` from `bond_in_mol` and from `atom_in_mol`
        bond_in_mol = tf.boolean_mask(
            bond_in_mol,
            tf.reduce_any(
                bond_in_mol,
                axis=0),
            axis=1)

        bond_in_mol = tf.boolean_mask(
            bond_in_mol,
            tf.reduce_any(
                bond_in_mol,
                axis=1),
            axis=0)

        atom_in_mol = tf.boolean_mask(
            atom_in_mol,
            tf.reduce_any(
                atom_in_mol,
                axis=0),
        axis=1)

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

            h_left = tf.gather(
                h_v,
                left_idxs)

            h_right = tf.gather(
                h_v,
                right_idxs)

            # (n_bonds, d_e)
            h_e = self.phi_e(h_e, h_e_0, h_left, h_right,
                tf.reduce_sum(
                    tf.boolean_mask(
                        tf.tile(
                            tf.expand_dims(
                                h_u, # (n_mols, d_u)
                                0), # (1, n_mols, d_u)
                            [tf.shape(h_e)[0], 1, 1]),
                        bond_in_mol),
                    axis=1,
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
                            [1, 1, tf.shape(h_u)[1]]),
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

        propagate_one_time(
            iter_idx,
            h_e, h_v, h_u,
            h_e_history, h_v_history, h_u_history)

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
                tf.TensorShape((None, None, None)),
                tf.TensorShape((None, None, None)),
                tf.TensorShape((None, None, None)),
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

    # TODO: need testing
    @staticmethod
    @tf.function
    def batch(
            mols_with_attributes,
            inner_batch_size=128,
            outer_batch_size=None,
            per_atom_attr=False,
            feature_dimension=0):
        """ Group molecules into batches.

        Parameters
        ----------
        mols_with_attributes : list
            molecules to be batched
        inner_batch_size : int
        outer_batch_size : int
        feature_dimension : int


        Returns
        -------
        batched_atoms
        batched_adjacency_map

        """

        # convert everything to tensor
        inner_batch_size = tf.convert_to_tensor(
            inner_batch_size,
            tf.int64)

        feature_dimension = tf.convert_to_tensor(
            feature_dimension,
            tf.int64)

        # define max number of molecules, to which all rows and columns
        # associated with molecules are padded
        max_n_mols = tf.math.floordiv(
            inner_batch_size,
            tf.constant(4, dtype=tf.int64))

        # and max number of bonds, to which all rows and columns
        # associated with bonds are padded
        max_n_bonds = tf.multiply(
            inner_batch_size,
            tf.constant(4, dtype=tf.int64))

        # we use state to denote count
        # we use a tensor with
        # shape = (2, )
        # dtype = int64
        # to record the state of the scan--
        # (key, count) pair
        state = tf.constant(
            [
                0, # key
                0 # count
            ],
            dtype=tf.int64)

        # we start by defining a scan function
        @tf.function
        def scan_func(state, input_element):
            # read key and count from the state
            key = state[0]
            count = state[1]

            # uplack
            atom, adjacency_map, attr = input_element

            # shape = (), dtype = tf.int64
            n_atoms = tf.shape(atom, tf.int64)[0]

            if tf.less_equal(
                tf.add(
                    count,
                    n_atoms),
                inner_batch_size):

                output_element = (atom, adjacency_map, attr, key)
                count = tf.add(count, n_atoms)

            else:
                key = tf.add(
                    key,
                    tf.constant(1, dtype=tf.int64))

                count = n_atoms

                output_element = (atom, adjacency_map, attr, key)

            state = tf.concat(
                [
                    tf.expand_dims(key, 0),
                    tf.expand_dims(count, 0)
                ],
                axis=0)

            return state, output_element

        # define the key func: grab the last element in an entry
        key_func = lambda atom, adjacency_map, attr, key: key

        @tf.function
        def init_func(key):
            if tf.equal(
                feature_dimension,
                tf.constant(0, dtype=tf.int64)):

                # initialize atoms as all -1
                # shape = (inner_batch_size, )
                # dtype = int64
                atoms = tf.multiply(
                    tf.ones(
                        (inner_batch_size, ),
                        dtype=tf.int64),
                    tf.constant(-1, dtype=tf.int64))

            else:
                # dtype = float32
                # shape = (inner_batch_size, feature_dimension)
                atoms = tf.multiply(
                    tf.ones(
                        (inner_batch_size, feature_dimension),
                        dtype=tf.int64),
                    tf.constant(-1, dtype=tf.int64))

            # initialize adjacency_map as all 0
            # shape = (inner_batch_size, inner_batch_size)
            # dtype = float32
            adjacency_map = tf.zeros(
                (
                    inner_batch_size,
                    inner_batch_size
                ),
                dtype=tf.float32)

            # shape = (inner_batch_size, max_n_mols)
            # dtype = Boolean
            atom_in_mol = tf.tile(
                [[False]],
                [inner_batch_size, max_n_mols])

            # shape = (max_n_bonds, max_n_mols)
            # dtype = Boolean
            bond_in_mol = tf.tile(
                [[False]],
                [max_n_bonds, max_n_mols])

            if per_atom_attr:
                # initialize attr like atoms
                # dtype = float32,
                # shape = (inner_batch_size, )
                attr = tf.multiply(
                    tf.ones(
                        (inner_batch_size, ),
                        dtype=tf.float32),
                    tf.constant(-1, dtype=tf.float32))

                # dtype = Boolean,
                # shape = (inner_batch_size, )
                attr_in_mol = tf.tile(
                    [[False]],
                    [inner_batch_size, max_n_mols])

            else:
                # initialize attr with number of molecules
                # dtype = float32,
                # shape = (max_n_mols, )
                attr = tf.multiply(
                    tf.ones(
                        (max_n_mols, ),
                        dtype=tf.float32),
                    tf.constant(-1, dtype=tf.float32))

                # dtype = Boolean
                # shape = (max_n_mols, max_n_mols)
                attr_in_mol = tf.tile(
                    [False],
                    [max_n_mols])

            # initialize atom and bond count
            # shape = (),
            # dtype = int64
            atom_count = tf.constant(0, dtype=tf.int64)
            bond_count = tf.constant(0, dtype=tf.int64)
            mol_count = tf.constant(0, dtype=tf.int64)

            return (
                atoms,
                adjacency_map,
                attr,
                atom_in_mol,
                bond_in_mol,
                attr_in_mol,
                atom_count,
                bond_count,
                mol_count)

        @tf.function
        def reduce_func(old_state, input):
            # grab all tensors from the batched state
            (
                atoms,
                adjacency_map,
                attr,
                atom_in_mol,
                bond_in_mol,
                attr_in_mol,
                atom_count,
                bond_count,
                mol_count
            ) = old_state

            # grab all tensors from input
            (
                _atoms,
                _adjacency_map,
                _attr,
                _key
            ) = input

            # get the count of atoms and bonds
            n_atoms = tf.shape(_atoms, tf.int64)[0]
            n_bonds = tf.math.count_nonzero(
                _adjacency_map)

            # TODO:
            # need testing, this might or might not support broadcasting
            # put local attributes into batched ones
            atoms = tf.tensor_scatter_nd_update(
                atoms,

                # idxs
                tf.expand_dims(
                    tf.range(
                        start=atom_count,
                        limit=tf.add(
                            atom_count,
                            n_atoms),
                        dtype=tf.int64),
                    axis=1),

                # update
                _atoms)

            # modify the masks
            atom_in_mol = tf.tensor_scatter_nd_update(
                atom_in_mol,

                # idxs
                tf.concat(
                    [
                        tf.expand_dims(
                            tf.range(
                                start=atom_count,
                                limit=tf.add(
                                    atom_count,
                                    n_atoms),
                                dtype=tf.int64),
                            axis=1),
                        tf.multiply(
                            mol_count,
                            tf.ones(
                                (n_atoms, 1),
                                dtype=tf.int64)),
                    ],
                    axis=1),

                # update
                tf.tile(
                    [True],
                    [n_atoms]))

            bond_in_mol = tf.tensor_scatter_nd_update(
                bond_in_mol,

                # idxs
                tf.concat(
                    [
                        tf.expand_dims(
                            tf.range(
                                start=bond_count,
                                limit=tf.add(
                                    bond_count,
                                    n_bonds),
                                dtype=tf.int64),
                            axis=1),
                        tf.multiply(
                            mol_count,
                            tf.ones(
                                (n_bonds, 1),
                                dtype=tf.int64)),
                    ],
                    axis=1),

                # update
                tf.tile(
                    [True],
                    [n_bonds]))

            adjacency_map = tf.tensor_scatter_nd_update(
                adjacency_map,

                # idxs
                tf.reshape(
                    tf.stack(
                        tf.meshgrid(
                            tf.range(
                                start=atom_count,
                                limit= tf.add(
                                    atom_count,
                                    n_atoms),
                                dtype=tf.int64),
                            tf.range(
                                start=atom_count,
                                limit= tf.add(
                                    atom_count,
                                    n_atoms),
                                dtype=tf.int64)),
                        axis=2),
                    [-1, 2]),

                # update
                tf.reshape(
                    tf.transpose(
                        _adjacency_map),
                    [-1]))

            if per_atom_attr:
                attr = tf.tensor_scatter_nd_update(
                    attr,

                    # idxs
                    tf.expand_dims(
                        tf.range(
                            start=atom_count,
                            limit=tf.add(
                                atom_count,
                                n_atoms),
                            dtype=tf.int64),
                        axis=1),

                    # update
                    _attr)

                attr_in_mol = tf.tensor_scatter_nd_update(
                    attr_in_mol,

                    # idxs
                    tf.concat(
                        [
                            tf.expand_dims(
                                tf.range(
                                    start=atom_count,
                                    limit=tf.add(
                                        atom_count,
                                        n_atoms),
                                    dtype=tf.int64),
                                axis=1),
                            tf.multiply(
                                mol_count,
                                tf.ones(
                                    (n_atoms, 1),
                                    dtype=tf.int64)),
                        ],
                        axis=1),

                    # update
                    tf.tile(
                        [True],
                        [n_atoms]))


            else:
                attr = tf.tensor_scatter_nd_update(
                    attr,

                    # idxs
                    tf.expand_dims(
                        tf.expand_dims(
                            mol_count,
                            axis=0),
                        axis=1),

                    tf.expand_dims(_attr, 0))

                attr_in_mol = tf.logical_or(
                    attr_in_mol,
                    tf.equal(
                        tf.range(max_n_mols),
                        mol_count))

            # update all the counts
            atom_count = tf.add(
                atom_count,
                n_atoms)

            bond_count = tf.add(
                bond_count,
                n_bonds)

            mol_count = tf.add(
                mol_count,
                tf.constant(1, dtype=tf.int64))

            return (
                atoms,
                adjacency_map,
                attr,
                atom_in_mol,
                bond_in_mol,
                attr_in_mol,
                atom_count,
                bond_count,
                mol_count
            )

        @tf.function
        def finalize_func(*state):
            (
                atoms,
                adjacency_map,
                attr,
                atom_in_mol,
                bond_in_mol,
                attr_in_mol,
                atom_count,
                bond_count,
                mol_count
            ) = state

            result = (
                atoms,
                adjacency_map,
                atom_in_mol,
                bond_in_mol,
                attr,
                attr_in_mol)

            return result

        # now we construct the reducer
        reducer = tf.data.experimental.Reducer(
            init_func=init_func,
            reduce_func=reduce_func,
            finalize_func=finalize_func)

        return mols_with_attributes.apply(
            tf.data.experimental.scan(
                state,
                scan_func)).apply(
            tf.data.experimental.group_by_reducer(
                key_func,
                reducer))

    @staticmethod
    @tf.function
    def get_number_batches(ds):
        """ Get the number of batches before they are put into dataset.
        """
        count = tf.constant(0, dtype=tf.int64)
        for sample in ds:
            count = tf.add(
                count,
                tf.constant(1, dtype=tf.int64))

        return count

    @staticmethod
    def batch_old(
            mols_with_attributes,
            inner_batch_size=128,
            outer_batch_size=None,
            per_atom_attr=False,
            feature_dimension=0):
        """ Group molecules into batches.

        Parameters
        ----------
        mols_with_attributes : list
            molecules to be batched
        inner_batch_size : int
        outer_batch_size : int
        feature_dimension : int


        Returns
        -------
        batched_atoms
        batched_adjacency_map

        """
        def _batch(
            mols_with_attributes,
            inner_batch_size=inner_batch_size,
            per_atom_attr=per_atom_attr):

            global n_batched_samples_total

            # init the sum of batch size
            atom_idx = tf.constant(0, dtype=tf.int64)
            bond_idx = tf.constant(0, dtype=tf.int64)
            mol_idx = tf.constant(0, dtype=tf.int64)

            if feature_dimension == 0:
                batched_atoms_cache = tf.constant(
                    -1,
                    shape=[inner_batch_size,],
                    dtype=tf.int64)

            else:
                batched_atoms_cache = tf.constant(
                    -1,
                    shape=[inner_batch_size, feature_dimension],
                    dtype=tf.float32)

            batched_adjacency_map_cache = tf.constant(
                0,
                shape=[inner_batch_size, inner_batch_size],
                dtype=tf.float32)

            batched_atom_in_mol_cache = tf.constant(
                False,
                shape=[inner_batch_size, inner_batch_size//4])

            batched_bond_in_mol_cache = tf.constant(
                False,
                shape=[
                        2 * inner_batch_size,
                        inner_batch_size // 4,
                    ]) # safe choice

            if per_atom_attr:
                batched_attr_cache = tf.constant(
                    -1,
                    shape=[inner_batch_size,],
                    dtype=tf.float32)

                batched_attr_in_mol_cache = tf.constant(
                    False,
                    shape=[inner_batch_size, inner_batch_size//4])

            else:
                batched_attr_cache = tf.tile(
                    tf.expand_dims(
                        tf.constant(-1, dtype=tf.float32),
                        0),
                    [inner_batch_size // 4])

                batched_attr_in_mol_cache = tf.tile(
                    tf.expand_dims(
                        tf.constant(False),
                        0),
                    [inner_batch_size // 4])

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

            batched_attr_in_mol = tf.expand_dims(
                batched_attr_in_mol_cache, 0)

            # loop through mols
            for atoms, adjacency_map, attr in mols_with_attributes:
                n_atoms = tf.shape(atoms, tf.int64)[0]
                n_bonds = tf.math.count_nonzero(adjacency_map)

                # NOTE:
                # here we exclude the single-atom molecule from our
                # dataset
                if tf.equal(
                    n_bonds,
                    tf.constant(0, dtype=tf.int64)):
                    continue

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

                    batched_attr_in_mol = tf.concat(
                        [
                            batched_attr_in_mol,
                            tf.expand_dims(batched_attr_in_mol_cache, 0)
                        ],
                        axis=0)

                    # use this one extra molecule to initiate the next batch
                    if feature_dimension == 0:
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

                    else:

                        batched_atoms_cache = tf.concat(
                            [
                                atoms,
                                tf.tile(
                                    tf.constant(
                                        -1,
                                        shape=(1, feature_dimension),
                                        dtype=tf.float32
                                    ),
                                    [inner_batch_size - n_atoms, 1])
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
                            [0, 2 * inner_batch_size - n_bonds],
                            [0, inner_batch_size//4 - 1]
                        ],
                        constant_values=False)

                    if per_atom_attr:
                        batched_attr_cache = tf.concat(
                            [
                                attr,
                                tf.tile(
                                    tf.constant(
                                        -1,
                                        shape=(1,),
                                        dtype=tf.float32),
                                    [inner_batch_size - n_atoms])
                            ],
                            axis=0)

                        batched_attr_in_mol_cache = tf.pad(
                            tf.tile(
                                [[True]],
                                [n_atoms, 1]),
                            [
                                [0, inner_batch_size - n_atoms],
                                [0, inner_batch_size//4 - 1]
                            ],
                            constant_values=False)

                    else:
                            batched_attr_cache = tf.concat(
                                [
                                    [attr],
                                    tf.tile(
                                        tf.constant(
                                            -1,
                                            shape=(1,),
                                            dtype=tf.float32),
                                        [inner_batch_size // 4 - 1])
                                ],
                                axis=0)

                            batched_attr_in_mol_cache = tf.concat(
                                [
                                    [True],
                                    tf.tile(
                                        tf.constant(
                                            False,
                                            shape=(1,)),
                                    [inner_batch_size // 4 -1])
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

                    if feature_dimension == 0:
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

                    else:
                        batched_atoms_cache = tf.where(
                            # cond
                            tf.tile(
                                tf.expand_dims(
                                    one_d_atom_mask,
                                    1),
                                [1, feature_dimension]),

                            # where True
                            tf.concat(
                                [
                                    tf.tile(
                                        tf.constant(
                                            -1,
                                            shape=(1, feature_dimension),
                                            dtype=tf.float32),
                                        [atom_idx, 1]),
                                    atoms,
                                    tf.tile(
                                        tf.constant(
                                            -1,
                                            shape=(1, feature_dimension),
                                            dtype=tf.float32),
                                        [inner_batch_size-atom_idx-n_atoms, 1])
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
                                    2 * inner_batch_size - bond_idx - n_bonds
                                ],
                                [
                                    mol_idx,
                                    inner_batch_size//4 - mol_idx - 1
                                ]
                            ],
                            constant_values=False),
                        batched_bond_in_mol_cache)

                    if per_atom_attr:
                        batched_attr_cache = tf.where(
                            # cond
                            one_d_atom_mask,

                            # where True
                            tf.concat(
                                [
                                    tf.tile(
                                        [tf.constant(-1, dtype=tf.float32)],
                                        [atom_idx]),
                                    attr,
                                    tf.tile(
                                        [tf.constant(-1, dtype=tf.float32)],
                                        [inner_batch_size - atom_idx - n_atoms])
                                ],
                                axis=0),

                            # where False
                            batched_attr_cache)

                        batched_attr_in_mol = tf.logical_or(
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
                            batched_attr_in_mol)

                    else:
                        batched_attr_cache = tf.where(
                            tf.equal(
                                tf.range(
                                    inner_batch_size // 4,
                                    dtype=tf.int64),
                                mol_idx),
                            attr * tf.ones(
                                (inner_batch_size // 4,),
                                dtype=tf.float32),
                            batched_attr_cache)

                        batched_attr_in_mol_cache = tf.logical_or(
                            batched_attr_in_mol_cache,
                            tf.equal(
                                tf.range(
                                    inner_batch_size // 4,
                                    dtype=tf.int64),
                                mol_idx))

                    atom_idx = atom_idx + n_atoms
                    bond_idx = bond_idx + n_bonds
                    mol_idx = mol_idx + 1

            inner_ds = tf.data.Dataset.from_tensor_slices(
                (
                    batched_atoms,
                    batched_adjacency_map,
                    batched_atom_in_mol,
                    batched_bond_in_mol,
                    batched_attr,
                    batched_attr_in_mol
                ))

            inner_ds = inner_ds.skip(1)
            n_batched_samples_total = tf.shape(
                batched_atoms,
                tf.int64)[0]

            return inner_ds

        inner_ds = mols_with_attributes.apply(_batch)

        if type(outer_batch_size) == type(None):
            return inner_ds, n_batched_samples_total

        else:
            outer_ds = inner_ds.batch(outer_batch_size, True)
            return outer_ds, n_batched_samples_total

    def call(self, *args, **kwargs):

        return self._call(*args, **kwargs)

    def switch(self, to_test=True):
        for fn in [self.rho_e_u, self.rho_e_v, self.rho_v_u]:
            if hasattr(fn, 'switch'):
                fn.switch()
