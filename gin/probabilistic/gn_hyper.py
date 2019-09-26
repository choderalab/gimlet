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
import gin

# =============================================================================
# utility functions
# =============================================================================
@tf.function
def get_geometric_idxs(atoms, adjacency_map):
    """ Find the bond, angles, and torsion indices in a molecular graph or
    graphs.

    Parameters
    ----------
    atoms : tf.Tensor, dtype=tf.int64,
        a tensor denoting the sequence of type of atoms
    adjacency_map : tf.Tensor, dtype=tf.int64,
        upper triangular tensor representing the adjacency map of the molecules

    Returns
    -------
    bond_idxs
    angle_idxs
    torsion_idxs
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

    # init the angles idxs to be all negative ones
    angle_idxs = tf.constant([[-1, -1, -1]], dtype=tf.int64)

    @tf.function
    def process_one_atom_if_there_is_angle(idx, angle_idxs,
            adjacency_map_full=adjacency_map_full):

        # get all the connection indices
        connection_idxs = tf.where(
            tf.greater(
                adjacency_map_full[idx, :],
                tf.constant(0, dtype=tf.float32)))

        # get the number of connections
        n_connections = tf.shape(connection_idxs)[0]

        # get the combinations from these connection indices
        connection_combinations = tf.gather_nd(
            tf.stack(
                tf.meshgrid(
                    connection_idxs,
                    connection_idxs),
                axis=2),
            tf.where(
                tf.greater(
                    tf.linalg.band_part(
                        tf.ones(
                            (
                                n_connections,
                                n_connections
                            ),
                            dtype=tf.int64),
                        0, -1),
                    tf.constant(0, dtype=tf.int64))))

        connection_combinations = tf.boolean_mask(
            connection_combinations,
            tf.greater(
                connection_combinations[:, 0] \
                 - connection_combinations[:, 1],
                tf.constant(0, dtype=tf.int64)))

        angle_idxs = tf.concat(
            [
                angle_idxs,
                tf.concat(
                    [
                        tf.expand_dims(
                            connection_combinations[:, 0],
                            1),
                        tf.expand_dims(
                            idx * tf.ones(
                                (tf.shape(connection_combinations)[0], ),
                                dtype=tf.int64),
                            1),
                        tf.expand_dims(
                            connection_combinations[:, 1],
                            1)
                    ],
                    axis=1)
            ],
            axis=0)

        return idx + 1, angle_idxs

    @tf.function
    def process_one_atom(idx, angle_idxs,
            adjacency_map_full=adjacency_map_full):

        if tf.less(
            tf.math.count_nonzero(adjacency_map_full[idx, :]),
            tf.constant(1, dtype=tf.int64)):
            return idx+1, angle_idxs

        else:
            return process_one_atom_if_there_is_angle(idx, angle_idxs)

    idx = tf.constant(0, dtype=tf.int64)
    # use while loop to update the indices forming the angles
    idx, angle_idxs = tf.while_loop(
        # condition
        lambda idx, angle_idxs: tf.less(idx, n_atoms),

        process_one_atom,

        [idx, angle_idxs],

        shape_invariants=[
            idx.get_shape(),
            tf.TensorShape((None, 3))])

    # discard the first row
    angle_idxs = angle_idxs[1:, ]

    n_angles = tf.shape(angle_idxs, tf.int64)[0]

    # init the torsion idxs to be all negative ones
    torsion_idxs = tf.constant([[-1, -1, -1, -1]], dtype=tf.int64)

    # for each bond, there is at least one torsion terms associated
    def process_one_bond_if_there_is_torsion(idx, torsion_idxs):
        bond = bond_idxs[idx]
        left_atom_connections = tf.where(
            tf.greater(
                adjacency_map_full[bond[0]],
                tf.constant(0, dtype=tf.float32)))

        right_atom_connections = tf.where(
            tf.greater(
                adjacency_map_full[bond[1]],
                tf.constant(0, dtype=tf.float32)))

        # get the combinations from these connection indices
        connection_combinations = tf.reshape(
            tf.stack(
                tf.meshgrid(
                    left_atom_connections,
                    right_atom_connections),
                axis=2),
            [-1, 2])

        torsion_idxs = tf.concat(
            [
                torsion_idxs,
                tf.concat(
                    [
                        tf.expand_dims(
                            connection_combinations[:, 0],
                            1),
                        bond[0] * tf.ones(
                            (tf.shape(connection_combinations)[0], 1),
                            dtype=tf.int64),
                        bond[1] * tf.ones(
                            (tf.shape(connection_combinations)[0], 1),
                            dtype=tf.int64),
                        tf.expand_dims(
                            connection_combinations[:, 1],
                            1)
                    ],
                    axis=1)
            ],
            axis=0)

        return idx + 1, torsion_idxs

    def process_one_bond(idx, torsion_idxs):
        if tf.logical_not(
            tf.logical_and(
                tf.greater(
                    tf.math.count_nonzero(
                        adjacency_map_full[bond_idxs[idx][0]]),
                    tf.constant(1, dtype=tf.int64)),
                tf.greater(
                    tf.math.count_nonzero(
                        adjacency_map_full[bond_idxs[idx][1]]),
                    tf.constant(1, dtype=tf.int64)))):
            return idx + 1, torsion_idxs

        else:
            return process_one_bond_if_there_is_torsion(
                idx, torsion_idxs)


    idx = tf.constant(0, dtype=tf.int64)
    idx, torsion_idxs = tf.while_loop(
        # condition
        lambda idx, _: tf.less(idx, tf.shape(bond_idxs, tf.int64)[0]),

        # body
        process_one_bond,

        # vars
        [idx, torsion_idxs],

        shape_invariants=[
            idx.get_shape(),
            tf.TensorShape([None, 4])
            ])

    # get rid of the first one
    torsion_idxs = torsion_idxs[1:, ]

    torsion_idxs = tf.boolean_mask(
        torsion_idxs,
        tf.logical_and(
            tf.logical_not(
                tf.equal(
                    torsion_idxs[:, 0] - torsion_idxs[:, 2],
                    tf.constant(0, dtype=tf.int64))),
            tf.logical_not(
                tf.equal(
                    torsion_idxs[:, 1] - torsion_idxs[:, 3],
                    tf.constant(0, dtype=tf.int64)))))

    return bond_idxs, angle_idxs, torsion_idxs

# =============================================================================
# module classes
# =============================================================================
class HyperGraphNet(tf.keras.Model):
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

    phi_a : function,
        applied per angle, with arguments $(v_0, v_1, v_2, u)$,
        and returns $a_k'$
    phi_t : function,
        applied per dihedral, with arguments $(v_1, v_2, v_3, v_4, u)$
        and returns $d'$.
    rho_a_u : function,
        applied to $A'$, and aggregates all angle updates, into $\bar{a'}$,
        which will then be used in next step's global update.
    rho_t_u : function,
        applied to $'D'$, and aggregates all dihedral updates, into $\bar{d'}$,
        which will then be used in next step's gloabl update.

    attention_k : function,
        applied to all atoms for dot-product attention.
    attention_q : function,
        applied to all atoms for dot-product attention.
    attention_v : function,
        applied to distance matrix for dot-product attention.

    """

    def __init__(
            self,

            # building blocks for GCN
            # update function
            phi_e=lambda *x: x[0],
            phi_u=lambda *x: x[0],
            phi_v=lambda *x: x[0],

            phi_a=lambda *x: x[0],
            phi_t=lambda *x: x[0],

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

            rho_a_u=(lambda h_a, angle_in_mol: tf.reduce_sum(
                tf.multiply(
                    tf.tile(
                        tf.expand_dims(
                            tf.where( # (n_bonds, n_mols)
                                tf.boolean_mask(
                                    angle_in_mol,
                                    tf.reduce_any(
                                        angle_in_mol,
                                        axis=1),
                                    axis=0),
                                tf.ones_like(
                                    tf.boolean_mask(
                                        angle_in_mol,
                                        tf.reduce_any(
                                            angle_in_mol,
                                            axis=1),
                                        axis=0),
                                    dtype=tf.float32),
                                tf.zeros_like(
                                    tf.boolean_mask(
                                        angle_in_mol,
                                        tf.reduce_any(
                                            angle_in_mol,
                                            axis=1),
                                        axis=0),
                                    dtype=tf.float32)),
                            2),
                        [1, 1, tf.shape(h_a)[1]]),
                    tf.tile( # (n_bonds, n_mols, d_e)
                        tf.expand_dims(
                            h_a, # (n_bonds, d_e)
                            1),
                        [1, tf.shape(angle_in_mol)[1], 1])),
                axis=0)),

            rho_t_u=(lambda h_d, diheral_in_mol: tf.reduce_sum(
                tf.multiply(
                    tf.tile(
                        tf.expand_dims(
                            tf.where( # (n_bonds, n_mols)
                                tf.boolean_mask(
                                    diheral_in_mol,
                                    tf.reduce_any(
                                        diheral_in_mol,
                                        axis=1),
                                    axis=0),
                                tf.ones_like(
                                    tf.boolean_mask(
                                        diheral_in_mol,
                                        tf.reduce_any(
                                            diheral_in_mol,
                                            axis=1),
                                        axis=0),
                                    dtype=tf.float32),
                                tf.zeros_like(
                                    tf.boolean_mask(
                                        diheral_in_mol,
                                        tf.reduce_any(
                                            diheral_in_mol,
                                            axis=1),
                                        axis=0),
                                    dtype=tf.float32)),
                            2),
                        [1, 1, tf.shape(h_d)[1]]),
                    tf.tile( # (n_bonds, n_mols, d_e)
                        tf.expand_dims(
                            h_d, # (n_bonds, d_e)
                            1),
                        [1, tf.shape(diheral_in_mol)[1], 1])),
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
            f_a=lambda x:x,
            f_t=lambda x:x,
            f_all=lambda x:x,

            repeat=3):

        super(HyperGraphNet, self).__init__()
        self.phi_e = phi_e
        self.rho_e_v = rho_e_v
        self.phi_v = phi_v
        self.rho_e_u = rho_e_u
        self.rho_v_u = rho_v_u
        self.rho_a_u = rho_a_u
        self.rho_t_u = rho_t_u
        self.phi_a = phi_a
        self.phi_t = phi_t
        self.phi_u = phi_u
        self.f_r = f_r
        self.f_e = f_e
        self.f_v = f_v
        self.f_u = f_u
        self.f_a = f_a
        self.f_t = f_t
        self.f_all = f_all
        self.repeat = repeat

    @tf.function
    def _call(
            self,
            atoms, # NOTE: here there could be more than one mol
            adjacency_map,
            coordinates,
            atom_in_mol=False, # (n_atoms, )
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

        bond_idxs, angle_idxs, torsion_idxs = get_geometric_idxs(
            atoms, adjacency_map)

        # get the dimensinos of the indices
        n_atoms = tf.shape(atoms, tf.int64)[0]
        n_bonds = tf.shape(bond_idxs, tf.int64)[0]
        n_angles = tf.shape(angle_idxs, tf.int64)[0]
        n_torsions = tf.shape(torsion_idxs, tf.int64)[0]

        # grab atoms that are at the two ends of a bond
        # (n_bonds, 2)
        left_idxs = bond_idxs[:, 0]

        # (n_bonds, 2)
        right_idxs = bond_idxs[:, 1]

        if tf.logical_not(tf.reduce_any(atom_in_mol)):
            atom_in_mol = tf.tile(
                [[True]],
                [n_atoms, 1])

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


        # (n_angles, n_atoms)
        angle_is_connected_to_atoms = tf.reduce_any(
            [
                tf.equal(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(n_atoms),
                            0),
                        [n_angles, 1]),
                    tf.tile(
                        tf.expand_dims(
                            angle_idxs[:, 0],
                            1),
                        [1, n_atoms])),
                tf.equal(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(n_atoms),
                            0),
                        [n_angles, 1]),
                    tf.tile(
                        tf.expand_dims(
                            angle_idxs[:, 1],
                            1),
                        [1, n_atoms])),
                tf.equal(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(n_atoms),
                            0),
                        [n_angles, 1]),
                    tf.tile(
                        tf.expand_dims(
                            angle_idxs[:, 2],
                            1),
                        [1, n_atoms]))
            ],
            axis=0)

        # (n_torsions, n_atoms)
        torsion_is_connected_to_atoms = tf.reduce_any(
            [
                tf.equal(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(n_atoms),
                            0),
                        [n_torsions, 1]),
                    tf.tile(
                        tf.expand_dims(
                            torsion_idxs[:, 0],
                            1),
                        [1, n_atoms])),
                tf.equal(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(n_atoms),
                            0),
                        [n_torsions, 1]),
                    tf.tile(
                        tf.expand_dims(
                            torsion_idxs[:, 1],
                            1),
                        [1, n_atoms])),
                tf.equal(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(n_atoms),
                            0),
                        [n_torsions, 1]),
                    tf.tile(
                        tf.expand_dims(
                            torsion_idxs[:, 2],
                            1),
                        [1, n_atoms])),
                tf.equal(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(n_atoms),
                            0),
                        [n_torsions, 1]),
                    tf.tile(
                        tf.expand_dims(
                            torsion_idxs[:, 3],
                            1),
                        [1, n_atoms]))
            ],
            axis=0)

        # (n_bonds, )
        # NOTE: here we use the same boolean mask as before, so they
        #       should be following the same order
        bond_orders = tf.boolean_mask(
            adjacency_map,
            is_bond)

        bond_distances = tf.boolean_mask(
            gin.deterministic.md.get_distance_matrix(coordinates),
            is_bond)

        angle_angles = gin.deterministic.md.get_angles(
            coordinates,
            angle_idxs)

        torsion_dihedrals = gin.deterministic.md.get_dihedrals(
            coordinates,
            torsion_idxs)

        # initialize the hidden layers
        # (n_bonds, ...)
        h_e = self.f_e(
            tf.expand_dims(bond_orders, 1))

        h_e_0 = h_e
        h_e_history = tf.expand_dims(h_e_0, 1)
        d_e = tf.shape(h_e, tf.int64)[1]

        # (n_atoms, ...)
        h_v = self.f_v(atoms)
        h_v_0 = h_v
        h_v_history = tf.expand_dims(h_v_0, 1)
        d_v = tf.shape(h_v, tf.int64)[1]

        # (n_angles, ...)
        h_a = self.f_a(
            tf.concat(
                [
                    tf.gather(
                        h_v,
                        angle_idxs[:, 1]),
                    tf.math.add(
                        tf.gather(
                            h_v,
                            angle_idxs[:, 0]),
                        tf.gather(
                            h_v,
                            angle_idxs[:, 1]))
                ],
                axis=1))

        h_a_0 = h_a
        h_a_history = tf.expand_dims(h_a_0, 1)
        d_a = tf.shape(h_a, tf.int64)[1]

        # (n_torsions, ...)
        h_t = self.f_t(
            tf.concat(
                [
                    tf.math.add(
                        tf.gather(
                            h_v,
                            torsion_idxs[:, 0]),
                        tf.gather(
                            h_v,
                            torsion_idxs[:, 3])),
                    tf.math.add(
                        tf.gather(
                            h_v,
                            torsion_idxs[:, 1]),
                        tf.gather(
                            h_v,
                            torsion_idxs[:, 2]))
                ],
                axis=1))
        h_t_0 = h_t
        h_t_history = tf.expand_dims(h_t_0, 1)
        d_t = tf.shape(h_t, tf.int64)[1]

        # (n_mols, ...)
        # NOTE: here $h_u$ could have more than one first dimensions
        h_u = self.f_u(atoms, adjacency_map, batched_attr_in_mol)
        h_u_0 = h_u
        h_u_history = tf.expand_dims(h_u_0, 1)
        d_u = tf.shape(h_u, tf.int64)[1]
        n_mols = tf.shape(h_u, tf.int64)[0]

        # specify what we know about the shape of the mask
        atom_in_mol.set_shape([None, None])

        atom_in_mol = tf.boolean_mask(
            atom_in_mol,
            tf.reduce_any(
                atom_in_mol,
                axis=0),
        axis=1)

        bond_in_mol = tf.greater(
            tf.matmul(
                tf.where(
                    bond_is_connected_to_atoms,
                    tf.ones_like(
                        bond_is_connected_to_atoms,
                        tf.int64),
                    tf.zeros_like(
                        bond_is_connected_to_atoms,
                        tf.int64)),
                tf.where(
                    atom_in_mol,
                    tf.ones_like(
                        atom_in_mol,
                        tf.int64),
                    tf.zeros_like(
                        atom_in_mol,
                        tf.int64))),
            tf.constant(0, dtype=tf.int64))

        angle_in_mol = tf.greater(
            tf.matmul(
                tf.where(
                    angle_is_connected_to_atoms,
                    tf.ones_like(
                        angle_is_connected_to_atoms,
                        tf.int64),
                    tf.zeros_like(
                        angle_is_connected_to_atoms,
                        tf.int64)),
                tf.where(
                    atom_in_mol,
                    tf.ones_like(
                        atom_in_mol,
                        tf.int64),
                    tf.zeros_like(
                        atom_in_mol,
                        tf.int64))),
            tf.constant(0, dtype=tf.int64))

        torsion_in_mol = tf.greater(
            tf.matmul(
                tf.where(
                    torsion_is_connected_to_atoms,
                    tf.ones_like(
                        torsion_is_connected_to_atoms,
                        tf.int64),
                    tf.zeros_like(
                        torsion_is_connected_to_atoms,
                        tf.int64)),
                tf.where(
                    atom_in_mol,
                    tf.ones_like(
                        atom_in_mol,
                        tf.int64),
                    tf.zeros_like(
                        atom_in_mol,
                        tf.int64))),
            tf.constant(0, dtype=tf.int64))

        def propagate_one_time(
                iter_idx,
                h_v, h_e, h_a, h_t, h_u,
                h_v_history, h_e_history, h_a_history,
                h_t_history, h_u_history,
                atom_in_mol=atom_in_mol, # (n_atoms, n_mols)
                bond_in_mol=bond_in_mol, # (n_bonds, n_mols)
                angle_in_mol=angle_in_mol,
                torsion_in_mol=torsion_in_mol
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

            h_left_right = h_left + h_right

            # (n_bonds, d_e)
            h_e = self.phi_e(h_e, h_e_0, h_left_right,
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

            h_v_center = tf.gather(
                h_v,
                angle_idxs[:, 1])

            h_v_sides = tf.math.add(
                tf.gather(
                    h_v,
                    angle_idxs[:, 0]),
                tf.gather(
                    h_v,
                    angle_idxs[:, 2]))

            h_a = self.phi_a(
                h_a,
                h_a_0,
                h_v_center,
                h_v_sides,
                tf.reduce_sum(
                    tf.boolean_mask(
                        tf.tile(
                            tf.expand_dims(
                                h_u, # (n_mols, d_u)
                                0), # (1, n_mols, d_u)
                            [tf.shape(h_a)[0], 1, 1]),
                        angle_in_mol),
                    axis=1,
                    keepdims=True))

            h_a_history = tf.concat(
                [
                    h_a_history,
                    tf.expand_dims(h_a, 1)
                ],
                axis=1)

            h_v_center = tf.math.add(
                tf.gather(
                    h_v,
                    torsion_idxs[:, 1]),
                tf.gather(
                    h_v,
                    torsion_idxs[:, 2]))

            h_v_sides = tf.math.add(
                tf.gather(
                    h_v,
                    torsion_idxs[:, 0]),
                tf.gather(
                    h_v,
                    torsion_idxs[:, 2]))

            h_t = self.phi_t(
                h_t,
                h_t_0,
                h_v_center,
                h_v_sides,
                tf.reduce_sum(
                    tf.boolean_mask(
                        tf.tile(
                            tf.expand_dims(
                                h_u, # (n_mols, d_u)
                                0), # (1, n_mols, d_u)
                            [tf.shape(h_t)[0], 1, 1]),
                        torsion_in_mol),
                    axis=1,
                    keepdims=True))

            h_t_history = tf.concat(
                [
                    h_t_history,
                    tf.expand_dims(h_t, 1)
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

            # aggregate $ \bar{a'} $
            h_a_bar = self.rho_a_u(h_a, angle_in_mol)

            # aggregate $ \bar{t} $
            h_t_bar = self.rho_t_u(h_t, torsion_in_mol)

            # update $ u' $
            # $$
            # u' = \phi^u (\bar{e'}, \bar{v'}, u)
            # $$
            # (n_mols, d_u)
            h_u = self.phi_u(
                h_u,
                h_u_0,
                h_e_bar,
                h_v_bar,
                h_a_bar,
                h_t_bar)

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
                h_v, h_e, h_a, h_t, h_u,
                h_v_history, h_e_history, h_a_history,
                h_t_history, h_u_history)

        a = propagate_one_time(0, h_v, h_e, h_a, h_t, h_u, \
        h_v_history, h_e_history, h_a_history, \
        h_t_history, h_u_history)

        # use while loop to execute the graph multiple times
        iter_idx = tf.constant(0, dtype=tf.int64)

        iter_idx, h_v, h_e, h_a, h_t, h_u, \
        h_v_history, h_e_history, h_a_history, \
        h_t_history, h_u_history \
            = tf.while_loop(
            # condition
            lambda \
                iter_idx, \
                h_v, h_e, h_a, h_t, h_u, \
                h_v_history, h_e_history, h_a_history, h_t_history, \
                h_u_history: \
                    tf.less(iter_idx, self.repeat),

            # loop body
            propagate_one_time,

            # loop vars
            [
                iter_idx,
                h_v, h_e, h_a, h_t, h_u,
                h_v_history, h_e_history, h_a_history,
                h_t_history, h_u_history
            ],

            # shape_invariants
            shape_invariants = [
                iter_idx.get_shape(),
                h_v.get_shape(),
                h_e.get_shape(),
                h_a.get_shape(),
                h_t.get_shape(),
                h_u.get_shape(),
                tf.TensorShape((None, None, None)),
                tf.TensorShape((None, None, None)),
                tf.TensorShape((None, None, None)),
                tf.TensorShape((None, None, None)),
                tf.TensorShape((None, None, None)),
                ])

        y_bar = self.f_r(
            h_v, h_e, h_a, h_t, h_u,
            h_v_history, h_e_history, h_a_history,
            h_t_history, h_u_history,
            atom_in_mol, bond_in_mol, angle_in_mol, torsion_in_mol,
            adjacency_map, coordinates)

        return y_bar

    # TODO: need testing
    @staticmethod
    @tf.function
    def batch(
            mols_with_coordinates_and_attributes,
            inner_batch_size=128,
            attr_dimension=0):
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

        attr_dimension = tf.convert_to_tensor(
            attr_dimension,
            tf.int64)

        # define max number of molecules, to which all rows and columns
        # associated with molecules are padded
        max_n_mols = tf.math.floordiv(
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
            atom, adjacency_map, coordinates, attr = input_element

            # shape = (), dtype = tf.int64
            n_atoms = tf.shape(atom, tf.int64)[0]

            if tf.less_equal(
                tf.add(
                    count,
                    n_atoms),
                inner_batch_size):

                output_element = (atom, adjacency_map, coordinates, attr, key)
                count = tf.add(count, n_atoms)

            else:
                key = tf.add(
                    key,
                    tf.constant(1, dtype=tf.int64))

                count = n_atoms

                output_element = (atom, adjacency_map, coordinates, attr, key)

            state = tf.concat(
                [
                    tf.expand_dims(key, 0),
                    tf.expand_dims(count, 0)
                ],
                axis=0)

            return state, output_element

        # define the key func: grab the last element in an entry
        key_func = lambda atom, adjacency_map,coordinates, attr, key: key

        @tf.function
        def init_func(key):
            # initialize atoms as all -1
            # shape = (inner_batch_size, )
            # dtype = int64
            atoms = tf.multiply(
                tf.ones(
                    (inner_batch_size, ),
                    dtype=tf.int64),
                tf.constant(-1, dtype=tf.int64))

            coordinates = tf.multiply(
                tf.ones(
                    (inner_batch_size, 3),
                    dtype=tf.float32),
                tf.constant(-1, dtype=tf.float32))

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

            if tf.equal(
                attr_dimension,
                tf.constant(0, dtype=tf.int64)):
                # initialize attr with number of molecules
                # dtype = float32,
                # shape = (max_n_mols, )
                attr = tf.multiply(
                    tf.ones(
                        (max_n_mols, ),
                        dtype=tf.float32),
                    tf.constant(-1, dtype=tf.float32))

            else:
                attr = tf.multiply(
                    tf.ones(
                        (max_n_mols, attr_dimension),
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
            mol_count = tf.constant(0, dtype=tf.int64)

            return (
                atoms,
                adjacency_map,
                coordinates,
                attr,
                atom_in_mol,
                attr_in_mol,
                atom_count,
                mol_count)

        # @tf.function
        def reduce_func(old_state, input):
            # grab all tensors from the batched state
            (
                atoms,
                adjacency_map,
                coordinates,
                attr,
                atom_in_mol,
                attr_in_mol,
                atom_count,
                mol_count
            ) = old_state

            # grab all tensors from input
            (
                _atoms,
                _adjacency_map,
                _coordinates,
                _attr,
                _key
            ) = input

            # get the count of atoms and bonds
            n_atoms = tf.shape(_atoms, tf.int64)[0]

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

            coordinates = tf.tensor_scatter_nd_update(
                coordinates,

                # idxs
                tf.expand_dims(
                    tf.range(
                        start=atom_count,
                        limit=tf.add(
                            atom_count,
                            n_atoms),
                        dtype=tf.int64),
                    axis=1),

                _coordinates)

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

            mol_count = tf.add(
                mol_count,
                tf.constant(1, dtype=tf.int64))

            return (
                atoms,
                adjacency_map,
                coordinates,
                attr,
                atom_in_mol,
                attr_in_mol,
                atom_count,
                mol_count
            )

        # @tf.function
        def finalize_func(*state):
            (
                atoms,
                adjacency_map,
                coordinates,
                attr,
                atom_in_mol,
                attr_in_mol,
                atom_count,
                mol_count
            ) = state

            result = (
                atoms,
                adjacency_map,
                coordinates,
                attr,
                atom_in_mol,
                attr_in_mol)

            return result

        # now we construct the reducer
        reducer = tf.data.experimental.Reducer(
            init_func=init_func,
            reduce_func=reduce_func,
            finalize_func=finalize_func)

        return mols_with_coordinates_and_attributes.apply(
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

    def call(self, *args, **kwargs):

        return self._call(*args, **kwargs)

    def switch(self, to_test=True):
        for fn in [self.rho_e_u, self.rho_e_v, self.rho_v_u]:
            if hasattr(fn, 'switch'):
                fn.switch()
