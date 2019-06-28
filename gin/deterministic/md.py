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
# import gin.deterministic.forcefields

# =============================================================================
# utility functions
# =============================================================================
# @tf.function
def get_distance_matrix(coordinates):
    """ Calculate the distance matrix from coordinates.

    $$
    D_{ij}^2 = <X_i, X_i> - 2<X_i, X_j> + <X_j, X_j>

    Parameters
    ----------
    coordinates: tf.Tensor, shape=(n_atoms, 3)

    """

    X_2 = tf.reshape(
        tf.reduce_sum(
            coordinates * coordinates,
            1),
        [-1, 1])

    return tf.math.sqrt(
        X_2 - 2 * tf.matmul(coordinates, tf.transpose(coordinates)) \
            + tf.transpose(X_2))

def get_angles(coordinates, angle_idxs):
    """ Calculate the angles from coordinates and angle indices.

    Parameters
    ----------
    coordinates: tf.Tensor, shape=(n_atoms, 3)

    """

    # get the coordinates of the atoms forming the angle
    # (n_angles, 3, 3)
    angle_coordinates = tf.gather(coordinates, angle_idxs)

    # (n_angles, 3)
    angle_left = angle_coordinates[:, 1, :] \
        - angle_coordinates[:, 0, :]

    # (n_angles, 3)
    angle_right = angle_coordinates[:, 1, :] \
        - angle_coordinates[:, 2, :]

    # (n_angles, )
    angles = tf.math.acos(
        tf.clip_by_value(
            tf.math.divide(
                tf.reduce_sum(
                    angle_left * angle_right,
                    axis=1),
                tf.norm(angle_left, axis=1) \
                    * tf.norm(angle_right, axis=1)),
            -1, 1))

    return angles

def get_dihedrals(coordinates, torsion_idxs):
    """ Calculate the dihedrals based on coordinates and the indices of
    the torsions.

    Parameters
    ----------
    coordinates: tf.Tensor, shape=(n_atoms, 3)
    """
    # get the coordinates of the atoms forming the dihedral
    # (n_torsions, 4, 3)
    torsion_idxs = tf.gather(coordinates, torsion_idxs)

    # (n_torsions, 3)
    normal_left = tf.linalg.cross(
        torsion_idxs[:, 1] - torsion_idxs[:, 0],
        torsion_idxs[:, 1] - torsion_idxs[:, 2])

    # (n_torsions, 3)
    normal_right = tf.linalg.cross(
        torsion_idxs[:, 2] - torsion_idxs[:, 3],
        torsion_idxs[:, 2] - torsion_idxs[:, 1])

    # (n_torsions, )
    dihedrals = tf.math.acos(
        tf.math.divide(
            tf.reduce_sum(
                normal_left * normal_right,
                axis=1),
            tf.norm(normal_left, axis=1) \
                * tf.norm(normal_right, axis=1)))

    return dihedrals

# =============================================================================
# module classes
# =============================================================================
class SingleMoleculeMechanicsSystem:
    """
    A single molecule system that could be calculated in MD calculation.

    """
    def __init__(
            self,
            mol,
            typing=None,
            forcefield=None):

        self.mol = mol
        self.atoms = mol[0]
        self.n_atoms = tf.shape(self.atoms, tf.int64)[0]
        self.adjacency_map = mol[1]
        self.typing = typing
        self.forcefield = forcefield

        if self.typing == None:
            self.typing = gin.deterministic.typing.TypingGAFF

        if self.forcefield == None:
            self.forcefield = gin.deterministic.forcefields.gaff

        if len(mol) == 3:
            self.coordinates = tf.math.divide(
                mol[2],
                tf.constant(10, dtype=tf.float32))

        else:
            self.coordinates = gin.deterministic.conformer.Conformers(
                mol,
                self.forcefield,
                self.typing).get_conformers_from_distance_geometry(1)[0]

        # get the types
        self.typing_assignment = self.typing(self.mol).get_assignment()

        self.get_bond_params()
        self.get_angle_params()
        self.get_torsion_params()
        self.get_nonbonded_params()

    def energy(self, coordinates=None):
        """ Compute the total energy of a small molecule.

        $$
        E = E_\mathtt{bonded} + E_\mathtt{nonbonded} + E_\mathtt{angles} \
            + E_\mathtt{torsions}
        $$
        """
        if type(coordinates) == type(None):
            coordinates = self.coordinates

        # get all the vars needed
        distance_matrix = get_distance_matrix(coordinates)

        angles = get_angles(coordinates, self.angle_idxs)
        torsions = get_dihedrals(coordinates, self.torsion_idxs)

        # bond energy
        # $$
        # E_\mathtt{bonded}
        # = \frac{1}{2} k (x - x_0) ^ 2
        # $$
        # NOTE: it's impossible for bonds to return NaN

        bond_energy = 0.5 * self.bond_k \
            * tf.pow(
                tf.gather_nd(
                    distance_matrix,
                    self.bond_idxs) \
                    - self.bond_length,
                2)

        bond_energy = tf.reduce_sum(bond_energy)

        # angle energy
        # $$
        # E_\mathtt{angle}
        # = \frac{1}{2} k (\theta - theta_0) ^ 2
        # $$
        angle_energy = 0.5 * self.angle_k \
            * tf.pow(
                angles - self.angle_angle,
                2)

        angle_energy = tf.reduce_sum(angle_energy)

        # torsion energy
        # $$
        # E_\texttt{torsion}
        # = k (1 + cos(n\theta - theta_0))
        # $$
        proper_torsion_energy = \
            self.torsion_proper_k1 * (
                1 + tf.math.cos(
                    self.torsion_proper_periodicity1 * torsions \
                        - self.torsion_proper_phase1)) \
            + self.torsion_proper_k2 * (
                1 + tf.math.cos(
                    self.torsion_proper_periodicity2 * torsions \
                        - self.torsion_proper_phase2)) \
            + self.torsion_proper_k3 * (
                1 + tf.math.cos(
                    self.torsion_proper_periodicity3 * torsions \
                        - self.torsion_proper_phase3))

        # get rid of nan
        proper_torsion_energy = tf.where(
            tf.math.is_nan(proper_torsion_energy),
            tf.zeros_like(proper_torsion_energy),
            proper_torsion_energy)

        proper_torsion_energy = tf.reduce_sum(proper_torsion_energy)

        improper_torsion_energy = \
            self.torsion_improper_k1 * (
                1 + tf.math.cos(
                    self.torsion_improper_periodicity1 * torsions \
                        - self.torsion_improper_phase1)) \
            + self.torsion_improper_k2 * (
                1 + tf.math.cos(
                    self.torsion_improper_periodicity2 * torsions \
                        - self.torsion_improper_phase2)) \
            + self.torsion_improper_k3 * (
                1 + tf.math.cos(
                    self.torsion_improper_periodicity3 * torsions \
                        - self.torsion_improper_phase3))

        # get rid of nan
        improper_torsion_energy = tf.where(
            tf.math.is_nan(improper_torsion_energy),
            tf.zeros_like(improper_torsion_energy),
            improper_torsion_energy)

        improper_torsion_energy = tf.reduce_sum(improper_torsion_energy)

        # lenard-jones
        # $$
        # E = 4 \epsilon ((\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^{6})
        # $$

        # (n_atoms, n_atoms)
        sigma_over_r = tf.clip_by_value(
            tf.math.divide_no_nan(
                self.nonbonded_sigma,
                distance_matrix),
            -1e10, 1e10)

        # (n_atoms, n_atoms)
        lj_energy_matrix = tf.stop_gradient(4 * self.nonbonded_epsilon) \
            * (tf.pow(sigma_over_r, 12) - tf.pow(sigma_over_r, 6))

        lj_energy = tf.math.add(
            tf.reduce_sum(
                tf.boolean_mask(
                    lj_energy_matrix,
                    self.is_nonbonded)),
            tf.reduce_sum(
                tf.boolean_mask(
                    lj_energy_matrix,
                    self.is_onefour)))

        constraint = tf.constant(100, dtype=tf.float32) * tf.reduce_sum(
            tf.pow(
                coordinates,
                2))

        energy_tot = bond_energy \
            + angle_energy \
            + proper_torsion_energy \
            + improper_torsion_energy \
            + lj_energy

        return energy_tot

    def force(self, coordinates=None):
        """ Explicitly calculate the gradient of the energy w.r.t.
        the coordinates.

        """

        pass

    def minimize(
            self,
            method='adam',
            coordinates=None,
            max_iter=10000,
            **kwargs):
        """ Minimize the energy.
        """

        max_iter = tf.constant(max_iter, dtype=tf.int64)

        if type(coordinates) == type(None):
            coordinates = self.coordinates

        if method == 'adam':
            # put coordinates into a variable
            coordinates = tf.Variable(coordinates)

            # keep a history
            recent_ten = tf.zeros((10, ), dtype=tf.float32)

            # get the Adam optimizer
            optimizer = tf.keras.optimizers.Adam(1000)

            # init
            iter_idx = tf.constant(0, dtype=tf.int64)

            while tf.less(iter_idx, max_iter):
                with tf.GradientTape() as tape:
                    energy = self.energy(coordinates)

                print(energy)

                recent_ten = tf.concat(
                    [
                        recent_ten[1:],
                        tf.expand_dims(energy, 0)
                    ],
                    axis=0)

                grad = tape.gradient(energy, coordinates)

                grad = tf.where(
                    tf.math.is_nan(grad),
                    tf.zeros_like(grad),
                    grad)

                optimizer.apply_gradients(zip([grad], [coordinates]))


                if tf.logical_and(
                    tf.greater(iter_idx, 100),
                    tf.less(
                        tf.math.reduce_std(recent_ten),
                        1e-3)):
                    break

                iter_idx += 1

            gin.i_o.to_sdf.write_sdf(
                [[
                    self.atoms,
                    self.adjacency_map,
                    tf.constant(10, dtype=tf.float32) * (
                        coordinates - tf.reduce_mean(coordinates, 0))
                ]],
                'caffeine_out.sdf')

    def get_bond_params(self):
        """ Get the config of all the bonds in the system.

        """
        # find the positions at which there is a bond
        is_bond = tf.greater(
            self.adjacency_map,
            tf.constant(0, dtype=tf.float32))

        self.is_bond = is_bond

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

        # get the specs of the bond
        bond_specs = tf.map_fn(
            lambda bond: tf.convert_to_tensor(
                    self.forcefield.get_bond(
                        int(
                            tf.gather(
                                self.typing_assignment, bond[0]).numpy()),
                        int(
                            tf.gather(
                                self.typing_assignment, bond[1]).numpy()))),
            bond_idxs,
            dtype=tf.float32)

        # (n_bonds, 2)
        self.bond_idxs = bond_idxs

        # (n_bonds, )
        self.bond_length = bond_specs[:, 0]

        # (n_bonds, )
        self.bond_k = bond_specs[:, 1]



    def get_angle_params(self):
        """ Get all the angles in the system.

        """
        # get the full adjacency_map
        full_adjacency_map = tf.transpose(self.adjacency_map) \
            + self.adjacency_map

        # init the angles idxs to be all negative ones
        angle_idxs = tf.constant([[-1, -1, -1]], dtype=tf.int64)

        @tf.function
        def process_one_atom_if_there_is_angle(idx, angle_idxs,
                full_adjacency_map=full_adjacency_map):

            # get all the connection indices
            connection_idxs = tf.where(
                tf.greater(
                    full_adjacency_map[idx, :],
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
                full_adjacency_map=full_adjacency_map):

            if tf.less(
                tf.math.count_nonzero(full_adjacency_map[idx, :]),
                tf.constant(1, dtype=tf.int64)):
                return idx+1, angle_idxs

            else:
                return process_one_atom_if_there_is_angle(idx, angle_idxs)


        idx = tf.constant(0, dtype=tf.int64)
        # use while loop to update the indices forming the angles
        idx, angle_idxs = tf.while_loop(
            # condition
            lambda idx, angle_idxs: tf.less(idx, self.n_atoms),

            process_one_atom,

            [idx, angle_idxs],

            shape_invariants=[
                idx.get_shape(),
                tf.TensorShape((None, 3))])

        # discard the first row
        angle_idxs = angle_idxs[1:, ]

        # get the specs of the angle
        angle_specs = tf.map_fn(
            lambda angle: tf.convert_to_tensor(
                    self.forcefield.get_angle(
                        int(
                            tf.gather(
                                self.typing_assignment, angle[0]).numpy()),
                        int(
                            tf.gather(
                                self.typing_assignment, angle[1]).numpy()),
                        int(
                            tf.gather(
                                self.typing_assignment, angle[2]).numpy()))),
            angle_idxs,
            dtype=tf.float32)

        # put everything into the attributes of the object

        # (n_angles, )
        self.angle_idxs = angle_idxs

        # (n_angles, )
        self.angle_angle = angle_specs[:, 0]

        # (n_angles, )
        self.angle_k = angle_specs[:, 1]

    def get_torsion_params(self):
        """ Get the torsion parameters.

        """
        # get the full adjacency_map
        full_adjacency_map = tf.transpose(self.adjacency_map) \
            + self.adjacency_map

        # init the torsion idxs to be all negative ones
        torsion_idxs = tf.constant([[-1, -1, -1, -1]], dtype=tf.int64)

        # for each bond, there is at least one torsion terms associated
        def process_one_bond_if_there_is_torsion(idx, torsion_idxs):
            bond = self.bond_idxs[idx]
            left_atom_connections = tf.where(
                tf.greater(
                    full_adjacency_map[bond[0]],
                    tf.constant(0, dtype=tf.float32)))

            right_atom_connections = tf.where(
                tf.greater(
                    full_adjacency_map[bond[1]],
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
                            full_adjacency_map[self.bond_idxs[idx][0]]),
                        tf.constant(1, dtype=tf.int64)),
                    tf.greater(
                        tf.math.count_nonzero(
                            full_adjacency_map[self.bond_idxs[idx][1]]),
                        tf.constant(1, dtype=tf.int64)))):
                return idx + 1, torsion_idxs

            else:
                return process_one_bond_if_there_is_torsion(
                    idx, torsion_idxs)


        idx = tf.constant(0, dtype=tf.int64)
        idx, torsion_idxs = tf.while_loop(
            # condition
            lambda idx, _: tf.less(idx, tf.shape(self.bond_idxs, tf.int64)[0]),

            # body
            process_one_bond,

            # vars
            [idx, torsion_idxs],

            shape_invariants=[idx.get_shape(), [4, None]])

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

        self.torsion_idxs = torsion_idxs

        # get the specs of the torsion
        torsion_proper_specs = tf.map_fn(
            lambda torsion: tf.convert_to_tensor(
                    self.forcefield.get_proper(
                        int(
                            tf.gather(
                                self.typing_assignment,
                                torsion[0]).numpy()),
                        int(
                            tf.gather(
                                self.typing_assignment,
                                torsion[1]).numpy()),
                        int(
                            tf.gather(
                                self.typing_assignment,
                                torsion[2]).numpy()),
                        int(
                            tf.gather(
                                self.typing_assignment,
                                torsion[3]).numpy()))),
            torsion_idxs,
            dtype=tf.float32)

        # get the specs of the torsion
        torsion_improper_specs = tf.map_fn(
            lambda torsion: tf.convert_to_tensor(
                    self.forcefield.get_improper(
                        int(
                            tf.gather(
                                self.typing_assignment,
                                torsion[0]).numpy()),
                        int(
                            tf.gather(
                                self.typing_assignment,
                                torsion[1]).numpy()),
                        int(
                            tf.gather(
                                self.typing_assignment,
                                torsion[2]).numpy()),
                        int(
                            tf.gather(
                                self.typing_assignment,
                                torsion[3]).numpy()))),
            torsion_idxs,
            dtype=tf.float32)

        # put the specs as the attributes of the object
        self.torsion_proper_periodicity1 = torsion_proper_specs[:, 0]
        self.torsion_proper_phase1 = torsion_proper_specs[:, 1]
        self.torsion_proper_k1 = torsion_proper_specs[:, 2]
        self.torsion_proper_periodicity2 = torsion_proper_specs[:, 3]
        self.torsion_proper_phase2 = torsion_proper_specs[:, 4]
        self.torsion_proper_k2 = torsion_proper_specs[:, 5]
        self.torsion_proper_periodicity3 = torsion_proper_specs[:, 6]
        self.torsion_proper_phase3 = torsion_proper_specs[:, 7]
        self.torsion_proper_k3 = torsion_proper_specs[:, 8]

        self.torsion_improper_periodicity1 = torsion_improper_specs[:, 0]
        self.torsion_improper_phase1 = torsion_improper_specs[:, 1]
        self.torsion_improper_k1 = torsion_improper_specs[:, 2]
        self.torsion_improper_periodicity2 = torsion_improper_specs[:, 3]
        self.torsion_improper_phase2 = torsion_improper_specs[:, 4]
        self.torsion_improper_k2 = torsion_improper_specs[:, 5]
        self.torsion_improper_periodicity3 = torsion_improper_specs[:, 6]
        self.torsion_improper_phase3 = torsion_improper_specs[:, 7]
        self.torsion_improper_k3 = torsion_improper_specs[:, 8]

    def get_nonbonded_params(self):
        """ Get the nonbonded parameters.

        """
        full_adjacency_map = tf.transpose(self.adjacency_map) \
            + self.adjacency_map

        # property of adjacency_map:
        # raise to the n'th power, entries greater than zero
        # indicates that two atoms could be connected by n bonds
        self.is_nonbonded = tf.reduce_all(
            [
                tf.equal(
                    full_adjacency_map,
                    tf.constant(0, dtype=tf.float32)),
                tf.equal(
                    tf.matmul(
                        full_adjacency_map,
                        full_adjacency_map),
                    tf.constant(0, dtype=tf.float32)),
                tf.equal(
                    tf.matmul(
                        full_adjacency_map,
                        tf.matmul(
                            full_adjacency_map,
                            full_adjacency_map)),
                    tf.constant(0, dtype=tf.float32)),
                tf.equal(
                    tf.matmul(
                        full_adjacency_map,
                        tf.matmul(
                            full_adjacency_map,
                            tf.matmul(
                                full_adjacency_map,
                                full_adjacency_map))),
                    tf.constant(0, dtype=tf.float32))
            ],
            axis=0)

        self.is_onefour = tf.greater(
            tf.matmul(
                full_adjacency_map,
                tf.matmul(
                    full_adjacency_map,
                    full_adjacency_map)),
            tf.constant(0, dtype=tf.float32))


        self.onefour_scaling = self.forcefield.get_onefour_scaling()

        nonbonded_specs = tf.map_fn(
            lambda atom: tf.convert_to_tensor(
                    self.forcefield.get_nonbonded(
                        int(tf.gather(self.typing_assignment, atom).numpy()))),
            self.atoms,
            dtype=tf.float32)

        # get the constants of nonbonded forces and put as attributes
        # $$
        # sigma = \frac{\sigma_1 + \sigma_2}{2}
        # epsilon = \sqrt{\epsilon_1 \epsilon_2}
        # $$
        self.nonbonded_sigma = tf.math.divide(
            tf.math.add(
                tf.tile(
                    tf.expand_dims(
                        nonbonded_specs[:, 0],
                        0),
                    [self.n_atoms, 1]),
                tf.tile(
                    tf.expand_dims(
                        nonbonded_specs[:, 0],
                        1),
                    [1, self.n_atoms])),
            2)

        self.nonbonded_epsilon = tf.math.sqrt(
            tf.math.multiply(
                tf.tile(
                    tf.expand_dims(
                        nonbonded_specs[:, 1],
                        0),
                    [self.n_atoms, 1]),
                tf.tile(
                    tf.expand_dims(
                        nonbonded_specs[:, 1],
                        1),
                    [1, self.n_atoms])))
