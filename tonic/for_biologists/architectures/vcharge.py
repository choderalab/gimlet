"""
vcharge.py

VCharge model pubslihed by Gilson et al. (2003)
doi://10.1021/ci034148o

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
import gin
import tensorflow as tf

# =============================================================================
# utility functions
# =============================================================================

# =============================================================================
# module classes
# =============================================================================
# define special typing
class VChargeTyping(gin.deterministic.typing.TypingBase):
    """ Define the atom typing designed in Gilson et al. paper.

    """
    def __init__(self, mol):
        super(VChargeTyping, self).__init__(mol)

    def is_1(self):
        """ H1
        """
        return self.is_hydrogen

    def is_2(self):
        """ C3
        """
        return tf.logical_and(
            self.is_carbon,
            self.is_sp3)

    def is_3(self):
        """ C2
        """
        return tf.logical_and(
            self.is_carbon,
            tf.logical_and(
                self.is_sp2,
                tf.logical_not(
                    self.is_aromatic)))

    def is_4(self):
        """ C1a
        """
        return tf.logical_and(
            self.is_carbon,
            tf.reduce_any(
                tf.greater(
                    self.adjacency_map_full,
                    tf.constant(2, dtype=tf.float32)),
                axis=0))

    def is_5(self):
        """ C1b
        """
        return tf.logical_and(
            self.is_carbon,
            tf.logical_and(
                self.is_sp1,
                tf.logical_not(
                    tf.reduce_any(
                        tf.greater(
                            self.adjacency_map_full,
                            tf.constant(2, dtype=tf.float32)),
                        axis=0))))

    def is_6(self):
        """ Car
        """
        return tf.logical_and(
            self.is_carbon,
            self.is_aromatic)

    def is_7(self):
        """ O3
        """
        return tf.logical_and(
            self.is_oxygen,
            self.is_sp3)

    def is_8(self):
        """ O2
        """
        return tf.logical_and(
            self.is_oxygen,
            self.is_sp2,)

    def is_9(self):
        """ O3n
        """
        return tf.logical_and(
            self.is_oxygen,
            tf.logical_and(
                self.is_sp3,
                self.is_connected_to_1))

    def is_10(self):
        """ Oar
        """
        return tf.logical_and(
            self.is_oxygen,
            self.is_aromatic)

    def is_11(self):
        """ N3
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_sp3,
                tf.logical_and(
                    tf.logical_not(
                        self.is_connected_to_aromatic),
                    tf.logical_not(
                        self.is_connected_to_4))))

    def is_12(self):
        """ N3s
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_sp3,
                tf.logical_and(
                    self.is_connected_to_aromatic,
                    tf.logical_not(
                        self.is_connected_to_4))))

    def is_13(self):
        """ N2
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_sp2,
                tf.logical_and(
                    tf.logical_not(
                        self.is_connected_to_4),
                    tf.logical_not(
                        self.is_aromatic))))

    def is_14(self):
        """ N1
        """
        return tf.logical_and(
            self.is_nitrogen,
            self.is_sp1)

    def is_15(self):
        """ N3p
        """
        return tf.logical_and(
            self.is_nitrogen,
            self.is_connected_to_4)

    def is_16(self):
        """ N2p
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_sp2,
                self.is_connected_to_4))

    def is_17(self):
        """ N1pa
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_sp1,
                tf.greater(
                    tf.math.count_nonzero(
                        tf.greater(
                            self.adjacency_map_full,
                            tf.constant(1, dtype=tf.float32)),
                        axis=0),
                    tf.constant(2, dtype=tf.int64))))

    def is_18(self):
        """ N1pb
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_sp1,
                tf.greater(
                    tf.math.count_nonzero(
                        tf.greater(
                            self.adjacency_map_full,
                            tf.constant(0, dtype=tf.float32)),
                        axis=0),
                    tf.constant(2, dtype=tf.int64))))

    def is_19(self):
        """ Nar3
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_connected_to_3,
                tf.logical_and(
                    self.is_aromatic,
                    tf.reduce_all(
                        tf.less(
                            self.adjacency_map_full,
                            tf.constant(2, dtype=tf.float32))))))

    def is_20(self):
        """ Nar2
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_connected_to_2,
                self.is_aromatic))

    def is_21(self):
        """ Narp
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_connected_to_2,
                tf.logical_and(
                    self.is_aromatic,
                    tf.reduce_all(
                        tf.less_equal(
                            self.adjacency_map_full,
                            tf.constant(1, dtype=tf.float32)),
                        axis=0))))

    def is_22(self):
        """ N1m
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_connected_to_1,
                self.is_sp2))

    def is_23(self):
        """ N2m
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_connected_to_2,
                tf.logical_and(
                    tf.logical_not(
                        self.is_in_ring),
                    tf.reduce_all(
                        tf.less_equal(
                            self.adjacency_map_full,
                            tf.constant(1, dtype=tf.float32)),
                        axis=0))))

    def is_24(self):
        """ N2mR
        """
        return tf.logical_and(
            self.is_nitrogen,
            tf.logical_and(
                self.is_connected_to_2,
                tf.logical_and(
                    self.is_in_ring,
                    tf.reduce_all(
                        tf.less_equal(
                            self.adjacency_map_full,
                            tf.constant(1, dtype=tf.float32)),
                        axis=0))))

    def is_25(self):
        """ Cl3
        """
        return self.is_chlorine

    def is_26(self):
        """ F3
        """
        return self.is_flourine

    def is_27(self):
        """ Br3
        """
        return self.is_bromine

    def is_28(self):
        """ S3
        """
        return tf.logical_and(
            self.is_sulfur,
            tf.logical_and(
                self.is_sp3,
                self.is_connected_to_2))

    def is_29(self):
        """ S3p
        """
        return tf.logical_and(
            self.is_sulfur,
            tf.logical_and(
                self.is_sp3,
                self.is_connected_to_3))

    def is_30(self):
        """ S4
        """
        return tf.logical_and(
            self.is_sulfur,
            tf.logical_and(
                tf.equal(
                    tf.math.count_nonzero(
                        tf.equal(
                            self.adjacency_map_full,
                            tf.constant(1, dtype=tf.float32)),
                        axis=0),
                    tf.constant(2, dtype=tf.int64)),
                tf.equal(
                    tf.math.count_nonzero(
                        tf.equal(
                            self.adjacency_map_full,
                            tf.constant(2, dtype=tf.float32)),
                        axis=0),
                    tf.constant(1, dtype=tf.int64))))

    def is_31(self):
        """ S6
        """
        return tf.logical_and(
            self.is_sulfur,
            tf.logical_and(
                tf.equal(
                    tf.math.count_nonzero(
                        tf.equal(
                            self.adjacency_map_full,
                            tf.constant(1, dtype=tf.float32)),
                        axis=0),
                    tf.constant(2, dtype=tf.int64)),
                tf.equal(
                    tf.math.count_nonzero(
                        tf.equal(
                            self.adjacency_map_full,
                            tf.constant(2, dtype=tf.float32)),
                        axis=0),
                    tf.constant(2, dtype=tf.int64))))

    def is_32(self):
        """ Sar
        """
        return tf.logical_and(
            self.is_sulfur,
            self.is_aromatic)

    def is_33(self):
        """ S3n
        """
        return tf.logical_and(
            self.is_sulfur,
            tf.logical_and(
                self.is_sp3,
                self.is_connected_to_1))

    def is_34(self):
        """ S2a
        """
        return tf.logical_and(
            self.is_sulfur,
            tf.logical_and(
                self.is_sp2,
                tf.logical_not(
                    self.is_aromatic)))

    def is_35(self):
        """ P3
        """
        return tf.logical_and(
            self.is_phosphorus,
            tf.logical_and(
                self.is_sp3,
                tf.logical_and(
                    tf.logical_not(
                        self.is_connected_to_aromatic),
                    tf.logical_not(
                        self.is_connected_to_4))))

    def is_36(self):
        """ P3p
        """
        return tf.logical_and(
            self.is_phosphorus,
            tf.logical_and(
                self.is_connected_to_4,
                self.is_sp3))

    def is_37(self):
        """ P5
        """
        return tf.logical_and(
            self.is_phosphorus,
            tf.logical_and(
                tf.equal(
                    tf.math.count_nonzero(
                        tf.equal(
                            self.adjacency_map_full,
                            tf.constant(1, dtype=tf.float32)),
                        axis=0),
                    tf.constant(3, dtype=tf.int64)),
                tf.equal(
                    tf.math.count_nonzero(
                        tf.greater_equal(
                            self.adjacency_map_full,
                            tf.constant(2, dtype=tf.float32)),
                        axis=0),
                    tf.constant(1, dtype=tf.int64))))

    def is_38(self):
        """ I
        """
        return tf.logical_and(
            self.is_iodine,
            self.is_connected_to_1)

    def is_39(self):
        """ Ip
        """
        return tf.logical_and(
            self.is_iodine,
            self.is_connected_to_2)

    def get_assignment(self):
        fn_array = [getattr(self, 'is_' + str(idx)) for idx in range(1, 40)]

        # use paralleled while loop for this assignment
        idx = tf.constant(1, dtype=tf.int64)
        assignment = tf.constant(-1, dtype=tf.int64) \
            * tf.ones_like(
                self.atoms,
                dtype=tf.int64)

        def loop_body(idx, assignment):
            # get the function
            fn = fn_array[idx - 1]
            is_this_type = fn()
            assignment = tf.where(
                is_this_type,

                # when it is of this type
                idx * tf.ones(
                    (tf.shape(assignment)[0], ),
                    dtype=tf.int64),

                # when it is not
                assignment)

            return idx + 1, assignment

        _, assignment = tf.while_loop(
            lambda idx, _: tf.less(idx, 40),

            loop_body,

            [idx, assignment])

        self.assignment = assignment
        return self.assignment

# =============================================================================
# module classes
# =============================================================================
class VCharge(tf.keras.Model):
    """ V Charge model. Pubslihed by Gilson et al. (2003)
    doi://10.1021/ci034148o

    Attributes
    ----------
    e : tf.Variable, shape = (34, ), dtype=float32,
        electronegtivity.
    s : tf.Variable, shape = (34, ), dtype=float32,
        hardness.
    alpha_1 : tf.Variable, shape = (), dtype=tf.float32,
        coefficient before single terms.
    alpha_2 : tf.Variable, shape = (), dtype=tf.float32,
        coefficient before double terms.
    alpha_3 : tf.Variable, shape = (), dtype=tf.float32,
        coefficient before triple terms.
    alpha_4 : tf.Variable, shape = (), dtype=tf.float32,
        coefficient before aromatic terms.
    alpha_5 : tf.Variable, shape = (), dtype=tf.float32,
        coefficient before 1-3 terms.
    beta : tf.Variable, shape = (), dtype=tf.float32,
        exponential scaling coefficient.

    """

    def __init__(self):
        super(VCharge, self).__init__()

        # $ e_i^0 $
        self.e = tf.Variable(
            tf.random.normal(
                shape=(39, ),
                dtype=tf.float32),
            name='e')

        # $ s_i $
        self.s = tf.Variable(
            tf.random.normal(
                shape=(39, ),
                dtype=tf.float32),
            name='s')

        # $ \alpha_1 $
        self.alpha_1 = tf.Variable(
            tf.random.normal(
                shape=(),
                dtype=tf.float32),
            name='alpha_1')

        # $ \alpha_2 $
        self.alpha_2 = tf.Variable(
            tf.random.normal(
                shape=(),
                dtype=tf.float32),
            name='alpha_2')

        # $ \alpha_3 $
        self.alpha_3 = tf.Variable(
            tf.random.normal(
                shape=(),
                dtype=tf.float32),
            name='alpha_3')

        # $ \alpha_4 $
        self.alpha_4 = tf.Variable(
            tf.random.normal(
                shape=(),
                dtype=tf.float32),
            name='alpha_4')

        # $ \alpha_5 $
        self.alpha_5 = tf.Variable(
            tf.random.normal(
                shape=(),
                dtype=tf.float32),
            name='alpha_5')

        # $ \beta $
        self.beta = tf.Variable(
            tf.random.normal(
                shape=(),
                dtype=tf.float32),
            name='beta')


    # @tf.function
    def get_charges(self, e, s, Q):
        """ Solve the function to get the absolute charges of atoms in a
        molecule from parameters.

        Parameters
        ----------
        e : tf.Tensor, dtype = tf.float32, shape = (34, ),
            electronegtivity.
        s : tf.Tensor, dtype = tf.float32, shape = (34, ),
            hardness.
        Q : tf.Tensor, dtype = tf.float32, shape=(),
            total charge of a molecule.

        We use Lagrange multipliers to analytically give the solution.

        $$

        U({\bf q})
        &= \sum_{i=1}^N \left[ e_i q_i +  \frac{1}{2}  s_i q_i^2\right]
            - \lambda \, \left( \sum_{j=1}^N q_j - Q \right) \\
        &= \sum_{i=1}^N \left[
            (e_i - \lambda) q_i +  \frac{1}{2}  s_i q_i^2 \right
            ] + Q

        $$

        This gives us:

        $$

        q_i^*
        &= - e_i s_i^{-1}
        + \lambda s_i^{-1} \\
        &= - e_i s_i^{-1}
        + s_i^{-1} \frac{
            Q +
             \sum\limits_{i=1}^N e_i \, s_i^{-1}
            }{\sum\limits_{j=1}^N s_j^{-1}}

        $$

        """

        return tf.math.add(
            tf.math.multiply(
                tf.math.negative(
                    e),
                tf.math.pow(
                    s,
                    -1)),
            tf.math.multiply(
                tf.math.pow(
                    s,
                    -1),
                tf.math.divide(
                    tf.math.add(
                        Q,
                        tf.reduce_sum(
                            tf.math.multiply(
                                e,
                                tf.math.pow(
                                    s,
                                    -1)))),
                    tf.reduce_sum(
                        tf.math.pow(
                            s,
                            -1)))))


    # @tf.function
    def call(self, atoms, adjacency_map, Q):
        """ Main method of the module.

        Take the topology of an molecule and give the charges of each atoms.

        Parameters
        ----------
        atoms : tf.Tensor,
            the tensor containing all the atoms.
            Note that here atoms could also be one-hot encoding of atom types.
        adjacency_map: tf.Tensor, dtype=tf.float32.
        Q : tf.Tensor, dtype = tf.float32, shape = (),
            total charge.
        """
        # full adjacency map could be calculated by
        # $ X^T + X, $
        # where $X$ is the adjacency map.
        adjacency_map_full = tf.math.add(
            adjacency_map,
            tf.transpose(
                adjacency_map))

        # read the electronegtivity of all atoms from the attributes.
        e_all_atoms = tf.gather(
            self.e,
            atoms)

        # $ D_{ij} = e_i - e_j $
        delta_e_all_atoms = tf.math.subtract(
            tf.expand_dims(
                e_all_atoms,
                0),
            tf.expand_dims(
                e_all_atoms,
                1))

        # $ S_ij | e_i^0 - e_j^0 | $
        delta_e_all_atoms_scaled = tf.math.multiply(
            tf.math.sign(
                delta_e_all_atoms),
            tf.math.pow(
                tf.math.abs(
                    delta_e_all_atoms),
                self.beta))

        # get all the masks
        is_connected_by_single = tf.where(
            tf.equal(
                adjacency_map_full,
                tf.constant(1, dtype=tf.float32)),
            tf.ones_like(adjacency_map_full),
            tf.zeros_like(adjacency_map_full))

        is_connected_by_double = tf.where(
            tf.equal(
                adjacency_map_full,
                tf.constant(2, dtype=tf.float32)),
            tf.ones_like(adjacency_map_full),
            tf.zeros_like(adjacency_map_full))

        is_connected_by_triple = tf.where(
            tf.equal(
                adjacency_map_full,
                tf.constant(3, dtype=tf.float32)),
            tf.ones_like(adjacency_map_full),
            tf.zeros_like(adjacency_map_full))

        # theorem:
        # for a graph with adjacency map X,
        # two points are connected by $n$ edges,
        # is equivalent to
        # $X^n > 0 $
        # here $^n$ denotes matrix multiplication
        is_onethree = tf.where(
            tf.greater(
                tf.matmul(
                    adjacency_map_full,
                    adjacency_map_full),
                tf.constant(0, dtype=tf.float32)),
            tf.ones_like(adjacency_map_full),
            tf.zeros_like(adjacency_map_full))

        # deal with aromaticity
        # under such atom typing, only 6, 10, 19, 20, 21, 22, and 32
        # are aromatic (altogether seven types)
        is_aromatic = tf.reduce_any(
            tf.equal(
                tf.tile(
                    tf.expand_dims(
                        atoms,
                        1),
                    [1, 7]),
                tf.tile(
                    tf.expand_dims(
                        tf.constant(
                            [
                                6,
                                10,
                                19,
                                20,
                                21,
                                22,
                                32
                            ],
                            dtype=tf.int64),
                        0),
                    [
                        tf.shape(atoms, tf.int64)[0],
                        1
                    ])),
            axis=1)

        is_connected_by_aromatic = tf.where(
            tf.logical_and(
                tf.greater(
                    adjacency_map_full,
                    tf.constant(0, dtype=tf.float32)),
                tf.logical_and(
                    tf.tile(
                        tf.expand_dims(
                            is_aromatic,
                            0),
                        [tf.shape(atoms, tf.int64)[0], 1]),
                    tf.tile(
                        tf.expand_dims(
                            is_aromatic,
                            1),
                        [1, tf.shape(atoms, tf.int64)[0]]))),
            tf.ones_like(adjacency_map_full),
            tf.zeros_like(adjacency_map_full))

        # get the final results of
        # $ e_i $
        e_i = tf.reduce_sum(
            tf.reduce_sum(
                [
                    tf.math.multiply(
                        self.alpha_1,
                        tf.math.multiply(
                            is_connected_by_single,
                            delta_e_all_atoms_scaled)),
                    tf.math.multiply(
                        self.alpha_2,
                        tf.math.multiply(
                            is_connected_by_double,
                            delta_e_all_atoms_scaled)),
                    tf.math.multiply(
                        self.alpha_3,
                        tf.math.multiply(
                            is_connected_by_triple,
                            delta_e_all_atoms_scaled)),
                    tf.math.multiply(
                        self.alpha_4,
                        tf.math.multiply(
                            is_connected_by_aromatic,
                            delta_e_all_atoms_scaled)),
                    tf.math.multiply(
                        self.alpha_5,
                        tf.math.multiply(
                            is_onethree,
                            delta_e_all_atoms_scaled))
                ],
                axis=0),
            axis=0)

        # get $ s_i $
        s_i = tf.gather(
            self.s,
            atoms)

        charges = self.get_charges(e_i, s_i, Q)
        return charges


def train(ds, charge_model, n_epochs=10):
    """ Main function to execute. Fit
    """
    optimizer = optimizer = tf.keras.optimizers.Adam(1e-6)
    logger = tf.get_logger()
    for dummy_idx in range(n_epochs):
        # loop through the molecules in the dataset
        for atoms, adjacency_map, coordinates, q_i in ds:
            q = tf.reduce_sum(q_i)
            with tf.GradientTape() as tape:
                # get predicted
                q_i_hat = charge_model(atoms, adjacency_map, q)
                loss = tf.losses.mean_squared_error(
                    q_i,
                    q_i_hat)

            # logger.log(msg=loss, level=0)
            print(loss)
            # backprop

            grad = tape.gradient(loss, variables)
            optimizer.apply_gradients(
                zip(grad, variables))

    return charge_model
