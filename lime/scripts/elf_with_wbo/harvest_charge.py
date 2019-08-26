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

from sklearn import metrics
import tensorflow as tf
import gin
import lime
import time
import pandas as pd
import numpy as np
import os

N_EPOCHS = 20

# =============================================================================
# utility functions
# =============================================================================
@tf.function
def get_charges(e, s, Q):
    """ Solve the function to get the absolute charges of atoms in a
    molecule from parameters.

    Parameters
    ----------
    e : tf.Tensor, dtype = tf.float32, shape = (34, ),
        electronegativity.
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


@tf.function
def get_q_i_hat_total_per_mol(e, s, Qs, attr_in_mol):
    """ Calculate the charges per molecule based on
    `attr_in_mol`.

    """
    attr_in_mol.set_shape([None, None])

    attr_in_mol = tf.boolean_mask(
        attr_in_mol,
        tf.reduce_any(
            attr_in_mol,
            axis=1),
        axis=0)

    attr_in_mol = tf.boolean_mask(
        attr_in_mol,
        tf.reduce_any(
            attr_in_mol,
            axis=0),
    axis=1)

    q_i = tf.tile(
        tf.expand_dims(
            tf.constant(
                0,
                dtype=tf.float32),
            0),
        [tf.shape(attr_in_mol, tf.int64)[0]])

    def loop_body(q_i, idx,
            e=e,
            s=s,
            Qs=Qs,
            attr_in_mol=attr_in_mol):

        # get attr
        _attr_in_mol = attr_in_mol[:, idx]

        # get the attributes of each molecule
        _Qs = Qs[idx]

        _e = tf.boolean_mask(
            e,
            _attr_in_mol)

        _s = tf.boolean_mask(
            s,
            _attr_in_mol)

        _idxs = tf.where(_attr_in_mol)

        # update
        q_i = tf.tensor_scatter_nd_update(
            q_i,

            # idxs
            _idxs,

            # update
            tf.reshape(
                    get_charges(
                        _e,
                        _s,
                        _Qs),
                [-1]))

        return q_i, tf.add(idx, tf.constant(1, dtype=tf.int64))

    idx = tf.constant(0, dtype=tf.int64)

    # loop_body(q_i, idx)


    q_i, idx = tf.while_loop(
        lambda _, idx: tf.less(
            idx,
            tf.shape(attr_in_mol, tf.int64)[1]),

        loop_body,

        [q_i, idx])


    return q_i

@tf.function
def get_q_total_per_mol(q_i, attr_in_mol):
    # attr_in_mol.set_shape([None, None])

    q_i = tf.boolean_mask(
        q_i,
        tf.reduce_any(
            attr_in_mol,
            axis=1))

    attr_in_mol = tf.boolean_mask(
        attr_in_mol,
        tf.reduce_any(
            attr_in_mol,
            axis=1),
        axis=0)

    attr_in_mol = tf.boolean_mask(
        attr_in_mol,
        tf.reduce_any(
            attr_in_mol,
            axis=0),
    axis=1)

    attr_in_mol = tf.where(
        attr_in_mol,

        tf.ones_like(
            attr_in_mol,
            dtype=tf.float32),

        tf.zeros_like(
            attr_in_mol,
            dtype=tf.float32))

    q_per_mol = tf.reduce_sum(
        tf.multiply(
            attr_in_mol,
            tf.tile(
                tf.expand_dims(
                        q_i,
                        1),
                [
                    1,
                    tf.shape(attr_in_mol, tf.int64)[1]
                ])),
        axis=0)

    return q_per_mol

oe_mols = gin.i_o.utils.file_to_oemols('/home/chodera/charge-datasets/datasets/riniker/ChEMBL_AM1BCC.oeb')
oe_mol_dicts = []
for oe_mol in oe_mols:
    try:
        oe_mol_dict =\
            gin.i_o.utils.oemol_to_dict(oe_mol, read_wbo=True)
        oe_mol_dicts.append(oe_mol_dict)
    except:
        continue

# oe_mol_dicts = [gin.i_o.utils.oemol_to_dict(oe_mol, wbo=True) for oe_mol in oe_mols]
n_samples = len(oe_mol_dicts)
ds_idxs = tf.data.Dataset.from_tensor_slices(
    tf.expand_dims(
        tf.convert_to_tensor(
            list(range(n_samples))),
        1))

def read_one_mol(idx):
    atoms = oe_mol_dicts[int(idx.numpy())]['atomic_symbols']
    atoms = tf.expand_dims(tf.convert_to_tensor(
            atoms,
            tf.string),
        1)
    atoms = tf.cast(
        tf.map_fn(
            lambda x: TRANSLATION[x.numpy()[0]],
            atoms,
            tf.int32),
        tf.int64)

    atoms = tf.reshape(
        atoms,
        [-1])

    n_atoms = tf.shape(atoms, tf.int64)[0]

    bonds = tf.convert_to_tensor(
        oe_mol_dicts[int(idx.numpy())]['connectivity'],
        dtype=tf.float32)

    adjacency_map = tf.zeros(
        (n_atoms, n_atoms),
        tf.float32)

    adjacency_map = tf.tensor_scatter_nd_update(
        adjacency_map,

        tf.cast(
            bonds[:, :2],
            tf.int64),

        bonds[:, 2])

    adjacency_map = gin.i_o.utils.conjugate_average(atoms, adjacency_map)

    charges = tf.convert_to_tensor(
        oe_mol_dicts[int(idx.numpy())]['partial_charges'],
        tf.float32)

    return atoms, adjacency_map, charges

ds_mols = ds_idxs.map(
    lambda idx: tf.py_function(
        read_one_mol,
        [idx],
        [tf.int64, tf.float32, tf.float32])).shuffle(
        n_samples,
        seed=2666)


ds_all = gin.probabilistic.gn.GraphNet.batch(
    ds_mols, 256, per_atom_attr=True).cache(
    str(os.getcwd()) + '/temp')


# get the number of samples
# NOTE: there is no way to get the number of samples in a dataset
# except loop through one time, unfortunately
n_batches = gin.probabilistic.gn.GraphNet.get_number_batches(
    ds_all)

n_batches = int(n_batches)
n_global_te = int(0.2 * n_batches)
ds_tr = ds_all.skip(n_global_te)
ds_te = ds_all.take(n_global_te)


point = {
    'D_V': 32,
    'D_E': 32,
    'D_U': 64,

    'd_sigma_units': 16,
    'd_pi_units': 16,

    'phi_v_units': 64,
    'phi_v_activation': 'sigmoid',

    'phi_e_units': 32,
    'phi_e_activation': 'leaky_relu',

    'phi_u_units': 32,
    'phi_u_activation': 'tanh',

    'f_r_units': 64,

    'learning_rate': 1e-3
}

def init(point):
    global gn
    global optimizer

    class f_v(tf.keras.Model):
        """ Featurization of nodes.
        Here we simply featurize atoms using one-hot encoding.

        """
        def __init__(self, units=point['D_V']):
            super(f_v, self).__init__()
            self.d = tf.keras.layers.Dense(units)

        @tf.function
        def call(self, x):
            x = tf.one_hot(x, 8)
            # set shape because Dense doesn't like variation
            x.set_shape([None, 8])
            return self.d(x)

    class f_e(tf.keras.Model):
        """ Featurization of edges.
        Here we split the $\sigma$ and $\pi$ component of bonds
        into two channels, and featurize them seperately.

        """
        def __init__(
                self,
                d_sigma_units=point['d_sigma_units'],
                d_pi_units=point['d_pi_units'],
                D_E=point['D_E']):

            super(f_e, self).__init__()
            self.D_E = D_E

            # sigma
            self.d_sigma_0 = tf.Variable(
                tf.zeros(
                    shape=(1, d_sigma_units),
                    dtype=tf.float32))
            self.d_sigma_1 = tf.keras.layers.Dense(
                int(self.D_E // 2))

            # pi
            self.d_pi_0 = tf.keras.layers.Dense(
                d_pi_units)
            self.d_pi_1 = tf.keras.layers.Dense(
                int(self.D_E // 2))

        @tf.function
        def call(self, x):
            # determine whether there is $\pi$ component in the bond
            has_pi = tf.greater(
                x,
                tf.constant(1, dtype=tf.float32))

            # calculate the sigma component of the bond
            x_sigma = tf.tile(
                self.d_sigma_1(self.d_sigma_0),
                [tf.shape(x, tf.int64)[0], 1])

            # calculate the pi component of the bond
            x_pi = tf.where(
                has_pi,

                # if has pi:
                self.d_pi_1(
                    self.d_pi_0(
                        tf.math.subtract(
                            x,
                            tf.constant(1, dtype=tf.float32)))),

                # else:
                tf.zeros(
                    shape=(self.D_E // 2, ),
                    dtype=tf.float32))

            x = tf.concat(
                [
                    x_sigma,
                    x_pi
                ],
                axis=1)

            return x

    f_u=(lambda atoms, adjacency_map, batched_attr_mask: \
        tf.tile(
            tf.zeros((1, point['D_U'])),
            [
                 tf.math.count_nonzero(
                     tf.reduce_any(
                         batched_attr_mask,
                         axis=0)),
                 1
            ]
        ))

    phi_v = lime.nets.for_gn.ConcatenateThenFullyConnect(
        (
            point['phi_v_units'],
            point['phi_v_activation'],
            point['phi_v_units'],
            point['D_V']
        ))


    phi_e = lime.nets.for_gn.ConcatenateThenFullyConnect(
        (
            point['phi_e_units'],
            point['phi_e_activation'],
            point['phi_e_units'],
            point['D_E']
        ))

    class phi_u(tf.keras.Model):
        def __init__(self, config=(
                    point['phi_u_units'],
                    point['phi_u_activation'],
                    point['phi_u_units'],
                    point['D_U']
                )):
            super(phi_u, self).__init__()
            self.d = lime.nets.for_gn.ConcatenateThenFullyConnect(config)

        @tf.function
        def call(self, h_u, h_u_0, h_e_bar, h_v_bar):
            return self.d(h_u, h_u_0, h_e_bar, h_v_bar)


    class f_r(tf.keras.Model):
        """ Readout function.
        """

        def __init__(self, units=point['f_r_units']):
            super(f_r, self).__init__()
            self.d_e_0 = tf.keras.layers.Dense(units)
            self.d_s_0 = tf.keras.layers.Dense(units)
            self.d_e_1 = tf.keras.layers.Dense(1)
            self.d_s_1 = tf.keras.layers.Dense(1)

        @tf.function
        def call(self,
                h_e, h_v, h_u,
                h_e_history, h_v_history, h_u_history,
                atom_in_mol, bond_in_mol):

            # although this could take many many arguments,
            # we only take $h_e$ for now
            e = self.d_e_1(self.d_e_0(h_v))
            s = self.d_s_1(self.d_s_0(h_v))

            return e, s

    gn = gin.probabilistic.gn.GraphNet(
        f_e=f_e(),
        f_v=f_v(),
        f_u=f_u,
        phi_e=phi_e,
        phi_v=phi_v,
        phi_u=phi_u(),
        f_r=f_r(),
        repeat=5)

    optimizer = tf.keras.optimizers.Adam(point['learning_rate'])

init(point)

for dummy_idx in range(N_EPOCHS):
    for atoms, adjacency_map, \
        atom_in_mol, bond_in_mol, q_i, attr_in_mol \
        in ds_tr:
        with tf.GradientTape() as tape:
            Qs = get_q_total_per_mol(q_i, attr_in_mol)

            e, s = gn(
                atoms, adjacency_map,
                atom_in_mol, bond_in_mol, attr_in_mol)

            e = tf.boolean_mask(
                e,
                tf.reduce_any(
                    attr_in_mol,
                    axis=1))

            s = tf.boolean_mask(
                s,
                tf.reduce_any(
                    attr_in_mol,
                    axis=1))

            q_i_hat = get_q_i_hat_total_per_mol(
                                e, s, Qs, attr_in_mol)

            q_i = tf.boolean_mask(
                q_i,
                tf.reduce_any(
                    attr_in_mol,
                    axis=1))

            loss = tf.losses.mean_squared_error(
                q_i,
                q_i_hat)

        variables = gn.variables
        grad = tape.gradient(loss, variables)
        optimizer.apply_gradients(
            zip(grad, variables))




gn.save_weights('partial_charge_weights')
